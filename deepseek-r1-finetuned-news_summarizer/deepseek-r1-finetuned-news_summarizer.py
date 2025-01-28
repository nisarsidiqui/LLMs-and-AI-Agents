# -*- coding: utf-8 -*-

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from trl import SFTTrainer
from evaluate import load
import numpy as np
from huggingface_hub import HfFolder

# Set your Hugging Face token
HF_TOKEN = "your_token_here"  # Replace with your token
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
OUTPUT_MODEL_NAME = "your-username/deepseek-summarizer"  # Replace with your desired model name

def compute_metrics(eval_preds):
    rouge = load("rouge")
    
    predictions, labels = eval_preds
    # SFTTrainer outputs raw logits, need to get predicted token ids
    predictions = predictions.argmax(axis=-1)
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(pred.split()) for pred in decoded_preds]
    decoded_labels = ["\n".join(label.split()) for label in decoded_labels]
    
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    return {k: round(v * 100, 4) for k, v in result.items()}

def prepare_dataset():
    # Load a smaller subset of the dataset for faster training
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    
    # Take a smaller subset for training
    train_dataset = dataset["train"].select(range(1000))  # Adjust size as needed
    val_dataset = dataset["validation"].select(range(100))
    
    return train_dataset, val_dataset

def format_prompt(example):
    return f"""Summarize the following article:

    Article: {example['article']}

    Summary: {example['highlights']}"""


def train_model():
    # Login to Hugging Face
    HfFolder.save_token(HF_TOKEN)

    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Load base model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Configure LoRA
    lora_config = LoraConfig(
        r=16,  # rank
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters info
    model.print_trainable_parameters()
    
    # Prepare dataset
    train_dataset, val_dataset = prepare_dataset()
    
    # Preprocess datasets
    tokenized_train = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    tokenized_val = val_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        learning_rate=2e-4,
        fp16=True,
        push_to_hub=True,
        hub_model_id=OUTPUT_MODEL_NAME,
        optim="paged_adamw_32bit"  # Memory efficient optimizer
    )
    
    # Initialize SFT trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        formatting_func=format_prompt,
        tokenizer=tokenizer,
        max_seq_length=512,
        compute_metrics=compute_metrics,
        packing=False,
    )
    
    # Train the model
    trainer.train()
    
        # Save the LoRA adapter
    trainer.model.save_pretrained("./lora_adapter")
    tokenizer.save_pretrained("./lora_adapter")

def merge_and_push_model():
    """Separate function for merging and pushing the model"""
    print("Starting model merge process...")
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model, 
        "./lora_adapter",
        device_map="auto"
    )
    
    print("Merging weights...")
    merged_model = model.merge_and_unload()
    
    print("Saving merged model...")
    merged_model.save_pretrained(
        "./merged_model",
        safe_serialization=True,
        max_shard_size="500MB"  # Split into smaller files
    )
    
    print("Pushing to Hub...")
    merged_model.push_to_hub(OUTPUT_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.push_to_hub(OUTPUT_MODEL_NAME)
    
    print("Merge and push completed!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--merge', action='store_true', help='Run merge process instead of training')
    args = parser.parse_args()
    
    if args.merge:
        merge_and_push_model()
    else:
        train_model()