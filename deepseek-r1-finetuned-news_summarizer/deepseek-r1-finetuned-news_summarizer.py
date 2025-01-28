# -*- coding: utf-8 -*-

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import torch
from evaluate import load
import numpy as np
from huggingface_hub import HfFolder
from transformers import TrainingArguments

# Set your Hugging Face token
HF_TOKEN = "your_token_here"  # Replace with your token
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
OUTPUT_MODEL_NAME = "your-username/deepseek-summarizer"  # Replace with your desired model name

def compute_metrics(eval_pred):
    rouge = load("rouge")
    predictions, labels = eval_pred
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    return {k: round(v * 100, 4) for k, v in result.items()}

def prepare_dataset():
    # Load a smaller subset of the dataset for faster training
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    
    # Take a smaller subset for training
    train_dataset = dataset["train"].select(range(1000))  # Adjust size as needed
    val_dataset = dataset["validation"].select(range(100))
    
    return train_dataset, val_dataset

def format_prompt(article, summary=None):
    prompt = f"Summarize the following article:\n\nArticle: {article}\n\nSummary:"
    if summary:
        return prompt + f" {summary}"
    return prompt

def preprocess_function(examples):
    articles = examples["article"]
    summaries = examples["highlights"]
    
    # Format prompts
    prompts = [format_prompt(article) for article in articles]
    model_inputs = tokenizer(prompts, max_length=512, truncation=True, padding="max_length")
    
    # Format targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(summaries, max_length=128, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def train_model():
    # Login to Hugging Face
    HfFolder.save_token(HF_TOKEN)
    
    # Load tokenizer and model
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
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
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        gradient_accumulation_steps=4,
        push_to_hub=True,
        hub_model_id=OUTPUT_MODEL_NAME,
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    # Train the model
    trainer.train()
    
    # Push to Hub
    trainer.push_to_hub()

if __name__ == "__main__":
    train_model()
