# -*- coding: utf-8 -*-


!pip install datasets transformers peft

!huggingface-cli login

# loading librarires
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch

# Load the dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

dataset["train"]

# Load the model and tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

# Tokenization function for summarization
def preprocess_function(examples):
    inputs = [ex for ex in examples["article"]]
    targets = [ex for ex in examples["highlights"]]

    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding=True)
    labels = tokenizer(targets, max_length=150, truncation=True, padding=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply the tokenization to the dataset
training_datasets = dataset["train"].map(preprocess_function, batched=True)

eval_datasets = dataset["test"].map(preprocess_function, batched=True)

from transformers import Trainer, TrainingArguments

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=eval_datasets,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()