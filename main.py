import pandas as pd
import torch
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from tqdm import tqdm

# Load MRPC dataset
def load_mrpc_dataset():
    dataset = load_dataset("glue", "mrpc")
    return dataset

# Load the tokenizer
def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

# Tokenize the datasets
def tokenize_dataset(dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

# Prepare the DataLoader
def prepare_dataloader(tokenized_dataset, batch_size):
    train_dataloader = DataLoader(tokenized_dataset["train"], shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(tokenized_dataset["validation"], batch_size=batch_size)
    return train_dataloader, eval_dataloader

# Load model
def load_model(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    return model

# Fine-tune the model
def fine_tune_model(model, train_dataloader, eval_dataloader, learning_rate, num_epochs, scheduler_lambda):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = LambdaLR(optimizer, lr_lambda=scheduler_lambda)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            outputs = model(**{k: v.to(model.device) for k, v in batch.items()})
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

        evaluate_model(model, eval_dataloader)

# Evaluate model
def evaluate_model(model, eval_dataloader):
    metric = load_metric("glue", "mrpc")
    model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**{k: v.to(model.device) for k, v in batch.items()})
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    final_score = metric.compute()
    print(f"Validation accuracy: {final_score['accuracy']}")

# Main function to execute the above steps
def main():
    model_name = "microsoft/deberta-v3-base"
    dataset = load_mrpc_dataset()
    tokenizer = load_tokenizer(model_name)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    train_dataloader, eval_dataloader = prepare_dataloader(tokenized_dataset, batch_size=16)
    model = load_model(model_name)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Define a lambda function for the scheduler
    scheduler_lambda = lambda step: 1

    # Fine-tuning the model
    fine_tune_model(model, train_dataloader, eval_dataloader, learning_rate=2e-5, num_epochs=3, scheduler_lambda=scheduler_lambda)

if __name__ == "__main__":
    main()
