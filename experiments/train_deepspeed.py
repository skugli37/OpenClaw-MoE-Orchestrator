import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import deepspeed
import os

def main():
    # Konfiguracija modela (mali model za demonstraciju u sandboxu)
    config = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=6,
        n_head=12
    )
    model = GPT2LMHeadModel(config)

    # Putanja do DeepSpeed konfiguracije
    ds_config_path = "ds_config.json"

    # Argumenti za trening
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        save_steps=100,
        save_total_limit=2,
        deepspeed=ds_config_path,
        fp16=True,
        logging_steps=10,
        report_to="none"
    )

    # Dummy podaci za demonstraciju brzine
    from torch.utils.data import Dataset
    class DummyDataset(Dataset):
        def __init__(self, size, seq_len):
            self.size = size
            self.seq_len = seq_len
            self.input_ids = torch.randint(0, 50257, (size, seq_len))
            self.labels = self.input_ids.clone()

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {"input_ids": self.input_ids[idx], "labels": self.labels[idx]}

    train_dataset = DummyDataset(100, 128)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    print("Starting DeepSpeed training...")
    trainer.train()
    print("Training complete.")

if __name__ == "__main__":
    main()
