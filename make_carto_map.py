import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import torch
import csv
import matplotlib.pyplot as plt
import json

NUM_PREPROCESSING_WORKERS = 2

def main():
    # Load pretrained model checkpoint
    model_checkpoint = "trained_model/checkpoint-206013"
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    # Load custom dataset
    dataset_path = "snli_validation_examples_modified.jsonl"
    dataset = datasets.load_dataset("json", data_files=dataset_path)["train"]

    # Preprocess dataset
    def prepare_dataset(example):
        tokenized = tokenizer(
            example["premise"],
            example["hypothesis"],
            max_length=128,
            truncation=True,
            padding="max_length",
        )
        tokenized["label"] = example["label"]
        return tokenized

    processed_dataset = dataset.map(prepare_dataset, batched=False)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        do_train=True,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        save_strategy="no",
        logging_dir="./logs",
        logging_steps=50,
        eval_strategy="no",
        save_total_limit=1,
    )

    # Preallocate probabilities tensor
    num_examples = len(processed_dataset)
    correct_label_probs = np.zeros((num_examples, 3), dtype=np.float32)  # 3 is the number of classes
    example_index = 0

    def compute_full_label_probs(logits):
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()  # Convert logits to probabilities
        return probs

    # Custom trainer
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            nonlocal example_index
            outputs = model(**inputs)
            logits = outputs.logits
            probs = compute_full_label_probs(logits.detach().cpu().numpy())

            labels = inputs["labels"].cpu().numpy()
            for i in range(len(labels)):
                if example_index + i < correct_label_probs.shape[0]:
                    correct_label_probs[example_index + i, labels[i]] = probs[i, labels[i]]

            example_index = (example_index + len(labels)) % correct_label_probs.shape[0]
            loss = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
            return (loss, outputs) if return_outputs else loss

    # Initialize Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        tokenizer=tokenizer,
    )



    # Train the model
    trainer.train()

    # Calculate statistics
    mean_probs = np.mean(correct_label_probs, axis=1)
    std_devs = np.std(correct_label_probs, axis=1)

    # Categorize data points
    hard_to_learn = (mean_probs < 0.4) & (std_devs < 0.28)
    easy_to_learn = (mean_probs > 0.7) & (std_devs < 0.28)
    ambiguous = std_devs >= 0.28

    # Find Index and return sentences into .jsonl file
    # Save ambiguous points to a .jsonl file
    ambiguous_points = []
    for i in range(len(ambiguous)):
        if ambiguous[i]:
            ambiguous_points.append(dataset[i])  # Append the original data point

    output_path = "ambiguous_data_points.jsonl"
    with open(output_path, "w") as f:
        for point in ambiguous_points:
            f.write(json.dumps(point) + "\n")

    print(f"Ambiguous data points saved to '{output_path}'.")

    # Plotting the scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(std_devs[hard_to_learn], mean_probs[hard_to_learn], c='blue', label='Hard-to-Learn', alpha=0.6)
    plt.scatter(std_devs[easy_to_learn], mean_probs[easy_to_learn], c='red', label='Easy-to-Learn', alpha=0.6)
    plt.scatter(std_devs[ambiguous], mean_probs[ambiguous], c='green', label='Ambiguous', alpha=0.6)

    # Add labels to regions
    bb = lambda c: dict(boxstyle="round,pad=0.3", ec=c, lw=2, fc="white")
    plt.text(0.02, 0.85, "easy-to-learn", fontsize=12, bbox=bb('r'))
    plt.text(0.02, 0.18, "hard-to-learn", fontsize=12, bbox=bb('b'))
    plt.text(0.31, 0.5, "ambiguous", fontsize=12, bbox=bb('g'))

    plt.xlabel("Variability")
    plt.ylabel("Confidence")
    plt.title("SNLI-ELECTRA-small Data Map  ")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
