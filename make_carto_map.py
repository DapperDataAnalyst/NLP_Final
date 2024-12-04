import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import torch
import csv
import matplotlib.pyplot as plt
import json
import seaborn as sns

import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
NUM_PREPROCESSING_WORKERS = 2

def main():
    # Load pretrained model checkpoint
    model_checkpoint = "trained_model/checkpoint-206013"
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    # Load SNLI training dataset
    dataset = datasets.load_dataset("snli")["train"]
    dataset = dataset.filter(lambda ex: ex['label'] != -1)  # Remove unlabeled examples
    # Load custom dataset
    #dataset_path = "snli_validation_examples.jsonl"
    #dataset = datasets.load_dataset("json", data_files=dataset_path)["train"]

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
        # Convert logits to torch tensor
        logits = torch.tensor(logits)
        # Stabilize logits by subtracting the maximum logit value 
        logits = logits - torch.max(logits, dim=-1, keepdim=True).values
        # Apply softmax to compute probabilities
        probs = torch.softmax(logits, dim=-1).numpy()
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
    hard_to_learn = (mean_probs < 0.30)
    easy_to_learn = (mean_probs > 0.65) 
    ambiguous = (mean_probs >= 0.30) & (mean_probs <= 0.65) & (std_devs > 0.25)

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

    # Define correctness levels and custom bins
    custom_correctness_levels = [0.0, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0]  # Explicit correctness levels
    correctness_labels = np.digitize(mean_probs, custom_correctness_levels) - 1  # Map mean_probs to custom bins

    # Define marker styles for correctness levels
    markers = ['o', '*', 's', 'd', '^', 'v', 'P']  # Circle, star, square, diamond, triangle up, triangle down, plus
    colors = {'hard_to_learn': 'blue', 'easy_to_learn': 'red', 'ambiguous': 'green'}

    # Create legend entries for custom correctness levels
    legend_entries = []
    for level, marker in zip(custom_correctness_levels, markers):
        legend_entries.append(
            mlines.Line2D(
                [], [], color='black', marker=marker, linestyle='None', markersize=8, label=f"{level:.1f}"
            )
        )

    # Plot each group (hard-to-learn, easy-to-learn, ambiguous) with custom correctness levels
    for group, mask, label_color in [
        ('hard_to_learn', hard_to_learn, colors['hard_to_learn']),
        ('easy_to_learn', easy_to_learn, colors['easy_to_learn']),
        ('ambiguous', ambiguous, colors['ambiguous']),
    ]:
        for i, marker in enumerate(markers):
            group_mask = (correctness_labels == i) & mask
            plt.scatter(
                std_devs[group_mask],
                mean_probs[group_mask],
                c=label_color,
                marker=marker,
                alpha=0.6
            )

    # Add the custom legend for correctness levels
    plt.legend(
        handles=legend_entries,
        title="correct.",
        loc="upper right",
        fontsize=10,
        title_fontsize=11,
        frameon=True,
        shadow=False,
        edgecolor="black"
    )

    # Highlight KDE for ambiguous points (optional)
    sns.kdeplot(x=std_devs[ambiguous], y=mean_probs[ambiguous], cmap="Greens", fill=True, alpha=0.3)

    # Add labels to regions
    bb = lambda c: dict(boxstyle="round,pad=0.3", ec=c, lw=2, fc="white")
    plt.text(0.02, 0.85, "easy-to-learn", fontsize=12, bbox=bb('red'))
    plt.text(0.02, 0.18, "hard-to-learn", fontsize=12, bbox=bb('blue'))
    plt.text(0.31, 0.5, "ambiguous", fontsize=12, bbox=bb('green'))

    # Add labels, grid, and title
    plt.xlabel("Variability (Standard Deviation)", fontsize=12)
    plt.ylabel("Confidence (Mean Probability)", fontsize=12)
    plt.title("SNLI-ELECTRA-small Data Map", fontsize=14)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
