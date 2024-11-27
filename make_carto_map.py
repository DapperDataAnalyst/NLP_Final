import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import torch
import csv
import matplotlib.pyplot as plt  # Import matplotlib for plotting

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
        eval_strategy="no",  # Disable evaluation
        save_total_limit=1,
    )

    # Preallocate probabilities tensor
    num_examples = len(processed_dataset)
    correct_label_probs = np.zeros((num_examples, 3), dtype=np.float32)  # 3 is the number of classes
    example_index = 0  # Tracks position in the dataset

    def compute_full_label_probs(logits):
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()  # Convert logits to probabilities
        return probs

    # Custom trainer
    class CustomTrainer(Trainer):
        def on_epoch_end(self, args, state, control, logs=None):
            # Print the probabilities of the correct label for all rows at the end of the epoch
            for i in range(len(correct_label_probs)):
                correct_label = processed_dataset[i]["label"]  # Get the label from the dataset
                prob_of_correct_label = correct_label_probs[i, correct_label]  # Get the probability of the correct label
                print(f"Example {i}: Correct Label = {correct_label}, Probability = {prob_of_correct_label:.4f}")

        def compute_loss(self, model, inputs, return_outputs=False):
            nonlocal example_index
            # Compute outputs and logits
            outputs = model(**inputs)
            logits = outputs.logits
            probs = compute_full_label_probs(logits.detach().cpu().numpy())

            # Store the probability of the correct label in the correct_label_probs array
            labels = inputs["labels"].cpu().numpy()  # Ensure the tensor is on CPU
            for i in range(len(labels)):
                if example_index + i < correct_label_probs.shape[0]:
                    correct_label_probs[example_index + i, labels[i]] = probs[i, labels[i]]
                else:
                    # Optionally, handle the situation when the index exceeds the array bounds
                    print(f"Warning: Index {example_index + i} out of bounds.")
            
            # Update the example index
            example_index = (example_index + len(labels)) % correct_label_probs.shape[0]

            # Compute loss using the superclass method
            loss = super().compute_loss(model, inputs, return_outputs=return_outputs)
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

    # Validate shape
    print(f"Shape of correct_label_probs: {correct_label_probs.shape}")

    # Calculate statistics
    mean_probs = np.mean(correct_label_probs, axis=1)
    std_devs = np.std(correct_label_probs, axis=1)

    print(f"Mean Probabilities Across Classes: {mean_probs}")
    print(f"Standard Deviations Across Classes: {std_devs}")

    # Create list of coordinate pairs (mean, std_dev)
    coordinate_pairs = list(zip(mean_probs, std_devs))

    # Save the coordinate pairs to a CSV file
    with open("mean_std_pairs.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Mean Probability", "Standard Deviation"])  # Header
        for pair in coordinate_pairs:
            writer.writerow(pair)  # Write each pair as a row

    print("Mean and standard deviation pairs saved to 'mean_std_pairs.csv'.")

    # Plotting the scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(std_devs, mean_probs, color='blue', alpha=0.6)  # Scatter plot
    # plt.title("Scatter Plot: Mean vs. Standard Deviation of Label Probabilities")
    plt.xlabel("Variability")
    plt.ylabel("Confidence")
    plt.grid(True)
    plt.show()  # Display the plot

if __name__ == "__main__":
    main()
