from datasets import load_dataset

# Load the SNLI dataset
dataset = load_dataset('snli')

# Select a split to view examples (e.g., 'train', 'validation', 'test')
split = 'validation'
snli_data = dataset[split]

# Extract and display some examples
num_examples = 5  # Number of examples to display

for i in range(num_examples):
    example = snli_data[i]
    premise = example['premise']
    hypothesis = example['hypothesis']
    label = example['label']  # Labels: 0 = entailment, 1 = neutral, 2 = contradiction

    # Print in a readable format
    print(f"Example {i + 1}:")
    print(f"Premise: {premise}")
    print(f"Hypothesis: {hypothesis}")
    print(f"Label: {label} ({['entailment', 'neutral', 'contradiction'][label]})")
    print("-" * 50)
