from datasets import load_dataset
import json
import random

# Load the SNLI dataset
dataset = load_dataset('snli')

# Select the validation set
validation_data = dataset['validation']

# Number of examples to extract
num_examples = 600

# Ensure reproducibility with a fixed random seed
random.seed(42)

# Randomly select examples
random_indices = random.sample(range(len(validation_data)), num_examples)
random_examples = [validation_data[i] for i in random_indices]

# Open a text file for writing
with open('snli_validation_examples.txt', 'w', encoding='utf-8') as f:
    # Write each randomly selected example in the specified format
    for example in random_examples:
        example_dict = {
            'premise': example['premise'],
            'hypothesis': example['hypothesis'],
            'label': example['label']  # Labels: 0 = entailment, 1 = neutral, 2 = contradiction
        }
        # Convert the dictionary to a JSON string and write it to the file
        f.write(json.dumps(example_dict, ensure_ascii=False) + '\n')

print(f"Saved {num_examples} random examples to 'snli_validation_examples.txt'")
