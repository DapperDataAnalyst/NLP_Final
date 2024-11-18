from datasets import load_dataset
import json

# Load the SNLI dataset
dataset = load_dataset('snli')

# Select the validation set
validation_data = dataset['validation']

# Number of examples to extract
num_examples = 600

# Open a text file for writing
with open('snli_validation_examples.txt', 'w', encoding='utf-8') as f:
    # Extract and write each example in the specified format
    for i in range(min(num_examples, len(validation_data))):
        example = validation_data[i]
        example_dict = {
            'premise': example['premise'],
            'hypothesis': example['hypothesis'],
            'label': example['label']  # Labels: 0 = entailment, 1 = neutral, 2 = contradiction
        }
        # Convert the dictionary to a JSON string and write it to the file
        f.write(json.dumps(example_dict, ensure_ascii=False) + '\n')

print(f"Saved {num_examples} examples to 'snli_validation_examples.txt'")
