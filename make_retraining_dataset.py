import json
import random

# File paths
hard_data_path = "hard_data_points.jsonl"
ambiguous_data_path = "ambiguous_data_points.jsonl"
easy_data_path = "easy_data_points.jsonl"
output_path = "new_train_dataset.jsonl"

# Load JSONL files into lists
def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]

# Save a list of dictionaries to a JSONL file
def save_jsonl(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

# Load data
hard_data = load_jsonl(hard_data_path)
ambiguous_data = load_jsonl(ambiguous_data_path)
easy_data = load_jsonl(easy_data_path)

# Randomly select required number of lines
selected_hard = random.sample(hard_data, min(900, len(hard_data)))
selected_ambiguous = random.sample(ambiguous_data, min(1500, len(ambiguous_data)))
selected_easy = random.sample(easy_data, min(600, len(easy_data)))

# Combine all selected lines
new_train_dataset = selected_hard + selected_ambiguous + selected_easy

# Save the combined dataset to a new JSONL file
save_jsonl(output_path, new_train_dataset)

print(f"New train dataset saved to '{output_path}' with {len(new_train_dataset)} total lines.")

