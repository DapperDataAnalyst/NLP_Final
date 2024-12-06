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

# Total samples in the final dataset
total_samples = 3000

# Percentage composition (adjust these as needed)
hard_percentage = 20  # % of hard examples
ambiguous_percentage = 60  # % of ambiguous examples
easy_percentage = 20  # % of easy examples

# Calculate number of samples for each category
num_hard = int((hard_percentage / 100) * total_samples)
num_ambiguous = int((ambiguous_percentage / 100) * total_samples)
num_easy = int((easy_percentage / 100) * total_samples)

# Ensure the total adds up to `total_samples`
if num_hard + num_ambiguous + num_easy != total_samples:
    raise ValueError("The percentages do not add up to 100%. Please adjust them.")

# Randomly select required number of lines
selected_hard = random.sample(hard_data, min(num_hard, len(hard_data)))
selected_ambiguous = random.sample(ambiguous_data, min(num_ambiguous, len(ambiguous_data)))
selected_easy = random.sample(easy_data, min(num_easy, len(easy_data)))

# Combine all selected lines
new_train_dataset = selected_hard + selected_ambiguous + selected_easy

# Remove "idx" field from each item
for item in new_train_dataset:
    if "idx" in item:
        del item["idx"]

# Save the cleaned dataset to a new JSONL file
save_jsonl(output_path, new_train_dataset)

print(f"New train dataset saved to '{output_path}' with {len(new_train_dataset)} total lines.")
