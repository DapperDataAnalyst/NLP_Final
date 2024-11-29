import json
from collections import defaultdict

# File paths
file_b_path = '.\eval_output\mismatched_predictions.jsonl'  # Replace with actual path to File A
file_a_path = '.\eval_output\snli_validation_examples_modified_withlabel_change.jsonl'  # Replace with actual path to File B

# Load File A into a dictionary with premises as keys
file_a_data = {}
with open(file_a_path, 'r', encoding='utf-8') as file_a:
    for line in file_a:
        # Split the line to separate the comment
        json_part, comment = line.rsplit("//", 1)
        data = json.loads(json_part.strip())
        file_a_data[data["premise"]] = comment.strip()

# Count matches for each comment type
comment_counts = defaultdict(int)

# Process File B
with open(file_b_path, 'r', encoding='utf-8') as file_b:
    for line in file_b:
        data = json.loads(line.strip())
        premise = data.get("premise")
        if premise in file_a_data:
            comment = file_a_data[premise]
            comment_counts[comment] += 1

# Display the results
print("Counts of matches by comment type:")
for comment, count in comment_counts.items():
    print(f"{comment}: {count}")
