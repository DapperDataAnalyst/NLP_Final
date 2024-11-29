# Load the JSONL file and filter lines where "label" does not match "predicted_label"
import json

# Path to the JSONL file
file_path = '.\eval_output\eval_predictions.jsonl'

# Open and process the file
mismatched_lines = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        if data.get("label") != data.get("predicted_label"):
            mismatched_lines.append(data)

# Path to the output JSONL file for mismatched lines
output_file_path = '.\eval_output\mismatched_predictions.jsonl'

# Process the file again and write mismatched lines to a new JSONL file
with open(file_path, 'r', encoding='utf-8') as file, open(output_file_path, 'w', encoding='utf-8') as output_file:
    for line in file:
        data = json.loads(line)
        if data.get("label") != data.get("predicted_label"):
            output_file.write(json.dumps(data) + '\n')

# Return the output file path for download
output_file_path


# Display the mismatched lines
#print(len(mismatched_lines))
#print(mismatched_lines[:5])  # Display total count and a sample of mismatched lines
