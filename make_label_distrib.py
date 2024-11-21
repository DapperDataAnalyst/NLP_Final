import json
from collections import Counter
import matplotlib.pyplot as plt

# Define a function to count labels in a file
def count_labels(file_path):
    label_counter = Counter()
    with open(file_path, 'r') as file:
        for line in file:
            # Parse the dictionary
            example = json.loads(line.strip())
            # Increment the label counter
            label_counter[example['label']] += 1
    return label_counter

# File paths
file1 = 'snli_validation_examples.txt'
file2 = 'snli_validation_examples_modified.txt'

# Count labels for each file
counts_file1 = count_labels(file1)
counts_file2 = count_labels(file2)

# Display results
print("Label counts in file 1:")
for label in range(0, 3):  # Labels 0, 1, 2
    print(f"Label {label}: {counts_file1.get(label, 0)}")

print("\nLabel counts in file 2:")
for label in range(0, 3):  # Labels 0, 1, 2
    print(f"Label {label}: {counts_file2.get(label, 0)}")

# Prepare data for visualization
labels = ['Label 0', 'Label 1', 'Label 2']
counts1 = [counts_file1.get(i, 0) for i in range(3)]
counts2 = [counts_file2.get(i, 0) for i in range(3)]

# Create bar chart
x = range(len(labels))  # Position of bars

plt.figure(figsize=(8, 5))
plt.bar(x, counts1, width=0.4, label='Original', color='skyblue', align='center')
plt.bar([pos + 0.4 for pos in x], counts2, width=0.4, label='Contrast', color='orange', align='center')

# Customize chart
plt.xticks([pos + 0.2 for pos in x], labels)  # Add labels at center of grouped bars
plt.xlabel('Labels')
plt.ylabel('Counts')
plt.title('Label Counts')
plt.legend()

# Show plot
plt.tight_layout()
plt.show()
