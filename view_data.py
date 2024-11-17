from datasets import load_dataset

# Load the SQuAD dataset
squad_dataset = load_dataset('squad')

# Print a sample from the train split
print("Sample from the training set:")
print(squad_dataset['train'][0])

# Print a sample from the validation split
print("\nSample from the validation set:")
print(squad_dataset['validation'][0])

print("Train dataset size:", len(squad_dataset['train']))
print("Validation dataset size:", len(squad_dataset['validation']))