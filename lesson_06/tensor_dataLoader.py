import torch
from torch.utils.data import TensorDataset, DataLoader

# Create five input examples
X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])

# Create their corresponding target values
y = torch.tensor([[20.0], [40.0], [60.0], [80.0], [100.0]])

# Combine inputs and targets
dataset = TensorDataset(X, y)

# Create a DataLoader that returns two examples per batch
loader = DataLoader(dataset, batch_size=2, shuffle=False)

# Loop through the batches
for batch_X, batch_y in loader:

    # Display the inputs in the current batch
    print("Inputs:", batch_X)

    # Display their corresponding targets
    print("Targets:", batch_y)

    # Separate batches for easier reading
    print("--------------")