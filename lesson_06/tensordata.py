
import torch
from torch.utils.data import TensorDataset

# Create input data: number of study hours
X = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0]
])

# Create target data: actual exam scores
y = torch.tensor([
    [20.0],
    [40.0],
    [60.0],
    [80.0],
    [100.0]
])

# Combine inputs and targets into one dataset
dataset = TensorDataset(X, y)

# Find the total number of examples
print("Dataset size:", len(dataset))

# Get the first example from the dataset
print("First example:", dataset[0])
