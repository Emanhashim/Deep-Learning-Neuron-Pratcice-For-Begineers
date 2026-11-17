import torch
import torch.nn as nn

# Build the network by listing layers in execution order
model = nn.Sequential(
    nn.Linear(2, 4),  # 2 input features → 4 outputs
    nn.ReLU(),        # Apply activation
    nn.Linear(4, 1)   # 4 inputs → 1 output
)

# Create two input examples
X = torch.tensor([
    [0.5, 0.8],
    [0.2, 0.4]
])

# Pass data through all layers automatically
predictions = model(X)

# Display predictions and their shape
print(predictions)
print("Shape:", predictions.shape)