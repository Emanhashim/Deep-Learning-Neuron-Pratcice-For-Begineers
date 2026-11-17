
import torch
import torch.nn as nn

# Import tools for organizing and loading data
from torch.utils.data import TensorDataset, DataLoader

# Create the input data: study hours
X = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0]
])

# Create the expected exam scores
y = torch.tensor([
    [20.0],
    [40.0],
    [60.0],
    [80.0],
    [100.0]
])

# Combine inputs and expected outputs into one dataset
dataset = TensorDataset(X, y)

# Load two examples at a time and shuffle their order
train_loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

# Define a simple neural network
class ScorePredictor(nn.Module):

    def __init__(self):
        super().__init__()

        # One input feature and one output value
        self.layer = nn.Linear(1, 1)

    def forward(self, x):

        # Calculate the predicted score
        return self.layer(x)


# Create the model
model = ScorePredictor()

# Select Mean Squared Error as our loss function
loss_function = nn.MSELoss()

# Select Adam as our optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

# Train the model for 500 complete epochs
for epoch in range(500):

    # Process every batch in the training dataset
    for batch_X, batch_y in train_loader:

        # Generate predictions for the current batch
        predictions = model(batch_X)

        # Measure the difference between predictions and targets
        loss = loss_function(predictions, batch_y)

        # Clear gradients from the previous batch
        optimizer.zero_grad()

        # Calculate gradients using backpropagation
        loss.backward()

        # Update the model's weights and biases
        optimizer.step()

    # Display the loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(
            "Epoch:", epoch + 1,
            "Loss:", round(loss.item(), 4)
        )

# Put the model into evaluation mode
model.eval()

# Create a new input: 6 study hours
new_input = torch.tensor([[6.0]])

# Disable gradient tracking because we are only predicting
with torch.no_grad():

    # Generate a prediction
    prediction = model(new_input)

# Display the prediction
print("Predicted score:", prediction.item())
