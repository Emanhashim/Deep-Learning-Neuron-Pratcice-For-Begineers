
import torch
import torch.nn as nn

# Define our neural network class
class MyNeuralNetwork(nn.Module):

    # Set up the layers when the model is created
    def __init__(self):
        super().__init__()

        # Transform 2 input features into 4 hidden values
        self.hidden = nn.Linear(2, 4)

        # Apply ReLU to introduce nonlinearity
        self.activation = nn.ReLU()

        # Transform the 4 hidden values into 1 output
        self.output = nn.Linear(4, 1)

    # Define how information flows through the network
    def forward(self, x):

        # Pass input through the hidden linear layer
        x = self.hidden(x)

        # Apply ReLU to the hidden layer's output
        x = self.activation(x)

        # Pass the result through the output layer
        x = self.output(x)

        # Return the final prediction
        return x


# Create an instance of our neural network
model = MyNeuralNetwork()

# Create two examples, each with two input features
X = torch.tensor([
    [0.5, 0.8],
    [0.2, 0.4]
])

# Pass the examples through the model
predictions = model(X)

# Display the input data
print("Input shape:", X.shape)

# Display the final predictions
print("Prediction shape:", predictions.shape)

# Display the predictions
print("Predictions:", predictions)
