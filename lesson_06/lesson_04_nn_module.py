
# Import PyTorch
import torch

# Import the neural network module from PyTorch
import torch.nn as nn


# Create our own neural network class
# nn.Module is the base class for PyTorch neural networks
class MyNeuralNetwork(nn.Module):

    # This method initializes our neural network
    def __init__(self):

        # Initialize the parent nn.Module class
        # This allows PyTorch to track our layers and parameters
        super().__init__()

        # Create the first layer
        # 2 = number of input features
        # 4 = number of neurons in the hidden layer
        self.layer1 = nn.Linear(2, 4)

        # Create the second layer
        # 4 = input values received from the hidden layer
        # 1 = one output neuron
        self.layer2 = nn.Linear(4, 1)


    # Define how data moves through our neural network
    def forward(self, x):

        # Pass the input through the first linear layer
        # Each of the 4 neurons calculates weighted sum + bias
        x = self.layer1(x)

        # Apply ReLU activation
        # Negative values become zero
        # Positive values remain unchanged
        x = torch.relu(x)

        # Pass the hidden layer output into the final layer
        # This produces one output for each input example
        x = self.layer2(x)

        # Return the final prediction
        return x


# Create an instance of our neural network
model = MyNeuralNetwork()


# Create two training examples
# Each example has 2 input features
X = torch.tensor([
    [0.5, 0.8],
    [0.2, 0.4]
])


# Pass the input data through the neural network
# PyTorch automatically calls forward()
predictions = model(X)


# Display the input and predictions
print("Input:")
print(X)

print("Predictions:")
print(predictions)


# Display the shape of the predictions
print("Prediction shape:", predictions.shape)


# Display all trainable weights and biases
for name, parameter in model.named_parameters():

    # Print the parameter name
    print("\nParameter:", name)

    # Print the actual randomly initialized values
    print(parameter)

    # Print the shape of each parameter
    print("Shape:", parameter.shape)
