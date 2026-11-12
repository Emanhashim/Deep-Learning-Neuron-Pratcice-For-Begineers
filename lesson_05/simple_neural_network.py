import numpy as np

# Input data
X = np.array([0.8, 0.9])

# Weights connecting input layer to hidden layer
W1 = np.array([
    [0.6, 0.2],
    [0.3, 0.7]
])

# Biases for hidden layer
b1 = np.array([0.1, 0.1])

# Weights connecting hidden layer to output layer
W2 = np.array([0.5, 0.4])

# Bias for output layer
b2 = 0.2