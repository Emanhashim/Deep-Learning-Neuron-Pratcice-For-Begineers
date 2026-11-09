import numpy as np

# Inputs
X = np.array([0.8, 0.9])

# Weights
W = np.array([0.6, 0.3])

# Bias
b = 0.2

# Calculate weighted sum
z = np.dot(X, W) + b

print("Weighted sum:", z)