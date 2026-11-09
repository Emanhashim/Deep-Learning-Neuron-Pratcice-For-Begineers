import numpy as np

# Input data: 3 students, 2 features
X = np.array([
    [5, 0.90],
    [3, 0.70],
    [8, 0.95]
])

# Weights: 2 input features, 2 neurons
W = np.array([
    [0.5, -0.3],
    [0.4, 0.8]
])

# Matrix multiplication
Z = X @ W

print("Input matrix:")
print(X)

print("\nWeight matrix:")
print(W)

print("\nResult after matrix multiplication:")
print(Z)

print("\nResult shape:", Z.shape)

A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

print(A @ B)
print(A * B)