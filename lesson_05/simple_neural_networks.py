
import numpy as np

# 1. INPUT DATA
X = np.array([0.8, 0.9])

# 2. WEIGHTS AND BIASES
W1 = np.array([
    [0.6, 0.2],
    [0.3, 0.7]
])

b1 = np.array([0.1, 0.1])

W2 = np.array([0.5, 0.4])
b2 = 0.2

# 3. ACTIVATION FUNCTION
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 4. HIDDEN LAYER
z1 = X @ W1 + b1
hidden_output = sigmoid(z1)

# 5. OUTPUT LAYER
z2 = hidden_output @ W2 + b2
prediction = sigmoid(z2)

# 6. DISPLAY RESULTS
print("Input:", X)
print("Hidden weighted sums:", z1)
print("Hidden outputs:", hidden_output)
print("Output weighted sum:", z2)
print("Final prediction:", prediction)
