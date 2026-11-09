import numpy as np

# Different input values
z = np.array([-3, -1, 0, 1, 3])

# 1. Sigmoid
sigmoid = 1 / (1 + np.exp(-z))

# 2. ReLU
relu = np.maximum(0, z)

# 3. Tanh
tanh = np.tanh(z)

print("Input:", z)
print("Sigmoid:", sigmoid)
print("ReLU:", relu)
print("Tanh:", tanh)

# 4. Softmax

scores = np.array([2.0, 1.0, 0.1])

exp_scores = np.exp(scores)
softmax = exp_scores / np.sum(exp_scores)

print("Softmax:", softmax)
print("Softmax total:", np.sum(softmax))