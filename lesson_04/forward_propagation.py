import numpy as np

# 1. Input values
X = np.array([4, 0.8])

# 2. Weights
W = np.array([0.5, 0.3])

# 3. Bias
b = 0.2

# 4. Calculate weighted sum
z = np.dot(X, W) + b

# 5. Apply sigmoid activation
prediction = 1 / (1 + np.exp(-z))

# 6. Display results
print("Weighted sum:", z)
print("Prediction:", prediction)
print("Prediction percentage:", prediction * 100, "%")