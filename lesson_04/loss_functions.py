import numpy as np

# Actual answers
actual = np.array([1, 0, 1, 1])

# Model predictions
predictions = np.array([0.92, 0.20, 0.70, 0.40])

# Calculate errors
errors = actual - predictions

# Square the errors
squared_errors = errors ** 2

# Calculate mean squared error
loss = np.mean(squared_errors)

print("Actual:", actual)
print("Predictions:", predictions)
print("Errors:", errors)
print("Squared errors:", squared_errors)
print("MSE Loss:", loss)