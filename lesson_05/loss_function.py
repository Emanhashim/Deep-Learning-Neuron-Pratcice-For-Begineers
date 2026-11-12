
import numpy as np

# Actual answers
y_actual = np.array([1, 0, 1, 1])

# Model predictions
y_predicted = np.array([0.90, 0.20, 0.70, 0.40])

# Calculate errors
errors = y_predicted - y_actual

# Square the errors
squared_errors = errors ** 2

# Calculate mean squared error
mse = np.mean(squared_errors)

print("Actual:", y_actual)
print("Predictions:", y_predicted)
print("Errors:", errors)
print("Squared errors:", squared_errors)
print("Mean Squared Error:", mse)
