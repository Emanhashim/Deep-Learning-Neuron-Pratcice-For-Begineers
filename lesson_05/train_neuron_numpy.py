
import numpy as np

# 1. Training data
X = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([2.0, 4.0, 6.0, 8.0])

# 2. Initialize parameters
weight = 0.5
bias = 0.5

# 3. Set learning rate
learning_rate = 0.05

# 4. Training loop
for epoch in range(1000):

    # Forward propagation
    predictions = weight * X + bias

    # Calculate loss
    errors = predictions - y
    loss = np.mean(errors ** 2)

    # Backpropagation: calculate gradients
    dw = (2 / len(X)) * np.sum(errors * X)
    db = (2 / len(X)) * np.sum(errors)

    # Gradient descent: update parameters
    weight = weight - learning_rate * dw
    bias = bias - learning_rate * db

    # Display progress
    if (epoch + 1) % 100 == 0:
        print(
            "Epoch:", epoch + 1,
            "Loss:", round(loss, 6),
            "Weight:", round(weight, 4),
            "Bias:", round(bias, 4)
        )

# 5. Test the trained neuron
test_input = 5.0
prediction = weight * test_input + bias

print("\nFinal weight:", weight)
print("Final bias:", bias)
print("Prediction for input 5:", prediction)
print("Expected answer:", 10.0)
