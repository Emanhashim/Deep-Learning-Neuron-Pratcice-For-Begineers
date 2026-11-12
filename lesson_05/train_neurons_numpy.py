
import numpy as np

# =================================
# 1. TRAINING DATA
# =================================

X = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([2.0, 4.0, 6.0, 8.0])


# =================================
# 2. INITIALIZE PARAMETERS
# =================================

weight = 0.5
bias = 0.5

learning_rate = 0.05


# =================================
# 3. TRAINING LOOP
# =================================

for epoch in range(1000):

    # Forward propagation
    predictions = weight * X + bias

    # Calculate loss
    errors = predictions - y
    loss = np.mean(errors ** 2)

    # Calculate gradients
    dw = (2 / len(X)) * np.sum(errors * X)
    db = (2 / len(X)) * np.sum(errors)

    # Update parameters
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


# =================================
# 4. TEST THE TRAINED MODEL
# =================================

test_input = 5.0

prediction = weight * test_input + bias

print("\nFinal weight:", weight)
print("Final bias:", bias)
print("Prediction:", prediction)
print("Expected:", 10.0)
