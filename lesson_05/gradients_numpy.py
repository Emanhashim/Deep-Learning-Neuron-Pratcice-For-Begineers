
import numpy as np

# 1. Input and correct answer
x = 2.0
y = 10.0

# 2. Initial weight and bias
weight = 3.0
bias = 0.0

# 3. Forward propagation
prediction = weight * x + bias

# 4. Calculate loss
loss = (prediction - y) ** 2

# 5. Calculate gradients
error = prediction - y

dw = 2 * error * x
db = 2 * error

# 6. Display results
print("Prediction:", prediction)
print("Loss:", loss)
print("Weight gradient:", dw)
print("Bias gradient:", db)
