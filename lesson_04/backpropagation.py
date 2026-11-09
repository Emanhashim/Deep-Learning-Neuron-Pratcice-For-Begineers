# 1. Starting values
x = 2
w = 0.2
b = 0
actual = 1

# 2. Forward propagation
z = w * x + b
prediction = z

# 3. Calculate loss
loss = (actual - prediction) ** 2

# 4. Backpropagation

# Derivative of loss with respect to prediction
dL_dprediction = 2 * (prediction - actual)

# Derivative of prediction with respect to z
dprediction_dz = 1

# Derivative of z with respect to weight
dz_dw = x

# Apply the chain rule
gradient = (
    dL_dprediction
    * dprediction_dz
    * dz_dw
)

# 5. Display results
print("Prediction:", prediction)
print("Loss:", loss)
print("Loss derivative:", dL_dprediction)
print("Gradient:", gradient)