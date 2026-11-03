import math

# Step 1: Define inputs
study_hours = 0.7
attendance = 0.85

# Step 2: Define weights
weight_1 = 0.6
weight_2 = 0.4

# Step 3: Define bias
bias = 0.1

# Step 4: Calculate weighted sum
z = (
    study_hours * weight_1
    + attendance * weight_2
    + bias
)

# Step 5: Apply sigmoid
output = 1 / (1 + math.exp(-z))

# Step 6: Display results
print("Weighted sum:", z)
print("Neuron output:", output)
print("Probability:", output * 100, "%")