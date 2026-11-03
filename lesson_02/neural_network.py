import math

# Activation function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# Input layer
study_hours = 0.7
attendance = 0.85


# Hidden neuron 1
z1 = (study_hours * 0.9) + (attendance * 0.1) + 0.1
h1 = sigmoid(z1)


# Hidden neuron 2
z2 = (study_hours * -0.3) + (attendance * 0.8) - 0.2
h2 = sigmoid(z2)


# Output neuron
z3 = (h1 * 0.6) + (h2 * 0.9) - 0.5
prediction = sigmoid(z3)


print("Hidden neuron 1:", h1)
print("Hidden neuron 2:", h2)
print("Final prediction:", prediction)