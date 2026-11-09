import numpy as np

z = 0.86

sigmoid = 1 / (1 + np.exp(-z))

print("Sigmoid:", sigmoid)