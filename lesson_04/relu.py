import numpy as np

z = np.array([-5, -2, 0, 3, 8])

relu = np.maximum(0, z)

print("ReLU:", relu)