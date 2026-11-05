import numpy as np

# 1. Scalar
scalar = np.array(10)

# 2. Vector
vector = np.array([2, 4, 6, 8, 10])

# 3. Matrix: 3 rows and 2 columns
matrix = np.array([
    [1, 2],
    [3, 4], 
    [5, 6]
])

# 4. Three-dimensional tensor
tensor = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])

# Inspect the data
for name, data in [
    ("Scalar", scalar),
    ("Vector", vector),
    ("Matrix", matrix),
    ("Tensor", tensor)
]:
    print(f"\n{name}")
    print("Data:", data)
    print("Shape:", data.shape)
    print("Dimensions:", data.ndim)
    print("Total elements:", data.size)

