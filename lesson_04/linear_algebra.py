import numpy as np

# 1. Create two vectors
A = np.array([2, 4, 6])
B = np.array([1, 3, 5])

# 2. Add the vectors
addition = A + B

# 3. Multiply vector A by 2
multiplication = 2 * A

# 4. Create a matrix with 3 rows and 2 columns
students = np.array([
    [5, 0.90],
    [3, 0.70],
    [8, 0.95]
])

# 5. Inspect the matrix
# 5. Inspect the matrix
print("Addition:", addition)
print("Scalar multiplication:", multiplication)
print("Student matrix:", students)
print("Shape:", students.shape)
print("Dimensions:", students.ndim)
print("Total elements:", students.size)