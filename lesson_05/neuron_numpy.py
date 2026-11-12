
import numpy as np

# =====================================
# 1. CREATE INPUT DATA
# =====================================

# Hours studied and attendance rate
X = np.array([0.8, 0.9])


# =====================================
# 2. CREATE WEIGHTS
# =====================================

W = np.array([0.6, 0.3])


# =====================================
# 3. CREATE BIAS
# =====================================

b = 0.2


# =====================================
# 4. CALCULATE WEIGHTED SUM
# =====================================

z = np.dot(X, W) + b


# =====================================
# 5. APPLY SIGMOID ACTIVATION
# =====================================

output = 1 / (1 + np.exp(-z))


# =====================================
# 6. DISPLAY RESULTS
# =====================================

print("Inputs:", X)
print("Weights:", W)
print("Bias:", b)

print("Weighted sum:", z)
print("Neuron output:", output)
