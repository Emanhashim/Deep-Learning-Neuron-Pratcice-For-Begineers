import numpy as np

scores = np.array([2.0, 1.0, 0.1])

exp_scores = np.exp(scores)

softmax = exp_scores / np.sum(exp_scores)

print("Softmax:", softmax)
print("Total:", np.sum(softmax))