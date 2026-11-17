import torch

# 0D tensor
scalar = torch.tensor(5)

# 1D tensor
vector = torch.tensor([1, 2, 3])

# 2D tensor
matrix = torch.tensor([
    [1, 2],
    [3, 4]
])

# 3D tensor
tensor_3d = torch.tensor([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])

print("Scalar:", scalar)
print("Vector:", vector)
print("Matrix:", matrix)
print("3D Tensor:", tensor_3d)