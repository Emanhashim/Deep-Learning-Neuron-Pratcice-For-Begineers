
import torch

# 1. Create a tensor containing zeros
a = torch.zeros(2, 3)

# 2. Create a tensor containing ones
b = torch.ones(2, 3)

# 3. Create a sequence from 1 to 8
c = torch.arange(1, 9)

# 4. Create a random tensor with 2 rows and 2 columns
d = torch.rand(2, 2)

# 5. Create an integer tensor
e = torch.tensor([10, 20, 30])

# 6. Convert e into float32
e = e.float()

# 7. Create two tensors for mathematical operations
x = torch.tensor([2.0, 4.0, 6.0])
y = torch.tensor([1.0, 2.0, 3.0])

print("Zeros:", a)
print("Ones:", b)
print("Sequence:", c)
print("Random:", d)
print("Converted tensor:", e)
print("Data type:", e.dtype)

print("Addition:", x + y)
print("Subtraction:", x - y)
print("Multiplication:", x * y)
print("Division:", x / y)
print("Mean:", torch.mean(x))
print("Sum:", torch.sum(x))
