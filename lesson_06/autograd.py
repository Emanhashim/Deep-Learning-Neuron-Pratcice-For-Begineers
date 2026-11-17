
import torch

weight = torch.tensor(2.0, requires_grad=True)

x = torch.tensor(3.0)
y = torch.tensor(12.0)

# Step 1: Calculate prediction
prediction = weight * x

# Step 2: Calculate loss
loss = (prediction - y) ** 2

# Step 3: Calculate gradients
loss.backward()

# Step 4: Print results
print("Prediction:", prediction.item())
print("Loss:", loss.item())
print("Gradient:", weight.grad.item())
