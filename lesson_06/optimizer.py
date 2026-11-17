
import torch

# Create a trainable parameter with an initial value of 3
weight = torch.nn.Parameter(torch.tensor(3.0))

# Create an SGD optimizer responsible for updating the weight
optimizer = torch.optim.SGD([weight], lr=0.1)

# Repeat the learning process five times
for step in range(5):

    # Calculate the loss using the current weight
    loss = weight ** 2

    # Clear gradients left over from previous iterations
    optimizer.zero_grad()

    # Calculate the gradient using automatic differentiation
    loss.backward()

    # Update the weight using the calculated gradient
    optimizer.step()

    # Display the updated weight and the loss from this iteration
    print(
        "Step:", step + 1,
        "Weight:", round(weight.item(), 4),
        "Loss:", round(loss.item(), 4)
    )
