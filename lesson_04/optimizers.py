import torch

# Create a simple trainable weight
weight = torch.nn.Parameter(torch.tensor(3.0))

# Choose an optimizer
optimizer = torch.optim.SGD(
    [weight],
    lr=0.1
)

# Training example
for step in range(5):

    # Calculate loss
    loss = weight ** 2

    # Calculate gradients
    optimizer.zero_grad()
    loss.backward()

    # Update weight
    optimizer.step()

    print(
        "Step:", step + 1,
        "Weight:", round(weight.item(), 4),
        "Loss:", round(loss.item(), 4)
    )