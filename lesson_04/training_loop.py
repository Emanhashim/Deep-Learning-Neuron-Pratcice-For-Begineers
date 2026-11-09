
import torch

# ==========================================
# 1. CREATE TRAINING DATA
# ==========================================

# Input data
X = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0]
])

# Correct answers
y = torch.tensor([
    [2.0],
    [4.0],
    [6.0],
    [8.0]
])


# ==========================================
# 2. CREATE THE MODEL
# ==========================================

model = torch.nn.Linear(1, 1)


# ==========================================
# 3. CHOOSE LOSS FUNCTION
# ==========================================

loss_fn = torch.nn.MSELoss()


# ==========================================
# 4. CHOOSE OPTIMIZER
# ==========================================

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.03
)


# ==========================================
# 5. TRAINING LOOP
# ==========================================

for epoch in range(500):

    # Forward propagation
    predictions = model(X)

    # Calculate loss
    loss = loss_fn(predictions, y)

    # Clear old gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update model weights and bias
    optimizer.step()

    # Display training progress every 50 epochs
    if (epoch + 1) % 50 == 0:
        print(
            "Epoch:", epoch + 1,
            "Loss:", round(loss.item(), 6)
        )


# ==========================================
# 6. TEST THE TRAINED MODEL
# ==========================================

# New input
test = torch.tensor([[5.0]])

# Make prediction
prediction = model(test)

print("\n========== FINAL RESULTS ==========")

print("Input:", test.item())

print("Predicted output:", prediction.item())

print("Expected output:", 10.0)

print("Learned weight:", model.weight.item())

print("Learned bias:", model.bias.item())
