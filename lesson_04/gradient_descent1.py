# Loss function
def loss(w):
    return w ** 2

# Derivative of loss function
def gradient(w):
    return 2 * w

# Starting weight
w = 3.0

# Learning rate
learning_rate = 0.1

# Repeat the learning process
for step in range(10):

    # 1. Calculate gradient
    grad = gradient(w)

    # 2. Update weight
    w = w - learning_rate * grad

    # 3. Calculate new loss
    current_loss = loss(w)

    # 4. Display results
    print(
        "Step:", step + 1,
        "Weight:", round(w, 4),
        "Loss:", round(current_loss, 4)
    )