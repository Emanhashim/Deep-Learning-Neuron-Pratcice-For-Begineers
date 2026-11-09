def loss(w):
    return w ** 2

def gradient(w):
    return 2 * w

w = 3.0
learning_rate = 0.1

for step in range(10):

    grad = gradient(w)

    w = w - learning_rate * grad

    print(
        "Step:", step + 1,
        "Weight:", round(w, 4),
        "Loss:", round(loss(w), 4)
    )