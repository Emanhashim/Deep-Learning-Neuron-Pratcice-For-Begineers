def loss(w):
    return w ** 2

def gradient(w):
    return 2 * w


learning_rates = [0.01, 0.1, 0.5]

for lr in learning_rates:

    w = 3.0

    print("\nLearning rate:", lr)

    for step in range(5):

        grad = gradient(w)

        w = w - lr * grad

        print(
            "Step:", step + 1,
            "Weight:", round(w, 4),
            "Loss:", round(loss(w), 4)
        )