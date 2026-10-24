import random
import numpy as np

import nn
from src.backends.tensor import Tensor
from nn import Linear, ReLU, optim


class EqualityModel:
    def __init__(self):
        # One input feature: (a - b)^2
        self.linear = Linear(
            in_features=1,
            out_features=1,
            bias=True,
        )

        self.relu = ReLU()

        # Set useful initial values explicitly.
        #
        # The desired behavior is:
        #   input = 0       -> output close to 1
        #   input > 0       -> output close to 0
        #
        # A negative weight and positive bias provide this behavior.
        self.linear._parameters["weight"].data = np.array(
            [[-0.1]],
            dtype=np.float32,
        )

        self.linear._parameters["bias"].data = np.array(
            [0.5],
            dtype=np.float32,
        )

    def forward(self, x: Tensor):
        x = self.linear(x)
        x = self.relu(x)
        return x

    def parameters(self):
        return self.linear._parameters.values()

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.grad = None


class SGD:
    def __init__(self, parameters, learning_rate=0.01):
        self.parameters = list(parameters)
        self.learning_rate = learning_rate

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = None

    def step(self):
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.data -= (
                    self.learning_rate * parameter.grad
                )


def make_example(a, b):
    """
    Return:

        input feature: (a - b)^2
        target:        1 if equal, otherwise 0
    """

    difference = a - b
    feature = float(difference * difference)
    target = 1.0 if a == b else 0.0

    # Linear expects the last input dimension to have size 1.
    x = Tensor([feature])
    y = Tensor([[target]])

    return x, y


def create_dataset():
    dataset = []

    # Equal examples.
    for value in range(5):
        dataset.append((value, value))

    # Unequal examples.
    unequal_examples = [
        (0, 1),
        (1, 0),
        (0, 2),
        (2, 0),
        (1, 3),
        (3, 1),
        (2, 4),
        (4, 2),
    ]

    dataset.extend(unequal_examples)
    return dataset


def scalar_value(data):
    """
    Convert a one-element NumPy array into a Python float.
    """
    return float(np.asarray(data).reshape(-1)[0])


def main():
    random.seed(42)

    model = EqualityModel()

    optimizer = optim.Adam(
        model.parameters(),
        learning_rate=0.1,
    )
    criterion = nn.loss.MSELoss()

    dataset = create_dataset()

    epochs = 500

    for epoch in range(epochs):
        random.shuffle(dataset)
        total_loss = 0.0

        for a, b in dataset:
            # Clear gradients from the previous example.
            optimizer.zero_grad()

            # Create input and target.
            x, target = make_example(a, b)

            # Forward pass.
            prediction = model.forward(Tensor(x))

            # Mean squared error.
            loss = criterion(prediction, Tensor(target))

            # Backward pass.
            loss.backward()

            # Update parameters.
            optimizer.step()

            total_loss += scalar_value(loss.data)

        if epoch % 50 == 0:
            average_loss = total_loss / len(dataset)

            weight = scalar_value(
                model.linear._parameters["weight"].data
            )
            bias = scalar_value(
                model.linear._parameters["bias"].data
            )

            print(
                f"epoch={epoch:03d}, "
                f"loss={average_loss:.6f}, "
                f"weight={weight:.4f}, "
                f"bias={bias:.4f}"
            )

    print("\nFinal predictions:")

    test_cases = [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
        (0, 1),
        (1, 3),
        (2, 4),
        (4, 0),
    ]

    for a, b in test_cases:
        x, target = make_example(a, b)
        prediction = model.forward(Tensor(x))

        score = scalar_value(prediction.data)
        predicted_label = 1 if score >= 0.5 else 0
        expected_label = 1 if a == b else 0

        print(
            f"{a} == {b}: "
            f"score={score:.4f}, "
            f"predicted={predicted_label}, "
            f"expected={expected_label}"
        )


if __name__ == "__main__":
    main()