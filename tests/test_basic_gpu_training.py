import numpy as np

from krnl import nn
from krnl import Tensor


def main():
    dx = []
    dy = []

    for i in range(10):
        dx.append(float(i))
        dy.append(float(i * 2 + 1))

    model = nn.Linear(
        in_features=1,
        out_features=1,
        bias=True,
        device="GPU"
    )

    criterion = nn.loss.MSELoss()

    optimizer = nn.optim.MuonAdamAux(
        model._parameters.values(),
        learning_rate=0.001,
    )

    weight = model._parameters["weight"]
    bias = model._parameters["bias"]

    print("Initial weight:", weight.data, "| Initial bias:", bias.data)

    for epoch in range(100):
        total_loss = 0.0

        for x_value, y_value in zip(dx, dy):
            optimizer.zero_grad()

            x = Tensor([[x_value]])
            target = Tensor([[y_value]])

            prediction = model(x)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()

            total_loss += float(np.asarray(loss.data).reshape(-1)[0])

        if epoch % 10 == 0:
            average_loss = total_loss / len(dx)
            print(
                f"epoch={epoch:03d}, "
                f"loss={average_loss:.6f}, "
                f"weight={weight.data}, "
                f"bias={bias.data}"
            )

    print("\nFinal predictions:")

    for x_value, expected_value in zip(dx, dy):
        x = Tensor([[x_value]])
        prediction = model(x)
        print(
            f"x={x_value:.1f}, "
            f"predicted={prediction.data}, "
            f"expected={expected_value:.1f}"
        )


if __name__ == "__main__":
    main()