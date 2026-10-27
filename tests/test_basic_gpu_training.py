import numpy as np

from krnl import nn
from krnl import Tensor, Device


def main():
    gpu = Device("GPU")

    dx = [float(i) for i in range(10)]
    dy = [float(i * 2 + 1) for i in range(10)]

    model = nn.Linear(
        in_features=1,
        out_features=1,
        bias=True,
        device=gpu,
    )

    criterion = nn.loss.MSELoss()

    optimizer = nn.optim.MuonAdamAux(
        model._parameters.values(),
        learning_rate=0.001,
    )

    weight = model._parameters["weight"]
    bias = model._parameters["bias"]

    print("Initial weight:", weight.data.get())
    print("Initial bias:", bias.data.get())

    for epoch in range(100):
        total_loss = 0.0

        for x_value, y_value in zip(dx, dy):
            optimizer.zero_grad()

            # Create both tensors directly on the GPU
            x = Tensor([[x_value]], device=gpu)
            target = Tensor([[y_value]], device=gpu)

            prediction = model(x)
            loss = criterion(prediction, target)

            loss.backward()
            optimizer.step()

            loss_value = float(
                np.asarray(loss.data.get()).reshape(-1)[0]
            )
            total_loss += loss_value

        if epoch % 10 == 0:
            average_loss = total_loss / len(dx)

            print(
                f"epoch={epoch:03d}, "
                f"loss={average_loss:.6f}, "
                f"weight={weight.data.get()}, "
                f"bias={bias.data.get()}"
            )

    print("\nFinal predictions:")

    for x_value, expected_value in zip(dx, dy):
        x = Tensor([[x_value]], device=gpu)
        prediction = model(x)

        print(
            f"x={x_value:.1f}, "
            f"predicted={prediction.data.get()}, "
            f"expected={expected_value:.1f}"
        )


if __name__ == "__main__":
    main()