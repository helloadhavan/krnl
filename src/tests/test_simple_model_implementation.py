from src.backends import tensor
from random import randint

dx = tensor.Tensor(randint(1, 10))
dy = dx * tensor.Tensor(randint(1, 4)) + tensor.Tensor(randint(1, 2))

weight = tensor.Tensor(randint(1, 10), requires_grad=True)
bias = tensor.Tensor(randint(1, 10), requires_grad=True)
print(f"Initial weight: {weight.data}, Initial bias: {bias.data}")


for i in range(100):
    # forward pass
    y = dx * weight + bias
    error = y - dy
    loss = error ** 2

    print(f"input: {dx.data}, desired: {dy.data}, predicted: {y.data}, error: {error.data}, loss: {loss.data}")

    # backward pass
    loss.backward()

    # update weights and biases (using SGD)
    learning_rate = 0.01
    weight.data -= learning_rate * weight.grad
    bias.data -= learning_rate * bias.grad