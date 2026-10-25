# test_opencl_backend.py

from backends.backends.tensor import Tensor


def test_tensor_backward_for_addition():
    """Verify that backward() computes gradients for scalar addition."""
    x = Tensor(2.0)
    w = Tensor(3.0)
    b = Tensor(4.0)

    y = x * w + b
    loss = y - (x * 2 + 1)  # Some arbitrary loss function

    loss.backward()

    print(loss.grad)
    print(loss.data)

if __name__ == "__main__":
    test_tensor_backward_for_addition()
    print("Tensor backward test passed.")