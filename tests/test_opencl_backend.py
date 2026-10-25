import numpy as np

from src.krnl.backends.GPU_processer import OpenCLBackend


def main():
    backend = OpenCLBackend()

    a = backend.upload(
        np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            dtype=np.float32,
        )
    )

    b = backend.upload(
        np.array(
            [
                [5.0, 6.0],
                [7.0, 8.0],
            ],
            dtype=np.float32,
        )
    )

    addition = backend.__add__(a, b)
    multiplication = backend.__mul__(a, b)
    matrix_multiplication = backend.__matmul__(a, b)

    print("a + b:")
    print(addition.get())

    print("\na * b:")
    print(multiplication.get())

    print("\na @ b:")
    print(matrix_multiplication.get())


if __name__ == "__main__":
    main()