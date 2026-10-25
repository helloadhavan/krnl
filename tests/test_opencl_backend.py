from src.krnl.backends.GPU_processer import OpenCLBackend
import numpy as np

backend = OpenCLBackend()

a = backend.upload(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
b = backend.upload(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32))


print(a + b)  # Should print the result of adding a and b
print(a * b)  # Should print the result of multiplying a and b
print(a @ b)  # Should print the result of multiplying a and b