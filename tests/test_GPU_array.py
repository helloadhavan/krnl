from src.krnl.backends import get_gpu_info, GPUArray

print(get_gpu_info())

# Basic test
a = GPUArray([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
b = a + GPUArray.ones((3, 3))

print(a)
print(b)