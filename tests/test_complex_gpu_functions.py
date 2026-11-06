from krnl.backends.GPU_processer import GPUArray

a = GPUArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = a.log()

print(a.get())
print(b.get())