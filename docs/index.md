# krnl Documentation

`krnl` is a lightweight, educational machine learning library written in Python on top of NumPy, with experimental OpenCL support for device-aware tensor computation.

## Table of Contents

- [Getting Started](getting_started.md) — installation, quick start, and first training loop
- [Tensor](tensor.md) — the core data structure and autograd engine
- [Device](device.md) — CPU and GPU device abstraction
- [Neural Network Modules](nn.md) — layers, activations, and the `Module` base class
- [Loss Functions](loss.md) — MSE, MAE, and Huber loss
- [Optimizers](optimizers.md) — SGD, Adam, Muon, and MuonAdam
- [Saving and Loading](saving_loading.md) — persisting and restoring model and optimizer state
- [OpenCL Backend](opencl_backend.md) — GPU computation via PyOpenCL
- [API Reference](api_reference.md) — complete public API listing
