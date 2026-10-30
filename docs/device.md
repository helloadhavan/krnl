# Device

The `Device` class abstracts over CPU and GPU execution targets. Every `Tensor` is associated with a device, and operations between tensors require them to be on the same device.

**Source:** [`src/krnl/backends/device.py`](../src/krnl/backends/device.py)

## Creating a Device

```python
from krnl import Device

cpu = Device("CPU")
gpu = Device("GPU", device_id=0)
```

## Constructor

```python
Device(device, device_id=0)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `device` | `"CPU"` or `"GPU"` | — | The target device type |
| `device_id` | `int` | `0` | GPU index (only relevant for `"GPU"`) |

## Querying Device Information

```python
device = Device("CPU")
info = device.get_device_info()
print(info)
```

For CPU devices, the returned dictionary contains:

| Key | Description |
|---|---|
| `platform` | Operating system name |
| `device` | Processor name |
| `max_compute_units` | Physical CPU core count |
| `max_work_group_size` | Logical CPU core count |
| `global_mem_size` | Total system RAM in bytes |
| `local_mem_size` | `None` (not applicable for CPU) |
| `max_clock_frequency` | Maximum CPU clock speed in MHz |

For GPU devices, the returned dictionary contains the same keys populated from the OpenCL device properties.

## Listing Available GPUs

```python
from krnl import get_gpu_info

gpus = get_gpu_info()
for gpu in gpus:
    print(gpu["device"], gpu["global_mem_size"])
```

## Moving Tensors Between Devices

```python
from krnl import Tensor, Device

t = Tensor([1.0, 2.0, 3.0], device=Device("CPU"))
t.to(Device("GPU"))   # transfers data to GPU in-place
t.to(Device("CPU"))   # transfers data back to CPU in-place
```

Note: `Tensor.to()` modifies the tensor in-place rather than returning a new tensor.
