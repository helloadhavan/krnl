# Saving and Loading

krnl provides `save` and `load` functions for persisting model and optimizer state to disk. Data is serialized using Python's `pickle` module with tensors converted to numpy arrays for safe storage.

**Source:** [`src/krnl/backends/save.py`](../src/krnl/backends/save.py) and [`src/krnl/backends/load.py`](../src/krnl/backends/load.py)

## Saving

```python
from krnl import save

save(data, path)
```

| Parameter | Type | Description |
|---|---|---|
| `data` | `dict` or `Tensor` | State dict or single tensor to save |
| `path` | `str` | File path to write to |

The `save` function recursively converts all `Tensor` and `GPUArray` values to numpy arrays before pickling. It handles:

- Flat dicts of tensors
- Nested dicts of tensors
- Lists and tuples of tensors
- Single tensors
- Mixed structures (dicts containing lists of tensors, etc.)

### Examples

```python
from krnl import save, nn, Device

model = nn.Linear(in_features=10, out_features=5, bias=True, device=Device("CPU"))
optimizer = nn.optim.Adam(model._parameters.values())

# Save model state
save(model.get_model_state_dict(), "my_model")

# Save optimizer state
save(optimizer.get_state_dict(), "my_optimizer")

# Save a single tensor
save(model._parameters["weight"], "weight_only")
```

## Loading

```python
from krnl import load

data = load(path, device=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | — | File path to read from |
| `device` | `Device` or `None` | `Device("CPU")` | Target device for reconstructed tensors |

The `load` function deserializes the pickle file and recursively converts numpy arrays back into `Parameter` objects (tensors with `requires_grad=True`). It handles the same nested structures as `save`.

### Examples

```python
from krnl import load, nn, Device

# Load on CPU (default)
model_state = load("my_model")
optim_state = load("my_optimizer")

# Load onto a specific device
model_state = load("my_model", device=Device("GPU"))
```

## Restoring State

### Model

Use `register_state_dict` to load saved parameters into a model:

```python
model = nn.Linear(in_features=10, out_features=5, bias=True)
model_state = load("my_model")
model.register_state_dict(model_state)
```

### Optimizer

Use `register_state_dict` to load saved state into an optimizer:

```python
optimizer = nn.optim.Adam(model._parameters.values())
optim_state = load("my_optimizer")
optimizer.register_state_dict(optim_state)
```

## Full Round-Trip Example

```python
from krnl import save, load, nn, Device

# --- Train and save ---
model = nn.Linear(in_features=4, out_features=2, bias=True, device=Device("CPU"))
optimizer = nn.optim.SGD(model._parameters.values(), learning_rate=0.01)

# ... training loop ...

save(model.get_model_state_dict(), "checkpoint_model")
save(optimizer.get_state_dict(), "checkpoint_optim")

# --- Later: restore and continue ---
model2 = nn.Linear(in_features=4, out_features=2, bias=True, device=Device("CPU"))
optimizer2 = nn.optim.SGD(model2._parameters.values(), learning_rate=0.01)

model2.register_state_dict(load("checkpoint_model"))
optimizer2.register_state_dict(load("checkpoint_optim"))

# model2 and optimizer2 now have the saved weights and state
```

## File Format

Files are standard Python pickle files. They can be inspected with:

```python
import pickle

with open("my_model", "rb") as f:
    data = pickle.load(f)

print(type(data))  # dict
print(data.keys()) # dict_keys(['states', 'parameters'])
```

All tensor data is stored as `numpy.ndarray` with `dtype=float32`.
