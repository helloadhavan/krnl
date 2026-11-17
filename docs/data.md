# Dataset and DataLoader

The `data` module provides utilities for organising training data into datasets and iterating over them in mini-batches.

**Source:** [`src/krnl/data/dataset.py`](../src/krnl/data/dataset.py), [`src/krnl/data/dataloader.py`](../src/krnl/data/dataloader.py)

## Dataset

`Dataset` is a simple container that pairs input and label tensors. It supports indexing so individual samples (or slices) can be retrieved.

```python
import numpy as np
from krnl import Tensor
from krnl.data import Dataset

inputs = Tensor(np.random.randn(100, 4).astype(np.float32))
labels = Tensor(np.random.randn(100, 1).astype(np.float32))

dataset = Dataset(inputs, labels)

# Retrieve a single sample
x, y = dataset[0]

# Retrieve a slice
x_batch, y_batch = dataset[0:8]
```

### Constructor

```python
Dataset(inputs: Tensor, labels: Tensor)
```

| Parameter | Type | Description |
|---|---|---|
| `inputs` | `Tensor` | Feature tensor — first axis is the sample dimension |
| `labels` | `Tensor` | Label tensor — first axis must match `inputs` |

Both arguments must be `Tensor` instances; a `TypeError` is raised otherwise.

### Methods

| Method | Returns | Description |
|---|---|---|
| `__getitem__(item)` | `(Tensor, Tensor)` | Index or slice into `inputs` and `labels` |

## DataLoader

`DataLoader` wraps a `Dataset` and yields mini-batches during iteration. It supports shuffling and optionally drops the last incomplete batch.

```python
from krnl.data import Dataset, DataLoader

loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=False)

print(len(loader))  # number of batches

for batch_inputs, batch_labels, batch_start in loader:
    # batch_inputs.shape  -> (16, 4)  (or smaller for the last batch)
    # batch_labels.shape  -> (16, 1)
    # batch_start         -> starting sample index of this batch
    pass
```

### Constructor

```python
DataLoader(dataset: Dataset, batch_size=32, shuffle=False, drop_last=False)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dataset` | `Dataset` | — | The dataset to load batches from |
| `batch_size` | `int` | `32` | Number of samples per batch |
| `shuffle` | `bool` | `False` | Shuffle sample indices at the start of each iteration pass |
| `drop_last` | `bool` | `False` | Drop the last batch when the dataset size is not evenly divisible by `batch_size` |

### Methods

| Method | Returns | Description |
|---|---|---|
| `__len__()` | `int` | Number of batches per full pass |
| `__iter__()` | iterator | Yields `(inputs, labels, batch_start)` tuples |

### Shuffling

When `shuffle=True`, the sample indices are randomly permuted at the **start of each iteration pass**. This means calling `for ... in loader` twice will produce different orderings.

### Drop Last

When `drop_last=True` and the dataset size is not evenly divisible by `batch_size`, the final incomplete batch is silently skipped. This is useful when downstream code requires a fixed batch size.

```python
# 10 samples, batch_size=3
loader = DataLoader(dataset, batch_size=3, drop_last=False)
print(len(loader))  # 4 (batches of 3, 3, 3, 1)

loader = DataLoader(dataset, batch_size=3, drop_last=True)
print(len(loader))  # 3 (last batch of 1 is dropped)
```

## Typical Usage with Trainer

```python
import numpy as np
from krnl import nn, Tensor
from krnl.data import Dataset, DataLoader
from krnl.train import Trainer, TrainerArguments

inputs = Tensor(np.random.randn(200, 4).astype(np.float32))
labels = Tensor(np.random.randn(200, 1).astype(np.float32))

dataset = Dataset(inputs, labels)
loader  = DataLoader(dataset, batch_size=32, shuffle=True)

model = nn.Sequential(
    nn.Linear(4, 16, bias=True),
    nn.ReLU(),
    nn.Linear(16, 1, bias=True),
)

args = TrainerArguments(
    learning_rate=0.01,
    num_epochs=20,
    optimizer="adam",
    loss_function="mse",
)

trainer = Trainer(model, loader, args)
trainer.train(log_epochs=5)
```
