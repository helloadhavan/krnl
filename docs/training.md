# Training

The `train` module provides a high-level training loop so you can train a model with a single method call instead of writing the forward/backward/step boilerplate yourself.

**Source:** [`src/krnl/train/trainer.py`](../src/krnl/train/trainer.py), [`src/krnl/train/args.py`](../src/krnl/train/args.py)

## TrainerArguments

`TrainerArguments` bundles the hyperparameters that control training: learning rate, number of epochs, which optimizer to use, and which loss function to apply.

```python
from krnl.train import TrainerArguments

args = TrainerArguments(
    learning_rate=0.01,
    batch_size=32,
    num_epochs=50,
    optimizer="adam",
    loss_function="mse",
)
```

### Constructor

```python
TrainerArguments(
    learning_rate=3e-4,
    batch_size=32,
    num_epochs=10,
    optimizer="adam",
    loss_function="mse",
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `learning_rate` | `float` | `3e-4` | Optimizer step size |
| `batch_size` | `int` | `32` | Batch size (informational — the actual batching is controlled by `DataLoader`) |
| `num_epochs` | `int` | `10` | Number of full passes over the dataset |
| `optimizer` | `str` | `"adam"` | Optimizer name: `"adam"`, `"sgd"`, `"muon"`, or `"muon_adam"` |
| `loss_function` | `str` or `BaseCriterion` | `"mse"` | Loss function name (`"mse"`, `"mae"`, `"huber"`) or a `BaseCriterion` instance |

### Methods

| Method | Returns | Description |
|---|---|---|
| `build_optimizer(parameters)` | `Optimizer` | Instantiate the configured optimizer with the given model parameters |

### Supported Optimizers

| String | Class |
|---|---|
| `"adam"` | `Adam` |
| `"sgd"` | `SGD` |
| `"muon"` | `Muon` |
| `"muon_adam"` | `MuonAdam` |

### Supported Loss Functions

| String | Class |
|---|---|
| `"mse"` | `MSELoss` |
| `"mae"` | `MAELoss` |
| `"huber"` | `HuberLoss` |

You can also pass a pre-constructed loss instance (e.g. `BCELoss()` or `CrossEntropy()`) directly:

```python
from krnl.nn.loss import CrossEntropy

args = TrainerArguments(loss_function=CrossEntropy())
```

## Trainer

`Trainer` ties together a model, a data loader, and training arguments into a complete training loop.

```python
from krnl.train import Trainer, TrainerArguments
from krnl.data import DataLoader

trainer = Trainer(model, data_loader, args)
trainer.train(log_epochs=10, log_steps=5)
```

### Constructor

```python
Trainer(model: Module, data_loader: DataLoader, args: TrainerArguments)
```

| Parameter | Type | Description |
|---|---|---|
| `model` | `Module` | The neural network module to train |
| `data_loader` | `DataLoader` | Mini-batch data loader |
| `args` | `TrainerArguments` | Hyperparameters and configuration |

The constructor automatically builds the optimizer by calling `args.build_optimizer(model._parameters.values())`.

### train()

```python
trainer.train(log_epochs=1, log_steps=None)
```

Runs the training loop for `args.num_epochs` epochs. Each epoch iterates over every batch from the data loader, performing:

1. **Forward pass** — `outputs = model(inputs)`
2. **Loss computation** — `loss = loss_function(outputs, labels)`
3. **Backward pass** — `optimizer.zero_grad()`, `loss.backward()`
4. **Parameter update** — `optimizer.step()`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `log_epochs` | `int` or `None` | `1` | Print average epoch loss every *n* epochs. `None` disables epoch logging. |
| `log_steps` | `int` or `None` | `None` | Print batch loss every *n* steps within each epoch. `None` disables step logging. |

When both `log_epochs` and `log_steps` are provided, both kinds of messages are emitted.

**Example output:**

```text
Epoch [1/50] Step [2/7], Loss: 1.2345
Epoch [1/50] Step [4/7], Loss: 0.9876
Epoch [1/50] Step [6/7], Loss: 0.8765
Epoch [1/50], Loss: 1.0234
```

### validate()

```python
val_loss = trainer.validate(val_inputs, val_targets)
```

Computes the loss on a held-out validation set **without** updating model parameters. Both inputs are detached from the computation graph before the forward pass.

| Parameter | Type | Description |
|---|---|---|
| `val_inputs` | `Tensor` | Validation input features |
| `val_targets` | `Tensor` | Validation target labels |

Returns a scalar `Tensor` containing the validation loss.

## Complete Example

```python
import numpy as np
from krnl import nn, Tensor
from krnl.data import Dataset, DataLoader
from krnl.train import Trainer, TrainerArguments

# Generate synthetic data: y = 2*x1 + 3*x2 - x3 + noise
np.random.seed(42)
X = np.random.randn(200, 3).astype(np.float32)
y = (2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + 0.1 * np.random.randn(200)).astype(np.float32)

inputs = Tensor(X)
labels = Tensor(y.reshape(-1, 1))

# Split into train / validation
train_ds = Dataset(Tensor(X[:160]), Tensor(y[:160].reshape(-1, 1)))
val_inputs = Tensor(X[160:])
val_labels = Tensor(y[160:].reshape(-1, 1))

loader = DataLoader(train_ds, batch_size=32, shuffle=True)

# Build model
model = nn.Sequential(
    nn.Linear(3, 16, bias=True),
    nn.ReLU(),
    nn.Linear(16, 1, bias=True),
)

# Configure training
args = TrainerArguments(
    learning_rate=0.01,
    num_epochs=100,
    optimizer="adam",
    loss_function="mse",
)

# Train
trainer = Trainer(model, loader, args)
trainer.train(log_epochs=20)

# Validate
val_loss = trainer.validate(val_inputs, val_labels)
print(f"Validation loss: {val_loss.item():.4f}")
```
