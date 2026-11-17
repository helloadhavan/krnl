import numpy as np
from krnl.backends.tensor import Tensor
from krnl.data.dataset import Dataset
from krnl.data.dataloader import DataLoader
from krnl.nn.model_utils import Linear, Sequential, GELU
from krnl.train.args import TrainerArguments
from krnl.train.trainer import Trainer


def test_trainer_train_and_validate():
    # Create dummy data: 20 samples, 3 features -> 1 output
    inputs = Tensor(np.random.randn(20, 3).astype(np.float32))
    labels = Tensor(np.random.randn(20, 1).astype(np.float32))

    ds = Dataset(inputs, labels)
    dl = DataLoader(ds, batch_size=5, shuffle=True)

    model = Sequential(
        Linear(3, 4),
        GELU(),
        Linear(4, 1),
    )

    args = TrainerArguments(
        learning_rate=0.01,
        batch_size=5,
        num_epochs=3,
        optimizer="Muon", # test case-insensitivity
        loss_function="MAE",
    )

    trainer = Trainer(model, dl, args)

    # Should run without errors, with both epoch and step logging
    trainer.train(log_epochs=1, log_steps=2)

    # Validation should return a scalar loss
    val_loss = trainer.validate(inputs, labels)
    assert val_loss.data.size == 1, "Validation loss should be a scalar"
    print(f"Validation loss: {val_loss.item():.4f}")
    print("All tests passed!")


if __name__ == "__main__":
    test_trainer_train_and_validate()
