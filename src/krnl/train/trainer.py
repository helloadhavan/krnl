from krnl.train.args import TrainerArguments
from krnl.data.dataloader import DataLoader
from krnl.nn.model_utils import Module
from krnl.backends import Tensor


class Trainer(object):
    """High-level training loop around a :class:`Module` and :class:`DataLoader`.

    Parameters
    ----------
    model : Module
        The neural-network module to train.
    data_loader : DataLoader
        Mini-batch data loader that yields ``(inputs, labels, batch_index)``
        tuples.
    args : TrainerArguments
        Hyper-parameters (learning rate, epochs, loss function, optimizer).
    """

    def __init__(self, model: Module, data_loader: DataLoader, args: TrainerArguments):
        self.model = model
        self.args = args
        self.data_loader = data_loader
        self.optimizer = args.build_optimizer(model._parameters.values())

    def train(self, log_epochs: int | None = 1, log_steps: int | None = None):
        """Run the training loop for ``self.args.num_epochs`` epochs.

        Parameters
        ----------
        log_epochs : int or None, optional
            If not ``None``, print a summary every *log_epochs* epochs
            (default ``1``).
        log_steps : int or None, optional
            If not ``None``, print a summary every *log_steps* mini-batch
            steps within each epoch.  When both *log_epochs* and *log_steps*
            are provided, both kinds of messages are emitted.
        """
        num_batches = len(self.data_loader)

        for epoch in range(self.args.num_epochs):
            epoch_loss = 0.0
            steps_in_epoch = 0

            for inputs, labels, batch_num in self.data_loader:
                # Forward pass
                outputs = self.model(inputs)
                loss = self.args.loss_function(outputs, labels)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss = loss.item()
                epoch_loss += batch_loss
                steps_in_epoch += 1

                # Per-step logging
                if log_steps is not None and steps_in_epoch % log_steps == 0:
                    print(
                        f"Epoch [{epoch + 1}/{self.args.num_epochs}] "
                        f"Step [{steps_in_epoch}/{num_batches}], "
                        f"Loss: {batch_loss:.4f}"
                    )

            # Per-epoch logging
            avg_loss = epoch_loss / max(steps_in_epoch, 1)
            if log_epochs is not None and (epoch + 1) % log_epochs == 0:
                print(
                    f"Epoch [{epoch + 1}/{self.args.num_epochs}], "
                    f"Loss: {avg_loss:.4f}"
                )

    def validate(self, val_inputs: Tensor, val_targets: Tensor):
        """Compute the loss on a held-out validation set (no gradient update).

        Parameters
        ----------
        val_inputs : Tensor
            Validation input features.
        val_targets : Tensor
            Validation target labels.

        Returns
        -------
        Tensor
            The scalar validation loss.
        """
        val_inputs = val_inputs.detach()
        val_targets = val_targets.detach()
        prediction = self.model(val_inputs)
        val_loss = self.args.loss_function(prediction, val_targets)
        return val_loss
