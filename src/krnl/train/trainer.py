from krnl.train.args import TrainerArguments
from krnl.nn.model_utils import Module
from krnl.backends import Tensor


class Trainer(object):
    def __init__(self, model: Module, inputs: Tensor, targets: Tensor, args: TrainerArguments):
        self.model = model
        self.inputs = inputs
        self.targets = targets
        self.args = args
        self.optimizer = args.build_optimizer(model._parameters.values())

    def train(self, epochs: int | None = None, log_epochs: int = 1):
        if epochs is None:
            epochs = self.args.num_epochs
        for epoch in range(epochs):
            # Forward pass
            outputs = self.model(self.inputs)
            loss = self.args.loss_function(outputs, self.targets)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % log_epochs == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    def validate(self, val_inputs: Tensor, val_targets: Tensor):
        val_inputs = val_inputs.detach()
        val_targets = val_targets.detach()
        prediction = self.model(val_inputs)
        val_loss = self.args.loss_function(prediction, val_targets)
        return val_loss
