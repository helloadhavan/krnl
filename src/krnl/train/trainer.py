from krnl.train.args import TrainerArguments
from krnl.nn.model_utils import Module
from krnl.backends import Tensor

class Trainer(object):
    def __init__(self, model: Module, inputs: Tensor, targets: Tensor, args: TrainerArguments):
        self.model = model
        self.inputs = inputs
        self.targets = targets
        self.args = args

    def train(self, epochs: int | None = None, log_epochs: int = 1):
        if epochs is None:
            epochs = self.args.num_epochs
        for epoch in range(epochs):
            # Forward pass
            outputs = self.model(self.inputs)
            loss = self.args.loss_function(outputs, self.targets)

            # Backward pass and optimization
            self.args.optimizer.zero_grad()
            loss.backward()
            self.args.optimizer.step()

            if (epoch + 1) % log_epochs == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')