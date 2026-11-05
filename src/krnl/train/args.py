from krnl.nn.optim import Optimizer, Adam, SGD, Muon, MuonAdam
from krnl.nn.loss import BaseCriterion, MSELoss, MAELoss, HuberLoss

optims = {
    "adam": Adam,
    "sgd": SGD,
    "muon": Muon,
    "muon_adam": MuonAdam
}

losses = {
    "mse": MSELoss,
    "mae": MAELoss,
    "huber": HuberLoss
}

class TrainerArguments:
    def __init__(
        self,
        learning_rate: float = 3e-4,
        batch_size: int = 32,
        num_epochs: int = 10,
        optimizer: str = "adam",
        loss_function: BaseCriterion | str = "mse",
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self._optimizer_name = optimizer
        self.loss_function = loss_function if isinstance(loss_function, BaseCriterion) else losses[loss_function.lower()]()

    def build_optimizer(self, parameters) -> Optimizer:
        """Build the optimizer with model parameters. Must be called before training."""
        if isinstance(self._optimizer_name, Optimizer):
            return self._optimizer_name
        optimizer_cls = optims[self._optimizer_name.lower()]
        self.optimizer = optimizer_cls(parameters, learning_rate=self.learning_rate)
        return self.optimizer
