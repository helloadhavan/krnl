from krnl.nn.optim import Optimizer, Adam, SGD, Muon, MuonAdam
from krnl.nn.loss import BaseCriterion, MSELoss, MAELoss, HuberLoss

optims = {
    "adam": Adam,
    "sdg": SGD,
    "muon": Muon,
    "muon_adam": MuonAdam
}

losses = {
    "mse": MSELoss,
    "mae": MAELoss,
    "huber": HuberLoss
}

class TrainerArguments:
    def __init__(self, learning_rate=0.001, batch_size=32, num_epochs=10, optimizer: Optimizer | str = "Adam", loss_function: BaseCriterion | str ='cross_entropy'):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.optimizer = optimizer if isinstance(optimizer, Optimizer) else optims[optimizer.lower()](learning_rate)
        self.loss_function = loss_function if isinstance(loss_function, BaseCriterion) else losses[loss_function.lower()]()