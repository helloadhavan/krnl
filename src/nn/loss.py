from src.backends.tensor import Tensor

class BaseCriterion:
    """
    Base class for all loss functions.
    """
    def __init__(self):
        pass

    def __call__(self, predicted: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError("Subclasses must implement the __call__ method.")

class MSELoss(BaseCriterion):
    """
    Mean Squared Error Loss.
    Applies the formula: MSE = mean((predicted - target)^2)
    """
    def __init__(self):
        super().__init__()

    def __call__(self, predicted: Tensor, target: Tensor) -> Tensor:
        if predicted.data.shape != target.data.shape:
            raise ValueError("Predicted and target tensors must have the same shape.")
        return (predicted - target) ** 2

class MAE(BaseCriterion):
    """
    Mean Absolute Error Loss.
    Applies the formula: MAE = mean(|predicted - target|)
    """
    def __init__(self):
        super().__init__()

    def __call__(self, predicted: Tensor, target: Tensor) -> Tensor:
        if predicted.data.shape != target.data.shape:
            raise ValueError("Predicted and target tensors must have the same shape.")
        return (predicted - target).abs()
