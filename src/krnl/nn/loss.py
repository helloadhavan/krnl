from ..backends.tensor import Tensor


class BaseCriterion:
    def __init__(self):
        pass

    def __call__(self, predicted: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError("Subclasses must implement the __call__ method.")


class MSELoss(BaseCriterion):
    def __init__(self):
        super().__init__()

    def __call__(self, predicted: Tensor, target: Tensor) -> Tensor:
        if predicted.data.shape != target.data.shape:
            raise ValueError("Predicted and target tensors must have the same shape.")
        return (predicted - target) ** 2


class MAELoss(BaseCriterion):
    def __init__(self):
        super().__init__()

    def __call__(self, predicted: Tensor, target: Tensor) -> Tensor:
        if predicted.data.shape != target.data.shape:
            raise ValueError("Predicted and target tensors must have the same shape.")
        return abs(predicted - target)


class HuberLoss(BaseCriterion):
    def __init__(self):
        super().__init__()

    def __call__(self, predicted: Tensor, target: Tensor, mae_toggle_error=1) -> Tensor:
        if predicted.data.shape != target.data.shape:
            raise ValueError("Predicted and target tensors must have the same shape.")
        error = predicted - target
        return abs(error) if mae_toggle_error > error else error ** 2


__all__ = ["BaseCriterion", "MSELoss", "MAELoss", "HuberLoss"]
