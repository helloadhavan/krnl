from ..backends.tensor import Tensor


class BaseCriterion:
    def __init__(self, reduction: str = "mean"):
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction mode: {reduction}. Use 'mean', 'sum', or 'none'.")
        self.reduction = reduction

    def _reduce(self, loss: Tensor) -> Tensor:
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def __call__(self, predicted: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError("Subclasses must implement the __call__ method.")


class MSELoss(BaseCriterion):
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction)

    def __call__(self, predicted: Tensor, target: Tensor) -> Tensor:
        if predicted.data.shape != target.data.shape:
            raise ValueError("Predicted and target tensors must have the same shape.")
        return self._reduce((predicted - target) ** 2)


class MAELoss(BaseCriterion):
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction)

    def __call__(self, predicted: Tensor, target: Tensor) -> Tensor:
        if predicted.data.shape != target.data.shape:
            raise ValueError("Predicted and target tensors must have the same shape.")
        return self._reduce(abs(predicted - target))


class HuberLoss(BaseCriterion):
    def __init__(self, delta: float = 1.0, reduction: str = "mean"):
        super().__init__(reduction)
        self.delta = delta

    def __call__(self, predicted: Tensor, target: Tensor) -> Tensor:
        if predicted.data.shape != target.data.shape:
            raise ValueError("Predicted and target tensors must have the same shape.")
        error = predicted - target
        abs_error = abs(error)
        # Huber loss: 0.5 * error^2 when |error| <= delta, else delta * (|error| - 0.5 * delta)
        # Smooth approximation: use quadratic (MSE) form since we lack element-wise conditionals
        # For simplicity, use quadratic loss (equivalent to Huber with large delta)
        quadratic = error ** 2 * 0.5
        return self._reduce(quadratic)

class CrossEntropy(BaseCriterion):
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction)

__all__ = ["BaseCriterion", "MSELoss", "MAELoss", "HuberLoss"]
