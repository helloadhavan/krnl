from backends.tensor import Tensor
from backends.device import Device

class Parameter(Tensor):
    """
    A Tensor intended to be trained as part of a Module.
    """

    def __init__(self, data, device=None):

        super().__init__(
            data=data,
            device=device if device is not None else Device("CPU"),
            requires_grad=True,
        )

class Module:
    """
    Base class for all neural network modules.
    """

    def __init__(self):
        self._parameters = {}

    def register_parameter(self, name: str, parameter: Parameter) -> None:
        if not isinstance(parameter, Parameter):
            raise TypeError("Only instances of Parameter can be registered.")
        self._parameters[name] = parameter

class Linear(Module):
    """
    A simple linear layer: y = xW^T + b
    """

    def __init__(self, in_features: int, out_features: int, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device if device is not None else Device("CPU")

        # Initialize weights and bias
        weight_data = np.random.randn(out_features, in_features) * 0.01
        bias_data = np.zeros(out_features)

        self.register_parameter("weight", Parameter(weight_data, device=self.device))
        self.register_parameter("bias", Parameter(bias_data, device=self.device))

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError("Input must be a Tensor.")
        if x.data.shape[-1] != self.in_features:
            raise ValueError(f"Expected input with {self.in_features} features, got {x.data.shape[-1]}.")

        return x @ self._parameters["weight"] + self._parameters["bias"]


