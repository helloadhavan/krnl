from ..backends.tensor import Tensor, rand
from ..backends.device import Device


class Parameter(Tensor):
    def __init__(self, data, device=None):
        super().__init__(
            data=data,
            device=device if device is not None else Device("CPU"),
            requires_grad=True,
        )


class Module:
    def __init__(self):
        self._parameters = {}

    def register_parameter(self, name: str, parameter: Parameter) -> None:
        if not isinstance(parameter, Parameter):
            raise TypeError("Only instances of Parameter can be registered.")
        self._parameters[name] = parameter

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("Subclasses must implement the forward method.")

    def zero_grad(self) -> None:
        for param in self._parameters.values():
            param.grad = None

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias=False, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device if device is not None else Device("CPU")
        self.bias = bias

        weight_data = rand((out_features, in_features), device=self.device)
        bias_data = rand((out_features,), device=self.device) if bias else None

        self.register_parameter("weight", Parameter(weight_data, device=self.device))
        if bias:
            self.register_parameter("bias", Parameter(bias_data, device=self.device))

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError("Input must be a Tensor.")

        if x.data.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected input with {self.in_features} features, "
                f"got {x.data.shape[-1]}."
            )

        weight = self._parameters["weight"]
        output = x @ weight

        if self.bias:
            bias = self._parameters["bias"]

            if len(bias.data.shape) == 1 and len(output.data.shape) == 2:
                bias = bias.reshape((1, self.out_features))

            output = output + bias

        return output


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError("Input must be a Tensor.")
        return x.relu()


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError("Input must be a Tensor.")
        return x.sigmoid()


__all__ = [
    "Tensor",
    "Parameter",
    "Module",
    "Linear",
    "ReLU",
    "Sigmoid",
]
