from __future__ import annotations

import numpy as np

from src.nn.model_utils import Parameter


class Optimizer:
    """
    Base class for optimizers.
    """

    def __init__(self, parameters, defaults=None):
        self.param = list(parameters)
        self.state = defaults if defaults is not None else {}

    def zero_grad(self):
        """
        Clear gradients before a new backward pass.
        """
        for parameter in self.param:
            parameter.grad = None

    def step(self):
        raise NotImplementedError(
            "Optimizer subclasses must implement step()."
        )


class SGD(Optimizer):
    """
    Stochastic Gradient Descent.

        parameter = parameter - learning_rate * gradient
    """

    def __init__(self, parameters, learning_rate=0.01):
        self.learning_rate = learning_rate

        defaults = {
            "lr": learning_rate,
        }

        super().__init__(
            parameters,
            defaults,
        )

    def step(self):
        """
        Update every parameter with its gradient.
        """
        for parameter in self.param:
            if parameter.grad is None:
                continue

            parameter.data -= (
                self.learning_rate * parameter.grad
            )


class Adam(Optimizer):
    """
    Adam optimizer.

    Adam maintains:

        m_t = beta1 * m_(t-1) + (1 - beta1) * gradient
        v_t = beta2 * v_(t-1) + (1 - beta2) * gradient²

    Then applies bias correction before updating parameters.
    """

    def __init__(
        self,
        parameters,
        learning_rate=0.001,
        betas=(0.9, 0.999),
        epsilon=1e-8,
    ):
        parameters = list(parameters)

        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "beta1 must be between 0 and 1."
            )

        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "beta2 must be between 0 and 1."
            )

        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be greater than zero."
            )

        if epsilon <= 0:
            raise ValueError(
                "epsilon must be greater than zero."
            )

        self.learning_rate = learning_rate
        self.beta1, self.beta2 = betas
        self.epsilon = epsilon

        self.step_count = 0

        self.state = {}

        super().__init__(
            parameters,
            {
                "lr": learning_rate,
                "betas": betas,
                "eps": epsilon,
            },
        )

        for parameter in parameters:
            self.state[id(parameter)] = {
                "m": self._zeros_like_data(parameter),
                "v": self._zeros_like_data(parameter),
            }

    @staticmethod
    def _zeros_like_data(parameter):
        """
        Create a zero array with the same shape, dtype, and device
        as parameter.data.

        This supports both NumPy CPU arrays and PyOpenCL arrays.
        """
        data = parameter.data

        if hasattr(data, "queue"):
            import pyopencl.array as cl_array

            return cl_array.zeros(
                data.queue,
                data.shape,
                dtype=np.float32,
            )

        return np.zeros_like(
            data,
            dtype=np.float32,
        )

    def step(self):
        """
        Apply one Adam update to every parameter.
        """
        self.step_count += 1

        beta1 = self.beta1
        beta2 = self.beta2

        for parameter in self.param:
            if parameter.grad is None:
                continue

            parameter_id = id(parameter)

            if parameter_id not in self.state:
                self.state[parameter_id] = {
                    "m": self._zeros_like_data(parameter),
                    "v": self._zeros_like_data(parameter),
                }

            state = self.state[parameter_id]

            gradient = parameter.grad
            m = state["m"]
            v = state["v"]

            m *= beta1
            m += (1.0 - beta1) * gradient

            v *= beta2
            v += (1.0 - beta2) * (gradient * gradient)

            m_hat = m / (
                1.0 - beta1 ** self.step_count
            )

            v_hat = v / (
                1.0 - beta2 ** self.step_count
            )

            parameter.data -= (
                self.learning_rate
                * m_hat
                / (np.sqrt(v_hat) + self.epsilon)
            )