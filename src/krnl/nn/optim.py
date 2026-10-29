from __future__ import annotations

import numpy as np

from .model_utils import Parameter


class Optimizer:
    def __init__(self, parameters, defaults=None):
        self.param = list(parameters)
        self.state = defaults if defaults is not None else {}

    def zero_grad(self):
        for parameter in self.param:
            parameter.grad = None

    def step(self):
        raise NotImplementedError("Optimizer subclasses must implement step().")

    def get_state_dict(self) -> dict:
        return {"parameters": self.param, "state": self.state}


class SGD(Optimizer):
    def __init__(self, parameters, learning_rate=0.01):
        self.learning_rate = learning_rate
        defaults = {"lr": learning_rate}
        super().__init__(parameters, defaults)

    def step(self):
        for parameter in self.param:
            if parameter.grad is None:
                continue
            parameter.data -= self.learning_rate * parameter.grad


class Adam(Optimizer):
    def __init__(self, parameters, learning_rate=0.001, betas=(0.9, 0.999), epsilon=1e-8):
        parameters = list(parameters)

        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("beta1 must be between 0 and 1.")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("beta2 must be between 0 and 1.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")
        if epsilon <= 0:
            raise ValueError("epsilon must be greater than zero.")

        self.learning_rate = learning_rate
        self.beta1, self.beta2 = betas
        self.epsilon = epsilon
        self.step_count = 0
        self.state = {}

        super().__init__(parameters, {"lr": learning_rate, "betas": betas, "eps": epsilon})

        for parameter in parameters:
            self.state[id(parameter)] = {
                "m": self._zeros_like_data(parameter),
                "v": self._zeros_like_data(parameter),
            }

    @staticmethod
    def _zeros_like_data(parameter):
        data = parameter.data
        if hasattr(data, "queue"):
            import pyopencl.array as cl_array
            return cl_array.zeros(data.queue, data.shape, dtype=np.float32)
        return np.zeros_like(data, dtype=np.float32)

    def step(self):
        self.step_count += 1
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

            m *= self.beta1
            m += (1.0 - self.beta1) * gradient
            v *= self.beta2
            v += (1.0 - self.beta2) * (gradient * gradient)

            m_hat = m / (1.0 - self.beta1 ** self.step_count)
            v_hat = v / (1.0 - self.beta2 ** self.step_count)
            parameter.data -= self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.epsilon))


class Muon(Optimizer):
    def __init__(self, parameters, learning_rate=0.02, momentum=0.9, weight_decay=0.0, eps=1e-8):
        parameters = list(parameters)

        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in the range [0, 1).")
        if eps <= 0:
            raise ValueError("eps must be greater than zero.")

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.eps = eps

        super().__init__(parameters, {"lr": learning_rate, "momentum": momentum, "weight_decay": weight_decay, "eps": eps})

        for parameter in parameters:
            self.state[id(parameter)] = {"momentum": self._zeros_like_data(parameter)}

    @staticmethod
    def _zeros_like_data(parameter):
        data = parameter.data
        if hasattr(data, "queue"):
            import pyopencl.array as cl_array
            return cl_array.zeros(data.queue, data.shape, dtype=np.float32)
        return np.zeros_like(data, dtype=np.float32)

    @staticmethod
    def _to_numpy(array):
        if hasattr(array, "get"):
            return np.asarray(array.get(), dtype=np.float32)
        return np.asarray(array, dtype=np.float32)

    @staticmethod
    def _to_parameter_dtype(array, parameter):
        if hasattr(parameter.data, "queue"):
            import pyopencl.array as cl_array
            return cl_array.to_device(parameter.data.queue, np.asarray(array, dtype=np.float32))
        return np.asarray(array, dtype=np.float32)

    @staticmethod
    def _orthogonalize(matrix):
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim == 0:
            return matrix
        if matrix.ndim == 1:
            norm = np.linalg.norm(matrix)
            if norm <= 0:
                return matrix
            return matrix / norm
        if matrix.ndim == 2:
            u, _, vh = np.linalg.svd(matrix, full_matrices=False)
            return u @ vh

        flat = matrix.reshape(matrix.shape[0], -1)
        u, _, vh = np.linalg.svd(flat, full_matrices=False)
        return (u @ vh).reshape(matrix.shape)

    def step(self):
        for parameter in self.param:
            if parameter.grad is None:
                continue

            parameter_id = id(parameter)
            if parameter_id not in self.state:
                self.state[parameter_id] = {"momentum": np.zeros_like(self._to_numpy(parameter.data), dtype=np.float32)}

            state = self.state[parameter_id]
            grad = self._to_numpy(parameter.grad)
            parameter_data = self._to_numpy(parameter.data)
            momentum = self._to_numpy(state["momentum"])

            if self.weight_decay != 0.0:
                grad = grad + (self.weight_decay * parameter_data)

            momentum = self.momentum * momentum + (1.0 - self.momentum) * grad
            update = self._orthogonalize(momentum)
            update_norm = np.linalg.norm(update)
            if update_norm > self.eps:
                update = update / (update_norm + self.eps)

            state["momentum"] = momentum

            if hasattr(parameter.data, "queue"):
                gpu_update = self._to_parameter_dtype(update, parameter)
                parameter.data -= self.learning_rate * gpu_update
            else:
                parameter.data -= self.learning_rate * update


class MuonAdam(Optimizer):
    def __init__(self, parameters, learning_rate=0.02, betas=(0.9, 0.999), epsilon=1e-8, weight_decay=0.0):
        parameters = list(parameters)

        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("beta1 must be between 0 and 1.")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("beta2 must be between 0 and 1.")
        if epsilon <= 0:
            raise ValueError("epsilon must be greater than zero.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")

        self.learning_rate = learning_rate
        self.beta1, self.beta2 = betas
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.step_count = 0

        super().__init__(parameters, {"lr": learning_rate, "betas": betas, "eps": epsilon, "weight_decay": weight_decay})

        for parameter in parameters:
            self.state[id(parameter)] = {"m": self._zeros_like_data(parameter), "v": self._zeros_like_data(parameter)}

    @staticmethod
    def _zeros_like_data(parameter: Parameter):
        data = parameter.data
        if hasattr(data, "queue"):
            import pyopencl.array as cl_array
            return cl_array.zeros(data.queue, data.shape, dtype=np.float32)
        return np.zeros_like(data, dtype=np.float32)

    @staticmethod
    def _to_numpy(array):
        if hasattr(array, "get"):
            return np.asarray(array.get(), dtype=np.float32)
        return np.asarray(array, dtype=np.float32)

    @staticmethod
    def _to_parameter_dtype(array, parameter):
        if hasattr(parameter.data, "queue"):
            import pyopencl.array as cl_array
            return cl_array.to_device(parameter.data.queue, np.asarray(array, dtype=np.float32))
        return np.asarray(array, dtype=np.float32)

    @staticmethod
    def _orthogonalize(matrix):
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim == 0:
            return matrix
        if matrix.ndim == 1:
            norm = np.linalg.norm(matrix)
            if norm <= 0:
                return matrix
            return matrix / norm
        if matrix.ndim == 2:
            u, _, vh = np.linalg.svd(matrix, full_matrices=False)
            return u @ vh

        flat = matrix.reshape(matrix.shape[0], -1)
        u, _, vh = np.linalg.svd(flat, full_matrices=False)
        return (u @ vh).reshape(matrix.shape)

    def step(self):
        self.step_count += 1

        for parameter in self.param:
            if parameter.grad is None:
                continue

            parameter_id = id(parameter)
            if parameter_id not in self.state:
                self.state[parameter_id] = {"m": self._zeros_like_data(parameter), "v": self._zeros_like_data(parameter)}

            state = self.state[parameter_id]
            grad = self._to_numpy(parameter.grad)
            parameter_data = self._to_numpy(parameter.data)
            m = self._to_numpy(state["m"])
            v = self._to_numpy(state["v"])

            if self.weight_decay != 0.0:
                grad = grad + (self.weight_decay * parameter_data)

            m = self.beta1 * m + (1.0 - self.beta1) * grad
            v = self.beta2 * v + (1.0 - self.beta2) * (grad * grad)
            m_hat = m / (1.0 - self.beta1 ** self.step_count)
            v_hat = v / (1.0 - self.beta2 ** self.step_count)
            update = m_hat / (np.sqrt(v_hat) + self.epsilon)
            update = self._orthogonalize(update)
            update_norm = np.linalg.norm(update)
            if update_norm > self.epsilon:
                update = update / (update_norm + self.epsilon)

            state["m"] = m
            state["v"] = v

            if hasattr(parameter.data, "queue"):
                gpu_update = self._to_parameter_dtype(update, parameter)
                parameter.data -= self.learning_rate * gpu_update
            else:
                parameter.data -= self.learning_rate * update


MuonWithAdamAux = MuonAdam
MuonWithAdamAuxiliary = MuonAdam
MuonAdamAux = MuonAdam
AdamAux = MuonAdam

__all__ = [
    "Optimizer",
    "SGD",
    "Adam",
    "Muon",
    "MuonAdam",
    "MuonWithAdamAux",
    "MuonWithAdamAuxiliary",
    "MuonAdamAux",
    "AdamAux",
]
