from krnl.backends.tensor import Tensor

class Dataset(object):
    def __init__(self, inputs: Tensor, labels: Tensor):
        self.inputs = inputs
        self.labels = labels
        if not isinstance(inputs, Tensor) or isinstance(labels, Tensor):
            raise TypeError(f"inputs and labels must be tensors")

    def __getitem__(self, item) -> tuple[Tensor, Tensor]:
        return self.inputs[item], self.labels[item]
