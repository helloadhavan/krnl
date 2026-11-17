from krnl.backends.tensor import Tensor

class Dataset(object):
    def __init__(self, inputs: Tensor | list[Tensor], labels: Tensor | list[Tensor]):
        self.inputs = inputs
        self.labels = labels
