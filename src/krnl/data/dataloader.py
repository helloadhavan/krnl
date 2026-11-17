from __future__ import annotations

import math
import numpy as np
from krnl.data.dataset import Dataset


class DataLoader:
    """Iterable mini-batch loader for a :class:`Dataset`.

    Parameters
    ----------
    dataset : Dataset
        The dataset to load batches from.
    batch_size : int, optional
        Number of samples per batch (default ``32``).
    shuffle : bool, optional
        Whether to shuffle the sample indices at the start of each
        iteration pass (default ``False``).
    drop_last : bool, optional
        If ``True``, the last batch is dropped when the dataset size is
        not evenly divisible by ``batch_size`` (default ``False``).
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        if not isinstance(dataset, Dataset):
            raise TypeError("dataset must be an instance of Dataset")
        if batch_size < 1:
            raise ValueError("batch_size must be a positive integer")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self._num_samples = self.dataset.inputs.shape[0]

    def __len__(self) -> int:
        """Return the number of batches per full pass over the dataset."""
        if self.drop_last:
            return self._num_samples // self.batch_size
        return math.ceil(self._num_samples / self.batch_size)

    def __iter__(self):
        """Yield ``(inputs, labels)`` :class:`Tensor` pairs, one batch at a time."""
        indices = np.arange(self._num_samples)

        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, self._num_samples, self.batch_size):
            end = start + self.batch_size

            if end > self._num_samples and self.drop_last:
                break

            batch_indices = indices[start:end].tolist()
            batch_inputs = self.dataset.inputs[batch_indices]
            batch_labels = self.dataset.labels[batch_indices]

            yield batch_inputs, batch_labels, start
