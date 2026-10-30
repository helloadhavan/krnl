import pickle
import numpy as np
from ..backends.tensor import Tensor
from ..backends.device import Device
from ..nn.model_utils import Parameter


def _numpy_to_tensor(value, device):
    """Convert a numpy array back into a Parameter (Tensor with requires_grad)."""
    if isinstance(value, np.ndarray):
        return Parameter(value, device=device)
    return value


def _reconstruct_value(value, device):
    """Recursively reconstruct Tensors/Parameters from serialized numpy data."""
    if isinstance(value, dict):
        return {k: _reconstruct_value(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        reconstructed = [_reconstruct_value(item, device) for item in value]
        return type(value)(reconstructed)
    return _numpy_to_tensor(value, device)


def load(path: str, device: Device | None = None) -> dict | Parameter:
    """Load a saved state dict or tensor from a pickle file.

    Parameters
    ----------
    path : str
        File path to load from (as written by ``save``).
    device : Device | None
        Target device for reconstructed tensors.  Defaults to CPU.

    Returns
    -------
    dict | Parameter
        The deserialized data with numpy arrays wrapped back into
        ``Parameter`` objects.
    """
    if device is None:
        device = Device("CPU")

    with open(path, "rb") as f:
        raw = pickle.load(f)

    if isinstance(raw, dict):
        return _reconstruct_value(raw, device)

    return _numpy_to_tensor(raw, device)
