import pickle
import numpy as np
from ..backends.tensor import Tensor
from ..backends.GPU_processer import GPUArray


def _tensor_to_serializable(value):
    """Extract raw numpy data from a Tensor or GPUArray."""
    if isinstance(value, Tensor):
        data = value.data
        if isinstance(data, GPUArray):
            return np.asarray(data.get(), dtype=np.float32)
        return np.asarray(data, dtype=np.float32)
    return value


def _convert_value(value):
    """Recursively convert a value so it is safe for pickling."""
    if isinstance(value, dict):
        return {k: _convert_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        converted = [_convert_value(item) for item in value]
        return type(value)(converted)
    return _tensor_to_serializable(value)


def save(data: dict | Tensor, path: str) -> None:
    if isinstance(data, dict):
        safe_data = {k: _convert_value(v) for k, v in data.items()}
    else:
        safe_data = _tensor_to_serializable(data)

    with open(path, "wb") as f:
        pickle.dump(safe_data, f)