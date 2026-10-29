import pickle
from ..backends.tensor import Tensor

def save(data: dict | Tensor, path: str) -> None:
    safe_data = {} if isinstance(data, dict) else data.tolist()
    if isinstance(data, dict):
        for k, v in data.items():
            safe_data[k] = v if not Tensor(v) else v.tolist()
    pickle.dump(safe_data, open(path, "wb"))