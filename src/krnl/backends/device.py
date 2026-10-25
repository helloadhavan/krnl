import platform
import psutil
from typing import Literal

from .GPU_processer import get_gpu_info


cpu_freq = psutil.cpu_freq()
max_clock = cpu_freq.max if cpu_freq else 0
virtual_mem = psutil.virtual_memory()

info = {
    "platform": platform.system(),
    "device": platform.processor(),
    "max_compute_units": psutil.cpu_count(logical=False),
    "max_work_group_size": psutil.cpu_count(logical=True),
    "global_mem_size": virtual_mem.total,
    "local_mem_size": None,
    "max_clock_frequency": max_clock,
}


class Device(object):
    def __init__(self, device: Literal["CPU", "GPU"], device_id: int = 0):
        self.device = device
        self.device_id = device_id

    def __repr__(self):
        return f"Device(device={self.device}, device_id={self.device_id})"

    def get_device_info(self):
        if self.device == "CPU":
            return info
        elif self.device == "GPU":
            gpu_info = get_gpu_info()
            if self.device_id < len(gpu_info):
                return gpu_info[self.device_id]
            else:
                raise ValueError(f"Invalid GPU device_id: {self.device_id}")
        else:
            raise ValueError(f"Invalid device type: {self.device}")
