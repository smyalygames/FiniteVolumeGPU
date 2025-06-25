from os import environ

__env_name = 'GPU_LANG'

if __env_name in environ and environ.get(__env_name).lower() == "cuda":
    from .cuda_simulator import CudaSimulator as BaseSimulator
else:
    from .hip_simulator import HIPSimulator as BaseSimulator

from .boundary import BoundaryCondition
