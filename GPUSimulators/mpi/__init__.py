from os import environ

__env_name = 'GPU_LANG'

if __env_name in environ and environ.get(__env_name).lower() == "cuda":
    from .cuda_simulator import CudaMPISimulator as MPISimulator
else:
    from .hip_simulator import HIPMPISimulator as MPISimulator

from .grid import MPIGrid
