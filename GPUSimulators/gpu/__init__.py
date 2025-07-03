from os import environ

__env_name = 'GPU_LANG'

if __env_name in environ and environ.get(__env_name).lower() == "cuda":
    from .cuda_context import CudaContext as KernelContext
    from .cuda_handler import CudaHandler as GPUHandler
    from .cuda_event import CudaEvent as Event
else:
    from .hip_context import HIPContext as KernelContext
    from .hip_handler import HIPHandler as GPUHandler
    from .hip_event import HIPEvent as Event