from os import environ

__env_name = 'GPU_LANG'

if __env_name in environ and environ.get(__env_name).lower() == "cuda":
    from .cuda.arkawa2d import CudaArakawaA2D as ArakawaA2D
    from .cuda.array2d import CudaArray2D as Array2D
    from .cuda.array3d import CudaArray3D as Array3D
else:
    from .hip.arkawa2d import HIPArakawaA2D as ArakawaA2D
    from .hip.array2d import HIPArray2D as Array2D
