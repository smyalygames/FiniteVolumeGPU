import pycuda.driver as cuda

from GPUSimulators.gpu import KernelContext
from . import simulator, BoundaryCondition


class CudaSimulator(simulator.BaseSimulator):
    def __init__(self,
                 context: KernelContext,
                 nx: int, ny: int,
                 dx: int, dy: int,
                 boundary_conditions: BoundaryCondition,
                 cfl_scale: float,
                 num_substeps: int,
                 block_width: int, block_height: int):
        super().__init__(context, nx, ny, dx, dy, boundary_conditions, cfl_scale, num_substeps, block_width,
                         block_height)

        # Create a CUDA stream
        self.stream = cuda.Stream()
        self.internal_stream = cuda.Stream()

    def synchronize(self):
        self.stream.synchronize()
