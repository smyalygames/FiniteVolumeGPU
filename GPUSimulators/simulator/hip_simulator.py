from hip import hip

from GPUSimulators.common import hip_check
from GPUSimulators.gpu import KernelContext
from . import BaseSimulator, BoundaryCondition


class HIPSimulator(BaseSimulator):
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

        # Create a HIP stream
        self.stream = hip_check(hip.hipStreamCreate())
        self.internal_stream = hip_check(hip.hipStreamCreate())

    def __del__(self):
        # Destroy the streams
        hip_check(hip.hipStreamDestroy(self.stream))
        hip_check(hip.hipStreamDestroy(self.internal_stream))

    def synchronize(self):
        hip_check(hip.hipStreamSynchronize(self.stream))
