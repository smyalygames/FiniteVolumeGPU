import numpy as np
from pycuda import gpuarray

from GPUSimulators.gpu.handler import BaseGPUHandler
from GPUSimulators.gpu import KernelContext


class CudaHandler(BaseGPUHandler):
    def __init__(self, context: KernelContext, module, function, arguments,
                 grid_size):
        super().__init__(context, module, function, arguments, grid_size)

        self.arguments = arguments

        self.kernel = module.get_function(function)
        self.kernel.prepare(arguments)

        self.cfl_data = gpuarray.GPUArray(grid_size, dtype=np.float32)

    def prepared_call(self, grid_size, block_size, stream, args: list):
        # if len(args) != len(self.arguments):
        #     raise ValueError("The parameters do not match the defined arguments.")

        self.kernel.prepared_async_call(grid_size, block_size, stream, *args)

    def array_fill(self, data, stream):
        self.cfl_data.fill(data, stream=stream)

    def array_min(self, stream):
        return gpuarray.min(self.cfl_data, stream=stream).get()
