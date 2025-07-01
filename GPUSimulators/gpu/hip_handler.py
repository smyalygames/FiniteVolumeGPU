import ctypes

import numpy as np
from hip import hip, hipblas

from GPUSimulators.common import hip_check
from GPUSimulators.gpu.handler import BaseGPUHandler
from GPUSimulators.gpu import KernelContext


class HIPHandler(BaseGPUHandler):
    def __init__(self, context: KernelContext, module, function, arguments,
                 grid_size):
        super().__init__(context, module, function, arguments, grid_size)

        self.kernel = hip_check(hip.hipModuleGetFunction(module, bytes(function, "utf-8")))
        self.context = context

        self.dtype = np.float32
        self.cfl_data_h = np.empty(grid_size, dtype=self.dtype)

        self.num_bytes = self.cfl_data_h.size * self.cfl_data_h.itemsize
        self.cfl_data = hip_check(hip.hipMalloc(self.num_bytes)).configure(
            typestr=np.finfo(self.dtype).dtype.name, shape=grid_size
        )

    def __del__(self):
        hip_check(hip.hipFree(self.cfl_data))

    def prepared_call(self, grid_size, block_size, stream, args):
        if len(grid_size) < 3:
            grid_size = (*grid_size, 1)

        for i in range(len(args)):
            val = args[i]
            if isinstance(val, int) or isinstance(val, np.int32):
                args[i] = ctypes.c_int(val)
            elif isinstance(val, float) or isinstance(val, np.float32):
                args[i] = ctypes.c_float(val)

        args = tuple(args)

        hip_check(hip.hipModuleLaunchKernel(
            self.kernel,
            *grid_size,
            *block_size,
            0,
            stream,
            None,
            args
        ))


    def array_fill(self, data, stream):
        self.cfl_data_h.fill(data)

        hip_check(
            hip.hipMemcpyAsync(self.cfl_data, self.cfl_data_h, self.num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice,
                               stream))

    def array_min(self, stream):
        handle = hip_check(hipblas.hipblasCreate())

        value = np.empty(1, self.dtype)
        hip_check(hipblas.hipblasIsamin(handle, self.cfl_data.size, self.cfl_data, 1, value))
        hip_check(hipblas.hipblasDestroy(handle))

        hip_check(hip.hipMemcpy(value, self.cfl_data, self.cfl_data_h.itemsize, hip.hipMemcpyKind.hipMemcpyDeviceToHost))

        return value[0]
