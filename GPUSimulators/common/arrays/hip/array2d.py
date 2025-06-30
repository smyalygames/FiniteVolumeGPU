import numpy as np
from hip import hip

from ...hip_check import hip_check
from ..array2d import BaseArray2D


class HIPArray2D(BaseArray2D):
    """
    Class that holds 2D HIP data
    """

    def __init__(self, stream, nx, ny, x_halo, y_halo, cpu_data=None, dtype: np.dtype = np.float32):
        """
        Uploads initial data to the HIP device
        """

        super().__init__(nx, ny, x_halo, y_halo, cpu_data)
        # self.logger.debug("Allocating [%dx%d] buffer", self.nx, self.ny)
        self.dtype = dtype
        self.data_h = np.zeros(self.shape, self.dtype)
        self.num_bytes = self.data_h.size * self.data_h.itemsize

        self.data = hip_check(hip.hipMalloc(self.num_bytes)).configure(
            typestr=np.finfo(self.dtype).dtype.name, shape=self.shape
        )

        # If there is no data to append, just leave this array as allocated
        if cpu_data is None:
            return

        # Create a copy object from host to device
        x = (self.shape[0] - cpu_data.shape[1]) // 2
        y = (self.shape[1] - cpu_data.shape[0]) // 2
        self.upload(stream, cpu_data, extent=[x, y, cpu_data.shape[1], cpu_data.shape[0]])
        # self.logger.debug("Buffer <%s> [%dx%d]: Allocated ", int(self.data.gpudata), self.nx, self.ny)

    def __del__(self, *args):
        # self.logger.debug("Buffer <%s> [%dx%d]: Releasing ", int(self.data.gpudata), self.nx, self.ny)
        hip_check(hip.hipFree(self.data))

    def download(self, stream, cpu_data=None, asynch=False, extent=None):
        """
        Enables downloading data from GPU to Python
        """

        if extent is None:
            x = self.x_halo
            y = self.y_halo
            nx = self.nx
            ny = self.ny
        else:
            x, y, nx, ny = extent

        if cpu_data is None:
            # self.logger.debug("Downloading [%dx%d] buffer", self.nx, self.ny)
            # Allocate host memory
            cpu_data = np.empty((ny, nx), dtype=self.dtype)

        self.check(x, y, nx, ny, cpu_data)

        if not asynch:
            hip_check(hip.hipStreamSynchronize(stream))

        hip_check(
            hip.hipMemcpyAsync(self.data, cpu_data, self.num_bytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost, stream))

        return cpu_data

    def upload(self, stream, cpu_data, extent=None):
        if extent is None:
            x = self.x_halo
            y = self.y_halo
            nx = self.nx
            ny = self.ny
        else:
            x, y, nx, ny = extent

        self.check(x, y, nx, ny, cpu_data)

        # TODO implement non-async to test if it actually works - avoid errors
        # Create a copy object from device to host
        hip_check(hip.hipMemcpyAsync(self.data, self.data_h, self.num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice,
                                     stream))
