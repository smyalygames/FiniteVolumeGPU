import numpy as np

import pycuda.gpuarray
import pycuda.driver as cuda
from pycuda.tools import PageLockedMemoryPool

from GPUSimulators.common.arrays.array2d import BaseArray2D


class CudaArray2D(BaseArray2D):
    """
    Class that holds 2D CUDA data
    """

    def __init__(self, stream, nx, ny, x_halo, y_halo, cpu_data=None, dtype=np.float32):
        """
        Uploads initial data to the CUDA device
        """

        super().__init__(nx, ny, x_halo, y_halo, cpu_data)
        # self.logger.debug("Allocating [%dx%d] buffer", self.nx, self.ny)
        # Should perhaps use pycuda.driver.mem_alloc_data.pitch() here
        self.data = pycuda.gpuarray.zeros(self.shape, dtype)

        # For returning to download
        self.memorypool = PageLockedMemoryPool()

        # Create a copy object from host to device
        x = (self.shape[0] - cpu_data.shape[1]) // 2
        y = (self.shape[1] - cpu_data.shape[0]) // 2
        self.upload(stream, cpu_data, extent=[x, y, cpu_data.shape[1], cpu_data.shape[0]])
        # self.logger.debug("Buffer <%s> [%dx%d]: Allocated ", int(self.data.gpudata), self.nx, self.ny)

    def __del__(self, *args):
        # self.logger.debug("Buffer <%s> [%dx%d]: Releasing ", int(self.data.gpudata), self.nx, self.ny)
        self.data.gpudata.free()
        self.data = None

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
            # The following fails, don't know why (crashes python)
            cpu_data = cuda.pagelocked_empty((int(ny), int(nx)), dtype=np.float32,
                                             mem_flags=cuda.host_alloc_flags.PORTABLE)
            # Non-pagelocked: cpu_data = np.empty((ny, nx), dtype=np.float32)
            # cpu_data = self.memorypool.allocate((ny, nx), dtype=np.float32)

        self.check(x, y, nx, ny, cpu_data)

        # Create a copy object from device to host
        copy = cuda.Memcpy2D()
        copy.set_src_device(self.data.gpudata)
        copy.set_dst_host(cpu_data)

        # Set offsets and pitch of a source
        copy.src_x_in_bytes = int(x) * self.data.strides[1]
        copy.src_y = int(y)
        copy.src_pitch = self.data.strides[0]

        # Set width in bytes to copy for each row and
        # number of rows to copy
        copy.width_in_bytes = int(nx) * cpu_data.itemsize
        copy.height = int(ny)

        copy(stream)
        if not asynch:
            stream.synchronize()

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

        # Create a copy object from device to host
        copy = cuda.Memcpy2D()
        copy.set_dst_device(self.data.gpudata)
        copy.set_src_host(cpu_data)

        # Set offsets and pitch of a source
        copy.dst_x_in_bytes = int(x) * self.data.strides[1]
        copy.dst_y = int(y)
        copy.dst_pitch = self.data.strides[0]

        # Set width in bytes to copy for each row and
        # number of rows to copy
        copy.width_in_bytes = int(nx) * cpu_data.itemsize
        copy.height = int(ny)

        copy(stream)
