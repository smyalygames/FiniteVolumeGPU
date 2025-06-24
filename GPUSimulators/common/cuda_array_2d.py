import logging

import numpy as np

import pycuda.gpuarray
import pycuda.driver as cuda
from pycuda.tools import PageLockedMemoryPool


class CudaArray2D:
    """
    Class that holds 2D CUDA data
    """

    def __init__(self, stream, nx, ny, x_halo, y_halo, cpu_data=None, dtype=np.float32):
        """
        Uploads initial data to the CUDA device
        """

        self.logger = logging.getLogger(__name__)
        self.nx = nx
        self.ny = ny
        self.x_halo = x_halo
        self.y_halo = y_halo

        nx_halo = nx + 2 * x_halo
        ny_halo = ny + 2 * y_halo

        # self.logger.debug("Allocating [%dx%d] buffer", self.nx, self.ny)
        # Should perhaps use pycuda.driver.mem_alloc_data.pitch() here
        self.data = pycuda.gpuarray.zeros((ny_halo, nx_halo), dtype)

        # For returning to download
        self.memorypool = PageLockedMemoryPool()

        # If we don't have any data, just allocate and return
        if cpu_data is None:
            return

        # Make sure data is in proper format
        if cpu_data.shape != (ny_halo, nx_halo) and cpu_data.shape != (self.ny, self.nx):
            raise ValueError(
                f"Wrong shape of data {str(cpu_data.shape)} vs {str((self.ny, self.nx))} / {str((ny_halo, nx_halo))}")

        if cpu_data.itemsize != 4:
            raise ValueError("Wrong size of data type")

        if np.isfortran(cpu_data):
            raise TypeError("Wrong datatype (Fortran, expected C)")

        # Create a copy object from host to device
        x = (nx_halo - cpu_data.shape[1]) // 2
        y = (ny_halo - cpu_data.shape[0]) // 2
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

        assert nx == cpu_data.shape[1]
        assert ny == cpu_data.shape[0]
        assert x + nx <= self.nx + 2 * self.x_halo
        assert y + ny <= self.ny + 2 * self.y_halo

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

        assert (nx == cpu_data.shape[1])
        assert (ny == cpu_data.shape[0])
        assert (x + nx <= self.nx + 2 * self.x_halo)
        assert (y + ny <= self.ny + 2 * self.y_halo)

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