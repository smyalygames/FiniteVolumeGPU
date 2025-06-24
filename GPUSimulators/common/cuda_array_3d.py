import logging

import numpy as np
import pycuda.gpuarray
import pycuda.driver as cuda
from pycuda.tools import PageLockedMemoryPool


class CudaArray3D:
    """
    Class that holds 3D data
    """

    def __init__(self, stream, nx, ny, nz, x_halo, y_halo, z_halo, cpu_data=None, dtype=np.float32):
        """
        Uploads initial data to the CL device
        """

        self.logger = logging.getLogger(__name__)
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.x_halo = x_halo
        self.y_halo = y_halo
        self.z_halo = z_halo

        nx_halo = nx + 2 * x_halo
        ny_halo = ny + 2 * y_halo
        nz_halo = nz + 2 * z_halo

        # self.logger.debug("Allocating [%dx%dx%d] buffer", self.nx, self.ny, self.nz)
        # Should perhaps use pycuda.driver.mem_alloc_data.pitch() here
        self.data = pycuda.gpuarray.zeros((nz_halo, ny_halo, nx_halo), dtype)

        # For returning to download
        self.memorypool = PageLockedMemoryPool()

        # If we don't have any data, just allocate and return
        if cpu_data is None:
            return

        # Make sure data is in proper format
        if (cpu_data.shape != (nz_halo, ny_halo, nx_halo)
                and cpu_data.shape != (self.nz, self.ny, self.nx)):
            raise ValueError(f"Wrong shape of data {str(cpu_data.shape)} vs {str((self.nz, self.ny, self.nx))} / {str((nz_halo, ny_halo, nx_halo))}")

        if cpu_data.itemsize != 4:
            raise ValueError("Wrong size of data type")

        if np.isfortran(cpu_data):
            raise TypeError("Wrong datatype (Fortran, expected C)")

        # Create a copy object from host to device
        copy = cuda.Memcpy3D()
        copy.set_src_host(cpu_data)
        copy.set_dst_device(self.data.gpudata)

        # Set offsets of destination
        x_offset = (nx_halo - cpu_data.shape[2]) // 2
        y_offset = (ny_halo - cpu_data.shape[1]) // 2
        z_offset = (nz_halo - cpu_data.shape[0]) // 2
        copy.dst_x_in_bytes = x_offset * self.data.strides[1]
        copy.dst_y = y_offset
        copy.dst_z = z_offset

        # Set pitch of destination
        copy.dst_pitch = self.data.strides[0]

        # Set width in bytes to copy for each row and
        # number of rows to copy
        width = max(self.nx, cpu_data.shape[2])
        height = max(self.ny, cpu_data.shape[1])
        depth = max(self.nz, cpu - data.shape[0])
        copy.width_in_bytes = width * cpu_data.itemsize
        copy.height = height
        copy.depth = depth

        # Perform the copy
        copy(stream)

        # self.logger.debug("Buffer <%s> [%dx%d]: Allocated ", int(self.data.gpudata), self.nx, self.ny)

    def __del__(self, *args):
        # self.logger.debug("Buffer <%s> [%dx%d]: Releasing ", int(self.data.gpudata), self.nx, self.ny)
        self.data.gpudata.free()
        self.data = None

    def download(self, stream, asynch=False):
        """
        Enables downloading data from GPU to Python
        """

        # self.logger.debug("Downloading [%dx%d] buffer", self.nx, self.ny)
        # Allocate host memory
        # cpu_data = cuda.pagelocked_empty((self.ny, self.nx), np.float32)
        # cpu_data = np.empty((self.nz, self.ny, self.nx), dtype=np.float32)
        cpu_data = self.memorypool.allocate((self.nz, self.ny, self.nx), dtype=np.float32)

        # Create a copy object from device to host
        copy = cuda.Memcpy2D()
        copy.set_src_device(self.data.gpudata)
        copy.set_dst_host(cpu_data)

        # Set offsets and pitch of a source
        copy.src_x_in_bytes = self.x_halo * self.data.strides[1]
        copy.src_y = self.y_halo
        copy.src_z = self.z_halo
        copy.src_pitch = self.data.strides[0]

        # Set width in bytes to copy for each row and
        # number of rows to copy
        copy.width_in_bytes = self.nx * cpu_data.itemsize
        copy.height = self.ny
        copy.depth = self.nz

        copy(stream)
        if not asynch:
            stream.synchronize()

        return cpu_data