from enum import Enum

import numpy as np
from hip import hip

from ...hip_check import hip_check
from ..array2d import BaseArray2D


class TransferType(Enum):
    HOST_TO_DEVICE = 0
    DEVICE_TO_HOST = 1


class HIPArray2D(BaseArray2D):
    """
    Class that holds 2D HIP data
    """

    def __init__(self, stream: hip.ihipStream_t, nx: int, ny: int, x_halo: int, y_halo: int,
                 cpu_data: np.ndarray = None, dtype: np.dtype = np.float32()):
        """
        Uploads initial data to the HIP device
        """

        super().__init__(nx, ny, x_halo, y_halo, cpu_data)
        # self.logger.debug("Allocating [%dx%d] buffer", self.nx, self.ny)
        self.dtype = dtype

        self.data_h = np.zeros(self.shape, self.dtype)

        shape_x = self.shape[0]
        shape_y = self.shape[1]

        self.width = shape_x * self.dtype.itemsize
        self.height = shape_y

        self.num_bytes = self.width * self.height

        self.data, self.pitch_d = hip_check(hip.hipMallocPitch(self.width, self.height))

        # Initialise the memory with an array of zeros.
        init_h = np.zeros(self.shape, self.dtype)
        self.pitch_h = shape_x * init_h.itemsize
        hip_check(hip.hipMemcpy2DAsync(self.data, self.pitch_d,
                                       init_h, self.pitch_h,
                                       self.width, self.height,
                                       hip.hipMemcpyKind.hipMemcpyHostToDevice, stream))

        # If there is no data to append, just leave this array as allocated
        if cpu_data is None:
            return

        host_x = cpu_data.shape[1]
        host_y = cpu_data.shape[0]

        # Create a copy object from host to device
        x = (shape_x - host_y) // 2
        y = (shape_y - host_x) // 2
        self.upload(stream, cpu_data, extent=(x, y, host_x, host_y))
        # self.logger.debug("Buffer <%s> [%dx%d]: Allocated ", int(self.data.gpudata), self.nx, self.ny)

    def __del__(self, *args):
        # self.logger.debug("Buffer <%s> [%dx%d]: Releasing ", int(self.data.gpudata), self.nx, self.ny)
        hip_check(hip.hipFree(self.data))

    def download(self, stream: hip.ihipStream_t, cpu_data: np.ndarray = None, asynch=False,
                 extent: tuple[int, int, int, int] = None) -> np.ndarray:
        """
        Enables downloading data from GPU to Python
        Args:
            stream: The GPU stream to add the memory copy to.
            cpu_data: The array to store the data copied from GPU memory.
            asynch: Synchronize the stream before returning `cpu_data`.
            extent: Parameters for where in the GPU memory to copy from.
        Returns:
            `cpu_data` with the data from the GPU memory.
            Note the data in `cpu_data` may be uninitialized if `asynch` was not set to `True`.
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
            cpu_data = np.zeros((ny, nx), dtype=self.dtype)

        self.check(x, y, nx, ny, cpu_data)

        pitch_h, width, height = self.__get_array_vars(cpu_data, nx, ny)

        # Parameters to copy to GPU memory
        copy = hip.hip_Memcpy2D(
            srcDevice=self.data,
            srcPitch=self.pitch_d,
            srcXInBytes=x * self.dtype.itemsize,
            srcY=y,
            srcMemoryType=hip.hipMemoryType.hipMemoryTypeDevice,

            dstHost=cpu_data,
            dstPitch=pitch_h,
            dstMemoryType=hip.hipMemoryType.hipMemoryTypeHost,

            WidthInBytes=width,
            Height=height
        )

        hip_check(hip.hipMemcpyParam2DAsync(copy, stream))

        if not asynch:
            hip_check(hip.hipStreamSynchronize(stream))

        return cpu_data

    def upload(self, stream: hip.ihipStream_t, cpu_data: np.ndarray, extent: tuple[int, int, int, int] = None):
        if extent is None:
            x = self.x_halo
            y = self.y_halo
            nx = self.nx
            ny = self.ny
        else:
            x, y, nx, ny = extent

        pitch_h, width, height = self.__get_array_vars(cpu_data, nx, ny)

        self.check(x, y, nx, ny, cpu_data)

        # Parameters to copy to GPU memory
        copy = hip.hip_Memcpy2D(
            srcHost = cpu_data,
            srcPitch = pitch_h,
            srcMemoryType = hip.hipMemoryType.hipMemoryTypeHost,

            dstDevice = self.data,
            dstPitch = self.pitch_d,
            dstXInBytes = x * self.dtype.itemsize,
            dstY = y,
            dstMemoryType = hip.hipMemoryType.hipMemoryTypeDevice,

            WidthInBytes = width,
            Height = height
        )

        hip_check(hip.hipMemcpyParam2DAsync(copy, stream))

    def get_strides(self) -> tuple[int, int]:
        return self.pitch_d, self.dtype.itemsize

    def get_pitch(self) -> int:
        return self.pitch_d

    def __get_array_vars(self, cpu_data: np.ndarray, nx: int = None, ny: int = None) -> tuple[int, int, int]:
        """
        Gets the variables used for defining the array.
        Args:
            nx: Height of the array, in elements.
            ny: Width of the array, in elements.
        """

        if nx is None and ny is None:
            width = self.nx * cpu_data.itemsize
            height = self.ny
        elif nx is not None and ny is not None:
            width = int(nx) * cpu_data.itemsize
            height = int(ny)
        else:
            raise ValueError("Can only get variables if either all variables are parsed to the function, or none. " +
                             "Cannot only have 1 variable parsed into the function.")

        pitch_h = cpu_data.strides[0]

        return pitch_h, width, height
