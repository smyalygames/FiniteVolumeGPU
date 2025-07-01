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
            cpu_data = np.zeros((ny, nx), dtype=self.dtype)

        copy_args = hip.hip_Memcpy2D(**self.__get_copy_info(x, y, nx, ny, cpu_data, True))

        hip_check(hip.hipMemcpyParam2DAsync(copy_args, stream))

        if not asynch:
            hip_check(hip.hipStreamSynchronize(stream))

        return cpu_data

    def upload(self, stream, cpu_data, extent=None):
        if extent is None:
            x = self.x_halo
            y = self.y_halo
            nx = self.nx
            ny = self.ny
        else:
            x, y, nx, ny = extent

        copy_param = hip.hip_Memcpy2D(**self.__get_copy_info(x, y, nx, ny, cpu_data))

        hip_check(hip.hipMemcpyParam2DAsync(copy_param, stream))

    def get_strides(self) -> tuple[int, ...]:
        strides = []
        for i in range(len(self.data_h.shape)):
            strides.append(self.data_h.shape[i] * np.float32().itemsize)

        return tuple(strides)

    def __get_copy_info(self, x, y, nx, ny, host, to_host=False):
        self.check(x, y, nx, ny, host)

        # Arguments for the host data
        src_args = [
            'Host',
            0,
            0,
            hip.hipMemoryType.hipMemoryTypeHost,
            host,
            host.strides[0]

        ]
        # Arguments for the device
        dst_args = [
            'Device',
            int(x) * np.float32().itemsize,
            int(y),
            hip.hipMemoryType.hipMemoryTypeDevice,
            self.data,
            self.get_strides()[0],
        ]

        if to_host:
            src_args, dst_args = dst_args, src_args

        args = {
            'srcXInBytes': src_args[1],
            'srcY': src_args[2],
            'srcMemoryType': src_args[3],
            f'src{src_args[0]}': src_args[4],
            'srcPitch': src_args[5],

            'dstXInBytes': dst_args[1],
            'dstY': dst_args[2],
            'dstMemoryType': dst_args[3],
            f'dst{dst_args[0]}': dst_args[4],
            'dstPitch': dst_args[5],

            'WidthInBytes': int(nx) * np.float32().itemsize,
            'Height': int(ny)
        }

        return args
