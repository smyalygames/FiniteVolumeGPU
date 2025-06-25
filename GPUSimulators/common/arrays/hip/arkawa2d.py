import numpy as np
from hip import hip, hipblas

from ....common import hip_check
from ..arkawa2d import BaseArakawaA2D
from .array2d import HIPArray2D


def _sum_array(array: HIPArray2D):
    """
    Sum all the elements in HIPArray2D using hipblas.
    Args:
        array: A HIPArray2D to compute the sum of.
    """
    data_h = array.data_h
    num_bytes = array.dtype.itemsize

    result_d = hip_check(hip.hipMalloc(num_bytes))
    result_h = array.dtype.type(0)

    # Sum the ``data_h`` array using hipblas
    handle = hip_check(hipblas.hipblasCreate())
    hip_check(hipblas.hipblasSasum(handle, data_h.size, data_h.data, 1, result_d))
    hip_check(hipblas.hipblasDestroy(handle))

    # Copy over the result from the device
    hip_check(hip.hipMemcpy(result_h, result_d, num_bytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))

    hip_check(hip.hipFree(result_d))

    return result_h


class ArakawaA2D(BaseArakawaA2D):
    """
    A class representing an Arakawa A type (unstaggered, logically Cartesian) grid
    """

    def __init__(self, stream, nx, ny, halo_x, halo_y, cpu_variables):
        """
        Uploads initial data to the GPU device
        """
        super().__init__(stream, nx, ny, halo_x, halo_y, cpu_variables)

    def check(self):
        """
        Checks that data is still sane
        """
        for i, gpu_variable in enumerate(self.gpu_variables):
            var_sum = _sum_array(gpu_variable)
            self.logger.debug(f"Data {i} with size [{gpu_variable.nx} x {gpu_variable.ny}] "
                              + f"has average {var_sum / (gpu_variable.nx * gpu_variable.ny)}")

            if np.isnan(var_sum):
                raise ValueError("Data contains NaN values!")
