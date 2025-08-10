import ctypes

import numpy as np
from hip import hip, hipblas

from ...hip_check import hip_check
from ..arkawa2d import BaseArakawaA2D
from .array2d import HIPArray2D


class HIPArakawaA2D(BaseArakawaA2D):
    """
    A class representing an Arakawa A type (unstaggered, logically Cartesian) grid
    """

    def __init__(self, stream, nx, ny, halo_x, halo_y, cpu_variables):
        """
        Uploads initial data to the GPU device
        """
        super().__init__(stream, nx, ny, halo_x, halo_y, cpu_variables, HIPArray2D)

    def check(self):
        """
        Checks that data is still sane
        """
        for i, gpu_variable in enumerate(self.gpu_variables):
            var_sum = self.__sum_array(gpu_variable)
            self.logger.debug(f"Data {i} with size [{gpu_variable.nx} x {gpu_variable.ny}] "
                              + f"has average {var_sum / (gpu_variable.nx * gpu_variable.ny)}")

            if np.isnan(var_sum):
                raise ValueError("Data contains NaN values!")

    @staticmethod
    def __sum_array(array: HIPArray2D) -> np.ndarray[tuple[int]]:
        """
        Sum all the elements in HIPArray2D using hipblas.
        Args:
            array: A HIPArray2D to compute the sum of.
        Returns:
            The sum of all the elements in ``array``.
        """
        dtype = array.dtype
        result_h = np.zeros(1, dtype=dtype)
        num_bytes = dtype.itemsize
        result_d = hip_check(hip.hipMalloc(num_bytes))

        # Sum the ``data_h`` array using hipblas
        handle = hip_check(hipblas.hipblasCreate())

        # Using pitched memory, so we need to sum row by row
        total_sum_d = hip_check(hip.hipMalloc(num_bytes))
        hip_check(hip.hipMemset(total_sum_d, 0, num_bytes))

        width, height = array.shape

        for y in range(height):
            row_ptr = int(array.data) + y * array.pitch_d

            hip_check(hipblas.hipblasSasum(handle, width, row_ptr, 1, result_d))

            hip_check(hipblas.hipblasSaxpy(handle, 1, ctypes.c_float(1.0), result_d, 1, total_sum_d, 1))

            hip_check(hip.hipMemcpy(result_h, total_sum_d, num_bytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))

        # Copy over the result from the device
        hip_check(hip.hipMemcpy(result_h, total_sum_d, num_bytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))

        # Cleanup
        hip_check(hipblas.hipblasDestroy(handle))
        hip_check(hip.hipFree(result_d))
        hip_check(hip.hipFree(total_sum_d))

        return result_h
