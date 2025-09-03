import ctypes
from typing import Union

import numpy as np
from hip import hip, hipblas

from ...hip_check import hip_check
from ..arkawa2d import BaseArakawaA2D
from .array2d import HIPArray2D


class HIPArakawaA2D(BaseArakawaA2D):
    """
    A class representing an Arakawa A type (unstaggered, logically Cartesian) grid
    """

    def __init__(self, stream: hip.ihipStream_t, nx: int, ny: int, halo_x: int, halo_y: int, cpu_variables: list[Union[np.ndarray, None]]):
        """
        Uploads initial data to the GPU device
        """
        super().__init__(stream, nx, ny, halo_x, halo_y, cpu_variables, HIPArray2D)

        # Variables for ``__sum_array``
        # TODO should have a way of not hardcoding the dtype
        dtype = np.float32
        self.__result_h = np.zeros(1, dtype=dtype)
        self.__num_bytes = self.__result_h.itemsize
        self.__result_d = hip_check(hip.hipMalloc(self.__num_bytes))
        self.__total_sum_d = hip_check(hip.hipMalloc(self.__num_bytes))

        self.__handle = hip_check(hipblas.hipblasCreate())

    def __del__(self):
        # Cleanup GPU variables in ``__sum_array``
        hip_check(hipblas.hipblasDestroy(self.__handle))
        hip_check(hip.hipFree(self.__result_d))
        hip_check(hip.hipFree(self.__total_sum_d))

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

    def __sum_array(self, array: HIPArray2D) -> np.ndarray[tuple[int]]:
        """
        Sum all the elements in HIPArray2D using hipblas.
        Args:
            array: A HIPArray2D to compute the sum of.
        Returns:
            The sum of all the elements in ``array``.
        """

        # Using pitched memory, so we need to sum row by row
        hip_check(hip.hipMemset(self.__total_sum_d, 0, self.__num_bytes))

        width, height = array.shape

        for y in range(height):
            row_ptr = int(array.data) + y * array.pitch_d

            hip_check(hipblas.hipblasSasum(self.__handle, width, row_ptr, 1, self.__result_d))

            hip_check(
                hipblas.hipblasSaxpy(self.__handle, 1, ctypes.c_float(1.0), self.__result_d, 1, self.__total_sum_d, 1))

        # Copy over the result from the device
        hip_check(hip.hipMemcpy(self.__result_h, self.__total_sum_d, self.__num_bytes,
                                hip.hipMemcpyKind.hipMemcpyDeviceToHost))

        return self.__result_h
