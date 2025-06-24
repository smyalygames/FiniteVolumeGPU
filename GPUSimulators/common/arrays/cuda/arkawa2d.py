import numpy as np
import pycuda.gpuarray

from GPUSimulators.common.arrays.arkawa2d import BaseArakawaA2D


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
            var_sum = pycuda.gpuarray.sum(gpu_variable.data).get()
            self.logger.debug(f"Data {i} with size [{gpu_variable.nx} x {gpu_variable.ny}] "
                              + f"has average {var_sum / (gpu_variable.nx * gpu_variable.ny)}")

            if np.isnan(var_sum):
                raise ValueError("Data contains NaN values!")
