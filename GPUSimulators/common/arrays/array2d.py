import logging

import numpy as np


class BaseArray2D(object):
    """
    A base class that holds 2D data. To be used depending on the GPGPU language.
    """

    def __init__(self, nx, ny, x_halo, y_halo, cpu_data=None):
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

        self.shape = (nx_halo, ny_halo)

        # If we don't have any data, just allocate and return
        if cpu_data is None:
            return

        # Make sure data is in proper format
        if cpu_data.shape != (ny_halo, nx_halo) and cpu_data.shape != (self.ny, self.nx):
            raise ValueError(
                f"Wrong shape of data {str(cpu_data.shape)} vs {str((self.ny, self.nx))} / "
                + f"{str((ny_halo, nx_halo))}")

        if cpu_data.itemsize != 4:
            raise ValueError("Wrong size of data type")

        if np.isfortran(cpu_data):
            raise TypeError("Wrong datatype (Fortran, expected C)")

    def check(self, x, y, nx, ny, cpu_data):
        if nx != cpu_data.shape[1]:
            raise ValueError
        if ny != cpu_data.shape[0]:
            raise ValueError
        if x + nx > self.nx + 2 * self.x_halo:
            raise ValueError
        if y + ny > self.ny + 2 * self.y_halo:
            raise ValueError
