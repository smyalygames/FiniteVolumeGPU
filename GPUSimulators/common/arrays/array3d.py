import logging

import numpy as np


class BaseArray3D(object):
    """
    A base class that holds 3D data. To be used depending on the GPGPU language.
    """
    def __init__(self, nx, ny, nz, x_halo, y_halo, z_halo, cpu_data=None):
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

        self.nx_halo = nx + 2 * x_halo
        self.ny_halo = ny + 2 * y_halo
        self.nz_halo = nz + 2 * z_halo

        # If we don't have any data, just allocate and return
        if cpu_data is None:
            return

        # Make sure data is in proper format
        if (cpu_data.shape != (self.nz_halo, self.ny_halo, self.nx_halo)
                and cpu_data.shape != (self.nz, self.ny, self.nx)):
            raise ValueError(f"Wrong shape of data {str(cpu_data.shape)} vs {str((self.nz, self.ny, self.nx))} / "
                             + f"{str((self.nz_halo, self.ny_halo, self.nx_halo))}")

        if cpu_data.itemsize != 4:
            raise ValueError("Wrong size of data type")

        if np.isfortran(cpu_data):
            raise TypeError("Wrong datatype (Fortran, expected C)")