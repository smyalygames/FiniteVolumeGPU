import logging


class BaseArakawaA2D(object):
    """
    A base class to be used to represent an Arakawa A type (unstaggered, logically Cartesian) grid.
    """

    def __init__(self, stream, nx, ny, halo_x, halo_y, cpu_variables, array_type):
        """
        Uploads initial data to the GPU device
        """
        self.logger = logging.getLogger(__name__)
        self.gpu_variables = []

        for cpu_variable in cpu_variables:
            self.gpu_variables += [array_type(stream, nx, ny, halo_x, halo_y, cpu_variable)]

    def __getitem__(self, key):
        if type(key) != int:
            raise TypeError("Indexing is int based")

        if key > len(self.gpu_variables) or key < 0:
            raise IndexError("Out of bounds")
        return self.gpu_variables[key]

    def download(self, stream, variables=None):
        """
        Enables downloading data from the GPU device to Python
        """
        if variables is None:
            variables = range(len(self.gpu_variables))

        cpu_variables = []
        for i in variables:
            if i >= len (self.gpu_variables):
                raise IndexError(f"Variable {i} is out of range")
            cpu_variables += [self.gpu_variables[i].download(stream, asynch=True)]

        # stream.synchronize()
        return cpu_variables
