import logging
import math

import numpy as np

from . import boundary
from GPUSimulators.gpu import KernelContext


def get_types(bc):
    types = {'north': boundary.BoundaryCondition.Type((bc >> 24) & 0x0000000F),
             'south': boundary.BoundaryCondition.Type((bc >> 16) & 0x0000000F),
             'east': boundary.BoundaryCondition.Type((bc >> 8) & 0x0000000F),
             'west': boundary.BoundaryCondition.Type((bc >> 0) & 0x0000000F)}
    return types


class BaseSimulator(object):

    def __init__(self,
                 context: KernelContext,
                 nx: int, ny: int,
                 dx: int, dy: int,
                 boundary_conditions: boundary.BoundaryCondition,
                 cfl_scale: float,
                 num_substeps: int,
                 block_width: int, block_height: int):
        """
        Initialization routine

        Args:
            context: GPU context to use
            kernel_wrapper: wrapper function of GPU kernel
            h0: Water depth incl ghost cells, (nx+1)*(ny+1) cells
            hu0: Initial momentum along x-axis incl ghost cells, (nx+1)*(ny+1) cells
            hv0: Initial momentum along y-axis incl ghost cells, (nx+1)*(ny+1) cells
            nx: Number of cells along x-axis
            ny: Number of cells along y-axis
            dx: Grid cell spacing along x-axis (20 000 m)
            dy: Grid cell spacing along y-axis (20 000 m)
            dt: Size of each timestep (90 s)
            cfl_scale: Courant number
            num_substeps: Number of substeps to perform for a full step
        """

        # Get logger
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)

        # Save input parameters
        # Notice that we need to specify them in the correct dataformat for the
        # GPU kernel
        self.context = context
        self.nx = np.int32(nx)
        self.ny = np.int32(ny)
        self.dx = np.float32(dx)
        self.dy = np.float32(dy)
        self.set_boundary_conditions(boundary_conditions)
        self.cfl_scale = cfl_scale
        self.num_substeps = num_substeps

        # Handle autotuning block size
        if self.context.autotuner:
            peak_configuration = self.context.autotuner.get_peak_performance(self.__class__)
            block_width = int(peak_configuration["block_width"])
            block_height = int(peak_configuration["block_height"])
            self.logger.debug(f"Used autotuning to get block size [{block_width} x {block_height}]")

        # Compute kernel launch parameters
        self.block_size = (block_width, block_height, 1)
        self.grid_size = (
            int(np.ceil(self.nx / float(self.block_size[0]))),
            int(np.ceil(self.ny / float(self.block_size[1])))
        )

        # Streams to be implemented in respective language classes
        self.stream = None
        self.internal_stream = None

        # Keep track of simulation time and number of timesteps
        self.t = 0.0
        self.nt = 0

    def __str__(self):
        return f"{self.__class__.__name__} [{self.nx}x{self.ny}]"

    def simulate(self, t, dt=None, tolerance=None, pbar=None):
        """
        Function which simulates t_end seconds using the step function
        Requires that the step() function is implemented in the subclasses

        Args:
            t: How long the simulation should run for.
            dt: Time steps.
            tolerance: How small should the time steps be before considering it an infinite loop.
            pbar: A tqdm progress bar to update time.
        """

        t_start = self.sim_time()
        t_end = t_start + t

        update_dt = True
        if dt is not None:
            update_dt = False
            self.dt = dt

        if tolerance is None:
            tolerance = 0.000000001

        while self.sim_time() < t_end:
            # Prevent an infinite loop from occurring from tiny numbers
            if abs(t_end - self.sim_time()) < tolerance:
                break

            if update_dt and (self.sim_steps() % 100 == 0):
                self.dt = self.compute_dt() * self.cfl_scale

            # Compute timestep for "this" iteration (i.e., shorten last timestep)
            current_dt = np.float32(min(self.dt, t_end - self.sim_time()))

            # Stop if end reached (should not happen)
            if current_dt <= 0.0:
                self.logger.warning(f"Timestep size {self.sim_steps()} is less than or equal to zero!")
                break

            # Step forward in time
            self.step(current_dt)

            # Update the progress bar
            if pbar is not None:
                pbar.update(float(current_dt))

    def step(self, dt: int):
        """
        Function which performs one single timestep of size dt

        Args:
            dt: Size of each timestep (seconds)
        """

        for i in range(self.num_substeps):
            self.substep(dt, i)

        self.t += dt
        self.nt += 1

    def download(self, variables=None):
        return self.get_output().download(self.stream, variables)

    def synchronize(self):
        raise NotImplementedError("Needs to be implemented in HIP/CUDA subclass")

    def internal_synchronize(self):
        raise NotImplementedError("Needs to be implemented in HIP/CUDA subclass")

    def sim_time(self):
        return self.t

    def sim_steps(self):
        return self.nt

    def get_extent(self):
        return [0, 0, self.nx * self.dx, self.ny * self.dy]

    def set_boundary_conditions(self, boundary_conditions):
        self.logger.debug(f"Boundary conditions set to {str(boundary_conditions)}")
        self.boundary_conditions = boundary_conditions.as_coded_int()

    def get_boundary_conditions(self):
        return boundary.BoundaryCondition(get_types(self.boundary_conditions))

    def substep(self, dt, step_number):
        """
        Function which performs one single substep with stepsize dt
        """

        raise NotImplementedError("Needs to be implemented in subclass")

    def get_output(self):
        raise NotImplementedError("Needs to be implemented in subclass")

    def check(self):
        self.logger.warning("check() is not implemented - please implement")
        # raise(NotImplementedError("Needs to be implemented in subclass"))

    def compute_dt(self):
        raise NotImplementedError("Needs to be implemented in subclass")
