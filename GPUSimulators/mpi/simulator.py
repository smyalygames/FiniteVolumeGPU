# -*- coding: utf-8 -*-

"""
This python module implements MPI simulator class

Copyright (C) 2018 SINTEF Digital

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import logging
from typing import Callable

import numpy as np
from mpi4py import MPI
import time

from GPUSimulators.simulator import BaseSimulator, BoundaryCondition


class BaseMPISimulator(BaseSimulator):
    """
    Class which handles communication between simulators on different MPI nodes
    """

    def __init__(self, sim, grid, data_func: Callable):
        self.profiling_data_mpi = {'start': {}, 'end': {}}
        self.profiling_data_mpi["start"]["t_mpi_halo_exchange"] = 0
        self.profiling_data_mpi["end"]["t_mpi_halo_exchange"] = 0
        self.profiling_data_mpi["start"]["t_mpi_halo_exchange_download"] = 0
        self.profiling_data_mpi["end"]["t_mpi_halo_exchange_download"] = 0
        self.profiling_data_mpi["start"]["t_mpi_halo_exchange_upload"] = 0
        self.profiling_data_mpi["end"]["t_mpi_halo_exchange_upload"] = 0
        self.profiling_data_mpi["start"]["t_mpi_halo_exchange_sendreceive"] = 0
        self.profiling_data_mpi["end"]["t_mpi_halo_exchange_sendreceive"] = 0
        self.profiling_data_mpi["start"]["t_mpi_step"] = 0
        self.profiling_data_mpi["end"]["t_mpi_step"] = 0
        self.profiling_data_mpi["n_time_steps"] = 0
        self.logger = logging.getLogger(__name__)

        autotuner = sim.context.autotuner
        sim.context.autotuner = None
        boundary_conditions = sim.get_boundary_conditions()
        super().__init__(sim.context,
                         sim.nx, sim.ny,
                         sim.dx, sim.dy,
                         boundary_conditions,
                         sim.cfl_scale,
                         sim.num_substeps,
                         sim.block_size[0], sim.block_size[1])
        sim.context.autotuner = autotuner

        self.sim = sim
        self.grid = grid

        # Get neighbor node ids
        self.east = grid.get_east()
        self.west = grid.get_west()
        self.north = grid.get_north()
        self.south = grid.get_south()

        # Get coordinate of this node
        # and handle global boundary conditions
        new_boundary_conditions = BoundaryCondition({
            'north': BoundaryCondition.Type.Dirichlet,
            'south': BoundaryCondition.Type.Dirichlet,
            'east': BoundaryCondition.Type.Dirichlet,
            'west': BoundaryCondition.Type.Dirichlet
        })
        gi, gj = grid.get_coordinate()
        # print("gi: " + str(gi) + ", gj: " + str(gj))
        if (gi == 0 and boundary_conditions.west != BoundaryCondition.Type.Periodic):
            self.west = None
            new_boundary_conditions.west = boundary_conditions.west
        if (gj == 0 and boundary_conditions.south != BoundaryCondition.Type.Periodic):
            self.south = None
            new_boundary_conditions.south = boundary_conditions.south
        if (gi == grid.x - 1 and boundary_conditions.east != BoundaryCondition.Type.Periodic):
            self.east = None
            new_boundary_conditions.east = boundary_conditions.east
        if (gj == grid.y - 1 and boundary_conditions.north != BoundaryCondition.Type.Periodic):
            self.north = None
            new_boundary_conditions.north = boundary_conditions.north
        sim.set_boundary_conditions(new_boundary_conditions)

        # Get number of variables
        self.nvars = len(self.get_output().gpu_variables)

        # Shorthands for computing extents and sizes
        gc_x = int(self.sim.get_output()[0].x_halo)
        gc_y = int(self.sim.get_output()[0].y_halo)
        nx = int(self.sim.nx)
        ny = int(self.sim.ny)

        # Set regions for ghost cells to read from
        # These have the format [x0, y0, width, height]
        self.read_e = np.array([nx, 0, gc_x, ny + 2 * gc_y])
        self.read_w = np.array([gc_x, 0, gc_x, ny + 2 * gc_y])
        self.read_n = np.array([gc_x, ny, nx, gc_y])
        self.read_s = np.array([gc_x, gc_y, nx, gc_y])

        # Set regions for ghost cells to write to
        self.write_e = self.read_e + np.array([gc_x, 0, 0, 0])
        self.write_w = self.read_w - np.array([gc_x, 0, 0, 0])
        self.write_n = self.read_n + np.array([0, gc_y, 0, 0])
        self.write_s = self.read_s - np.array([0, gc_y, 0, 0])

        self.in_e = None
        self.in_w = None
        self.in_n = None
        self.in_s = None

        # Allocate data for sending
        self.out_e = None
        self.out_w = None
        self.out_n = None
        self.out_s = None

        # Creates the page locked memory
        data_func()

        self.logger.debug(f"Simulator rank {self.grid.comm.rank} initialized on {MPI.Get_processor_name()}")

        self.full_exchange()
        sim.context.synchronize()

    def substep(self, dt, step_number):

        # nvtx.mark("substep start", color="yellow")

        self.profiling_data_mpi["start"]["t_mpi_step"] += time.time()

        # nvtx.mark("substep external", color="blue")
        self.sim.substep(dt, step_number, external=True, internal=False)  # only "internal ghost cells"

        # nvtx.mark("substep internal", color="red")
        self.sim.substep(dt, step_number, internal=True, external=False)  # "internal ghost cells" excluded

        # nvtx.mark("substep full", color="blue")
        # self.sim.substep(dt, step_number, external=True, internal=True)

        self.sim.swap_buffers()

        self.profiling_data_mpi["end"]["t_mpi_step"] += time.time()

        # nvtx.mark("exchange", color="blue")
        self.full_exchange()

        # nvtx.mark("sync start", color="blue")
        self.sim.synchronize()
        self.sim.internal_synchronize()
        # nvtx.mark("sync end", color="blue")

        self.profiling_data_mpi["n_time_steps"] += 1

    def get_output(self):
        return self.sim.get_output()

    def synchronize(self):
        self.sim.synchronize()

    def check(self):
        return self.sim.check()

    def compute_dt(self):
        local_dt = np.array([np.float32(self.sim.compute_dt())])
        global_dt = np.empty(1, dtype=np.float32)
        self.grid.comm.Allreduce(local_dt, global_dt, op=MPI.MIN)
        self.logger.debug(f"Local dt: {local_dt[0]}, global dt: {global_dt[0]}")
        return global_dt[0]

    def get_extent(self):
        """
        Function which returns the extent of node with rank 
        in the grid
        """

        width = self.sim.nx * self.sim.dx
        height = self.sim.ny * self.sim.dy
        x0 = self.grid.x * width
        y0 = self.grid.y * height
        x1 = x0 + width
        y1 = y0 + height
        return x0, x1, y0, y1

    def full_exchange(self):
        ####
        # First transfer internal cells north-south
        ####

        # Download from the GPU
        self.profiling_data_mpi["start"]["t_mpi_halo_exchange_download"] += time.time()

        if self.north is not None:
            for k in range(self.nvars):
                self.sim.u0[k].download(self.sim.stream, cpu_data=self.out_n[k, :, :], asynch=True, extent=self.read_n)
        if self.south is not None:
            for k in range(self.nvars):
                self.sim.u0[k].download(self.sim.stream, cpu_data=self.out_s[k, :, :], asynch=True, extent=self.read_s)
        self.sim.synchronize()

        self.profiling_data_mpi["end"]["t_mpi_halo_exchange_download"] += time.time()

        # Send/receive to north/south neighbors
        self.profiling_data_mpi["start"]["t_mpi_halo_exchange_sendreceive"] += time.time()

        comm_send = []
        comm_recv = []
        if self.north is not None:
            comm_send += [self.grid.comm.Isend(self.out_n, dest=self.north, tag=4 * self.nt + 0)]
            comm_recv += [self.grid.comm.Irecv(self.in_n, source=self.north, tag=4 * self.nt + 1)]
        if self.south is not None:
            comm_send += [self.grid.comm.Isend(self.out_s, dest=self.south, tag=4 * self.nt + 1)]
            comm_recv += [self.grid.comm.Irecv(self.in_s, source=self.south, tag=4 * self.nt + 0)]

        # Wait for incoming transfers to complete
        for comm in comm_recv:
            comm.wait()

        self.profiling_data_mpi["end"]["t_mpi_halo_exchange_sendreceive"] += time.time()

        # Upload to the GPU
        self.profiling_data_mpi["start"]["t_mpi_halo_exchange_upload"] += time.time()

        if self.north is not None:
            for k in range(self.nvars):
                self.sim.u0[k].upload(self.sim.stream, self.in_n[k, :, :], extent=self.write_n)
        if self.south is not None:
            for k in range(self.nvars):
                self.sim.u0[k].upload(self.sim.stream, self.in_s[k, :, :], extent=self.write_s)

        self.profiling_data_mpi["end"]["t_mpi_halo_exchange_upload"] += time.time()

        # Wait for sending to complete
        self.profiling_data_mpi["start"]["t_mpi_halo_exchange_sendreceive"] += time.time()

        for comm in comm_send:
            comm.wait()

        self.profiling_data_mpi["end"]["t_mpi_halo_exchange_sendreceive"] += time.time()

        ####
        # Then transfer east-west including ghost cells that have been filled in by north-south transfer above
        ####

        # Download from the GPU
        self.profiling_data_mpi["start"]["t_mpi_halo_exchange_download"] += time.time()

        if self.east is not None:
            for k in range(self.nvars):
                self.sim.u0[k].download(self.sim.stream, cpu_data=self.out_e[k, :, :], asynch=True, extent=self.read_e)
        if self.west is not None:
            for k in range(self.nvars):
                self.sim.u0[k].download(self.sim.stream, cpu_data=self.out_w[k, :, :], asynch=True, extent=self.read_w)
        self.sim.synchronize()

        self.profiling_data_mpi["end"]["t_mpi_halo_exchange_download"] += time.time()

        # Send/receive to east/west neighbors
        self.profiling_data_mpi["start"]["t_mpi_halo_exchange_sendreceive"] += time.time()

        comm_send = []
        comm_recv = []
        if self.east is not None:
            comm_send += [self.grid.comm.Isend(self.out_e, dest=self.east, tag=4 * self.nt + 2)]
            comm_recv += [self.grid.comm.Irecv(self.in_e, source=self.east, tag=4 * self.nt + 3)]
        if self.west is not None:
            comm_send += [self.grid.comm.Isend(self.out_w, dest=self.west, tag=4 * self.nt + 3)]
            comm_recv += [self.grid.comm.Irecv(self.in_w, source=self.west, tag=4 * self.nt + 2)]

        # Wait for incoming transfers to complete
        for comm in comm_recv:
            comm.wait()

        self.profiling_data_mpi["end"]["t_mpi_halo_exchange_sendreceive"] += time.time()

        # Upload to the GPU
        self.profiling_data_mpi["start"]["t_mpi_halo_exchange_upload"] += time.time()

        if self.east is not None:
            for k in range(self.nvars):
                self.sim.u0[k].upload(self.sim.stream, self.in_e[k, :, :], extent=self.write_e)
        if self.west is not None:
            for k in range(self.nvars):
                self.sim.u0[k].upload(self.sim.stream, self.in_w[k, :, :], extent=self.write_w)

        self.profiling_data_mpi["end"]["t_mpi_halo_exchange_upload"] += time.time()

        # Wait for sending to complete
        self.profiling_data_mpi["start"]["t_mpi_halo_exchange_sendreceive"] += time.time()

        for comm in comm_send:
            comm.wait()

        self.profiling_data_mpi["end"]["t_mpi_halo_exchange_sendreceive"] += time.time()

    def __create_pagelocked_memory(self):
        """
        Allocate data for receiving
        Note that east and west also transfer ghost cells
        whilst north/south only transfer internal cells
        Reuses the width/height defined in the read-extets above
        """
        raise NotImplementedError("This function needs to be implemented in a subclass.")
