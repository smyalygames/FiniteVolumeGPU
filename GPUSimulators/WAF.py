# -*- coding: utf-8 -*-

"""
This python module implements the Weighted average flux (WAF) described in
E. Toro, Shock-Capturing methods for free-surface shallow flows, 2001

Copyright (C) 2016  SINTEF ICT

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

#Import packages we need
from GPUSimulators import Simulator, Common
from GPUSimulators.Simulator import BaseSimulator, BoundaryCondition
import numpy as np

from pycuda import gpuarray


class WAF (Simulator.BaseSimulator):
    """
    Class that solves the SW equations using the Forward-Backward linear scheme
    """

    def __init__(self, 
                 context,
                 h0, hu0, hv0, 
                 nx, ny, 
                 dx, dy, 
                 g, 
                 cfl_scale=0.9,
                 boundary_conditions=BoundaryCondition(), 
                 block_width=16, block_height=16,
                 dt: float=None,
                 compile_opts: list[str]=[]):
        """
        Initialization routine

        Args:
            h0: Water depth incl ghost cells, (nx+1)*(ny+1) cells
            hu0: Initial momentum along x-axis incl ghost cells, (nx+1)*(ny+1) cells
            hv0: Initial momentum along y-axis incl ghost cells, (nx+1)*(ny+1) cells
            nx: Number of cells along x-axis
            ny: Number of cells along y-axis
            dx: Grid cell spacing along x-axis (20 000 m)
            dy: Grid cell spacing along y-axis (20 000 m)
            dt: Size of each timestep (90 s)
            g: Gravitational accelleration (9.81 m/s^2)
            compile_opts: Pass a list of nvcc compiler options
        """
                 
        # Call super constructor
        super().__init__(context, 
            nx, ny, 
            dx, dy, 
            boundary_conditions,
            cfl_scale,
            2,
            block_width, block_height);
        self.g = np.float32(g) 

        #Get kernels
        module = context.get_module("cuda/SWE2D_WAF.cu", 
                                        defines={
                                            'BLOCK_WIDTH': self.block_size[0], 
                                            'BLOCK_HEIGHT': self.block_size[1]
                                        }, 
                                        compile_args={
                                            'no_extern_c': True,
                                            'options': ["--use_fast_math"] + compile_opts, 
                                        }, 
                                        jit_compile_args={})
        self.kernel = module.get_function("WAFKernel")
        self.kernel.prepare("iiffffiiPiPiPiPiPiPiPiiii")
    
        #Create data by uploading to device
        self.u0 = Common.ArakawaA2D(self.stream, 
                        nx, ny, 
                        2, 2, 
                        [h0, hu0, hv0])
        self.u1 = Common.ArakawaA2D(self.stream, 
                        nx, ny, 
                        2, 2, 
                        [None, None, None])
        self.cfl_data = gpuarray.GPUArray(self.grid_size, dtype=np.float32)
        
        if dt == None:
            dt_x = np.min(self.dx / (np.abs(hu0/h0) + np.sqrt(g*h0)))
            dt_y = np.min(self.dy / (np.abs(hv0/h0) + np.sqrt(g*h0)))
            self.dt = min(dt_x, dt_y)
        else:
            self.dt = dt
        
        self.cfl_data.fill(self.dt, stream=self.stream)
    
    def substep(self, dt, step_number):
        self.substepDimsplit(dt*0.5, step_number)
        
    def substepDimsplit(self, dt, substep):
        self.kernel.prepared_async_call(self.grid_size, self.block_size, self.stream, 
                self.nx, self.ny, 
                self.dx, self.dy, dt, 
                self.g, 
                substep,
                self.boundary_conditions, 
                self.u0[0].data.gpudata, self.u0[0].data.strides[0], 
                self.u0[1].data.gpudata, self.u0[1].data.strides[0], 
                self.u0[2].data.gpudata, self.u0[2].data.strides[0], 
                self.u1[0].data.gpudata, self.u1[0].data.strides[0], 
                self.u1[1].data.gpudata, self.u1[1].data.strides[0], 
                self.u1[2].data.gpudata, self.u1[2].data.strides[0],
                self.cfl_data.gpudata,
                0, 0,
                self.nx, self.ny)
        self.u0, self.u1 = self.u1, self.u0

    def getOutput(self):
        return self.u0
        
    def check(self):
        self.u0.check()
        self.u1.check()
        
    def computeDt(self):
        max_dt = gpuarray.min(self.cfl_data, stream=self.stream).get();
        return max_dt*0.5