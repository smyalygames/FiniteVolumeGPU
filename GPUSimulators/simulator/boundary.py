# -*- coding: utf-8 -*-

"""
This python module implements the classical Lax-Friedrichs numerical
scheme for the shallow water equations

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

# Import packages we need
from enum import IntEnum

import numpy as np


class BoundaryCondition(object):
    """
    Class for holding boundary conditions for global boundaries
    """

    class Type(IntEnum):
        """
        Enum that describes the different types of boundary conditions
        WARNING: MUST MATCH THAT OF common.h IN CUDA
        """

        Dirichlet = 0,
        Neumann = 1,
        Periodic = 2,
        Reflective = 3

    def __init__(self, types: dict[str, Type]=None):
        """
        Constructor
        """
        if types is None:
            types = {
                    'north': self.Type.Reflective,
                    'south': self.Type.Reflective,
                    'east': self.Type.Reflective,
                    'west': self.Type.Reflective
                 }

        self.north = types['north']
        self.south = types['south']
        self.east = types['east']
        self.west = types['west']

        if (self.north == BoundaryCondition.Type.Neumann
                or self.south == BoundaryCondition.Type.Neumann
                or self.east == BoundaryCondition.Type.Neumann
                or self.west == BoundaryCondition.Type.Neumann):
            raise (NotImplementedError("Neumann boundary condition not supported"))

    def __str__(self):
        return f"[north={str(self.north)}, south={str(self.south)}, east={str(self.east)}, west={str(self.west)}]"

    def as_coded_int(self):
        """
        Helper function which packs four boundary conditions into one integer
        """

        bc = 0
        bc = bc | (self.north & 0x0000000F) << 24
        bc = bc | (self.south & 0x0000000F) << 16
        bc = bc | (self.east & 0x0000000F) << 8
        bc = bc | (self.west & 0x0000000F) << 0

        # for t in types:
        #    print("{0:s}, {1:d}, {1:032b}, {1:08b}".format(t, types[t]))
        # print("bc: {0:032b}".format(bc))

        return np.int32(bc)


