import logging

import numpy as np
from mpi4py import MPI


def get_grid(num_nodes, num_dims):
    if not isinstance(num_nodes, int):
        raise TypeError("Parameter `num_nodes` is not a an integer.")
    if not isinstance(num_dims, int):
        raise TypeError("Parameter `num_dims` is not a an integer.")

    # Adapted from https://stackoverflow.com/questions/28057307/factoring-a-number-into-roughly-equal-factors
    # Original code by https://stackoverflow.com/users/3928385/ishamael
    # Factorizes a number into n roughly equal factors

    # Dictionary to remember already computed permutations
    memo = {}

    def dp(n, left):  # returns tuple (cost, [factors])
        """
        Recursively searches through all factorizations
        """

        # Already tried: return an existing result
        if (n, left) in memo:
            return memo[(n, left)]

        # Spent all factors: return number itself
        if left == 1:
            return (n, [n])

        # Find a new factor
        i = 2
        best = n
        best_tuple = [n]
        while i * i < n:
            # If a factor found
            if n % i == 0:
                # Factorize remainder
                rem = dp(n // i, left - 1)

                # If new permutation better, save it
                if rem[0] + i < best:
                    best = rem[0] + i
                    best_tuple = [i] + rem[1]
            i += 1

        # Store calculation
        memo[(n, left)] = (best, best_tuple)
        return memo[(n, left)]

    grid = dp(num_nodes, num_dims)[1]

    if len(grid) < num_dims:
        # Split problematic 4
        if 4 in grid:
            grid.remove(4)
            grid.append(2)
            grid.append(2)

        # Pad with ones to guarantee num_dims
        grid = grid + [1] * (num_dims - len(grid))

    # Sort in descending order
    grid = np.sort(grid)
    grid = grid[::-1]

    # XXX: We only use vertical (north-south) partitioning for now
    grid[0] = 1
    grid[1] = num_nodes

    return grid


class MPIGrid(object):
    """
    Class which represents an MPI grid of nodes. Facilitates easy communication between
    neighboring nodes
    """

    def __init__(self, comm, ndims=2):
        self.logger = logging.getLogger(__name__)

        if ndims != 2:
            raise ValueError("Unsupported number of dimensions. Must be two at the moment")
        if comm.size < 1:
            raise ValueError("Must have at least one node")

        self.grid = get_grid(comm.size, ndims)
        self.comm = comm

        self.logger.debug(
            f"Created MPI grid: {self.grid}. Rank {self.comm.rank} has coordinate {self.get_coordinate()}")

    def get_coordinate(self, rank=None):
        if rank is None:
            rank = self.comm.rank
        i = (rank % self.grid[0])
        j = (rank // self.grid[0])
        return i, j

    def get_rank(self, i, j):
        return j * self.grid[0] + i

    def get_east(self):
        i, j = self.get_coordinate(self.comm.rank)
        i = (i + 1) % self.grid[0]
        return self.get_rank(i, j)

    def get_west(self):
        i, j = self.get_coordinate(self.comm.rank)
        i = (i + self.grid[0] - 1) % self.grid[0]
        return self.get_rank(i, j)

    def get_north(self):
        i, j = self.get_coordinate(self.comm.rank)
        j = (j + 1) % self.grid[1]
        return self.get_rank(i, j)

    def get_south(self):
        i, j = self.get_coordinate(self.comm.rank)
        j = (j + self.grid[1] - 1) % self.grid[1]
        return self.get_rank(i, j)

    def gather(self, data, root=0):
        out_data = None
        if self.comm.rank == root:
            out_data = np.empty([self.comm.size] + list(data.shape), dtype=data.dtype)
        self.comm.Gather(data, out_data, root)
        return out_data

    def get_local_rank(self):
        """
        Returns the local rank on this node for this MPI process
        """

        # This function has been adapted from
        # https://github.com/SheffieldML/PyDeepGP/blob/master/deepgp/util/parallel.py
        # by Zhenwen Dai released under BSD 3-Clause "New" or "Revised" License:
        #
        # Copyright (c) 2016, Zhenwen Dai
        # All rights reserved.
        #
        # Redistribution and use in source and binary forms, with or without
        # modification, are permitted provided that the following conditions are met:
        #
        # * Redistributions of source code must retain the above copyright notice, this
        #   list of conditions and the following disclaimer.
        #
        # * Redistributions in binary form must reproduce the above copyright notice,
        #   this list of conditions and the following disclaimer in the documentation
        #   and/or other materials provided with the distribution.
        #
        # * Neither the name of DGP nor the names of its
        #   contributors may be used to endorse or promote products derived from
        #   this software without specific prior written permission.
        #
        # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
        # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
        # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
        # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
        # FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
        # DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
        # SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
        # CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
        # OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
        # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

        # Get this ranks unique (physical) node name
        node_name = MPI.Get_processor_name()

        # Gather the list of all node names on all nodes
        node_names = self.comm.allgather(node_name)

        # Loop over all node names up until our rank
        # and count how many duplicates of our nodename we find
        local_rank = len([0 for name in node_names[:self.comm.rank] if name == node_name])

        return local_rank
