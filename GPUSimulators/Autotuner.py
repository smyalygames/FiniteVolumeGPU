# -*- coding: utf-8 -*-

"""
This python module implements the different helper functions and 
classes

Copyright (C) 2018  SINTEF ICT

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

import gc
import logging
import os
from socket import gethostname

import numpy as np
import pycuda.driver as cuda
from tqdm.auto import tqdm

from GPUSimulators import Simulator
from GPUSimulators.common import Timer
from GPUSimulators.gpu import KernelContext


def run_benchmark(simulator, arguments, timesteps=10, warmup_timesteps=2):
    """
    Runs a benchmark, and returns the number of megacells achieved
    """

    logger = logging.getLogger(__name__)

    # Initialize simulator
    try:
        sim = simulator(**arguments)
    except:
        # An exception raised - not possible to continue
        logger.debug(f"Failed creating {simulator.__name__} with arguments {str(arguments)}")
        # raise RuntimeError("Failed creating %s with arguments %s", simulator.__name__, str(arguments))
        return np.nan

    # Create timer events
    start = cuda.Event()
    end = cuda.Event()

    # Warmup
    for i in range(warmup_timesteps):
        sim.substep(sim.dt, i)

    # Run simulation with timer
    start.record(sim.stream)
    for i in range(timesteps):
        sim.substep(sim.dt, i)
    end.record(sim.stream)

    # Synchronize end event
    end.synchronize()

    # Compute megacells
    gpu_elapsed = end.time_since(start) * 1.0e-3
    megacells = (sim.nx * sim.ny * timesteps / (1000 * 1000)) / gpu_elapsed

    # Sanity check solution
    h, hu, hv = sim.download()
    sane = True
    sane = sane and sanity_check(0.3, 0.7)
    sane = sane and sanity_check(-0.2, 0.2)
    sane = sane and sanity_check(-0.2, 0.2)

    if sane:
        logger.debug(f"{simulator.__name__} [{arguments["block_width"]} x {arguments["block_height"]}] succeeded: "
                     + f"{megacells} megacells, gpu elapsed {gpu_elapsed}")
        return megacells
    else:
        logger.debug(f"{simulator.__name__} [{arguments["block_width"]} x {arguments["block_height"]}] failed: "
                     + f"gpu elapsed {gpu_elapsed}")
        # raise RuntimeError("%s [%d x %d] failed: gpu elapsed %f", simulator.__name__, arguments["block_width"], arguments["block_height"], gpu_elapsed)
        return np.nan


def gen_test_data(nx, ny, g):
    """
    Generates test dataset
    """

    width = 100.0
    height = 100.0
    dx = width / float(nx)
    dy = height / float(ny)

    x_center = dx * nx / 2.0
    y_center = dy * ny / 2.0

    # Create a gaussian "dam break" that will not form shocks
    size = width / 5.0
    dt = 10 ** 10

    h = np.zeros((ny, nx), dtype=np.float32)
    hu = np.zeros((ny, nx), dtype=np.float32)
    hv = np.zeros((ny, nx), dtype=np.float32)

    extent = 1.0 / np.sqrt(2.0)
    x = (dx * (np.arange(0, nx, dtype=np.float32) + 0.5) - x_center) / size
    y = (dy * (np.arange(0, ny, dtype=np.float32) + 0.5) - y_center) / size
    xv, yv = np.meshgrid(x, y, sparse=False, indexing='xy')
    r = np.minimum(1.0, np.sqrt(xv ** 2 + yv ** 2))
    xv = None
    yv = None
    gc.collect()

    # Generate highres
    cos = np.cos(np.pi * r)
    h = 0.5 + 0.1 * 0.5 * (1.0 + cos)
    hu = 0.1 * 0.5 * (1.0 + cos)
    hv = hu.copy()

    scale = 0.7
    max_h_estimate = 0.6
    max_u_estimate = 0.1 * np.sqrt(2.0)
    dx = width / nx
    dy = height / ny
    dt = scale * min(dx, dy) / (max_u_estimate + np.sqrt(g * max_h_estimate))

    return h, hu, hv, dx, dy, dt


def sanity_check(variable, bound_min, bound_max):
    """
    Checks that a variable is "sane"
    """

    maxval = np.amax(variable)
    minval = np.amin(variable)
    if (np.isnan(maxval)
            or np.isnan(minval)
            or maxval > bound_max
            or minval < bound_min):
        return False
    else:
        return True


def benchmark_single_simulator(simulator, arguments, block_widths, block_heights):
    """
    Runs a set of benchmarks for a single simulator
    """

    logger = logging.getLogger(__name__)

    megacells = np.empty((len(block_heights), len(block_widths)))
    megacells.fill(np.nan)

    logger.debug("Running %d benchmarks with %s", len(block_heights) * len(block_widths), simulator.__name__)

    sim_arguments = arguments.copy()

    with Timer(simulator.__name__) as t:
        for j, block_height in enumerate(tqdm(block_heights, desc='Autotuner Progress')):
            sim_arguments.update({'block_height': block_height})
            for i, block_width in enumerate(tqdm(block_widths, desc=f'Iteration {j} Progress', leave=False)):
                sim_arguments.update({'block_width': block_width})
                megacells[j, i] = run_benchmark(sim_arguments)

    logger.debug("Completed %s in %f seconds", simulator.__name__, t.secs)

    return megacells


class Autotuner:
    def __init__(self,
                 nx=2048, ny=2048,
                 block_widths=range(8, 32, 1),
                 block_heights=range(8, 32, 1)):
        logger = logging.getLogger(__name__)
        self.filename = f"autotuning_data_{gethostname()}.npz"
        self.nx = nx
        self.ny = ny
        self.block_widths = block_widths
        self.block_heights = block_heights
        self.performance = {}

    def benchmark(self, simulator, force=False):
        logger = logging.getLogger(__name__)

        # Run through simulators and benchmark
        key = str(simulator.__name__)
        logger.info(f"Benchmarking {key} to {self.filename}")

        # If this simulator has been benchmarked already, skip it
        if force == False and os.path.isfile(self.filename):
            with np.load(self.filename) as data:
                if key in data["simulators"]:
                    logger.info(f"{key} already benchmarked - skipping")
                    return

        # Set arguments to send to the simulators during construction
        context = KernelContext(autotuning=False)
        g = 9.81
        h0, hu0, hv0, dx, dy, dt = gen_test_data(ny=self.ny, g=g)
        arguments = {
            'context': context,
            'h0': h0, 'hu0': hu0, 'hv0': hv0,
            'nx': self.nx, 'ny': self.ny,
            'dx': dx, 'dy': dy, 'dt': 0.9 * dt,
            'g': g,
            'compile_opts': ['-Wno-deprecated-gpu-targets']
        }

        # Load existing data into memory
        benchmark_data = {
            "simulators": [],
        }
        if os.path.isfile(self.filename):
            with np.load(self.filename) as data:
                for k, v in data.items():
                    benchmark_data[k] = v

        # Run benchmark
        benchmark_data[key + "_megacells"] = benchmark_single_simulator(arguments, self.block_widths,
                                                                        self.block_heights)
        benchmark_data[key + "_block_widths"] = self.block_widths
        benchmark_data[key + "_block_heights"] = self.block_heights
        benchmark_data[key + "_arguments"] = str(arguments)

        existing_sims = benchmark_data["simulators"]
        if isinstance(existing_sims, np.ndarray):
            existing_sims = existing_sims.tolist()
        if key not in existing_sims:
            benchmark_data["simulators"] = existing_sims + [key]

        # Save to file
        np.savez_compressed(self.filename, **benchmark_data)

    def get_peak_performance(self, simulator):
        """
        Function which reads a numpy file with autotuning data
        and reports the maximum performance and block size
        """

        logger = logging.getLogger(__name__)

        assert issubclass(simulator, Simulator.BaseSimulator)
        key = simulator.__name__

        if key in self.performance:
            return self.performance[key]
        else:
            # Run simulation if required
            if not os.path.isfile(self.filename):
                logger.debug(f"Could not get autotuned peak performance for {key}: benchmarking")
                self.benchmark(simulator)

            with np.load(self.filename) as data:
                if key not in data['simulators']:
                    logger.debug(f"Could not get autotuned peak performance for {key}: benchmarking")
                    data.close()
                    self.benchmark(simulator)
                    data = np.load(self.filename)

                def find_max_index(megacells):
                    max_index = np.nanargmax(megacells)
                    return np.unravel_index(max_index, megacells.shape)

                megacells = data[key + '_megacells']
                block_widths = data[key + '_block_widths']
                block_heights = data[key + '_block_heights']
                j, i = find_max_index(megacells)

                self.performance[key] = {"block_width": block_widths[i],
                                         "block_height": block_heights[j],
                                         "megacells": megacells[j, i]}
                logger.debug(f"Returning {self.performance[key]} as peak performance parameters")
                return self.performance[key]

            # This should never happen
            raise "Something wrong: Could not get autotuning data!"
            return None
