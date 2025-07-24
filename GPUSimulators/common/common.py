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

import os
import time
import subprocess
import logging
import json

import numpy as np
from tqdm.auto import tqdm
from mpi4py import MPI

import GPUSimulators.mpi
from GPUSimulators.common.data_dumper import DataDumper
from GPUSimulators.common.timer import Timer


def safe_call(cmd):
    logger = logging.getLogger(__name__)
    try:
        # git rev-parse HEAD
        current_dir = os.path.dirname(os.path.realpath(__file__))
        params = dict()
        params['stderr'] = subprocess.STDOUT
        params['cwd'] = current_dir
        params['universal_newlines'] = True  # text=True in more recent python
        params['shell'] = False
        if os.name == 'nt':
            params['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
        stdout = subprocess.check_output(cmd, **params)
    except subprocess.CalledProcessError as e:
        output = e.output
        logger.error("Git failed, \nReturn code: " + str(e.returncode) + "\nOutput: " + output)
        raise e

    return stdout


def get_git_hash():
    return safe_call(["git", "rev-parse", "HEAD"])


def get_git_status():
    return safe_call(["git", "status", "--porcelain", "-uno"])


def to_json(in_dict, compressed=True):
    """
    Creates JSON string from a dictionary
    """

    logger = logging.getLogger(__name__)
    out_dict = in_dict.copy()
    for key in out_dict:
        if isinstance(out_dict[key], np.ndarray):
            out_dict[key] = out_dict[key].tolist()
        else:
            try:
                json.dumps(out_dict[key])
            except:
                value = str(out_dict[key])
                logger.warning(f"JSON: Converting {key} to string ({value})")
                out_dict[key] = value
    return json.dumps(out_dict)


def run_simulation(simulator, simulator_args, outfile, save_times, save_var_names=[], dt=None):
    """
    Runs a simulation, and store output in a netcdf file. Stores the times given in
    save_times, and saves all the variables in list save_var_names. Elements in
    save_var_names can be set to None if you do not want to save them
    """

    profiling_data_sim_runner = {'start': {}, 'end': {}}
    profiling_data_sim_runner["start"]["t_sim_init"] = 0
    profiling_data_sim_runner["end"]["t_sim_init"] = 0
    profiling_data_sim_runner["start"]["t_nc_write"] = 0
    profiling_data_sim_runner["end"]["t_nc_write"] = 0
    profiling_data_sim_runner["start"]["t_full_step"] = 0
    profiling_data_sim_runner["end"]["t_full_step"] = 0

    profiling_data_sim_runner["start"]["t_sim_init"] = time.time()

    logger = logging.getLogger(__name__)

    if len(save_times) <= 0:
        raise ValueError("Need to specify which times to save")

    with Timer("construct") as t:
        sim = simulator(**simulator_args)
    logger.info(f"Constructed in {str(t.secs)} seconds")

    # Create a netcdf file and simulate
    with DataDumper(outfile, mode='w', parallel=True, comm=MPI.COMM_WORLD, info=MPI.Info()) as outdata:
        logger.info("Created NetCDF4 file.")

        # Create attributes (metadata)
        outdata.nc.created = time.ctime(time.time())
        outdata.nc.git_hash = get_git_hash()
        outdata.nc.git_status = get_git_status()
        outdata.nc.simulator = str(simulator)

        # do not write fields to attributes (they are to large)
        simulator_args_for_ncfile = simulator_args.copy()
        del simulator_args_for_ncfile["rho"]
        del simulator_args_for_ncfile["rho_u"]
        del simulator_args_for_ncfile["rho_v"]
        del simulator_args_for_ncfile["E"]
        outdata.nc.sim_args = to_json(simulator_args_for_ncfile)

        # Create dimensions
        if isinstance(sim, GPUSimulators.mpi.MPISimulator):
            x_size = sim.grid.x
            y_size = sim.grid.y
            logger.info(f"Grid is - x: {x_size}, y: {y_size}")
        else:
            x_size = 1
            y_size = 1

        grid_x0 = sim.grid.x0
        grid_x1 = sim.grid.x1
        grid_y0 = sim.grid.y0
        grid_y1 = sim.grid.y1

        x = simulator_args['nx'] * x_size
        y = simulator_args['ny'] * y_size
        outdata.nc.createDimension('time', len(save_times))
        outdata.nc.createDimension('x', x)
        outdata.nc.createDimension('y', y)

        # Create variables for dimensions
        ncvars = {'time': outdata.nc.createVariable('time', 'f4', ('time',)),
                  'x': outdata.nc.createVariable('x', 'f4', ('x',)),
                  'y': outdata.nc.createVariable('y', 'f4', ('y',))}


        # Fill variables with proper values
        ncvars['time'][:] = save_times
        ncvars['time'].units = "s"
        x0, x1, y0, y1 = sim.get_extent()
        ncvars['x'][grid_x0:grid_x1] = np.linspace(x0, x1, simulator_args['nx'])
        ncvars['y'][grid_y0:grid_y1] = np.linspace(y0, y1, simulator_args['ny'])

        # Choose which variables to download (prune None from the list, but keep the index)
        download_vars = []
        for i, var_name in enumerate(save_var_names):
            if var_name is not None:
                download_vars += [i]
        save_var_names = list(save_var_names[i] for i in download_vars)

        # Create variables
        for var_name in save_var_names:
            ncvars[var_name] = outdata.nc.createVariable(
                var_name, 'f4', ('time', 'y', 'x'), zlib=True, least_significant_digit=3)
            ncvars[var_name].set_collective(True)

        # Create step sizes between each save
        t_steps = np.empty_like(save_times)
        t_steps[0] = save_times[0]
        t_steps[1:] = save_times[1:] - save_times[0:-1]

        profiling_data_sim_runner["end"]["t_sim_init"] = time.time()

        with tqdm(total=save_times[-1], desc="Simulation progress", unit="sim s") as pbar:
            # Start simulation loop
            for save_step, t_step in enumerate(t_steps):
                t_end = save_step

                # Sanity check simulator
                try:
                    sim.check()
                except AssertionError as e:
                    logger.error(f"Error after {sim.sim_steps()} steps (t={sim.sim_time()}: {str(e)}")
                    return outdata.filename

                profiling_data_sim_runner["start"]["t_full_step"] += time.time()

                # Simulate
                if t_step > 0.0:
                    sim.simulate(t_step, dt, pbar=pbar)

                profiling_data_sim_runner["end"]["t_full_step"] += time.time()

                profiling_data_sim_runner["start"]["t_nc_write"] += time.time()

                # Download
                save_vars = sim.download(download_vars)

                # Save to file
                for i, var_name in enumerate(save_var_names):
                    ncvars[var_name][save_step, grid_y0:grid_y1] = save_vars[i]

                profiling_data_sim_runner["end"]["t_nc_write"] += time.time()

        logger.debug(f"Simulated to t={t_end} in "
                     + f"{sim.sim_steps()} timesteps (average dt={sim.sim_time() / sim.sim_steps()})")

    return outdata.filename, profiling_data_sim_runner, sim.profiling_data_mpi
