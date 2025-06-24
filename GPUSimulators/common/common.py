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

import numpy as np
import time
import subprocess
import logging
import json

from GPUSimulators.common.data_dumper import DataDumper
from GPUSimulators.common.progress_printer import ProgressPrinter
from GPUSimulators.common.timer import Timer


def safe_call(cmd):
    logger = logging.getLogger(__name__)
    try:
        #git rev-parse HEAD
        current_dir = os.path.dirname(os.path.realpath(__file__))
        params = dict()
        params['stderr'] = subprocess.STDOUT
        params['cwd'] = current_dir
        params['universal_newlines'] = True #text=True in more recent python
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
                logger.warning("JSON: Converting {:s} to string ({:s})".format(key, value))
                out_dict[key] = value
    return json.dumps(out_dict)


def run_simulation(simulator, simulator_args, outfile, save_times, save_var_names=[], dt=None):
    """
    Runs a simulation, and store output in a netcdf file. Stores the times given in
    save_times, and saves all the variables in list save_var_names. Elements in
    save_var_names can be set to None if you do not want to save them
    """

    profiling_data_sim_runner = { 'start': {}, 'end': {} }
    profiling_data_sim_runner["start"]["t_sim_init"] = 0
    profiling_data_sim_runner["end"]["t_sim_init"] = 0
    profiling_data_sim_runner["start"]["t_nc_write"] = 0
    profiling_data_sim_runner["end"]["t_nc_write"] = 0
    profiling_data_sim_runner["start"]["t_full_step"] = 0
    profiling_data_sim_runner["end"]["t_full_step"] = 0

    profiling_data_sim_runner["start"]["t_sim_init"] = time.time()

    logger = logging.getLogger(__name__)

    if len(save_times <= 0):
        raise ValueError("Need to specify which times to save")

    with Timer("construct") as t:
        sim = simulator(**simulator_args)
    logger.info(f"Constructed in {str(t.secs)} seconds")

    #Create a netcdf file and simulate
    with DataDumper(outfile, mode='w', clobber=False) as outdata:
        
        #Create attributes (metadata)
        outdata.ncfile.created = time.ctime(time.time())
        outdata.ncfile.git_hash = get_git_hash()
        outdata.ncfile.git_status = get_git_status()
        outdata.ncfile.simulator = str(simulator)
        
        # do not write fields to attributes (they are to large)
        simulator_args_for_ncfile = simulator_args.copy()
        del simulator_args_for_ncfile["rho"]
        del simulator_args_for_ncfile["rho_u"]
        del simulator_args_for_ncfile["rho_v"]
        del simulator_args_for_ncfile["E"]
        outdata.ncfile.sim_args = to_json(simulator_args_for_ncfile)
        
        #Create dimensions
        outdata.ncfile.createDimension('time', len(save_times))
        outdata.ncfile.createDimension('x', simulator_args['nx'])
        outdata.ncfile.createDimension('y', simulator_args['ny'])

        #Create variables for dimensions
        ncvars = {'time': outdata.ncfile.createVariable('time', np.dtype('float32').char, 'time'),
                  'x': outdata.ncfile.createVariable('x', np.dtype('float32').char, 'x'),
                  'y': outdata.ncfile.createVariable('y', np.dtype('float32').char, 'y')}

        #Fill variables with proper values
        ncvars['time'][:] = save_times
        extent = sim.get_extent()
        ncvars['x'][:] = np.linspace(extent[0], extent[1], simulator_args['nx'])
        ncvars['y'][:] = np.linspace(extent[2], extent[3], simulator_args['ny'])
        
        #Choose which variables to download (prune None from the list, but keep the index)
        download_vars = []
        for i, var_name in enumerate(save_var_names):
            if var_name is not None:
                download_vars += [i]
        save_var_names = list(save_var_names[i] for i in download_vars)
        
        #Create variables
        for var_name in save_var_names:
            ncvars[var_name] = outdata.ncfile.createVariable(
                var_name, np.dtype('float32').char, ('time', 'y', 'x'), zlib=True, least_significant_digit=3)

        #Create step sizes between each save
        t_steps = np.empty_like(save_times)
        t_steps[0] = save_times[0]
        t_steps[1:] = save_times[1:] - save_times[0:-1]

        profiling_data_sim_runner["end"]["t_sim_init"] = time.time()

        # Start simulation loop
        progress_printer = ProgressPrinter(save_times[-1], print_every=10)
        for k in range(len(save_times)):
            # Get target time and step size there
            t_step = t_steps[k]
            t_end = save_times[k]
            
            # Sanity check simulator
            try:
                sim.check()
            except AssertionError as e:
                logger.error(f"Error after {sim.sim_steps()} steps (t={sim.sim_time()}: {str(e)}")
                return outdata.filename

            profiling_data_sim_runner["start"]["t_full_step"] += time.time()

            # Simulate
            if t_step > 0.0:
                sim.simulate(t_step, dt)

            profiling_data_sim_runner["end"]["t_full_step"] += time.time()

            profiling_data_sim_runner["start"]["t_nc_write"] += time.time()

            #Download
            save_vars = sim.download(download_vars)
            
            #Save to file
            for i, var_name in enumerate(save_var_names):
                ncvars[var_name][k, :] = save_vars[i]

            profiling_data_sim_runner["end"]["t_nc_write"] += time.time()

            #Write progress to screen
            print_string = progress_printer.get_print_string(t_end)
            if print_string:
                logger.debug(print_string)
                
        logger.debug(f"Simulated to t={t_end} in "
                     + f"{sim.sim_steps()} timesteps (average dt={sim.sim_time() / sim.sim_steps()})")

    return outdata.filename, profiling_data_sim_runner, sim.profiling_data_mpi
    