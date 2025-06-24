import gc
import logging
import os
import signal
import subprocess
import time

from GPUSimulators.common.popen_file_buffer import PopenFileBuffer


class IPEngine(object):
    """
    Class for starting IPEngines for MPI processing in IPython
    """

    def __init__(self, n_engines):
        self.logger = logging.getLogger(__name__)

        # Start ipcontroller
        self.logger.info("Starting IPController")
        self.c_buff = PopenFileBuffer()
        c_cmd = ["ipcontroller", "--ip='*'"]
        c_params = dict()
        c_params['stderr'] = self.c_buff.stderr
        c_params['stdout'] = self.c_buff.stdout
        c_params['shell'] = False
        if os.name == 'nt':
            c_params['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
        self.c = subprocess.Popen(c_cmd, **c_params)

        # Wait until the controller is running
        time.sleep(3)

        # Start engines
        self.logger.info("Starting IPEngines")
        self.e_buff = PopenFileBuffer()
        e_cmd = ["mpiexec", "-n", str(n_engines), "ipengine", "--mpi"]
        e_params = dict()
        e_params['stderr'] = self.e_buff.stderr
        e_params['stdout'] = self.e_buff.stdout
        e_params['shell'] = False
        if os.name == 'nt':
            e_params['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
        self.e = subprocess.Popen(e_cmd, **e_params)

        # attach to a running cluster
        import ipyparallel
        self.cluster = ipyparallel.Client()  # profile='mpi')
        time.sleep(3)
        while len(self.cluster.ids) != n_engines:
            time.sleep(0.5)
            self.logger.info("Waiting for cluster...")
            self.cluster = ipyparallel.Client()  # profile='mpi')

        self.logger.info("Done")

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        if self.e is not None:
            if os.name == 'nt':
                self.logger.warning("Sending CTRL+C to IPEngine")
                self.e.send_signal(signal.CTRL_C_EVENT)

            try:
                self.e.communicate(timeout=3)
                self.e.kill()
            except subprocess.TimeoutExpired:
                self.logger.warning("Killing IPEngine")
                self.e.kill()
                self.e.communicate()
            self.e = None

            cout, cerr = self.e_buff.read()
            self.logger.info(f"IPEngine cout: {cout}")
            self.logger.info(f"IPEngine cerr: {cerr}")
            self.e_buff = None

            gc.collect()

        if self.c is not None:
            if os.name == 'nt':
                self.logger.warning("Sending CTRL+C to IPController")
                self.c.send_signal(signal.CTRL_C_EVENT)

            try:
                self.c.communicate(timeout=3)
                self.c.kill()
            except subprocess.TimeoutExpired:
                self.logger.warning("Killing IPController")
                self.c.kill()
                self.c.communicate()
            self.c = None

            cout, cerr = self.c_buff.read()
            self.logger.info(f"IPController cout: {cout}")
            self.logger.info(f"IPController cerr: {cerr}")
            self.c_buff = None

            gc.collect()