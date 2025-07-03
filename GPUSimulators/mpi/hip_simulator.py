import numpy as np

from .simulator import BaseMPISimulator


class HIPMPISimulator(BaseMPISimulator):
    def __init__(self, sim, grid):
        super().__init__(sim, grid, self.__create_pagelocked_memory)

    def __create_pagelocked_memory(self):
        self.in_e = np.empty((int(self.nvars), int(self.read_e[3]), int(self.read_e[2])),
                             dtype=np.float32)  # np.empty((self.nvars, self.read_e[3], self.read_e[2]), dtype=np.float32)
        self.in_w = np.empty((int(self.nvars), int(self.read_w[3]), int(self.read_w[2])),
                             dtype=np.float32)  # np.empty((self.nvars, self.read_w[3], self.read_w[2]), dtype=np.float32)
        self.in_n = np.empty((int(self.nvars), int(self.read_n[3]), int(self.read_n[2])),
                             dtype=np.float32)  # np.empty((self.nvars, self.read_n[3], self.read_n[2]), dtype=np.float32)
        self.in_s = np.empty((int(self.nvars), int(self.read_s[3]), int(self.read_s[2])),
                             dtype=np.float32)  # np.empty((self.nvars, self.read_s[3], self.read_s[2]), dtype=np.float32)

        # Allocate data for sending
        self.out_e = np.empty((int(self.nvars), int(self.read_e[3]), int(self.read_e[2])),
                              dtype=np.float32)  # np.empty_like(self.in_e)
        self.out_w = np.empty((int(self.nvars), int(self.read_w[3]), int(self.read_w[2])),
                              dtype=np.float32)  # np.empty_like(self.in_w)
        self.out_n = np.empty((int(self.nvars), int(self.read_n[3]), int(self.read_n[2])),
                              dtype=np.float32)  # np.empty_like(self.in_n)
        self.out_s = np.empty((int(self.nvars), int(self.read_s[3]), int(self.read_s[2])),
                              dtype=np.float32)  # np.empty_like(self.in_s)
