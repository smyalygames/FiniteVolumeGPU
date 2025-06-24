import json
import logging
import os

import netCDF4
import numpy as np


def to_json(in_dict):
    out_dict = in_dict.copy()

    for key in out_dict:
        if isinstance(out_dict[key], np.ndarray):
            out_dict[key] = out_dict[key].tolist()
        else:
            try:
                json.dumps(out_dict[key])
            except:
                out_dict[key] = str(out_dict[key])

    return json.dumps(out_dict)


class DataDumper(object):
    """
    Simple class for holding a netCDF4 object
    (handles opening and closing nicely)
    Use as
    with DataDumper("filename") as data:
        ...
    """

    def __init__(self, filename, *args, **kwargs):
        self.logger = logging.getLogger(__name__)

        # Create directory if needed
        filename = os.path.abspath(filename)
        dirname = os.path.dirname(filename)
        if dirname and not os.path.isdir(dirname):
            self.logger.info("Creating directory " + dirname)
            os.makedirs(dirname)

        # Get mode of a file if we have that
        mode = None
        if args:
            mode = args[0]
        elif kwargs and 'mode' in kwargs.keys():
            mode = kwargs['mode']

        # Create a new unique file if writing
        if mode:
            if ("w" in mode) or ("+" in mode) or ("a" in mode):
                i = 0
                stem, ext = os.path.splitext(filename)
                while os.path.isfile(filename):
                    filename = f"{stem}_{str(i).zfill(4)}{ext}"
                    i = i + 1
        self.filename = os.path.abspath(filename)

        # Save arguments
        self.args = args
        self.kwargs = kwargs

        # Log output
        self.logger.info("Initialized " + self.filename)

    def __enter__(self):
        self.logger.info("Opening " + self.filename)
        if self.args:
            self.logger.info("Arguments: " + str(self.args))
        if self.kwargs:
            self.logger.info("Keyword arguments: " + str(self.kwargs))
        self.ncfile = netCDF4.Dataset(self.filename, *self.args, **self.kwargs)
        return self

    def __exit__(self, *args):
        self.logger.info("Closing " + self.filename)
        self.ncfile.close()

