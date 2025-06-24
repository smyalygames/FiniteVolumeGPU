import logging
import os
import io
import re
import logging
from hashlib import md5


class Context(object):
    """
    Class that manages either a HIP or CUDA context.
    """

    def __init__(self, language: str, device=0, context_flags=None, use_cache=True, autotuning=True):
        """
        Create a new context.
        """
        self.use_cache = use_cache
        self.logger = logging.getLogger(__name__)
        self.modules = {}

        self.module_path = os.path.dirname(os.path.realpath(__file__))

        self.autotuner = None

        # Creates cache directory if specified
        self.cache_path = os.path.join(self.module_path, f"{language}_cache")
        if self.use_cache:
            if not os.path.isdir(self.cache_path):
                os.mkdir(self.cache_path)
            self.logger.info(f"Using cache dir {self.cache_path}")

    def __del__(self):
        """
        Cleans up the context.
        """
        pass

    def __str__(self):
        """
        Gives the context id.
        """
        pass

    def hash_kernel(self, kernel_filename: str, include_dirs: list[str]) -> str:
        """
        Generate a kernel ID for the caches.

        Args:
            kernel_filename: Path to the kernel file.
            include_dirs: Directories to search for ``#include`` in the kernel file.

        Returns:
            MD5 has for the kernel in the cache.

        Raises:
            RuntimeError: When the number of ``#include``s surpassed the maximum (101) permitted ``#include``s.
        """

        num_includes = 0
        max_includes = 100
        kernel_hasher = md5()
        logger = logging.getLogger(__name__)

        # Loop over files and includes, and check if something has changed
        files = [kernel_filename]
        while len(files):
            if num_includes > max_includes:
                raise RuntimeError(f"Maximum number of includes reached.\n"
                                   + f"Potential circular include in {kernel_filename}?")

            filename = files.pop()

            modified = os.path.getmtime(filename)

            # Open the file
            with io.open(filename, "r") as file:
                # Search for ``#include <reference>`` and also hash the file
                file_str = file.read()
                kernel_hasher.update(file_str.encode('utf-8'))
                kernel_hasher.update(str(modified).encode('utf-8'))

                # Find all the includes
                includes = re.findall('^\W*#include\W+(.+?)\W*$', file_str, re.M)

            # Iterate through everything that looks like is an ``include``
            for include_file in includes:
                # Search through ``include`` directories for the file
                file_path = os.path.dirname(filename)
                for include_path in [file_path] + include_dirs:
                    # If found, add it to the list of files to check
                    temp_path = os.path.join(include_path, include_file)
                    if os.path.isfile(temp_path):
                        files = files + [temp_path]
                        # To avoid circular includes
                        num_includes = num_includes + 1
                        break

        return kernel_hasher.hexdigest()

    def get_module(self, kernel_filename: str,
                   include_dirs: dict = None,
                   defines: list[str] = None,
                   compile_args: dict = None,
                   jit_compile_args: dict = None):
        """
        Reads a text file and creates a kernel from that.
        """
        raise NotImplementedError("Needs to be implemented in subclass")

    def synchronize(self):
        """
        Synchronizes all the streams, etc.
        """
        raise NotImplementedError("Needs to be implemented in subclass")
