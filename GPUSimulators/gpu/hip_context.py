import hashlib
import io
import os.path

import hip as hip_main
from hip import hip

from GPUSimulators.common import Timer
from GPUSimulators.gpu.context import Context


class HIPContext(Context):
    """
    Class that manages the HIP context.
    """

    def __init__(self, device=0, context_flags=None, use_cache=True, autotuning=False):
        """
        Creates a new HIP context.
        """
        super().__init__("hip", device, context_flags, use_cache, autotuning)

        # Log information about HIP version
        self.logger.info(f"HIP Python version {hip_main.HIP_VERSION_NAME}")
        self.logger.info(f"ROCm version {hip_main.ROCM_VERSION_NAME}")

        # Device information
        props = hip.hipDeviceProp_t()
        self.__hip_check(hip.hipGetDeviceProperties(props, device))
        device_count = self.__hip_check(hip.hipGetDeviceCount())
        arch = props.gcnArchName
        self.logger.info(
            f"Using device {device}/{device_count} '{props.name.decode('ascii')} ({arch.decode('ascii')})'"
            + f" ({props.pciBusID})"
        )
        self.logger.debug(f" => total available memory: {int(props.totalGlobalMem / pow(1024, 2))} MiB")

        if autotuning:
            self.logger.info(
                "Autotuning enabled. It may take several minutes to run the code the first time: have patience")
            raise (NotImplementedError("Autotuner is not yet implemented for HIP."))
            # TODO Implement Autotuner for HIP
            # self.autotuner = Autotuner.Autotuner()

    def __hip_check(self, call_request):
        """
        Function that checks if the HIP function executed successfully.
        """

        err = call_request[0]
        result = call_request[1:]
        if len(result) == 1:
            result = result[0]
        if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
            self.logger.error(f"HIP Error: {str(err)}")
            raise RuntimeError(str(err))
        return result

    def get_module(self, kernel_filename: str,
                   include_dirs: dict=None,
                   defines:dict[str: int]=None,
                   compile_args:dict=None,
                   jit_compile_args:dict=None):
        """
        Reads a ``.hip`` file and creates a HIP kernel from that.

        Args:
            kernel_filename: The file to use for the kernel.
            include_dirs: List of directories for the ``#include``s referenced.
            defines: Adds ``#define`` tags to the kernel, such as: ``#define key value``.
            compile_args: Adds other compiler options (parameters) for ``pycuda.compiler.compile()``.
            jit_compile_args: Adds other just-in-time compilation options (parameters)
                for ``pycuda.driver.module_from_buffer()``.

        Returns:
            The kernel module (pycuda.driver.Module).
        """
        if defines is None:
            defines = {}
        if include_dirs is None:
            include_dirs = []
        if compile_args is None:
            compile_args = {'no_extern_c': True}
        if jit_compile_args is None:
            jit_compile_args = {}

        def compile_message_handler(compile_success_bool, info_str, error_str):
            self.logger.debug(f"Compilation success: {str(compile_success_bool)}")
            if info_str:
                self.logger.debug(f"Compilation info: {info_str}")
            if error_str:
                self.logger.debug(f"Compilation error: {error_str}")

        kernel_filename = os.path.normpath(kernel_filename)
        kernel_path = os.path.abspath(os.path.join(self.module_path, kernel_filename))

        # Create a hash of the kernel options
        options_hasher = hashlib.md5()
        options_hasher.update(str(defines).encode('utf-8') + str(compile_args).encode('utf-8'))
        options_hash = options_hasher.hexdigest()

        # Create hash of the kernel source
        source_hash = self.hash_kernel(kernel_path, include_dirs=[self.module_path] + include_dirs)

        # Create the final hash
        root, ext = os.path.splitext(kernel_filename)
        kernel_hash = root + "_" + source_hash + "_" + options_hash + ext
        cached_kernel_filename = os.path.join(self.cache_path, kernel_hash)

        # Checking if the kernel is already in the hashmap
        if kernel_hash in self.modules.keys():
            self.logger.debug(f"Found kernel {kernel_filename} cached in hashmap ({kernel_hash})")
            return self.modules[kernel_hash]
        elif self.use_cache and os.path.isfile(cached_kernel_filename):
            self.logger.debug(f"Found kernerl {kernel_filename} cached on disk ({kernel_hash})")

            with io.open(cached_kernel_filename, "rb") as file:
                file_str = file.read()
                # TODO add ``module`` to HIP
                module = None

            self.modules[kernel_hash] = module
            return module
        else:
            self.logger.debug(f"Compiling {kernel_filename} ({kernel_hash})")

            # Create kernel string
            kernel_string = ""
            for key, value in defines.items():
                kernel_string += f"#define {str(key)} {str(value)}\n"
            kernel_string += f"#include \"{os.path.join(self.module_path, kernel_filename)}\""

            if self.use_cache:
                cached_kernel_dir = os.path.dirname(cached_kernel_filename)
                if not os.path.isdir(cached_kernel_dir):
                    os.mkdir(cached_kernel_dir)
                with io.open(cached_kernel_filename + ".txt", "w") as file:
                    file.write(kernel_string)

            with Timer("compiler") as timer:
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="The CUDA compiler succeeded, but said the following:\nkernel.cu", category=UserWarning)
                    # TODO compile the binary file
                    bin = None

                # TODO get binary from buffer
                module = None
                if self.use_cache:
                    with io.open(cached_kernel_filename, "wb") as file:
                        file.write(bin)

            self.modules[kernel_hash] = module
            return module

    def synchronize(self):
        self.__hip_check(hip.hipDeviceSynchronize())


test = HIPContext()
