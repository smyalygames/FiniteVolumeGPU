import hashlib
import io
import os.path

import hip as hip_main
from hip import hip, hiprtc

from GPUSimulators.common import Timer, hip_check
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
        self.prog = {}

        # Log information about HIP version
        self.logger.info(f"HIP Python version {hip_main.HIP_VERSION_NAME}")
        self.logger.info(f"ROCm version {hip_main.ROCM_VERSION_NAME}")

        # Device information
        props = hip.hipDeviceProp_t()
        hip_check(hip.hipGetDeviceProperties(props, device))
        device_count = hip_check(hip.hipGetDeviceCount())
        self.arch = props.gcnArchName
        self.logger.info(
            f"Using device {device}/{device_count} '{props.name.decode()} ({self.arch.decode()})'"
            + f" ({props.pciBusID})"
        )
        self.logger.debug(f" => total available memory: {int(props.totalGlobalMem / pow(1024, 2))} MiB")

        if autotuning:
            self.logger.info(
                "Autotuning enabled. It may take several minutes to run the code the first time: have patience")
            raise NotImplementedError("Autotuner is not yet implemented for HIP.")
            # TODO Implement Autotuner for HIP
            # self.autotuner = Autotuner.Autotuner()

    def __del__(self):
        for module in self.modules.values():
            hip_check(hip.hipModuleUnload(module))

        for prog in self.prog.values():
            hip_check(hiprtc.hiprtcDestroyProgram(prog.createRef()))

    def get_module(self, kernel_filename: str,
                   function: str,
                   include_dirs: list[str] = None,
                   defines: dict[str: int] = None,
                   compile_args: dict[str: list] = None,
                   jit_compile_args: dict = None):
        """
        Reads a ``.hip`` file and creates a HIP kernel from that.

        Args:
            kernel_filename: The file to use for the kernel.
            function: The main function of the kernel.
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
            include_dirs = [os.path.join(self.module_path, "include")]
        if compile_args is None:
            compile_args = {'hip': []}
        if jit_compile_args is None:
            jit_compile_args = {}

        compile_args = compile_args.get('hip')

        compile_args = [bytes(arg, "utf-8")for arg in compile_args]
        compile_args.append(b"--offload-arch=" + self.arch)

        def compile_message_handler(compile_success_bool, info_str, error_str):
            self.logger.debug(f"Compilation success: {str(compile_success_bool)}")
            if info_str:
                self.logger.debug(f"Compilation info: {info_str}")
            if error_str:
                self.logger.debug(f"Compilation error: {error_str}")

        kernel_filename = os.path.normpath(kernel_filename + ".hip")
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

        # Checks if the module is already cached in the hash map
        if kernel_hash in self.modules.keys():
            self.logger.debug(f"Found kernel {kernel_filename} cached in hashmap ({kernel_hash}).")
            return self.modules[kernel_hash]
        elif self.use_cache and os.path.isfile(cached_kernel_filename):
            # Check if the cache is on the disk
            self.logger.debug(f"Found kernel {kernel_filename} cached on disk ({kernel_hash}).")

            with io.open(cached_kernel_filename, "rb") as file:
                code = file.read()
                module = hip_check(hip.hipModuleLoadData(code))

            self.modules[kernel_hash] = module
            return module
        else:
            # As it was not found in the cache, compile it.
            self.logger.debug(f"Compiling {kernel_filename} ({kernel_hash}) for {self.arch}.")

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
                prog = hip_check(
                    hiprtc.hiprtcCreateProgram(bytes(kernel_string, "utf-8"), bytes(function, "utf-8"),
                                               0, [], []))

                err, = hiprtc.hiprtcCompileProgram(prog, len(compile_args), compile_args)
                if err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
                    log_size = hip_check(hiprtc.hiprtcGetProgramLogSize(prog))
                    log = bytearray(log_size)
                    hip_check(hiprtc.hiprtcGetProgramLog(prog, log))
                    raise RuntimeError(log.decode())

                code_size = hip_check(hiprtc.hiprtcGetCodeSize(prog))
                code = bytearray(code_size)
                hip_check(hiprtc.hiprtcGetCode(prog, code))
                module = hip_check(hip.hipModuleLoadData(code))

                if self.use_cache:
                    with io.open(cached_kernel_filename, "wb") as file:
                        file.write(code)

            self.modules[kernel_hash] = module
            self.prog[kernel_hash] = prog
            return module

    def synchronize(self):
        hip_check(hip.hipDeviceSynchronize())


test = HIPContext()

test.get_module("SWE2D_HLL",
                "HLLKernel",
                defines={
                    'BLOCK_WIDTH': 8,
                    'BLOCK_HEIGHT': 8
                },
                jit_compile_args={})
