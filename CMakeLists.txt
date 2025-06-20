cmake_minimum_required(VERSION 3.21)
cmake_policy(VERSION 3.21.3...3.31.6)

if(NOT DEFINED ROCM_PATH)
    if(NOT DEFINED ENV{ROCM_PATH})
        execute_process(COMMAND hipconfig --rocmpath
                OUTPUT_VARIABLE ROCM_PATH)
        message(STATUS "ROCm SDK path: ${ROCM_PATH}")
        set(ROCM_PATH "${ROCM_PATH}" CACHE PATH
                "Path to where ROCm is installed")
    else()
        set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH
                "Path to where ROCm is installed")
    endif()
endif()
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -isystem ${ROCM_PATH}/include")

project(FiniteVolumeGPU LANGUAGES CXX HIP CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Dependencies

find_package(hip REQUIRED)



# HIP
if((DEFINED ENV{HIP_PLATFORM}) AND ("$ENV{HIP_PLATFORM}" STREQUAL "nvidia"))
    message("Building for NVIDIA backend.")

    enable_language(CUDA)
    set(CUDA_SEPARABLE_COMPILATION ON)

    find_package(CUDAToolkit REQUIRED)

    set(kernel_lang CUDA)
    set(kernel_libs CUDA::cudart CUDA::cuda_driver)

    set(CMAKE_CUDA_ARCHITECTURES $ENV{GPU_ARCH})

    # Remove any preprocessor definitions for AMD
    remove_definitions(-D__HIP_PLATFORM_HCC__ -D__HIP_PLATFORM_AMD__)
    # Add CUDA preprocessor definitions
    add_definitions(-D__HIP_PLATFORM_NVCC__ -D__HIP_PLATFORM_NVIDIA__)
else()
    message("Building for AMD backend.")

    set(kernel_lang HIP)
    set(kernel_libs hip::device)

    set(CMAKE_HIP_ARCHITECTURES $ENV{GPU_ARCH})

    # Remove any preprocessor definitions for CUDA
    remove_definitions(-D__HIP_PLATFORM_NVCC__ -D__HIP_PLATFORM_NVIDIA__)
    # Add AMD preprocessor definitions
    add_definitions(-D__HIP_PLATFORM_HCC__ -D__HIP_PLATFORM_AMD__)
endif()

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(GPUSimulators/gpu/hip)
add_subdirectory(GPUSimulators/gpu/cuda)
