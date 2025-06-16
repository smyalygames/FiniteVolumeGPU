cmake_minimum_required(VERSION 4.0)
project(FiniteVolumeGPU LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 52)
endif()

set(CMAKE_CXX_STANDARD 23)

include_directories(common)

add_library(cuda_comp SHARED
        common/common.h
        common/EulerCommon.h
        common/limiters.h
        common/SWECommon.h
        cuda/EE2D_KP07_dimsplit.cu
        cuda/SWE2D_FORCE.cu
        cuda/SWE2D_HLL.cu
        cuda/SWE2D_HLL2.cu
        cuda/SWE2D_KP07.cu
        cuda/SWE2D_KP07_dimsplit.cu
        cuda/SWE2D_LxF.cu
        cuda/SWE2D_WAF.cu
        cuda/EE2D_KP07_dimsplit.cu
        cuda/EE2D_KP07_dimsplit.cu
        cuda/EE2D_KP07_dimsplit.cu)

target_compile_definitions(cuda_comp PUBLIC
        BLOCK_WIDTH
        BLOCK_HEIGHT
)

target_compile_features(cuda_comp PRIVATE cuda_std_14)
