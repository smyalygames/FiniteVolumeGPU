cmake_minimum_required(VERSION 3.21)
cmake_policy(VERSION 3.21.3...3.31.6)
project(FiniteVolumeGPU LANGUAGES CXX HIP CUDA)

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(hip)
add_subdirectory(cuda)
