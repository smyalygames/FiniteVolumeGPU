include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

execute_process(COMMAND hipconfig --rocmpath
    OUTPUT_VARIABLE rocm_path)
message(STATUS "ROCm SDK path: ${rocm_path}")
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -isystem ${rocm_path}/include")


file(GLOB MODELS "${CMAKE_CURRENT_SOURCE_DIR}/*.hip")

add_library(hip_comp ${MODELS})

target_compile_definitions(hip_comp PUBLIC
        BLOCK_WIDTH
        BLOCK_HEIGHT
)

target_compile_features(hip_comp PRIVATE cxx_std_23)
