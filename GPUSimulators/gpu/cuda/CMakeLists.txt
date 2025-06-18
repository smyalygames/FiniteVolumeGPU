if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 52)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB MODELS "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

add_library(cuda_comp MODULE ${MODELS})

target_compile_definitions(cuda_comp PUBLIC
        BLOCK_WIDTH
        BLOCK_HEIGHT
)

target_compile_features(cuda_comp PRIVATE cuda_std_14)
