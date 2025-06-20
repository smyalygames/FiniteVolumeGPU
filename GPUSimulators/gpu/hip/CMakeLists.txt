file(GLOB MODELS "${CMAKE_CURRENT_SOURCE_DIR}/*.hip")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)


add_library(hip_comp STATIC ${MODELS})

set_target_properties(hip_comp PROPERTIES LINKER_LANGUAGE HIP)

target_compile_definitions(hip_comp PUBLIC
        BLOCK_WIDTH
        BLOCK_HEIGHT
)
