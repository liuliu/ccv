cmake_minimum_required(VERSION 2.6)

set(ccv_lib "ccv_lib")
set(ccv_bin "ccv_bin")

set(CMAKE_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG} -g -O0)
set(CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE} -O2)

add_subdirectory(lib)
add_subdirectory(bin)

