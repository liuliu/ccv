project(ccv C)
cmake_minimum_required(VERSION 2.8)
enable_testing()

# Add any custom CMake modules we include to the module path.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

# Unless specified explicitly, use the RelWithDebInfo build type which enables optimisations and
# debug information.
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif(NOT CMAKE_BUILD_TYPE)

# Compiler specific flags
if(NOT WIN32)
	set(CMAKE_C_FLAGS "${CMAKE_ANSI_CFLAGS} ${CMAKE_C_FLAGS} -std=c99")
endif(NOT WIN32)

# The CCV library itself
add_subdirectory(lib)

# Add the CCV library directory to the include path for subsequent steps
include_directories("${CMAKE_SOURCE_DIR}/lib")

# Tools provided by the library
add_subdirectory(bin)

# Testing
add_subdirectory(test)
