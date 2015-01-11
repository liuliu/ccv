cmake_minimum_required(VERSION 2.6)

project(${ccv_lib})

file(GLOB sources "*.c")
file(GLOB_RECURSE thirdparty "3rdparty/*.c")
file(GLOB_RECURSE cuda "cuda/*.c")

file(GLOB header "ccv.h")

set(DEP_LIBS m dl pthread) 

find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)

if(PNG_FOUND)
	add_definitions(-DHAVE_LIBPNG)
	list(APPEND DEP_LIBS png)
endif()

if(JPEG_FOUND)
	add_definitions(-DHAVE_LIBJPEG)
	list(APPEND DEP_LIBS jpeg)
endif()

find_library(LIBLINEAR_LIBRARY NAMES linear liblinear)

if(LIBLINEAR_LIBRARY)
	message("liblinear found")
	add_definitions(-DHAVE_LIBLINEAR)
	list(APPEND DEP_LIBS linear)
else()
	message("liblinear not found")
endif()

find_library(LIBGSL_LIBRARY NAMES gsl libgsl)

if(LIBGSL_LIBRARY)
	message("libgsl found")
	add_definitions(-DHAVE_GSL)
	list(APPEND DEP_LIBS gsl)
else()
	message("libgsl not found")
endif()

message("libs: ${DEP_LIBS}")

add_library(${ccv_lib} SHARED ${sources} ${thirdparty} ${cuda})
set_target_properties(${ccv_lib} PROPERTIES OUTPUT_NAME "ccv")
target_link_libraries(${ccv_lib} ${DEP_LIBS})

install(TARGETS ${ccv_lib} DESTINATION lib)
install(FILES ${header} DESTINATION "include")

