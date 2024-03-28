cmake_minimum_required(VERSION 2.6)

project(${ccv_bin})

set(exes aflw bbfcreate bbfdetect bbffmt cifar-10 cnnclassify dpmcreate dpmdetect icfcreate icfdetect icfoptimize image-net msermatch scdcreate scddetect siftmatch swtcreate swtdetect tld)

foreach(e ${exes})
	file(GLOB sources "${e}.c")

	add_executable(${e} ${sources})
	set_target_properties(${e} PROPERTIES OUTPUT_NAME ${e})

	add_dependencies(${e} ${ccv_lib})
	include_directories(${ccv_lib_SOURCE_DIR})
	target_link_libraries(${e} ${ccv_lib})

	install(TARGETS ${e} DESTINATION bin)
endforeach()
 
