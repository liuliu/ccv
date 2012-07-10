foreach(_target algebra.tests util.tests numeric.tests basic.tests memory.tests)
	add_executable(${_target} ${_target}.c)
	target_link_libraries(${_target} ccv)
	add_test(
		NAME ${_target}
		WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		COMMAND ${_target}
	)
endforeach(_target)
