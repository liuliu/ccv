add_executable(defects.l0.1.tests defects.l0.1.tests.c)
target_link_libraries(defects.l0.1.tests ccv)
add_test(defects.l0.1.tests defects.l0.1.tests)
