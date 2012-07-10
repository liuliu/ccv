# - Find CBLAS
# Find the native CBLAS headers and libraries.
#
#  CBLAS_LIBRARIES    - List of libraries when using cblas.
#  CBLAS_FOUND        - True if cblas found.
#
# Copyright 2009-2011 The VOTCA Development Team (http://www.votca.org)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

find_library(CBLAS_LIBRARY NAMES cblas gslcblas HINTS $ENV{CBLASDIR}/lib $ENV{CBLASDIR}/lib64 )

set(CBLAS_LIBRARIES ${CBLAS_LIBRARY} )

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set FFTW3_FOUND to TRUE
# if all listed variables are TRUE

find_package_handle_standard_args(CBLAS DEFAULT_MSG CBLAS_LIBRARY )

if (CBLAS_FOUND)
  include(CheckLibraryExists)
  check_library_exists("${CBLAS_LIBRARY}" cblas_dsyrk "" FOUND_DSYRK)
  if(NOT FOUND_DSYRK)
    message(FATAL_ERROR "Could not find cblas_dsyrk in ${CBLAS_LIBRARY}, take a look at the error message in ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log to find out what was going wrong. If you don't have pkg-config installed you will most likely have to set CBLAS_LIBRARY by hand (i.e. -DCBLAS_LIBRARY='/path/to/libcblas.so') !")
  endif(NOT FOUND_DSYRK)
endif (CBLAS_FOUND)

mark_as_advanced( CBLAS_LIBRARY )
