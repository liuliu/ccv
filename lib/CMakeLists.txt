find_package(JPEG)
if(JPEG_FOUND)
	include_directories(${JPEG_INCLUDE_DIR})
	list(APPEND _ccv_libs ${JPEG_LIBRARIES})
	add_definitions(-DHAVE_LIBJPEG)
else(JPEG_FOUND)
	message(STATUS "!!! libjpeg not found. Support for JPEG IO will be disabled.")
endif(JPEG_FOUND)

find_package(PNG)
if(PNG_FOUND)
	add_definitions(-DHAVE_LIBPNG)
	add_definitions(${PNG_DEFINITIONS})
	include_directories(${PNG_PNG_INCLUDE_DIR})
	list(APPEND _ccv_libs ${PNG_LIBRARIES})
else(PNG_FOUND)
	message(STATUS "!!! libpng not found. support for PNG IO will be disabled.")
endif(PNG_FOUND)

find_package(GSL)
if(GSL_FOUND)
	add_definitions(-DHAVE_GSL)
	include_directories(${GSL_INCLUDE_DIRS})
	list(APPEND _ccv_libs ${GSL_LIBRARIES})
else(GSL_FOUND)
	message(STATUS "!!! The GNU scientific library (GSL) not found. Support for it will not be compiled in.")
endif(GSL_FOUND)

find_package(CBLAS)
if(CBLAS_FOUND)
	add_definitions(-DHAVE_CBLAS)
	include_directories(${CBLAS_INCLUDE_DIRS})
	list(APPEND _ccv_libs ${CBLAS_LIBRARIES})
else(CBLAS_FOUND)
	message(STATUS "!!! The CBLAS library was not found. Support for it will not be compiled in.")
endif(CBLAS_FOUND)

find_package(FFTW)
if(FFTW_FOUND)
	include_directories(${FFTW_INCLUDES})
	list(APPEND _ccv_libs ${FFTW_LIB} ${FFTWF_LIB})
	add_definitions(-DHAVE_FFTW3)
else(FFTW_FOUND)
	message(STATUS "!!! The FFTW3 library was not found. The kissfft library will be used instead.")

	list(APPEND _fft_sources
		3rdparty/kissfft/kiss_fft.c
		3rdparty/kissfft/kissf_fft.c
		3rdparty/kissfft/kiss_fftr.c
		3rdparty/kissfft/kissf_fftr.c
		3rdparty/kissfft/kiss_fftndr.c
		3rdparty/kissfft/kissf_fftnd.c
		3rdparty/kissfft/kissf_fftndr.c
		3rdparty/kissfft/kiss_fftnd.c
	)
endif(FFTW_FOUND)

find_package(OpenMP)
if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else(OPENMP_FOUND)
	message(STATUS "!!! Your compiler does not appear to support OpenMP.")
endif(OPENMP_FOUND)

add_library(ccv
	ccv_classic.c
	ccv_memory.c
	ccv_numeric.c
	ccv_dpm.c
	ccv_mser.c
	ccv_bbf.c
	ccv_daisy.c
	ccv_io.c
	ccv_sift.c
	ccv_cache.c
	ccv_util.c
	ccv_basic.c
	ccv_algebra.c
	ccv_swt.c

	3rdparty/sha1/sha1.c
	${_fft_sources}
)
target_link_libraries(ccv ${_ccv_libs})
