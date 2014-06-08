cmake_minimum_required(VERSION 2.8)
include(CheckIncludeFile)
project(ccv)
 
find_package(png)
find_package(jpeg)
find_path(HAVE_FFTW fftw3.h)
find_path(HAVE_LIBLINEAR linear.h)
find_path(HAVE_CBLAS cblas.h)
find_path(HAVE_AVCODEC libavcodec/avcodec.h)
find_path(HAVE_AVFORMAT libavformat/avformat.h)
find_path(HAVE_SWSCALE libswscale/swscale.h)
check_include_file(xmmintrin.h HAVE_SSE)
 
if(JPEG_FOUND)
  add_definitions(-DHAVE_LIBJPEG)
  include_directories(${JPEG_INCLUDE_DIRS})
endif()
 
if(PNG_FOUND)
  add_definitions(-DHAVE_LIBPNG)
  include_directories(${PNG_INCLUDE_DIRS})
endif()
 
if(HAVE_FFTW)
  add_definitions(-DHAVE_FFTW3)
  include_directories(${HAVE_FFTW})
  message(STATUS "Found FFTW3: ${HAVE_FFTW}")
endif()
 
if(HAVE_LIBLINEAR)
  add_definitions(-DHAVE_LIBLINEAR)
  include_directories(${HAVE_LIBLINEAR})
  message(STATUS "Found LIBLINEAR: ${HAVE_LIBLINEAR}")
endif()
 
if(HAVE_CBLAS)
  add_definitions(-DHAVE_CBLAS)
  include_directories(${HAVE_CBLAS})
  message(STATUS "Found CBLAS: ${HAVE_CBLAS}")
endif()
 
if(HAVE_AVCODEC)
  add_definitions(-DHAVE_AVCODEC)
  include_directories(${HAVE_AVCODEC})
  message(STATUS "Found AVCODEC: ${HAVE_AVCODEC}")
endif()
 
if(HAVE_AVFORMAT)
  add_definitions(-DHAVE_AVFORMAT)
  include_directories(${HAVE_AVFORMAT})
  message(STATUS "Found AVFORMAT: ${HAVE_AVFORMAT}")
endif()
 
if(HAVE_SWSCALE)
  add_definitions(-DHAVE_SWSCALE)
  include_directories(${HAVE_SWSCALE})
  message(STATUS "Found SWSCALE: ${HAVE_SWSCALE}")
endif()
 
if(HAVE_SSE)
  add_definitions(-DHAVE_SSE2)
  message(STATUS "Found SSE2")
endif()
 
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  find_path(HAVE_DISPATCH dispatch/dispatch.h)  
  if(HAVE_DISPATCH)
    add_definitions(-DUSE_DISPATCH)
    message(STATUS "Found DISPATCH")
  endif()
endif()
 
# @todo - check for gsl
 
set(bd ${CMAKE_CURRENT_LIST_DIR})
set(sd ${bd}/lib)
 
set(ccv_sources
  ${sd}/ccv_algebra.c
  ${sd}/ccv_basic.c
  ${sd}/ccv_bbf.c
  ${sd}/ccv_cache.c
  ${sd}/ccv_classic.c
  ${sd}/ccv_convnet.c
  ${sd}/ccv_daisy.c
  ${sd}/ccv_dpm.c
  ${sd}/ccv_ferns.c
  ${sd}/ccv_icf.c
  ${sd}/ccv_io.c
  ${sd}/ccv_memory.c
  ${sd}/ccv_mser.c
  ${sd}/ccv_numeric.c
  ${sd}/ccv_resample.c
  ${sd}/ccv_sift.c
  ${sd}/ccv_sparse_coding.c
  ${sd}/ccv_swt.c
  ${sd}/ccv_tld.c
  ${sd}/ccv_transform.c
  ${sd}/ccv_util.c
 
  ${sd}/3rdparty/dsfmt/dSFMT.c
  ${sd}/3rdparty/sfmt/SFMT.c
  ${sd}/3rdparty/sha1/sha1.c
  ${sd}/3rdparty/sqlite3/sqlite3.c
  ${sd}/3rdparty/kissfft/kiss_fft.c
  ${sd}/3rdparty/kissfft/kiss_fftnd.c
  ${sd}/3rdparty/kissfft/kiss_fftndr.c
  ${sd}/3rdparty/kissfft/kiss_fftr.c
  ${sd}/3rdparty/kissfft/kissf_fft.c
  ${sd}/3rdparty/kissfft/kissf_fftnd.c
  ${sd}/3rdparty/kissfft/kissf_fftndr.c
  ${sd}/3rdparty/kissfft/kissf_fftr.c
) 
 
add_library(ccv ${ccv_sources})
 
install(TARGETS ccv ARCHIVE DESTINATION lib)
 
install(FILES 
  ${sd}/3rdparty/dsfmt/dSFMT-common.h 
  ${sd}/3rdparty/dsfmt/dSFMT-params.h
  ${sd}/3rdparty/dsfmt/dSFMT-params19937.h
  ${sd}/3rdparty/dsfmt/dSFMT.h
  DESTINATION include/
  )
 
install(FILES
  ${sd}/3rdparty/kissfft/_kiss_fft_guts.h
  ${sd}/3rdparty/kissfft/_kissf_fft_guts.h
  ${sd}/3rdparty/kissfft/kiss_fft.h
  ${sd}/3rdparty/kissfft/kiss_fftnd.h
  ${sd}/3rdparty/kissfft/kiss_fftndr.h
  ${sd}/3rdparty/kissfft/kiss_fftr.h
  ${sd}/3rdparty/kissfft/kissf_fft.h
  ${sd}/3rdparty/kissfft/kissf_fftnd.h
  ${sd}/3rdparty/kissfft/kissf_fftndr.h
  ${sd}/3rdparty/kissfft/kissf_fftr.h
  DESTINATION include/
  )
 
install(FILES
  ${sd}/3rdparty/sfmt/SFMT-alti.h
  ${sd}/3rdparty/sfmt/SFMT-common.h
  ${sd}/3rdparty/sfmt/SFMT-params.h
  ${sd}/3rdparty/sfmt/SFMT-params19937.h
  ${sd}/3rdparty/sfmt/SFMT-sse2.h
  ${sd}/3rdparty/sfmt/SFMT.h
  DESTINATION include/
)
 
install(FILES
  ${sd}/3rdparty/sha1/sha1.h
   DESTINATION include/
)
 
install(FILES
  ${sd}/3rdparty/sqlite3/sqlite3.h
   DESTINATION include/
)
