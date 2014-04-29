import os
import ycm_core
from clang_helpers import PrepareClangFlags

def FlagsForFile(filename):
	return {
		'flags' : ['-ffast-math', '-Wall', '-msse2', '-D HAVE_SSE2', '-D HAVE_LIBJPEG', '-D HAVE_LIBPNG', '-D HAVE_GSL', '-D HAVE_FFTW3', '-D HAVE_LIBLINEAR', '-D HAVE_CBLAS', '-D HAVE_AVCODEC', '-D HAVE_AVFORMAT', '-D HAVE_SWSCALE'],
		'do_cache' : True
	}
