CC = clang # -faddress-sanitizer -fno-omit-frame-pointer
CFLAGS = -O1 -msse2 -ffast-math -mtune=native -Wall -D HAVE_LIBJPEG -D HAVE_LIBPNG -D HAVE_FFTW3 -D HAVE_CBLAS -D HAVE_GSL -D HAVE_LIBLINEAR # -fopenmp -D USE_OPENMP

all: libccv.a

clean:
	rm -f *.o 3rdparty/sha1/*.o 3rdparty/kissfft/*.o libccv.a

libccv.a: ccv_cache.o ccv_memory.o 3rdparty/sha1/sha1.o 3rdparty/kissfft/kiss_fft.o 3rdparty/kissfft/kiss_fftnd.o 3rdparty/kissfft/kiss_fftr.o 3rdparty/kissfft/kiss_fftndr.o 3rdparty/kissfft/kissf_fft.o 3rdparty/kissfft/kissf_fftnd.o 3rdparty/kissfft/kissf_fftr.o 3rdparty/kissfft/kissf_fftndr.o ccv_io.o ccv_numeric.o ccv_algebra.o ccv_util.o ccv_basic.o ccv_daisy.o ccv_sift.o ccv_bbf.o ccv_swt.o ccv_dpm.o
	ar rcs $@ $^

ccv_io.o: ccv_io.c ccv.h ccv_internal.h io/*.c
	$(CC) $(INCFLAGS) $< -o $@ -c $(CFLAGS)

%.o: %.c ccv.h ccv_internal.h
	$(CC) $(INCFLAGS) $< -o $@ -c $(CFLAGS)
