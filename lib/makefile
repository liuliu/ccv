include config.mk

# CC += -g -fsanitize=address -fno-omit-frame-pointer # -fprofile-arcs -ftest-coverage
CFLAGS := -O3 -ffast-math -Wall $(CFLAGS)
NVFLAGS := -O3 $(NVFLAGS)

.PHONY: all clean

all: libccv.a ../samples/image-net-2012-vgg-d.sqlite3

clean:
	rm -f *.o 3rdparty/sha1/*.o 3rdparty/sfmt/*.o 3rdparty/kissfft/*.o 3rdparty/dsfmt/*.o 3rdparty/sqlite3/*.o cuda/*.o libccv.a

libccv.a: ccv_cache.o ccv_memory.o 3rdparty/sha1/sha1.o 3rdparty/kissfft/kiss_fft.o 3rdparty/kissfft/kiss_fftnd.o 3rdparty/kissfft/kiss_fftr.o 3rdparty/kissfft/kiss_fftndr.o 3rdparty/kissfft/kissf_fft.o 3rdparty/kissfft/kissf_fftnd.o 3rdparty/kissfft/kissf_fftr.o 3rdparty/kissfft/kissf_fftndr.o 3rdparty/dsfmt/dSFMT.o 3rdparty/sfmt/SFMT.o 3rdparty/sqlite3/sqlite3.o ccv_io.o ccv_numeric.o ccv_algebra.o ccv_util.o ccv_basic.o ccv_image_processing.o ccv_resample.o ccv_transform.o ccv_classic.o ccv_daisy.o ccv_sift.o ccv_bbf.o ccv_mser.o ccv_swt.o ccv_dpm.o ccv_tld.o ccv_ferns.o ccv_icf.o ccv_scd.o ccv_convnet.o ccv_output.o $(CUDA_OBJS)
	$(AR) rcs $@ $^

ccv_io.o: ccv_io.c ccv.h ccv_internal.h io/*.c
	$(CC) $< -o $@ -c $(CFLAGS)

3rdparty/sqlite3/sqlite3.o: 3rdparty/sqlite3/sqlite3.c
	$(CC) $< -o $@ -c -O3 -D SQLITE_THREADSAFE=0 -D SQLITE_OMIT_LOAD_EXTENSION

../samples/image-net-2012-vgg-d.sqlite3:
	../samples/download-vgg-d-model.sh

%.o: %.c ccv.h ccv_internal.h cuda/cwc.h cuda/cwc_helper.h
	$(CC) $< -o $@ -c $(CFLAGS)

%.o: %.cu ccv.h cuda/cwc.h cuda/cwc_internal.h cuda/cwc_helper.h
	$(NVCC) $< -o $@ -c $(NVFLAGS)
