include config.mk

CFLAGS := -O3 -ffast-math -Wall -I"." $(CFLAGS)
NVFLAGS := -O3 $(NVFLAGS)

SRCS := ccv_cache.c ccv_memory.c 3rdparty/siphash/siphash24.c 3rdparty/kissfft/kiss_fft.c 3rdparty/kissfft/kiss_fftnd.c 3rdparty/kissfft/kiss_fftr.c 3rdparty/kissfft/kiss_fftndr.c 3rdparty/kissfft/kissf_fft.c 3rdparty/kissfft/kissf_fftnd.c 3rdparty/kissfft/kissf_fftr.c 3rdparty/kissfft/kissf_fftndr.c 3rdparty/dsfmt/dSFMT.c 3rdparty/sfmt/SFMT.c 3rdparty/sqlite3/sqlite3.c ccv_io.c ccv_numeric.c ccv_algebra.c ccv_util.c ccv_basic.c ccv_image_processing.c ccv_resample.c ccv_transform.c ccv_classic.c ccv_daisy.c ccv_sift.c ccv_bbf.c ccv_mser.c ccv_swt.c ccv_dpm.c ccv_tld.c ccv_ferns.c ccv_icf.c ccv_scd.c ccv_convnet.c ccv_output.c

SRC_OBJS := $(patsubst %.c,%.o,$(SRCS))

CUDA_OBJS := $(patsubst %.c,%.o,$(patsubst %.cu,%.o,$(CUDA_SRCS)))

# Make nnc/libnnc.o a phony target therefore it will be triggered every time.
.PHONY: release all lib clean dep nnc/libnnc.o

release: all

include scheme.mk

all: lib ../samples/image-net-2012-vgg-d.sqlite3

lib: libccv.a

clean:
	${MAKE} -C ./nnc clean && rm -f $(SRC_OBJS) $(CUDA_OBJS) libccv.a

libccv.a: $(SRC_OBJS) $(CUDA_OBJS) nnc/libnnc.o
	$(AR) rcs $@ $^

nnc/libnnc.o:
	${MAKE} -C ./nnc

3rdparty/sqlite3/sqlite3.o: 3rdparty/sqlite3/sqlite3.c
	$(CC) $< -o $@ -c -O3 -D SQLITE_THREADSAFE=0 -D SQLITE_OMIT_LOAD_EXTENSION

../samples/image-net-2012-vgg-d.sqlite3:
	../samples/download-vgg-d-model.sh

%.o: %.c
	$(CC) $< -o $@ -c $(CFLAGS)

%.o: %.cu
	$(NVCC) $< -o $@ -c $(NVFLAGS)

dep: .dep.mk
.dep.mk: $(SRCS) $(CUDA_SRCS)
	$(CC) $(CFLAGS) -MM $(patsubst %.cu,,$^) > .dep.mk
ifneq ($(CUDA_SRCS),)
	$(NVCC) $(NVFLAGS) -M $(patsubst %.c,,$^) >> .dep.mk
endif

-include .dep.mk
