include config.mk

# CC += -g -fno-omit-frame-pointer -fsanitize=address# -fprofile-arcs -ftest-coverage
CFLAGS := -O3 -ffast-math -Wall -I"." $(CFLAGS)
NVFLAGS := -O3 $(NVFLAGS)

NNC_OBJS := nnc/ccv_nnc_cmd.o nnc/ccv_nnc_tensor.o nnc/ccv_nnc_graph.o nnc/cpu/ccv_nnc_cmd_ref.o nnc/cpu/ccv_nnc_cmd_opt.o

.PHONY: all lib clean

all: lib ../samples/image-net-2012-vgg-d.sqlite3

lib: libccv.a

clean:
	rm -f *.o 3rdparty/siphash/*.o 3rdparty/sfmt/*.o 3rdparty/kissfft/*.o 3rdparty/dsfmt/*.o 3rdparty/sqlite3/*.o cuda/*.o cuda/convnet/*.o $(NNC_OBJS) libccv.a

libccv.a: ccv_cache.o ccv_memory.o 3rdparty/siphash/siphash24.o 3rdparty/kissfft/kiss_fft.o 3rdparty/kissfft/kiss_fftnd.o 3rdparty/kissfft/kiss_fftr.o 3rdparty/kissfft/kiss_fftndr.o 3rdparty/kissfft/kissf_fft.o 3rdparty/kissfft/kissf_fftnd.o 3rdparty/kissfft/kissf_fftr.o 3rdparty/kissfft/kissf_fftndr.o 3rdparty/dsfmt/dSFMT.o 3rdparty/sfmt/SFMT.o 3rdparty/sqlite3/sqlite3.o ccv_io.o ccv_numeric.o ccv_algebra.o ccv_util.o ccv_basic.o ccv_image_processing.o ccv_resample.o ccv_transform.o ccv_classic.o ccv_daisy.o ccv_sift.o ccv_bbf.o ccv_mser.o ccv_swt.o ccv_dpm.o ccv_tld.o ccv_ferns.o ccv_icf.o ccv_scd.o ccv_convnet.o ccv_output.o $(NNC_OBJS) $(CUDA_OBJS)
	$(AR) rcs $@ $^

ccv_io.o: ccv_io.c ccv.h ccv_internal.h io/*.inc
	$(CC) $< -o $@ -c $(CFLAGS)

3rdparty/sqlite3/sqlite3.o: 3rdparty/sqlite3/sqlite3.c
	$(CC) $< -o $@ -c -O3 -D SQLITE_THREADSAFE=0 -D SQLITE_OMIT_LOAD_EXTENSION

../samples/image-net-2012-vgg-d.sqlite3:
	../samples/download-vgg-d-model.sh

%.o: %.c ccv.h ccv_internal.h cuda/cwc.h cuda/cwc_ext.h nnc/ccv_nnc_tfb.h nnc/ccv_nnc.h nnc/ccv_nnc_internal.h
	$(CC) $< -o $@ -c $(CFLAGS)

%.o: %.cu ccv.h cuda/cwc.h cuda/cwc_internal.h cuda/cwc_ext.h nnc/ccv_nnc_tfb.h nnc/ccv_nnc.h nnc/ccv_nnc_internal.h
	$(NVCC) $< -o $@ -c $(NVFLAGS)
