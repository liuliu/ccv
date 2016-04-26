include ../../config.mk

#CC += -g -fno-omit-frame-pointer -fsanitize=address# -fprofile-arcs -ftest-coverage
CFLAGS := -O3 -ffast-math -Wall -I"../../" $(CFLAGS)
NVFLAGS := -O3 -I"../../" $(NVFLAGS)

.PHONY: all lib clean libnnc-cuda.o

all: lib

lib: libnnc-cuda.o

clean:
	rm -f *.o

libnnc-cuda.o: ccv_nnc_compat.o ccv_nnc_cmd_ref.o ccv_nnc_cmd_cudnn.o
	ld -r $^ -o $@

%.o: %.c ccv_nnc_compat.h ../../ccv.h ../../ccv_internal.h ../ccv_nnc_tfb.h ../ccv_nnc.h ../ccv_nnc_easy.h ../ccv_nnc_internal.h ../ccv_nnc_init.inc
	$(CC) $< -o $@ -c $(CFLAGS)

%.o: %.cu ccv_nnc_compat.h ../../ccv.h ../ccv_nnc_tfb.h ../ccv_nnc.h ../ccv_nnc_easy.h ../ccv_nnc_internal.h ../ccv_nnc_init.inc
	$(NVCC) $< -o $@ -c $(NVFLAGS)
