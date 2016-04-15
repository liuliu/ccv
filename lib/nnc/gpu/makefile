include ../../config.mk

#CC += -g -fno-omit-frame-pointer -fsanitize=address# -fprofile-arcs -ftest-coverage
CFLAGS := -O3 -ffast-math -Wall -I"../../" $(CFLAGS)
NVFLAGS := -O3 $(NVFLAGS)

.PHONY: all lib clean

all: lib

lib: libnnc-cuda.o

clean:
	rm -f *.o

libnnc-cuda.o: ccv_nnc_cmd_ref.o
	ld -r $^ -o $@

%.o: %.c ../../ccv.h ../../ccv_internal.h ../ccv_nnc_tfb.h ../ccv_nnc.h ../ccv_nnc_easy.h ../ccv_nnc_internal.h ../ccv_nnc_init.inc
	$(CC) $< -o $@ -c $(CFLAGS)

%.o: %.cu ../../ccv.h ../ccv_nnc_tfb.h ../ccv_nnc.h ../ccv_nnc_easy.h ../ccv_nnc_internal.h ../ccv_nnc_init.inc
	$(NVCC) $< -o $@ -c $(NVFLAGS)
