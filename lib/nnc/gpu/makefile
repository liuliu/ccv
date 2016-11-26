include ../../config.mk

#CC += -g -fno-omit-frame-pointer -fsanitize=address# -fprofile-arcs -ftest-coverage
CFLAGS := -O3 -Wall -I"../../" $(CFLAGS)
NVFLAGS := -O3 -I"../../" $(NVFLAGS)

.PHONY: all lib clean libnnc-compat-cuda.o

all: lib

lib: libnnc-compat-cuda.o

clean:
	rm -f *.o

libnnc-compat-cuda.o: ccv_nnc_compat.o
	ld -r $^ -o $@

%.o: %.c ccv_nnc_compat.h ../../ccv.h ../../ccv_internal.h ../ccv_nnc_tfb.h ../ccv_nnc.h ../ccv_nnc_easy.h ../ccv_nnc_internal.h
	$(CC) $< -o $@ -c $(CFLAGS)

%.o: %.cu ccv_nnc_compat.h ../../ccv.h ../ccv_nnc_tfb.h ../ccv_nnc.h ../ccv_nnc_easy.h ../ccv_nnc_internal.h
	$(NVCC) $< -o $@ -c $(NVFLAGS)
