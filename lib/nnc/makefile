include ../config.mk

#CC += -g -fno-omit-frame-pointer -fsanitize=address# -fprofile-arcs -ftest-coverage
CFLAGS := -O3 -Wall -I"../" $(CFLAGS)
NVFLAGS := -O3 $(NVFLAGS)

.PHONY: all lib clean cmd libnnc.o cmd/libnnc-cmd.o gpu/libnnc-compat-cuda.o

all: lib

lib: libnnc.o

clean:
	rm -f *.o gpu/*.o cmd/*.o

libnnc.o: ccv_nnc_cmd.o ccv_nnc_tensor.o ccv_nnc_graph.o ccv_nnc_symbolic_graph.o cmd/libnnc-cmd.o $(CUDA_COMPAT_LIB)
	ld -r $^ -o $@

%.o: %.c gpu/ccv_nnc_cuda.h ../ccv.h ../ccv_internal.h ccv_nnc_tfb.h ccv_nnc.h ccv_nnc_easy.h ccv_nnc_internal.h
	$(CC) $< -o $@ -c $(CFLAGS)

%.o: %.cu gpu/ccv_nnc_cuda.h ../ccv.h ccv_nnc_tfb.h ccv_nnc.h ccv_nnc_easy.h ccv_nnc_internal.h
	$(NVCC) $< -o $@ -c $(NVFLAGS)

cmd/libnnc-cmd.o:
	${MAKE} -C ./cmd

cmd:
	${MAKE} -C ./cmd cmd

gpu/libnnc-compat-cuda.o:
	${MAKE} -C ./gpu

