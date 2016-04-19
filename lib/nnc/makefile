include ../config.mk

#CC += -g -fno-omit-frame-pointer -fsanitize=address# -fprofile-arcs -ftest-coverage
CFLAGS := -O3 -ffast-math -Wall -I"../" $(CFLAGS)
NVFLAGS := -O3 $(NVFLAGS)

.PHONY: all lib clean libnnc.o gpu/libnnc-cuda.o nnc-init

all: lib

lib: libnnc.o

clean:
	rm -f *.o cpu/*.o gpu/*.o cpu/conv/*.o cpu/fc/*.o

libnnc.o: ccv_nnc_cmd.o ccv_nnc_tensor.o ccv_nnc_graph.o cpu/ccv_nnc_cmd_ref.o cpu/ccv_nnc_cmd_opt.o cpu/conv/ccv_nnc_cmd_conv_opt.o cpu/conv/ccv_nnc_cmd_conv_4x4_3x3_winograd.o cpu/conv/ccv_nnc_cmd_conv_fft.o cpu/conv/ccv_nnc_cmd_conv_gemm.o cpu/fc/ccv_nnc_cmd_fc_gemm.o cpu/fc/ccv_nnc_cmd_fc_opt.o $(CUDA_TARGETS)
	ld -r $^ -o $@

%.o: %.c gpu/ccv_nnc_cuda.h ../ccv.h ../ccv_internal.h ccv_nnc_tfb.h ccv_nnc.h ccv_nnc_easy.h ccv_nnc_internal.h ccv_nnc_init.inc
	$(CC) $< -o $@ -c $(CFLAGS)

%.o: %.cu gpu/ccv_nnc_cuda.h ../ccv.h ccv_nnc_tfb.h ccv_nnc.h ccv_nnc_easy.h ccv_nnc_internal.h ccv_nnc_init.inc
	$(NVCC) $< -o $@ -c $(NVFLAGS)

gpu/libnnc-cuda.o:
	${MAKE} -C ./gpu

nnc-init:
	./nnc-init.rb cpu
	mv ccv_nnc_init.inc cpu/ccv_nnc_init.inc
	./nnc-init.rb cpu gpu
