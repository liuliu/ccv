include ../../config.mk
include config.mk

#CC += -g -fno-omit-frame-pointer -fsanitize=address# -fprofile-arcs -ftest-coverage
CFLAGS := -O3 -Wall -I"../../" $(CFLAGS)
NVFLAGS := -O3 -I"../../" $(NVFLAGS)

.PHONY: all lib clean libnnc-cmd.o $(CUDA_CMD_LIB)

all: lib

lib: libnnc-cmd.o

clean:
	rm -f *.o

libnnc-cmd.o: $(CMD_OBJS) $(CUDA_CMD_LIB)
	ld -r $^ -o $@

libnnc-cmd-cuda.o: $(CUDA_CMD_OBJS)
	ld -r $^ -o $@

%.o: %.c ../../ccv.h ../../ccv_internal.h ../ccv_nnc_tfb.h ../ccv_nnc.h ../ccv_nnc_easy.h ../ccv_nnc_internal.h
	$(CC) $< -o $@ -c $(CFLAGS)

%.o: %.cu ../../ccv.h ../ccv_nnc_tfb.h ../ccv_nnc.h ../ccv_nnc_easy.h ../ccv_nnc_internal.h
	$(NVCC) $< -o $@ -c $(NVFLAGS)
