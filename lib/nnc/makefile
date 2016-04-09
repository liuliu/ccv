include ../config.mk

#CC += -g -fno-omit-frame-pointer -fsanitize=address# -fprofile-arcs -ftest-coverage
CFLAGS := -O3 -ffast-math -Wall -I"../" -I"../" $(CFLAGS)
NVFLAGS := -O3 $(NVFLAGS)

.PHONY: all lib clean

all: lib

lib: libnnc.a

clean:
	rm -f *.o cpu/*.o gpu/*.o libnnc.a

libnnc.a: ccv_nnc_cmd.o ccv_nnc_tensor.o ccv_nnc_graph.o cpu/ccv_nnc_cmd_ref.o cpu/ccv_nnc_cmd_opt.o
	$(AR) rcs $@ $^

%.o: %.c ../ccv.h ../ccv_internal.h ccv_nnc_tfb.h ccv_nnc.h ccv_nnc_internal.h
	$(CC) $< -o $@ -c $(CFLAGS)

%.o: %.cu ../ccv.h ccv_nnc_tfb.h ccv_nnc.h ccv_nnc_internal.h
	$(NVCC) $< -o $@ -c $(NVFLAGS)
