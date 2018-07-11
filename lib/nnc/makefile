include ../config.mk

CFLAGS := -O3 -Wall -I"../" $(CFLAGS)
NVFLAGS := -O3 $(NVFLAGS)

SRCS := ccv_nnc_cmd.c ccv_nnc_tensor.c ccv_nnc_graph.c ccv_nnc_symbolic_graph.c ccv_nnc_symbolic_graph_io.c ccv_nnc_symbolic_graph_compile.c ccv_nnc_symbolic_graph_backward.c ccv_nnc_symbolic_graph_while.c ccv_nnc_graph_while.c ccv_nnc_tensor_tape.c ccv_nnc_symbolic_graph_case_of.c ccv_nnc_graph_case_of.c ccv_nnc_symbolic_graph_minimize.c ccv_nnc_symbolic_graph_simplify.c ccv_nnc_graph_run.c ccv_nnc_dynamic_graph.c ccv_nnc_dynamic_graph_backward.c ccv_nnc_dynamic_graph_minimize.c ccv_cnnp_dataframe.c ccv_cnnp_model.c ccv_cnnp_model_core.c

SRC_OBJS := $(patsubst %.c,%.o,$(SRCS))

.PHONY: release all lib clean cmd libnnc.o cmd/libnnc-cmd.o gpu/libnnc-compat-cuda.o

release: all

include ../scheme.mk

all: lib

lib: libnnc.o

clean:
	rm -f *.o gpu/*.o && ${MAKE} -C ./gpu clean && ${MAKE} -C ./cmd clean

libnnc.o: $(SRC_OBJS) cmd/libnnc-cmd.o $(CUDA_COMPAT_LIB)
	ld -r $^ -o $@

%.o: %.c
	$(CC) $< -o $@ -c $(CFLAGS)

%.o: %.cu
	$(NVCC) $< -o $@ -c $(NVFLAGS)

cmd/libnnc-cmd.o:
	${MAKE} -C ./cmd

cmd:
	${MAKE} -C ./cmd cmd

gpu/libnnc-compat-cuda.o:
	${MAKE} -C ./gpu

dep: .dep.mk
.dep.mk: $(SRCS)
	$(CC) $(CFLAGS) -MM $^ > .dep.mk

-include .dep.mk
