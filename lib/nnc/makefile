include ../config.mk

CFLAGS := -O3 -Wall -I"../" $(CFLAGS)
NVFLAGS := -O3 $(NVFLAGS)

SRCS := ccv_nnc_cmd.c ccv_nnc_tensor.c ccv_nnc_tensor_io.c ccv_nnc_stream.c ccv_nnc_graph.c ccv_nnc_symbolic_graph.c ccv_nnc_symbolic_graph_io.c ccv_nnc_symbolic_graph_compile.c ccv_nnc_symbolic_graph_backward.c ccv_nnc_symbolic_graph_while.c ccv_nnc_graph_while.c ccv_nnc_tensor_tape.c ccv_nnc_symbolic_graph_case_of.c ccv_nnc_graph_case_of.c ccv_nnc_symbolic_graph_minimize.c ccv_nnc_symbolic_graph_parallel.c ccv_nnc_symbolic_graph_simplify.c ccv_nnc_symbolic_graph_memory_compression.c ccv_nnc_graph_run.c ccv_nnc_dynamic_graph.c ccv_nnc_dynamic_graph_backward.c ccv_nnc_dynamic_graph_apply_gradients.c ccv_nnc_dynamic_graph_minimize.c ccv_nnc_dynamic_graph_evaluate.c ccv_cnnp_dataframe.c ccv_cnnp_dataframe_core.c ccv_cnnp_dataframe_addons.c ccv_cnnp_model.c ccv_cnnp_model_io.c ccv_cnnp_model_core.c

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
	echo '' > .dep.mk
	for SRC in $^ ; do \
		$(CC) $(CFLAGS) -MM $$SRC | sed -e 's/^.*\://g' | (echo "$${SRC%%.*}.o: \\" && cat) >> .dep.mk ; \
	done

-include .dep.mk
