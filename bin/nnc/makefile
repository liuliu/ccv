include ../../lib/config.mk

LDFLAGS := -L"../../lib" -lccv $(LDFLAGS)
CFLAGS := -O3 -Wall -I"../../lib" $(CFLAGS)
NVFLAGS := -O3 -I"../../lib" -lineinfo $(NVFLAGS)

TARGETS = nnc-e2e-verify nnc-e2e-sym-verify nnc-sym cifar-10 imagenet coco imdb iwslt wmt csv imdb_lstm sdpa_bench sdpa_grad_bench routed_segmented_gemm_bench moe_routing_bench mps_rowwise_gemm_bench na_int8_dynamic_quant_bench ane_rowwise_quant_transpose_bench # attention_kernel_gen gemm_kernel_gen square_attention_test laplacian_test adversarial_shape_test

FUZZ_TARGETS = csv_fuzz

TARGET_SRCS := $(patsubst %,%.c,$(TARGETS))

FUZZ_TARGET_SRCS := $(patsubst %,%.c,$(FUZZ_TARGETS))

.PHONY: release all clean dep

release: all

include ../../lib/scheme.mk

all: $(TARGETS)

fuzz: $(FUZZ_TARGETS)

clean:
	${MAKE} clean -C ../../lib ; rm -f *.o $(TARGETS) $(FUZZ_TARGETS)

$(TARGETS): %: %.o libccv.a
	$(CC) -o $@ $< $(LDFLAGS)

$(FUZZ_TARGETS): %: %.o libccv.a
	$(CC) -o $@ $< $(LDFLAGS)

libccv.a:
	${MAKE} -C ../../lib

%.o: %.c
	$(CC) $< -o $@ -c $(CFLAGS)

laplacian_test.o: laplacian_test.cpp
	$(CC) $< -o $@ -c $(CFLAGS) -std=c++17

adversarial_shape_test.o: adversarial_shape_test.cpp
	$(CC) $< -o $@ -c $(CFLAGS) -std=c++17

square_attention_test.o: square_attention_test.cpp
	$(CC) $< -o $@ -c $(CFLAGS) -std=c++17

gemm_kernel_gen.o: gemm_kernel_gen.cpp
	$(CC) $< -o $@ -c $(CFLAGS) -std=c++17

attention_kernel_gen.o: attention_kernel_gen.cpp
	$(CC) $< -o $@ -c $(CFLAGS) -std=c++17

na_int8_dynamic_quant_bench.o: na_int8_dynamic_quant_bench.cpp
	$(CC) $< -o $@ -c $(CFLAGS) -std=c++17

ane_rowwise_quant_transpose_bench.o: ane_rowwise_quant_transpose_bench.cpp
	$(CC) $< -o $@ -c $(CFLAGS) -std=c++17

.gitignore:
	echo $(TARGETS) | tr ' ' '\n' > .gitignore

dep: .dep.mk
.dep.mk: $(TARGET_SRCS)
	$(CC) $(CFLAGS) -MM $^ > .dep.mk

-include .dep.mk
