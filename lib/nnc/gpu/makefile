include ../../config.mk

CFLAGS := -O3 -Wall -I"../../" $(CFLAGS)
NVFLAGS := -O3 -I"../../" $(NVFLAGS) -I"3rdparty/cutlass/include"

FLASH_ATTN_SRCS := 3rdparty/flash_attn/flash_api.cu 3rdparty/flash_attn/src/flash_bwd_hdim128_bf16_sm80.cu 3rdparty/flash_attn/src/flash_bwd_hdim128_fp16_sm80.cu 3rdparty/flash_attn/src/flash_bwd_hdim160_bf16_sm80.cu 3rdparty/flash_attn/src/flash_bwd_hdim160_fp16_sm80.cu 3rdparty/flash_attn/src/flash_bwd_hdim192_bf16_sm80.cu 3rdparty/flash_attn/src/flash_bwd_hdim192_fp16_sm80.cu 3rdparty/flash_attn/src/flash_bwd_hdim224_bf16_sm80.cu 3rdparty/flash_attn/src/flash_bwd_hdim224_fp16_sm80.cu 3rdparty/flash_attn/src/flash_bwd_hdim256_bf16_sm80.cu 3rdparty/flash_attn/src/flash_bwd_hdim256_fp16_sm80.cu 3rdparty/flash_attn/src/flash_bwd_hdim32_bf16_sm80.cu 3rdparty/flash_attn/src/flash_bwd_hdim32_fp16_sm80.cu 3rdparty/flash_attn/src/flash_bwd_hdim64_bf16_sm80.cu 3rdparty/flash_attn/src/flash_bwd_hdim64_fp16_sm80.cu 3rdparty/flash_attn/src/flash_bwd_hdim96_bf16_sm80.cu 3rdparty/flash_attn/src/flash_bwd_hdim96_fp16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_hdim128_bf16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_hdim128_fp16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_hdim160_bf16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_hdim160_fp16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_hdim192_bf16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_hdim192_fp16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_hdim224_bf16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_hdim224_fp16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_hdim256_bf16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_hdim256_fp16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_hdim32_bf16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_hdim32_fp16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_hdim64_bf16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_hdim64_fp16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_hdim96_bf16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_hdim96_fp16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_split_hdim128_bf16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_split_hdim128_fp16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_split_hdim160_bf16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_split_hdim160_fp16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_split_hdim192_bf16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_split_hdim192_fp16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_split_hdim224_bf16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_split_hdim224_fp16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_split_hdim256_bf16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_split_hdim256_fp16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_split_hdim32_bf16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_split_hdim32_fp16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_split_hdim64_bf16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_split_hdim64_fp16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_split_hdim96_bf16_sm80.cu 3rdparty/flash_attn/src/flash_fwd_split_hdim96_fp16_sm80.cu

SRCS := ccv_nnc_compat.cu ccv_nnc_palettize.cu

SRC_OBJS := $(patsubst %.c,%.o,$(patsubst %.cu,%.o,$(SRCS)))

FLASH_ATTN_SRC_OBJS := $(patsubst %.c,%.o,$(patsubst %.cu,%.o,$(FLASH_ATTN_SRCS)))

.PHONY: release all lib clean libnnc-compat-cuda.o libnnc-compat-cuda-sm80.o

release: all

include ../../scheme.mk

all: lib

lib: libnnc-compat-cuda.o

clean:
	rm -f *.o

libnnc-compat-cuda.o: $(SRC_OBJS) $(CUDA_COMPAT_SM80_LIB)
	ld -r $^ -o $@

libnnc-compat-cuda-sm80.o: $(FLASH_ATTN_SRC_OBJS)
	ld -r $^ -o $@

%.o: %.c
	$(CC) $< -o $@ -c $(CFLAGS)

%.o: %.cu
	$(NVCC) $< -o $@ -c $(NVFLAGS)

dep: .dep.mk
.dep.mk: $(SRCS)
	echo '' > .dep.mk
	for SRC in $(patsubst %.c,,$^) ; do \
		$(NVCC) $(NVFLAGS) -M $$SRC | sed -e 's/^.*\://g' | (echo "$${SRC%%.*}.o: \\" && cat) >> .dep.mk ; \
	done

-include .dep.mk
