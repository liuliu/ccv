include ../../config.mk

CFLAGS := -std=c++17 -O3 -Wall -I"../../" $(CFLAGS)

SRCS := Metal.cpp ccv_nnc_mfa.cpp ccv_nnc_mfa_attention.cpp ccv_nnc_mfa_error.cpp ccv_nnc_mfa_gemm.cpp ccv_nnc_mfa_normalization.cpp ccv_nnc_mfa_depalettize.cpp ccv_nnc_mfa_adam.cpp ccv_nnc_mfa_cmul.cpp ccv_nnc_mfa_gelu.cpp ccv_nnc_mfa_swish.cpp ccv_nnc_mfa_gemv.cpp ccv_nnc_mfa_cast.cpp ccv_nnc_mfa_add.cpp 3rdparty/metal-cpp/Dispatch.cpp v2/CodeWriter.cpp v2/GEMMDescriptor.cpp v2/GEMMKernelDescriptor.cpp v2/GEMMHeaders.cpp v2/GEMMKernel.cpp v2/AttentionDescriptor.cpp v2/AttentionKernelDescriptor.cpp v2/AttentionKernel.cpp v2/CMulDescriptor.cpp v2/CMulKernel.cpp v2/GeluDescriptor.cpp v2/GeluKernel.cpp v2/CastDescriptor.cpp v2/CastKernel.cpp v2/AddDescriptor.cpp v2/AddKernel.cpp ccv_nnc_mfa_segmented_gemm.cpp v2/SegmentedGEMMPrologueDescriptor.cpp v2/SegmentedGEMMPrologueKernelDescriptor.cpp v2/SegmentedGEMMPrologueKernel.cpp v2/NAMatMulDescriptor.cpp v2/NAMatMulKernelDescriptor.cpp v2/NAMatMulKernel.cpp v2/NAAttentionDescriptor.cpp v2/NAAttentionKernelDescriptor.cpp v2/NAAttentionKernel.cpp v2/GEMMKernel+Precompiled.cpp v2/AttentionKernel+Precompiled.cpp v2/SwishKernel.cpp v2/SwishDescriptor.cpp

SRC_OBJS := $(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(SRCS)))

.PHONY: release all lib clean

release: all

include ../../scheme.mk

all: lib

lib: libnnc-compat-mfa.o

clean:
	rm -f *.o

libnnc-compat-mfa.o: $(SRC_OBJS)
	ld -r $^ -o $@

%.o: %.c
	$(CC) $< -o $@ -c $(CFLAGS)

%.o: %.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

dep: .dep.mk
.dep.mk: $(SRCS)
	echo '' > .dep.mk
	for SRC in $(patsubst %.c,,$^) ; do \
		$(CC) $(CFLAGS) -M $$SRC | sed -e 's/^.*\://g' | (echo "$${SRC%%.*}.o: \\" && cat) >> .dep.mk ; \
	done

-include .dep.mk

