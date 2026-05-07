include ../../config.mk

CFLAGS := -std=c++17 -O3 -Wall -I"../../" $(CFLAGS)

SRCS := Metal.cpp ccv_nnc_mfa.cpp ccv_nnc_mfa_attention.cpp ccv_nnc_mfa_error.cpp ccv_nnc_mfa_gemm.cpp ccv_nnc_mfa_scaled_gemm.cpp ccv_nnc_mfa_ane_rowwise_gemm.cpp ccv_nnc_mfa_ane_rowwise_coreml.mm ccv_nnc_mfa_segmented_scaled_gemm.cpp ccv_nnc_mfa_conv3d.cpp ccv_nnc_mfa_normalization.cpp ccv_nnc_mfa_rmsnorm_gated.cpp ccv_nnc_mfa_depalettize.cpp ccv_nnc_mfa_dequantize_8i_rowwise.cpp ccv_nnc_mfa_adam.cpp ccv_nnc_mfa_cmul.cpp ccv_nnc_mfa_gelu.cpp ccv_nnc_mfa_sigmoid.cpp ccv_nnc_mfa_swish.cpp ccv_nnc_mfa_swish_mul.cpp ccv_nnc_mfa_exp.cpp ccv_nnc_mfa_softplus.cpp ccv_nnc_mfa_gemv.cpp ccv_nnc_mfa_cast.cpp ccv_nnc_mfa_add.cpp ccv_nnc_mfa_rotate_half.cpp ccv_nnc_mfa_gated_delta.cpp 3rdparty/metal-cpp/Dispatch.cpp kernels/CodeWriter.cpp kernels/GEMMDescriptor.cpp kernels/GEMMKernelDescriptor.cpp kernels/GEMMHeaders.cpp kernels/GEMMKernel.cpp kernels/AttentionDescriptor.cpp kernels/AttentionKernelDescriptor.cpp kernels/AttentionKernel.cpp kernels/AttentionR1Descriptor.cpp kernels/AttentionR1Kernel.cpp kernels/CMulDescriptor.cpp kernels/CMulKernel.cpp kernels/GeluDescriptor.cpp kernels/GeluKernel.cpp kernels/CastDescriptor.cpp kernels/CastKernel.cpp kernels/AddDescriptor.cpp kernels/AddKernel.cpp kernels/RotateHalfDescriptor.cpp kernels/RotateHalfKernel.cpp kernels/GatedDeltaDescriptor.cpp kernels/GatedDeltaKernel.cpp ccv_nnc_mfa_segmented_gemm.cpp kernels/SegmentedGEMMPrologueDescriptor.cpp kernels/SegmentedGEMMPrologueKernelDescriptor.cpp kernels/SegmentedGEMMPrologueKernel.cpp kernels/SegmentedScaledGEMMPrologueDescriptor.cpp kernels/SegmentedScaledGEMMPrologueKernelDescriptor.cpp kernels/SegmentedScaledGEMMPrologueKernel.cpp kernels/NAMatMulDescriptor.cpp kernels/NAMatMulKernelDescriptor.cpp kernels/NAMatMulKernel.cpp kernels/NAMatMulSmallMDescriptor.cpp kernels/NAMatMulSmallMKernelDescriptor.cpp kernels/NAMatMulSmallMKernel.cpp kernels/NAInt8MatMulDescriptor.cpp kernels/NAInt8MatMulKernelDescriptor.cpp kernels/NAInt8MatMulKernel.cpp kernels/NAInt8MatMulSmallMDescriptor.cpp kernels/NAInt8MatMulSmallMKernelDescriptor.cpp kernels/NAInt8MatMulSmallMKernel.cpp kernels/NAAttentionDescriptor.cpp kernels/NAAttentionKernelDescriptor.cpp kernels/NAAttentionKernel.cpp kernels/NAInt8AttentionDescriptor.cpp kernels/NAInt8AttentionKernelDescriptor.cpp kernels/NAInt8AttentionKernel.cpp kernels/NAConv3DDescriptor.cpp kernels/NAConv3DKernelDescriptor.cpp kernels/NAConv3DKernel.cpp kernels/Conv3DDescriptor.cpp kernels/Conv3DKernelDescriptor.cpp kernels/Conv3DKernel.cpp kernels/GEMMKernel+Precompiled.cpp kernels/AttentionKernel+Precompiled.cpp kernels/SigmoidKernel.cpp kernels/SigmoidDescriptor.cpp kernels/SwishKernel.cpp kernels/SwishDescriptor.cpp kernels/SwishMulKernel.cpp kernels/SwishMulDescriptor.cpp kernels/RMSNormGatedKernel.cpp kernels/RMSNormGatedDescriptor.cpp kernels/ExpKernel.cpp kernels/ExpDescriptor.cpp kernels/SoftplusKernel.cpp kernels/SoftplusDescriptor.cpp kernels/GemvKernel.cpp kernels/GemvDescriptor.cpp kernels/DepalettizeKernel.cpp kernels/DepalettizeDescriptor.cpp kernels/Dequantize8iRowwiseKernel.cpp kernels/Dequantize8iRowwiseDescriptor.cpp kernels/ANERowwiseTransformKernel.cpp kernels/ANERowwiseTransformDescriptor.cpp kernels/ANERowwiseTransformKernelDescriptor.cpp kernels/AdamKernel.cpp kernels/AdamDescriptor.cpp kernels/NormalizationKernel.cpp kernels/NormalizationDescriptor.cpp

SRC_OBJS := $(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(patsubst %.mm,%.o,$(SRCS))))

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

%.o: %.mm
	$(CC) $< -o $@ -c $(CFLAGS)

dep: .dep.mk
.dep.mk: $(SRCS)
	echo '' > .dep.mk
	for SRC in $(patsubst %.c,,$^) ; do \
		$(CC) $(CFLAGS) -M $$SRC | sed -e 's/^.*\://g' | (echo "$${SRC%%.*}.o: \\" && cat) >> .dep.mk ; \
	done

-include .dep.mk
