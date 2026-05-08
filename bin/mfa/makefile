include ../../lib/config.mk

CFLAGS := -std=c++17 -O0 -g -Wall -I"../.." -I"../../lib" $(CFLAGS)
LDFLAGS := $(LDFLAGS) -framework QuartzCore

TARGETS = gemm_scaffold gemv_thin_tensorop_bench scaled_gemv_bench na_gemm_splitk_bench na_int8_attention_bench na_attention_causal_bench na_attention_decode1_bench na_int8_attention_backward_probe na_attention_backward_bench attention_backward_compare_probe sdpa_backward_compare_bench rotate_half_bench gated_delta_bench na_int8_matmul_bench ane_fp16_matmul_harness ane_int8_matmul_probe ane_int8_conv_matmul_harness ane_int8_rowwise_conv_harness ane_na_int8_matmul_harness dump_attention_source dump_na_attention_source dump_na_int8_attention_source conv3d_layout_scaffold implicit_conv3d_scaffold gemm_kernel_introspect implicit_conv3d_bench conv3d_kernel_bench conv3d_branch_validate

MPSGRAPH_LDFLAGS := $(LDFLAGS) -framework Foundation -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph
ANE_LDFLAGS := $(LDFLAGS) -framework IOSurface -ldl
ANE_ROWWISE_LDFLAGS := $(LDFLAGS) -framework Foundation -framework CoreML -framework CoreVideo -framework IOSurface -framework MetalPerformanceShaders

COMMON_OBJS = \
	Metal.local.o \
	Dispatch.local.o \
	ccv_nnc_mfa_error.local.o \
	CodeWriter.local.o \
	GEMMDescriptor.local.o \
	GEMMKernelDescriptor.local.o \
	GEMMHeaders.local.o \
	GEMMKernel.local.o \
	GEMMKernel_Precompiled.local.o \
	Conv3DDescriptor.local.o \
	Conv3DKernelDescriptor.local.o \
	Conv3DKernel.local.o

NA_GEMM_OBJS = \
	NAInt8MatMulDescriptor.local.o \
	NAInt8MatMulSmallMDescriptor.local.o \
	NAMatMulDescriptor.local.o \
	NAMatMulKernelDescriptor.local.o \
	NAMatMulKernel.local.o \
	NAInt8MatMulKernelDescriptor.local.o \
	NAInt8MatMulKernel.local.o \
	NAInt8MatMulSmallMKernelDescriptor.local.o \
	NAInt8MatMulSmallMKernel.local.o

ANE_ROWWISE_OBJS = \
	ANERowwiseTransformDescriptor.local.o \
	ANERowwiseTransformKernelDescriptor.local.o \
	ANERowwiseTransformKernel.local.o \
	ccv_nnc_mfa_ane_rowwise_coreml.local.o \
	ccv_nnc_mfa_ane_rowwise_gemm.local.o

GEMV_OBJS = \
	GemvDescriptor.local.o \
	GemvKernel.local.o

INT8_GEMV_OBJS = \
	GemvDescriptor.local.o \
	GemvKernel.local.o \
	Int8GemvDescriptor.local.o \
	Int8GemvKernel.local.o \
	Dequantize8iRowwiseDescriptor.local.o \
	Dequantize8iRowwiseKernel.local.o

NA_CONV_OBJS = \
	NAConv3DDescriptor.local.o \
	NAConv3DKernelDescriptor.local.o \
	NAConv3DKernel.local.o

NA_ATTENTION_OBJS = \
	NAInt8AttentionKernelDescriptor.local.o \
	NAInt8AttentionKernel.local.o

NA_BASELINE_ATTENTION_OBJS = \
	NAAttentionKernelDescriptor.local.o \
	NAAttentionKernel.local.o \
	NAAttentionDescriptor.local.o

GENERIC_ATTENTION_OBJS = \
	AttentionDescriptor.local.o \
	AttentionKernelDescriptor.local.o \
	AttentionKernel.local.o \
	AttentionKernel_Precompiled.local.o

ROTATE_HALF_OBJS = \
	RotateHalfDescriptor.local.o \
	RotateHalfKernel.local.o

GATED_DELTA_OBJS = \
	GatedDeltaDescriptor.local.o \
	GatedDeltaKernel.local.o

.PHONY: all clean

all: $(TARGETS)

clean:
	rm -f dump_attention_source.o dump_na_source.o dump_na_attention_source.o dump_na_int8_attention_source.o gemm_scaffold.o gemv_thin_tensorop_bench.o scaled_gemv_bench.o na_gemm_splitk_bench.o na_int8_attention_bench.o na_attention_causal_bench.o na_attention_decode1_bench.o na_int8_attention_backward_probe.o na_attention_backward_bench.o attention_backward_compare_probe.o sdpa_backward_compare_bench.o rotate_half_bench.o gated_delta_bench.o na_int8_matmul_bench.o ane_fp16_matmul_harness.o ane_int8_matmul_probe.o ane_int8_conv_matmul_harness.o ane_int8_rowwise_conv_harness.o ane_na_int8_matmul_harness.o conv3d_layout_scaffold.o implicit_conv3d_scaffold.o gemm_kernel_introspect.o implicit_conv3d_bench.o conv3d_kernel_bench.o conv3d_branch_validate.o $(COMMON_OBJS) $(GEMV_OBJS) $(INT8_GEMV_OBJS) $(NA_GEMM_OBJS) $(ANE_ROWWISE_OBJS) $(NA_CONV_OBJS) $(NA_ATTENTION_OBJS) $(NA_BASELINE_ATTENTION_OBJS) $(GENERIC_ATTENTION_OBJS) $(ROTATE_HALF_OBJS) $(GATED_DELTA_OBJS) $(TARGETS) dump_na_source

gemm_scaffold: gemm_scaffold.o $(COMMON_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

gemv_thin_tensorop_bench: gemv_thin_tensorop_bench.o Metal.local.o Dispatch.local.o ccv_nnc_mfa_error.local.o $(GEMV_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

scaled_gemv_bench: scaled_gemv_bench.o Metal.local.o Dispatch.local.o ccv_nnc_mfa_error.local.o $(INT8_GEMV_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

na_gemm_splitk_bench: na_gemm_splitk_bench.o $(COMMON_OBJS) $(NA_GEMM_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)
na_int8_attention_bench: na_int8_attention_bench.o $(COMMON_OBJS) $(NA_ATTENTION_OBJS) $(NA_BASELINE_ATTENTION_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)
na_attention_causal_bench: na_attention_causal_bench.o $(COMMON_OBJS) $(NA_BASELINE_ATTENTION_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)
na_attention_decode1_bench: na_attention_decode1_bench.o Metal.local.o Dispatch.local.o ccv_nnc_mfa_error.local.o
	$(CC) -o $@ $^ $(LDFLAGS)
na_int8_attention_backward_probe: na_int8_attention_backward_probe.o $(COMMON_OBJS) $(NA_ATTENTION_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)
na_attention_backward_bench: na_attention_backward_bench.o $(COMMON_OBJS) $(NA_BASELINE_ATTENTION_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)
attention_backward_compare_probe: attention_backward_compare_probe.o $(COMMON_OBJS) $(NA_BASELINE_ATTENTION_OBJS) $(GENERIC_ATTENTION_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)
sdpa_backward_compare_bench: sdpa_backward_compare_bench.o $(COMMON_OBJS) $(NA_ATTENTION_OBJS) $(NA_BASELINE_ATTENTION_OBJS)
	$(CC) -o $@ $^ $(MPSGRAPH_LDFLAGS)
rotate_half_bench: rotate_half_bench.o $(COMMON_OBJS) $(ROTATE_HALF_OBJS)
	$(CC) -o $@ $^ $(MPSGRAPH_LDFLAGS)
gated_delta_bench: gated_delta_bench.o Metal.local.o Dispatch.local.o ccv_nnc_mfa_error.local.o $(GATED_DELTA_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)
na_int8_matmul_bench: na_int8_matmul_bench.o $(COMMON_OBJS) $(NA_GEMM_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)
ane_fp16_matmul_harness: ane_fp16_matmul_harness.o
	$(CC) -o $@ $^ $(ANE_LDFLAGS)
ane_int8_matmul_probe: ane_int8_matmul_probe.o
	$(CC) -o $@ $^ $(ANE_LDFLAGS)
ane_int8_conv_matmul_harness: ane_int8_conv_matmul_harness.o
	$(CC) -o $@ $^ $(ANE_LDFLAGS)
ane_int8_rowwise_conv_harness: ane_int8_rowwise_conv_harness.o
	$(CC) -o $@ $^ $(ANE_LDFLAGS)
ane_na_int8_matmul_harness: ane_na_int8_matmul_harness.o $(COMMON_OBJS) $(NA_GEMM_OBJS) $(ANE_ROWWISE_OBJS)
	$(CC) -o $@ $^ $(ANE_ROWWISE_LDFLAGS)

dump_na_source: dump_na_source.o $(COMMON_OBJS) NAMatMulKernelDescriptor.local.o NAMatMulKernel.local.o
	$(CC) -o $@ $^ $(LDFLAGS)

dump_attention_source: dump_attention_source.o $(COMMON_OBJS) AttentionKernelDescriptor.local.o AttentionKernel.local.o AttentionKernel_Precompiled.local.o AttentionDescriptor.local.o
	$(CC) -o $@ $^ $(LDFLAGS)

dump_na_attention_source: dump_na_attention_source.o $(COMMON_OBJS) NAAttentionKernelDescriptor.local.o NAAttentionKernel.local.o NAAttentionDescriptor.local.o
	$(CC) -o $@ $^ $(LDFLAGS)

dump_na_int8_attention_source: dump_na_int8_attention_source.o $(COMMON_OBJS) NAInt8AttentionKernelDescriptor.local.o NAInt8AttentionKernel.local.o NAInt8AttentionDescriptor.local.o
	$(CC) -o $@ $^ $(LDFLAGS)

conv3d_layout_scaffold: conv3d_layout_scaffold.o $(COMMON_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

implicit_conv3d_scaffold: implicit_conv3d_scaffold.o $(COMMON_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

gemm_kernel_introspect: gemm_kernel_introspect.o $(COMMON_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

implicit_conv3d_bench: implicit_conv3d_bench.o $(COMMON_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

conv3d_kernel_bench: conv3d_kernel_bench.o $(COMMON_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

conv3d_branch_validate: conv3d_branch_validate.o $(COMMON_OBJS) $(NA_CONV_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

gemm_scaffold.o: gemm_scaffold.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

gemv_thin_tensorop_bench.o: gemv_thin_tensorop_bench.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

scaled_gemv_bench.o: scaled_gemv_bench.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

na_gemm_splitk_bench.o: na_gemm_splitk_bench.cpp
	$(CC) $< -o $@ -c $(CFLAGS)
na_int8_attention_bench.o: na_int8_attention_bench.cpp
	$(CC) $< -o $@ -c $(CFLAGS)
na_attention_causal_bench.o: na_attention_causal_bench.cpp
	$(CC) $< -o $@ -c $(CFLAGS)
na_attention_decode1_bench.o: na_attention_decode1_bench.cpp
	$(CC) $< -o $@ -c $(CFLAGS)
na_int8_attention_backward_probe.o: na_int8_attention_backward_probe.cpp
	$(CC) $< -o $@ -c $(subst -std=c++17,-std=gnu++17,$(CFLAGS))
na_attention_backward_bench.o: na_attention_backward_bench.cpp
	$(CC) $< -o $@ -c $(CFLAGS)
attention_backward_compare_probe.o: attention_backward_compare_probe.cpp
	$(CC) $< -o $@ -c $(CFLAGS)
sdpa_backward_compare_bench.o: sdpa_backward_compare_bench.mm
	$(CC) $< -o $@ -c $(subst -std=c++17,-std=gnu++17,$(CFLAGS))
rotate_half_bench.o: rotate_half_bench.mm
	$(CC) $< -o $@ -c $(subst -std=c++17,-std=gnu++17,$(CFLAGS))
gated_delta_bench.o: gated_delta_bench.cpp
	$(CC) $< -o $@ -c $(CFLAGS)
na_int8_matmul_bench.o: na_int8_matmul_bench.cpp
	$(CC) $< -o $@ -c $(CFLAGS)
ane_fp16_matmul_harness.o: ane_fp16_matmul_harness.mm
	$(CC) $< -o $@ -c $(subst -std=c++17,-std=gnu++17,$(CFLAGS)) -fobjc-arc
ane_int8_matmul_probe.o: ane_int8_matmul_probe.mm
	$(CC) $< -o $@ -c $(subst -std=c++17,-std=gnu++17,$(CFLAGS)) -fobjc-arc
ane_int8_conv_matmul_harness.o: ane_int8_conv_matmul_harness.mm
	$(CC) $< -o $@ -c $(subst -std=c++17,-std=gnu++17,$(CFLAGS)) -fobjc-arc
ane_int8_rowwise_conv_harness.o: ane_int8_rowwise_conv_harness.mm
	$(CC) $< -o $@ -c $(subst -std=c++17,-std=gnu++17,$(CFLAGS)) -fobjc-arc
ane_na_int8_matmul_harness.o: ane_na_int8_matmul_harness.mm
	$(CC) $< -o $@ -c $(subst -std=c++17,-std=gnu++17,$(CFLAGS)) -fobjc-arc

dump_na_source.o: dump_na_source.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

dump_attention_source.o: dump_attention_source.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

dump_na_attention_source.o: dump_na_attention_source.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

dump_na_int8_attention_source.o: dump_na_int8_attention_source.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

conv3d_layout_scaffold.o: conv3d_layout_scaffold.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

implicit_conv3d_scaffold.o: implicit_conv3d_scaffold.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

gemm_kernel_introspect.o: gemm_kernel_introspect.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

implicit_conv3d_bench.o: implicit_conv3d_bench.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

conv3d_kernel_bench.o: conv3d_kernel_bench.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

conv3d_branch_validate.o: conv3d_branch_validate.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

Metal.local.o: ../../lib/nnc/mfa/Metal.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

Dispatch.local.o: ../../lib/nnc/mfa/3rdparty/metal-cpp/Dispatch.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

ccv_nnc_mfa_error.local.o: ../../lib/nnc/mfa/ccv_nnc_mfa_error.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

GemvDescriptor.local.o: ../../lib/nnc/mfa/kernels/GemvDescriptor.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

GemvKernel.local.o: ../../lib/nnc/mfa/kernels/GemvKernel.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

Int8GemvDescriptor.local.o: ../../lib/nnc/mfa/kernels/Int8GemvDescriptor.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

Int8GemvKernel.local.o: ../../lib/nnc/mfa/kernels/Int8GemvKernel.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

Dequantize8iRowwiseDescriptor.local.o: ../../lib/nnc/mfa/kernels/Dequantize8iRowwiseDescriptor.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

Dequantize8iRowwiseKernel.local.o: ../../lib/nnc/mfa/kernels/Dequantize8iRowwiseKernel.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

CodeWriter.local.o: ../../lib/nnc/mfa/kernels/CodeWriter.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

GEMMDescriptor.local.o: ../../lib/nnc/mfa/kernels/GEMMDescriptor.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

GEMMKernelDescriptor.local.o: ../../lib/nnc/mfa/kernels/GEMMKernelDescriptor.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

GEMMHeaders.local.o: ../../lib/nnc/mfa/kernels/GEMMHeaders.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

GEMMKernel.local.o: ../../lib/nnc/mfa/kernels/GEMMKernel.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

GEMMKernel_Precompiled.local.o: ../../lib/nnc/mfa/kernels/GEMMKernel+Precompiled.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

Conv3DDescriptor.local.o: ../../lib/nnc/mfa/kernels/Conv3DDescriptor.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

Conv3DKernelDescriptor.local.o: ../../lib/nnc/mfa/kernels/Conv3DKernelDescriptor.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

Conv3DKernel.local.o: ../../lib/nnc/mfa/kernels/Conv3DKernel.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

NAMatMulDescriptor.local.o: ../../lib/nnc/mfa/kernels/NAMatMulDescriptor.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

NAInt8MatMulDescriptor.local.o: ../../lib/nnc/mfa/kernels/NAInt8MatMulDescriptor.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

NAInt8MatMulSmallMDescriptor.local.o: ../../lib/nnc/mfa/kernels/NAInt8MatMulSmallMDescriptor.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

NAMatMulKernelDescriptor.local.o: ../../lib/nnc/mfa/kernels/NAMatMulKernelDescriptor.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

NAMatMulKernel.local.o: ../../lib/nnc/mfa/kernels/NAMatMulKernel.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

NAInt8AttentionKernelDescriptor.local.o: ../../lib/nnc/mfa/kernels/NAInt8AttentionKernelDescriptor.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

NAInt8AttentionKernel.local.o: ../../lib/nnc/mfa/kernels/NAInt8AttentionKernel.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

NAInt8AttentionDescriptor.local.o: ../../lib/nnc/mfa/kernels/NAInt8AttentionDescriptor.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

NAAttentionKernelDescriptor.local.o: ../../lib/nnc/mfa/kernels/NAAttentionKernelDescriptor.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

NAAttentionKernel.local.o: ../../lib/nnc/mfa/kernels/NAAttentionKernel.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

NAAttentionDescriptor.local.o: ../../lib/nnc/mfa/kernels/NAAttentionDescriptor.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

AttentionDescriptor.local.o: ../../lib/nnc/mfa/kernels/AttentionDescriptor.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

AttentionKernelDescriptor.local.o: ../../lib/nnc/mfa/kernels/AttentionKernelDescriptor.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

AttentionKernel.local.o: ../../lib/nnc/mfa/kernels/AttentionKernel.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

AttentionKernel_Precompiled.local.o: ../../lib/nnc/mfa/kernels/AttentionKernel+Precompiled.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

RotateHalfDescriptor.local.o: ../../lib/nnc/mfa/kernels/RotateHalfDescriptor.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

RotateHalfKernel.local.o: ../../lib/nnc/mfa/kernels/RotateHalfKernel.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

GatedDeltaDescriptor.local.o: ../../lib/nnc/mfa/kernels/GatedDeltaDescriptor.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

GatedDeltaKernel.local.o: ../../lib/nnc/mfa/kernels/GatedDeltaKernel.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

NAInt8MatMulKernelDescriptor.local.o: ../../lib/nnc/mfa/kernels/NAInt8MatMulKernelDescriptor.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

NAInt8MatMulKernel.local.o: ../../lib/nnc/mfa/kernels/NAInt8MatMulKernel.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

NAInt8MatMulSmallMKernelDescriptor.local.o: ../../lib/nnc/mfa/kernels/NAInt8MatMulSmallMKernelDescriptor.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

NAInt8MatMulSmallMKernel.local.o: ../../lib/nnc/mfa/kernels/NAInt8MatMulSmallMKernel.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

ANERowwiseTransformDescriptor.local.o: ../../lib/nnc/mfa/kernels/ANERowwiseTransformDescriptor.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

ANERowwiseTransformKernelDescriptor.local.o: ../../lib/nnc/mfa/kernels/ANERowwiseTransformKernelDescriptor.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

ANERowwiseTransformKernel.local.o: ../../lib/nnc/mfa/kernels/ANERowwiseTransformKernel.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

ccv_nnc_mfa_ane_rowwise_coreml.local.o: ../../lib/nnc/mfa/ccv_nnc_mfa_ane_rowwise_coreml.mm
	$(CC) $< -o $@ -c $(subst -std=c++17,-std=gnu++17,$(CFLAGS)) -fno-objc-arc

ccv_nnc_mfa_ane_rowwise_gemm.local.o: ../../lib/nnc/mfa/ccv_nnc_mfa_ane_rowwise_gemm.cpp
	$(CC) $< -o $@ -c $(CFLAGS)
NAConv3DDescriptor.local.o: ../../lib/nnc/mfa/kernels/NAConv3DDescriptor.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

NAConv3DKernelDescriptor.local.o: ../../lib/nnc/mfa/kernels/NAConv3DKernelDescriptor.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

NAConv3DKernel.local.o: ../../lib/nnc/mfa/kernels/NAConv3DKernel.cpp
	$(CC) $< -o $@ -c $(CFLAGS)
