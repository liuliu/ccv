include ../../lib/config.mk

CFLAGS := -std=c++17 -O0 -g -Wall -I"../.." -I"../../lib" $(CFLAGS)
LDFLAGS := $(LDFLAGS) -framework QuartzCore

TARGETS = gemm_scaffold na_gemm_splitk_bench na_int8_attention_bench na_attention_backward_bench na_int8_matmul_bench dump_na_attention_source conv3d_layout_scaffold implicit_conv3d_scaffold gemm_kernel_introspect implicit_conv3d_bench conv3d_kernel_bench conv3d_branch_validate

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
	NAMatMulDescriptor.local.o \
	NAMatMulKernelDescriptor.local.o \
	NAMatMulKernel.local.o \
	NAInt8MatMulKernelDescriptor.local.o \
	NAInt8MatMulKernel.local.o

NA_CONV_OBJS = \
	NAConv3DDescriptor.local.o \
	NAConv3DKernelDescriptor.local.o \
	NAConv3DKernel.local.o

NA_ATTENTION_OBJS = \
	NAInt8AttentionKernelDescriptor.local.o \
	NAInt8AttentionKernel.local.o

NA_BASELINE_ATTENTION_OBJS = \
	NAAttentionKernelDescriptor.local.o \
	NAAttentionKernel.local.o

.PHONY: all clean

all: $(TARGETS)

clean:
	rm -f dump_na_source.o dump_na_attention_source.o gemm_scaffold.o na_gemm_splitk_bench.o na_int8_attention_bench.o na_attention_backward_bench.o na_int8_matmul_bench.o conv3d_layout_scaffold.o implicit_conv3d_scaffold.o gemm_kernel_introspect.o implicit_conv3d_bench.o conv3d_kernel_bench.o conv3d_branch_validate.o $(COMMON_OBJS) $(NA_GEMM_OBJS) $(NA_CONV_OBJS) $(NA_ATTENTION_OBJS) $(NA_BASELINE_ATTENTION_OBJS) $(TARGETS) dump_na_source

gemm_scaffold: gemm_scaffold.o $(COMMON_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

na_gemm_splitk_bench: na_gemm_splitk_bench.o $(COMMON_OBJS) $(NA_GEMM_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)
na_int8_attention_bench: na_int8_attention_bench.o $(COMMON_OBJS) $(NA_ATTENTION_OBJS) $(NA_BASELINE_ATTENTION_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)
na_attention_backward_bench: na_attention_backward_bench.o $(COMMON_OBJS) $(NA_BASELINE_ATTENTION_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)
na_int8_matmul_bench: na_int8_matmul_bench.o $(COMMON_OBJS) $(NA_GEMM_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

dump_na_source: dump_na_source.o $(COMMON_OBJS) NAMatMulKernelDescriptor.local.o NAMatMulKernel.local.o
	$(CC) -o $@ $^ $(LDFLAGS)

dump_na_attention_source: dump_na_attention_source.o $(COMMON_OBJS) NAAttentionKernelDescriptor.local.o NAAttentionKernel.local.o NAAttentionDescriptor.local.o
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

na_gemm_splitk_bench.o: na_gemm_splitk_bench.cpp
	$(CC) $< -o $@ -c $(CFLAGS)
na_int8_attention_bench.o: na_int8_attention_bench.cpp
	$(CC) $< -o $@ -c $(CFLAGS)
na_attention_backward_bench.o: na_attention_backward_bench.cpp
	$(CC) $< -o $@ -c $(CFLAGS)
na_int8_matmul_bench.o: na_int8_matmul_bench.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

dump_na_source.o: dump_na_source.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

dump_na_attention_source.o: dump_na_attention_source.cpp
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

NAMatMulKernelDescriptor.local.o: ../../lib/nnc/mfa/kernels/NAMatMulKernelDescriptor.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

NAMatMulKernel.local.o: ../../lib/nnc/mfa/kernels/NAMatMulKernel.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

NAInt8AttentionKernelDescriptor.local.o: ../../lib/nnc/mfa/kernels/NAInt8AttentionKernelDescriptor.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

NAInt8AttentionKernel.local.o: ../../lib/nnc/mfa/kernels/NAInt8AttentionKernel.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

NAAttentionKernelDescriptor.local.o: ../../lib/nnc/mfa/kernels/NAAttentionKernelDescriptor.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

NAAttentionKernel.local.o: ../../lib/nnc/mfa/kernels/NAAttentionKernel.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

NAAttentionDescriptor.local.o: ../../lib/nnc/mfa/kernels/NAAttentionDescriptor.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

NAInt8MatMulKernelDescriptor.local.o: ../../lib/nnc/mfa/kernels/NAInt8MatMulKernelDescriptor.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

NAInt8MatMulKernel.local.o: ../../lib/nnc/mfa/kernels/NAInt8MatMulKernel.cpp
	$(CC) $< -o $@ -c $(CFLAGS)
NAConv3DDescriptor.local.o: ../../lib/nnc/mfa/kernels/NAConv3DDescriptor.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

NAConv3DKernelDescriptor.local.o: ../../lib/nnc/mfa/kernels/NAConv3DKernelDescriptor.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

NAConv3DKernel.local.o: ../../lib/nnc/mfa/kernels/NAConv3DKernel.cpp
	$(CC) $< -o $@ -c $(CFLAGS)
