include ../../lib/config.mk

CFLAGS := -std=c++17 -O0 -g -Wall -I"../.." -I"../../lib" $(CFLAGS)
LDFLAGS := $(LDFLAGS) -framework QuartzCore

TARGETS = gemm_scaffold conv3d_layout_scaffold implicit_conv3d_scaffold gemm_kernel_introspect implicit_conv3d_bench conv3d_kernel_bench

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

.PHONY: all clean

all: $(TARGETS)

clean:
	rm -f gemm_scaffold.o conv3d_layout_scaffold.o implicit_conv3d_scaffold.o gemm_kernel_introspect.o implicit_conv3d_bench.o conv3d_kernel_bench.o $(COMMON_OBJS) $(TARGETS)

gemm_scaffold: gemm_scaffold.o $(COMMON_OBJS)
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

gemm_scaffold.o: gemm_scaffold.cpp
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
