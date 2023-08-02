CC := /opt/homebrew/Cellar/llvm@11/11.1.0_4/bin/clang
AR := ar
NVCC := 
CUDA_SRCS := 
CUDA_COMPAT_LIB := 
CUDA_CMD_LIB := 
MFA_COMPAT_LIB := mfa/libnnc-compat-mfa.o
MPS_COMPAT_LIB := mps/libnnc-compat-mps.o
MPS_CMD_LIB := libnnc-cmd-mps.o
DEFINE_MACROS := -D HAVE_PTHREAD -D HAVE_ACCELERATE_FRAMEWORK -D HAVE_SSE2 -D HAVE_MPS
prefix := /usr/local
exec_prefix := ${prefix}
CFLAGS :=  -msse2 $(DEFINE_MACROS) -I${prefix}/include
NVFLAGS := --use_fast_math -arch=sm_70 -std=c++14 $(DEFINE_MACROS)
LDFLAGS :=  -L${exec_prefix}/lib -lm -lpthread -framework Accelerate -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework Foundation -framework Metal -lc++
