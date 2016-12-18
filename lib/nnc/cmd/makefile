include ../../config.mk
include config.mk

CMD_OBJS := $(patsubst %.c,%.o,$(patsubst %.cu,%.o,$(CMD_SRCS)))
CUDA_CMD_OBJS := $(patsubst %.c,%.o,$(patsubst %.cu,%.o,$(CUDA_CMD_SRCS)))

CFLAGS := -O3 -Wall -I"../../" $(CFLAGS)
NVFLAGS := -O3 -I"../../" $(NVFLAGS)

.PHONY: release all lib clean cmd libnnc-cmd.o $(CUDA_CMD_LIB)

release: all

include ../../scheme.mk

all: lib

lib: libnnc-cmd.o

clean:
	find . -type f -name '*.o' -exec rm {} +

cmd:
	./build-cmd.rb .

libnnc-cmd.o: $(CMD_OBJS) $(CUDA_CMD_LIB)
	ld -r $^ -o $@

libnnc-cmd-cuda.o: $(CUDA_CMD_OBJS)
	ld -r $^ -o $@

%.o: %.c
	$(CC) $< -o $@ -c $(CFLAGS)

%.o: %.cu
	$(NVCC) $< -o $@ -c $(NVFLAGS)

dep: .dep.mk
.dep.mk: $(CMD_SRCS) $(CUDA_CMD_SRCS)
	$(CC) $(CFLAGS) -MM $(patsubst %.cu,,$^) > .dep.mk
ifneq ($(CUDA_CMD_SRCS),)
	$(NVCC) $(NVFLAGS) -M $(patsubst %.c,,$^) >> .dep.mk
endif

-include .dep.mk
