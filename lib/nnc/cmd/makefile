include ../../config.mk
include config.mk

CMD_OBJS := $(patsubst %.c,%.o,$(patsubst %.cu,%.o,$(CMD_SRCS)))
CUDA_CMD_OBJS := $(patsubst %.c,%.o,$(patsubst %.cu,%.o,$(CUDA_CMD_SRCS)))

CFLAGS := -O3 -Wall -I"../../" -I".." $(CFLAGS)
NVFLAGS := -O3 -I"../../" -I".." $(NVFLAGS)

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
	echo '' > .dep.mk
	for CMD_SRC in $(patsubst %.cu,,$^) ; do \
		$(CC) $(CFLAGS) -MM $$CMD_SRC | sed -e 's/^.*\://g' | (echo "$${CMD_SRC%%.*}.o: \\" && cat) >> .dep.mk ; \
	done
ifneq ($(CUDA_CMD_LIB),)
	for CUDA_CMD_SRC in $(patsubst %.c,,$^) ; do \
		$(NVCC) $(NVFLAGS) -M $$CUDA_CMD_SRC | sed -e 's/^.*\://g' | (echo "$${CUDA_CMD_SRC%%.*}.o: \\" && cat) >> .dep.mk ; \
	done
endif

-include .dep.mk
