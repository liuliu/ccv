include ../../config.mk

CFLAGS := -O3 -Wall -I"../../" $(CFLAGS)
NVFLAGS := -O3 -I"../../" $(NVFLAGS)

SRCS := ccv_nnc_compat.cu

SRC_OBJS := $(patsubst %.c,%.o,$(patsubst %.cu,%.o,$(SRCS)))

.PHONY: release all lib clean libnnc-compat-cuda.o

release: all

include ../../scheme.mk

all: lib

lib: libnnc-compat-cuda.o

clean:
	rm -f *.o

libnnc-compat-cuda.o: $(SRC_OBJS)
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
