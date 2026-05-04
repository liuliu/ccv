include ../../lib/config.mk

LDFLAGS := -L"../../lib" -lccv $(LDFLAGS)
NVFLAGS := -O3 -I"../../lib" -lineinfo $(NVFLAGS)

TARGETS = layer_norm_bench

.PHONY: release all clean

release: all

include ../../lib/scheme.mk

all: $(TARGETS)

clean:
	${MAKE} clean -C ../../lib ; rm -f *.o $(TARGETS)

$(TARGETS): %: %.o libccv.a
	$(CC) -o $@ $< $(LDFLAGS)

libccv.a:
	${MAKE} -C ../../lib

%.o: %.cu ../../lib/ccv.h
	$(NVCC) $< -o $@ -c $(NVFLAGS)
