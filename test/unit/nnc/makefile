include ../../../lib/config.mk

LDFLAGS := -L"../../../lib" -lccv $(LDFLAGS)
CFLAGS := -O3 -Wall -I"../../../lib" -I"../../" $(CFLAGS)
TARGETS = tfb.tests tensor.tests forward.tests backward.tests gradient.tests graph.tests winograd.tests transform.tests symbolic.graph.tests autograd.tests

.PHONY: release debug all test clean

release: all

debug: CFLAGS += -g -fno-omit-frame-pointer -fsanitize=address
debug: LDFLAGS += -g -fno-omit-frame-pointer -fsanitize=address
debug: all

all: $(TARGETS)

test:
	@for test in $(TARGETS) ; do ./"$$test" || exit ; done

clean:
	${MAKE} clean -C ../../../lib ; rm -f *.o $(TARGETS)

%.tests: %.tests.o libccv.a
	$(CC) -o $@ $< $(LDFLAGS)

libccv.a:
	${MAKE} -C ../../../lib

%.o: %.c ../../../lib/ccv.h ../../../lib/nnc/ccv_nnc_tfb.h
	$(CC) $< -o $@ -c $(CFLAGS)
