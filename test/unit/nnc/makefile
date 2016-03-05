include ../../../lib/config.mk

# CC += -g -fsanitize=address -fno-omit-frame-pointer # -fprofile-arcs -ftest-coverage
LDFLAGS := -L"../../../lib" -lccv $(LDFLAGS)
CFLAGS := -O3 -Wall -I"../../../lib" -I"../../" $(CFLAGS)
TARGETS = tfb.tests tensor.tests forward.tests backward.tests gradient.tests graph.tests

all: $(TARGETS)

test: all
	@for test in $(TARGETS) ; do ./"$$test" || exit ; done

clean:
	${MAKE} clean -C ../../../lib ; rm -f *.o $(TARGETS)

%.tests: %.tests.o libccv.a
	$(CC) -o $@ $< $(LDFLAGS)

libccv.a:
	${MAKE} -C ../../../lib

%.o: %.c ../../../lib/ccv.h ../../../lib/nnc/ccv_nnc_tfb.h
	$(CC) $< -o $@ -c $(CFLAGS)
