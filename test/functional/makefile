include ../../lib/config.mk

#CC +=# -fprofile-arcs -ftest-coverage
LDFLAGS := -L"../../lib" -lccv $(LDFLAGS)
CFLAGS := -O3 -Wall -I"../../lib" -I"../" $(CFLAGS)
TARGETS = algebra.tests util.tests numeric.tests basic.tests memory.tests io.tests transform.tests convnet.tests

test: all
	@for test in $(TARGETS) ; do ./"$$test" || exit ; done

all: $(TARGETS)

clean:
	${MAKE} clean -C ../../lib ; rm -f *.o $(TARGETS)

%.tests: %.tests.o libccv.a
	$(CC) -o $@ $< $(LDFLAGS)

libccv.a:
	${MAKE} -C ../../lib

%.o: %.c ../../lib/ccv.h
	$(CC) $< -o $@ -c $(CFLAGS)
