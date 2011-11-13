CC = clang
LDFLAGS = -L"../lib" -pthread -static -lccv -ljpeg -lpng -lfftw3 -lz -lgsl -lblas -lm # -lgslcblas
# CFLAGS = -Wall -I"../lib"
CFLAGS = -O3 -msse2 -Wall -I"../lib"
TARGETS = algebra.tests util.tests numeric.tests basic.tests memory.tests

all: $(TARGETS)

test:
	for test in $(TARGETS) ; do ./"$$test" ; done

clean:
	rm *.o ../lib/*.o ../lib/3rdparty/*.o ../lib/libccv.a $(TARGETS)

%.tests: %.tests.o libccv.a
	$(CC) -o $@ $< $(LDFLAGS)

libccv.a:
	${MAKE} -C ../lib

%.o: %.c ../lib/ccv.h
	$(CC) $(INCFLAGS) $< -o $@ -c $(CFLAGS)
