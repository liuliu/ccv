CC = clang # -faddress-sanitizer -fno-omit-frame-pointer
LDFLAGS = -L"../../lib" -lccv -pthread -ljpeg -lpng -lfftw3 -lfftw3f -lz -lgsl -lblas -lm
CFLAGS = -O1 -msse2 -Wall -I"../../lib" -I"../"
TARGETS = defects.l0.1.tests

test: all
	for test in $(TARGETS) ; do ./"$$test" ; done

all: $(TARGETS)

clean:
	rm *.o ../../lib/*.o ../../lib/3rdparty/*.o ../../lib/libccv.a $(TARGETS)

%.tests: %.tests.o libccv.a
	$(CC) -o $@ $< $(LDFLAGS)

libccv.a:
	${MAKE} -C ../../lib

%.o: %.c ../lib/ccv.h
	$(CC) $(INCFLAGS) $< -o $@ -c $(CFLAGS)
