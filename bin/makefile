CC = gcc
LDFLAGS = -L"../lib" -L"/opt/ati-stream-sdk/lib/x86_64" -pthread -lccv -ljpeg -lpng -lfftw3 -lz -lgsl -lblas -lm -lgomp -lOpenCL
CXXFLAGS = -O3 -msse2 -Wall -I"../lib" -fopenmp -D USE_OPENMP -D USE_OPENCL

TARGETS = sgfdetect sgfcreate sgffmt sgfopt siftmatch

all: libccv.a $(TARGETS)

clean:
	rm *.o ../lib/*.o ../lib/libccv.a $(TARGETS)

$(TARGETS): %: %.o libccv.a
	$(CC) -o $@ $< $(LDFLAGS)

libccv.a:
	${MAKE} -C ../lib

%.o: %.c ../lib/ccv.h
	$(CC) $(INCFLAGS) $< -o $@ -c $(CXXFLAGS)
