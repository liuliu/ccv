CC = gcc
LDFLAGS = -L"../src" -L"/opt/ati-stream-sdk/lib/x86_64" -pthread -static -lccv -ljpeg -lpng -lfftw3 -lz -lgsl -lblas -lm -lgomp # -lOpenCL
CXXFLAGS = -O3 -msse2 -Wall -I"../src" -fopenmp -D USE_OPENMP # -D USE_OPENCL

TARGETS = sgfdetect sgfcreate

all: libccv.a $(TARGETS)

clean:
	rm *.o ../src/*.o ../src/libccv.a $(TARGETS)

$(TARGETS): %: %.o libccv.a
	$(CC) -o $@ $< $(LDFLAGS)

libccv.a:
	${MAKE} -C ../src

%.o: %.c ../src/ccv.h
	$(CC) $(INCFLAGS) $< -o $@ -c $(CXXFLAGS)
