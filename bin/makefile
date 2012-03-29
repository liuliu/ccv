CC = clang
LDFLAGS = -L"../lib" -lccv -pthread -ljpeg -lpng -lfftw3 -lz -lgsl -lblas -lm # -lgomp # -lOpenCL -L"/opt/ati-stream-sdk/lib/x86_64"
CXXFLAGS = -O3 -msse2 -Wall -I"../lib" # -fopenmp -D USE_OPENMP # -D USE_OPENCL

TARGETS = bbffmt siftmatch bbfcreate bbfdetect swtdetect swtcreate convert

all: libccv.a $(TARGETS)

clean:
	rm *.o ../lib/*.o ../lib/3rdparty/*.o ../lib/libccv.a $(TARGETS)

$(TARGETS): %: %.o libccv.a
	$(CC) -o $@ $< $(LDFLAGS)

libccv.a:
	${MAKE} -C ../lib

%.o: %.c ../lib/ccv.h
	$(CC) $< -o $@ -c $(CXXFLAGS)
