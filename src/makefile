CC = gcc
CFLAGS = -O3 -msse2 -Wall -fopenmp -D USE_OPENMP # -D USE_OPENCL -I"/opt/ati-stream-sdk/include/"

all: libccv.a

clean:
	rm *.o libccv.a

libccv.a: ccv_memory.o 3rdparty/sha1.o ccv_io.o ccv_numeric.o ccv_algebra.o ccv_util.o ccv_basic.o ccv_daisy.o ccv_sgf.o
	ar rcs $@ $^

%.o: %.c ccv.h
	$(CC) $(INCFLAGS) $< -o $@ -c $(CFLAGS)
