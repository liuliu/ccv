CC = clang
LDFLAGS = -L"../lib" -lccv -pthread -ljpeg -lpng -lfftw3f -lfftw3 -lz -lgsl -lblas -lm # -lgomp
CFLAGS = -O3 -msse2 -Wall -I"../lib"

TARGETS = bbffmt siftmatch bbfcreate bbfdetect swtdetect swtcreate dpmdetect convert

all: libccv.a $(TARGETS)

clean:
	rm *.o ../lib/*.o ../lib/3rdparty/*.o ../lib/libccv.a $(TARGETS)

$(TARGETS): %: %.o libccv.a
	$(CC) -o $@ $< $(LDFLAGS)

libccv.a:
	${MAKE} -C ../lib

%.o: %.c ../lib/ccv.h
	$(CC) $< -o $@ -c $(CFLAGS)
