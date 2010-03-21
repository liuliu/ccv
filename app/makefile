CC = clang
LDFLAGS = -L"../src" -pthread -static -lccv -ljpeg -lpng -lfftw3 -lz -lgsl -lblas -lm
CXXFLAGS = -O3 -msse2 -Wall -I"../src"

TARGETS = sgfdetect

all: libccv.a $(TARGETS)

clean:
	rm *.o ../src/*.o ../src/libccv.a $(TARGETS)

$(TARGETS): %: %.o libccv.a
	$(CC) -o $@ $< $(LDFLAGS)

libccv.a:
	${MAKE} -C ../src

%.o: %.c ../src/ccv.h
	$(CC) $(INCFLAGS) $< -o $@ -c $(CXXFLAGS)
