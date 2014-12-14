include ../lib/config.mk

# CC += -g -fsanitize=address -fno-omit-frame-pointer # -fprofile-arcs -ftest-coverage
LDFLAGS := -L"../lib" -lccv $(LDFLAGS)
CFLAGS := -O3 -Wall -I"../lib" $(CFLAGS)

TARGETS = bbffmt msermatch siftmatch bbfcreate bbfdetect scdcreate scddetect swtcreate swtdetect dpmcreate dpmdetect tld icfcreate icfdetect icfoptimize cifar-10 image-net cnnclassify aflw

all: libccv.a $(TARGETS)

clean:
	${MAKE} clean -C ../lib ; rm -f *.o $(TARGETS)

$(TARGETS): %: %.o libccv.a
	$(CC) -o $@ $< $(LDFLAGS)

libccv.a:
	${MAKE} -C ../lib

%.o: %.c ../lib/ccv.h
	$(CC) $< -o $@ -c $(CFLAGS)
