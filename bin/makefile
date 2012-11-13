include ../config.mk

#CC += -faddress-sanitizer -fno-omit-frame-pointer
LDFLAGS := -L"../lib" -lccv $(LDFLAGS)
CFLAGS := -O3 -Wall -I"../lib" $(CFLAGS)

TARGETS := bbffmt msermatch siftmatch bbfcreate bbfdetect swtcreate swtdetect dpmcreate dpmdetect convert tld

all: ../lib/libccv.a $(TARGETS)

clean:
	${MAKE} clean -C ../lib ; rm *.o $(TARGETS)

$(TARGETS): %: %.o ../lib/libccv.a
	$(CC) -o $@ $< $(LDFLAGS)

../lib/libccv.a:
	${MAKE} -C ../lib

%.o: %.c ../lib/ccv.h
	$(CC) $< -o $@ -c $(CFLAGS)
