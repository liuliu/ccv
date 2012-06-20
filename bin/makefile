<<<<<<< HEAD
include ../lib/config.mk
=======
CC = `cat ../lib/.cc.conf`
LDFLAGS = -L"../lib" -lccv `cat ../lib/.l.conf` -lm
CFLAGS = -O3 -Wall -I"../lib" `cat ../lib/.def.conf`
>>>>>>> added a simple config script

#CC += -faddress-sanitizer -fno-omit-frame-pointer
LDFLAGS := -L"../lib" -lccv $(LDFLAGS)
CFLAGS := -O3 -Wall -I"../lib" $(CFLAGS)

TARGETS = bbffmt msermatch siftmatch bbfcreate bbfdetect swtcreate swtdetect dpmcreate dpmdetect convert tld

all: libccv.a $(TARGETS)

clean:
	${MAKE} clean -C ../lib ; rm -f *.o $(TARGETS)

$(TARGETS): %: %.o libccv.a
	$(CC) -o $@ $< $(LDFLAGS)

libccv.a:
	${MAKE} -C ../lib

%.o: %.c ../lib/ccv.h
	$(CC) $< -o $@ -c $(CFLAGS)
