CC = `cat ../lib/.cc.conf`
LDFLAGS = -L"../lib" -lccv `cat ../lib/.l.conf` -lm
CFLAGS = -O3 -Wall -I"../lib" `cat ../lib/.def.conf`

TARGETS = bbffmt siftmatch bbfcreate bbfdetect swtcreate swtdetect dpmcreate dpmdetect convert

all: libccv.a $(TARGETS)

clean:
	rm *.o ../lib/*.o ../lib/3rdparty/*.o ../lib/libccv.a $(TARGETS)

$(TARGETS): %: %.o libccv.a
	$(CC) -o $@ $< $(LDFLAGS)

libccv.a:
	${MAKE} -C ../lib

%.o: %.c ../lib/ccv.h
	$(CC) $< -o $@ -c $(CFLAGS)
