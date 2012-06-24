CC = `cat ../lib/.CC`# -faddress-sanitizer -fno-omit-frame-pointer
LDFLAGS = -L"../lib" -lccv `cat ../lib/.LN` -lm
CFLAGS = -O1 -Wall -I"../lib" `cat ../lib/.DEF`

TARGETS = bbffmt siftmatch bbfcreate bbfdetect swtcreate swtdetect dpmcreate dpmdetect convert

all: libccv.a $(TARGETS)

clean:
	${MAKE} clean -C ../lib ; rm *.o $(TARGETS)

$(TARGETS): %: %.o libccv.a
	@echo "$(CC) -o $@ $< $(LDFLAGS)"
	@$(CC) -o $@ $< $(LDFLAGS)

libccv.a:
	${MAKE} -C ../lib

%.o: %.c ../lib/ccv.h
	@echo "$(CC) $< -o $@ -c $(CFLAGS)"
	@$(CC) $< -o $@ -c $(CFLAGS)
