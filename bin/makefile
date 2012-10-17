CC = `cat ../lib/.CC`# -faddress-sanitizer -fno-omit-frame-pointer
LDFLAGS = -L"../lib" -lccv `cat ../lib/.LN`
CFLAGS = -O3 -Wall -I"../lib" `cat ../lib/.DEF`

TARGETS = bbffmt msermatch siftmatch bbfcreate bbfdetect swtcreate swtdetect dpmcreate dpmdetect convert tld

all: libccv.a $(TARGETS)

clean:
	${MAKE} clean -C ../lib ; rm *.o $(TARGETS)

$(TARGETS): %: %.o libccv.a
	@echo "$(CC) -o $@ $< $(LDFLAGS)"
	@$(CC) -o $@ $< $(LDFLAGS)

# lucas_kanade: lucas_kanade.o libccv.a
#	clang++ -o $@ $< $(LDFLAGS) -L"/usr/local/lib" -lopencv_highgui -lopencv_imgproc -lopencv_core -lopencv_video

libccv.a:
	${MAKE} -C ../lib

%.o: %.c ../lib/ccv.h
	@echo "$(CC) $< -o $@ -c $(CFLAGS)"
	@$(CC) $< -o $@ -c $(CFLAGS)

# %.o: %.cpp ../lib/ccv.h
#	clang++ $< -o $@ -c $(CFLAGS)
