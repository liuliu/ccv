include ../lib/config.mk
#CC += -faddress-sanitizer -fno-omit-frame-pointer
LDFLAGS := -L"../lib" -lccv -lev -ldispatch $(LDFLAGS)
CFLAGS := -O3 -Wall -I"../lib" $(CFLAGS)

TARGETS = ccv

all: libccv.a $(TARGETS)

clean:
	${MAKE} clean -C ../lib ; rm -f *.o $(TARGETS) -f

$(TARGETS): serve.o uri.o parsers.o bbf.o dpm.o icf.o sift.o swt.o tld.o convnet.o async.o ebb.o ebb_request_parser.o libccv.a
	$(CC) -o $@ serve.o uri.o parsers.o bbf.o dpm.o icf.o sift.o swt.o tld.o convnet.o async.o ebb.o ebb_request_parser.o $(LDFLAGS)

libccv.a:
	${MAKE} -C ../lib

ebb_request_parser.c: ebb_request_parser.rl
	ragel -s -G2 $< -o $@

%.o: %.c ../lib/ccv.h ebb.h ebb_request_parser.h
	$(CC) $< -o $@ -c $(CFLAGS)
