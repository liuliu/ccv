include ../lib/config.mk
#CC += -faddress-sanitizer -fno-omit-frame-pointer
LDFLAGS := -L"../lib" -lccv -lev -ldispatch $(LDFLAGS)
CFLAGS := -O3 -Wall -I"../lib" $(CFLAGS)

TARGETS = ccv

DEPS = serve.o uri.o parsers.o bbf.o dpm.o icf.o scd.o sift.o swt.o tld.o convnet.o async.o ebb.o ebb_request_parser.o

all: libccv.a $(TARGETS)

clean:
	${MAKE} clean -C ../lib ; rm -f *.o $(TARGETS) -f

$(TARGETS): $(DEPS) libccv.a
	$(CC) -o $@ $(DEPS) $(LDFLAGS)

libccv.a:
	${MAKE} -C ../lib

ebb_request_parser.c: ebb_request_parser.rl
	ragel -s -G2 $< -o $@

%.o: %.c ../lib/ccv.h ebb.h ebb_request_parser.h
	$(CC) $< -o $@ -c $(CFLAGS)
