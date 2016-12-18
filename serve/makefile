include ../lib/config.mk

LDFLAGS := -L"../lib" -lccv -lev -ldispatch $(LDFLAGS)
CFLAGS := -O3 -Wall -I"../lib" $(CFLAGS)

TARGETS = ccv

SRCS = serve.c uri.c parsers.c bbf.c dpm.c icf.c scd.c sift.c swt.c tld.c convnet.c async.c ebb.c ebb_request_parser.c

SRC_OBJS := $(patsubst %.c,%.o,$(SRCS))

.PHONY: release all clean dep

release: all

include ../lib/scheme.mk

all: libccv.a $(TARGETS)

clean:
	${MAKE} clean -C ../lib ; rm -f *.o $(TARGETS) -f

$(TARGETS): $(SRC_OBJS) libccv.a
	$(CC) -o $@ $(SRC_OBJS) $(LDFLAGS)

libccv.a:
	${MAKE} -C ../lib

ebb_request_parser.c: ebb_request_parser.rl
	ragel -s -G2 $< -o $@

%.o: %.c
	$(CC) $< -o $@ -c $(CFLAGS)

.gitignore:
	echo $(TARGETS) | tr ' ' '\n' > .gitignore

dep: .dep.mk
.dep.mk: $(SRCS)
	$(CC) $(CFLAGS) -MM $^ > .dep.mk

-include .dep.mk
