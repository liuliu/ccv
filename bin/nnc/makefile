include ../../lib/config.mk

LDFLAGS := -L"../../lib" -lccv $(LDFLAGS)
CFLAGS := -O3 -Wall -I"../../lib" $(CFLAGS)
NVFLAGS := -O3 -I"../../lib" -lineinfo $(NVFLAGS)

.PHONY: release debug all clean

release: all

debug: CFLAGS += -g -fno-omit-frame-pointer -fsanitize=address
debug: LDFLAGS += -g -fno-omit-frame-pointer -fsanitize=address
debug: all

all: nnc-e2e-verify nnc-e2e-sym-verify nnc-sym

clean:
	${MAKE} clean -C ../../lib ; rm -f *.o $(TARGETS)

nnc-e2e-verify: %: %.o libccv.a
	$(CC) -o $@ $< $(LDFLAGS)

nnc-e2e-sym-verify: %: %.o libccv.a
	$(CC) -o $@ $< $(LDFLAGS)

nnc-sym: %: %.o libccv.a
	$(CC) -o $@ $< $(LDFLAGS)

libccv.a:
	${MAKE} -C ../../lib

%.o: %.c ../../lib/ccv.h
	$(CC) $< -o $@ -c $(CFLAGS)
