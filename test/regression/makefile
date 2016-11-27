include ../../lib/config.mk

LDFLAGS := -L"../../lib" -lccv $(LDFLAGS)
CFLAGS := -O3 -Wall -I"../../lib" -I"../" $(CFLAGS)
TARGETS = defects.l0.1.tests

.PHONY: release debug all test clean

release: all

debug: CFLAGS += -g -fno-omit-frame-pointer -fsanitize=address
debug: LDFLAGS += -g -fno-omit-frame-pointer -fsanitize=address
debug: all

all: $(TARGETS)

test:
	@for test in $(TARGETS) ; do ./"$$test" || exit ; done

clean:
	${MAKE} clean -C ../../lib ; rm -f *.o $(TARGETS)

%.tests: %.tests.o libccv.a
	@echo "$(CC) -o $@ $< $(LDFLAGS)"
	@$(CC) -o $@ $< $(LDFLAGS)

libccv.a:
	${MAKE} -C ../../lib

%.o: %.c ../../lib/ccv.h ../../lib/nnc/ccv_nnc_tfb.h
	@echo "$(CC) $< -o $@ -c $(CFLAGS)"
	@$(CC) $< -o $@ -c $(CFLAGS)
