CC = `cat ../../lib/.CC`
LDFLAGS = -L"../../lib" -lccv `cat ../../lib/.LN`
CFLAGS = -O3 -msse2 -Wall -I"../../lib" -I"../" `cat ../../lib/.DEF`
TARGETS = defects.l0.1.tests

test: all
	@for test in $(TARGETS) ; do ./"$$test" ; done

all: $(TARGETS)

clean:
	${MAKE} clean -C ../../lib ; rm *.o $(TARGETS)

%.tests: %.tests.o libccv.a
	@echo "$(CC) -o $@ $< $(LDFLAGS)"
	@$(CC) -o $@ $< $(LDFLAGS)

libccv.a:
	${MAKE} -C ../../lib

%.o: %.c ../../lib/ccv.h
	@echo "$(CC) $< -o $@ -c $(CFLAGS)"
	@$(CC) $< -o $@ -c $(CFLAGS)
