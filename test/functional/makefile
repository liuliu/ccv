CC = `cat ../../lib/.CC`
LDFLAGS = -L"../../lib" -lccv -pthread `cat ../../lib/.LN`
CFLAGS = -O3 -msse2 -Wall -I"../../lib" -I"../" `cat ../../lib/.DEF`
TARGETS = algebra.tests util.tests numeric.tests basic.tests memory.tests

test: all
	@for test in $(TARGETS) ; do ./"$$test" ; done

all: $(TARGETS)

clean:
	rm *.o ../../lib/*.o ../../lib/3rdparty/*.o ../../lib/libccv.a $(TARGETS)

%.tests: %.tests.o libccv.a
	@echo "$(CC) -o $@ $< $(LDFLAGS)"
	@$(CC) -o $@ $< $(LDFLAGS)

libccv.a:
	${MAKE} -C ../../lib

%.o: %.c ../../lib/ccv.h
	@echo "$(CC) $(INCFLAGS) $< -o $@ -c $(CFLAGS)"
	@$(CC) $(INCFLAGS) $< -o $@ -c $(CFLAGS)
