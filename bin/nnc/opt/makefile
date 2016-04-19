include ../../../lib/config.mk

#CC += -fno-omit-frame-pointer -fsanitize=address
LDFLAGS := -L"../../../lib" -lccv $(LDFLAGS)
CFLAGS := -O3 -Wall -I"../../../lib" $(CFLAGS)
NVFLAGS := -O3 -I"../../../lib" -lineinfo $(NVFLAGS)

all: conv-4x4-3x3-winograd conv-opt auto-tune-conv conv-gemm fc-opt fc-gemm auto-tune-fc

clean:
	${MAKE} clean -C ../../../lib ; rm -f *.o $(TARGETS)

conv-4x4-3x3-winograd: %: %.o libccv.a
	$(CC) -o $@ $< $(LDFLAGS)

conv-gemm: %: %.o libccv.a
	$(CC) -o $@ $< $(LDFLAGS)

conv-opt: %: %.o libccv.a
	$(CC) -o $@ $< $(LDFLAGS)

auto-tune-conv: %: %.o libccv.a
	$(CC) -o $@ $< $(LDFLAGS)

fc-opt: %: %.o libccv.a
	$(CC) -o $@ $< $(LDFLAGS)

fc-gemm: %: %.o libccv.a
	$(CC) -o $@ $< $(LDFLAGS)

auto-tune-fc: %: %.o libccv.a
	$(CC) -o $@ $< $(LDFLAGS)

libccv.a:
	${MAKE} -C ../../../lib

%.o: %.c ../../../lib/ccv.h
	$(CC) $< -o $@ -c $(CFLAGS)
