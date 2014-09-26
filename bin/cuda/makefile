include ../../lib/config.mk

#CC += -faddress-sanitizer -fno-omit-frame-pointer
LDFLAGS := -L"../../lib" -lccv $(LDFLAGS)
CFLAGS := -O3 -Wall -I"../../lib" $(CFLAGS)
NVFLAGS := -O3 -I"../../lib" -lineinfo $(NVFLAGS)

all: libccv.a cwc-forwards cwc-backwards cwc-bench cwc-verify

clean:
	${MAKE} clean -C ../../lib ; rm -f *.o $(TARGETS)

cwc-bench: %: %.o cwc-bench-runtime.o libccv.a
	$(CC) -o $@ cwc-bench-runtime.o $< $(LDFLAGS)

cwc-verify: %: %.o cwc-verify-runtime.o libccv.a
	$(CC) -o $@ cwc-verify-runtime.o $< $(LDFLAGS)

cwc-forwards: %: %.o cwc-forwards-runtime.o libccv.a
	$(CC) -o $@ cwc-forwards-runtime.o $< $(LDFLAGS)

cwc-backwards: %: %.o cwc-backwards-runtime.o libccv.a
	$(CC) -o $@ cwc-backwards-runtime.o $< $(LDFLAGS)

libccv.a:
	${MAKE} -C ../../lib

%.o: %.c ../../lib/ccv.h
	$(CC) $< -o $@ -c $(CFLAGS)

%.o: %.cu ../../lib/ccv.h ../../lib/cuda/*.h ../../lib/cuda/*.cu
	$(NVCC) $< -o $@ -c $(NVFLAGS)
