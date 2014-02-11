include ../../lib/config.mk

#CC += -faddress-sanitizer -fno-omit-frame-pointer
LDFLAGS := -L"../../lib" -lccv $(LDFLAGS)
CFLAGS := -O3 -Wall -I"../../lib" $(CFLAGS)
NVFLAGS := -O3 -I"../../lib" $(NVFLAGS)

all: libccv.a cwc-bench

clean:
	${MAKE} clean -C ../../lib ; rm -f *.o $(TARGETS)

cwc-bench: %: %.o cwc-bench-runtime.o libccv.a
	$(CC) -o $@ cwc-bench-runtime.o $< $(LDFLAGS)

libccv.a:
	${MAKE} -C ../../lib

%.o: %.c ../../lib/ccv.h
	$(CC) $< -o $@ -c $(CFLAGS)

%.o: %.cu ../../lib/ccv.h ../../lib/cuda/*.h ../../lib/cuda/*.cu
	$(NVCC) $< -o $@ -c $(NVFLAGS)
