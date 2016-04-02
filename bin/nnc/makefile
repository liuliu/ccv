include ../../lib/config.mk

#CC += -fno-omit-frame-pointer -fsanitize=address
LDFLAGS := -L"../../lib" -lccv $(LDFLAGS)
CFLAGS := -O3 -Wall -I"../../lib" $(CFLAGS)
NVFLAGS := -O3 -I"../../lib" -lineinfo $(NVFLAGS)

all: nnc-x-verify winograd

clean:
	${MAKE} clean -C ../../lib ; rm -f *.o $(TARGETS)

nnc-x-verify: %: %.o libccv.a
	$(CC) -o $@ $< $(LDFLAGS)

winograd: %: %.o libccv.a
	$(CC) -o $@ $< $(LDFLAGS)

libccv.a:
	${MAKE} -C ../../lib

%.o: %.c ../../lib/ccv.h
	$(CC) $< -o $@ -c $(CFLAGS)

%.o: %.cu ../../lib/ccv.h ../../lib/cuda/*.h ../../lib/cuda/*.cu
	$(NVCC) $< -o $@ -c $(NVFLAGS)
