include ../../../lib/config.mk

LDFLAGS := -L"../../../lib" -lccv $(LDFLAGS)
CFLAGS := -O3 -Wall -I"../../../lib" $(CFLAGS)
NVFLAGS := -O3 -I"../../../lib" -lineinfo $(NVFLAGS)

.PHONY: release debug all clean

release: all

debug: CFLAGS += -g -fno-omit-frame-pointer -fsanitize=address
debug: LDFLAGS += -g -fno-omit-frame-pointer -fsanitize=address
debug: all

all: conv-cudnn-forw conv-cudnn-back

clean:
	${MAKE} clean -C ../../../lib ; rm -f *.o $(TARGETS)

conv-cudnn-forw: %: %.o libccv.a
	$(CC) -o $@ $< $(LDFLAGS)

conv-cudnn-back: %: %.o libccv.a
	$(CC) -o $@ $< $(LDFLAGS)

libccv.a:
	${MAKE} -C ../../../lib

%.o: %.c ../../../lib/ccv.h
	$(CC) $< -o $@ -c $(CFLAGS)
