include ../../../lib/config.mk

LDFLAGS := -L"../../../lib" -lccv $(LDFLAGS)
CFLAGS := -O3 -Wall -I"../../../lib" $(CFLAGS)
NVFLAGS := -O3 -I"../../../lib" -lineinfo $(NVFLAGS)

TARGETS = conv-cudnn-forw conv-cudnn-back

TARGET_SRCS := $(patsubst %,%.c,$(TARGETS))

.PHONY: release all clean dep

release: all

include ../../../lib/scheme.mk

all: $(TARGETS)

clean:
	${MAKE} clean -C ../../../lib ; rm -f *.o $(TARGETS)

$(TARGETS): %: %.o libccv.a
	$(CC) -o $@ $< $(LDFLAGS)

libccv.a:
	${MAKE} -C ../../../lib

%.o: %.c
	$(CC) $< -o $@ -c $(CFLAGS)

.gitignore:
	echo $(TARGETS) | tr ' ' '\n' > .gitignore

dep: .dep.mk
.dep.mk: $(TARGET_SRCS)
	$(CC) $(CFLAGS) -MM $^ > .dep.mk

-include .dep.mk
