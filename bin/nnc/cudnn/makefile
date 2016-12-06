include ../../../lib/config.mk

LDFLAGS := -L"../../../lib" -lccv $(LDFLAGS)
CFLAGS := -O3 -Wall -I"../../../lib" $(CFLAGS)
NVFLAGS := -O3 -I"../../../lib" -lineinfo $(NVFLAGS)

TARGETS = conv-cudnn-forw conv-cudnn-back

TARGET_SRCS := $(patsubst %,%.c,$(TARGETS))

DEBUG ?= 0
ifeq ($(DEBUG), 1)
	CFLAGS += -g -fno-omit-frame-pointer -fsanitize=address
endif

.PHONY: release debug all clean dep

release: all

debug: CFLAGS += -g -fno-omit-frame-pointer -fsanitize=address
debug: LDFLAGS += -g -fno-omit-frame-pointer -fsanitize=address
debug: export DEBUG = 1
debug: all

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
