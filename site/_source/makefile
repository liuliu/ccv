include ../../lib/config.mk

LDFLAGS := -L"../../lib" -lccv $(LDFLAGS)
CFLAGS := -O3 -Wall -I"../../lib" $(CFLAGS)

TARGETS = section-001-001 section-001-002

TARGET_SRCS := $(patsubst %,%.c,$(TARGETS))

DEBUG ?= 0
ifeq ($(DEBUG), 1)
	CFLAGS += -g -fno-omit-frame-pointer -fsanitize=address
endif

release: all

debug: CFLAGS += -g -fno-omit-frame-pointer -fsanitize=address
debug: LDFLAGS += -g -fno-omit-frame-pointer -fsanitize=address
debug: export DEBUG = 1
debug: all

all: libccv.a $(TARGETS)

clean:
	${MAKE} clean -C ../../lib ; rm -f *.o $(TARGETS)

$(TARGETS): %: %.o libccv.a
	$(CC) -o $@ $< $(LDFLAGS)

libccv.a:
	${MAKE} -C ../../lib

%.o: %.c
	$(CC) $< -o $@ -c $(CFLAGS)

.gitignore:
	echo $(TARGETS) | tr ' ' '\n' > .gitignore

dep: .dep.mk
.dep.mk: $(TARGET_SRCS)
	$(CC) $(CFLAGS) -MM $^ > .dep.mk

-include .dep.mk
