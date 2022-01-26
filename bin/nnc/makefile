include ../../lib/config.mk

LDFLAGS := -L"../../lib" -lccv $(LDFLAGS)
CFLAGS := -O3 -Wall -I"../../lib" $(CFLAGS)
NVFLAGS := -O3 -I"../../lib" -lineinfo $(NVFLAGS)

TARGETS = nnc-e2e-verify nnc-e2e-sym-verify nnc-sym cifar-10 imagenet coco imdb iwslt wmt csv imdb_lstm

FUZZ_TARGETS = csv_fuzz

TARGET_SRCS := $(patsubst %,%.c,$(TARGETS))

FUZZ_TARGET_SRCS := $(patsubst %,%.c,$(FUZZ_TARGETS))

.PHONY: release all clean dep

release: all

include ../../lib/scheme.mk

all: $(TARGETS)

fuzz: $(FUZZ_TARGETS)

clean:
	${MAKE} clean -C ../../lib ; rm -f *.o $(TARGETS) $(FUZZ_TARGETS)

$(TARGETS): %: %.o libccv.a
	$(CC) -o $@ $< $(LDFLAGS)

$(FUZZ_TARGETS): %: %.o libccv.a
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
