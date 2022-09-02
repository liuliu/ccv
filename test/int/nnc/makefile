include ../../../lib/config.mk

export LSAN_OPTIONS=suppressions=known-leaks.txt
LDFLAGS := -L"../../../lib" -lccv $(LDFLAGS)
CFLAGS := -O3 -Wall -I"../../../lib" -I"../../" $(CFLAGS)
TARGETS = cudnn.tests cublas.tests schedule.tests symbolic.graph.vgg.d.tests graph.vgg.d.tests dense.net.tests parallel.tests nccl.tests compression.tests index.tests dynamic.graph.tests cifar.tests imdb.tests loss.tests random.tests swish.tests sgd.tests rmsprop.tests adam.tests upsample.tests smooth_l1.tests roi_align.tests nms.tests compare.tests transform.tests cnnp.core.tests concat.tests lamb.tests reduce.tests lstm.tests gelu.tests

ifeq ($(shell uname 2>/dev/null || echo Unknown),Darwin)
	LDFLAGS += -Wl,-U,___test_case_setup,-U,___test_case_teardown
endif

TARGET_SRCS := $(patsubst %,%.c,$(TARGETS))

.PRECIOUS: $(patsubst %,%.o,$(TARGETS))

.PHONY: release all test clean dep

release: all

include ../../../lib/scheme.mk

all: $(TARGETS)

test:
	@for test in $(TARGETS) ; do LLVM_PROFILE_FILE="$$test.profraw" ./"$$test" || exit ; done

clean:
	${MAKE} clean -C ../../../lib ; rm -f *.o $(TARGETS)

%.tests: %.tests.o libccv.a
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
