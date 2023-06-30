include ../../config.mk

CFLAGS := -O3 -Wall -I"../../" $(CFLAGS)

SRCS := ccv_nnc_mfa.cpp

SRC_OBJS := $(patsubst %.c,%.o,$(patsubst %.m,%.o,$(SRCS)))

.PHONY: release all lib clean libnnc-compat-mfa.o

release: all

include ../../scheme.mk

all: lib

lib: libnnc-compat-mfa.o

clean:
	rm -f *.o

libnnc-compat-mfa.o: $(SRC_OBJS)
	ld -r $^ -o $@

%.o: %.c
	$(CC) $< -o $@ -c $(CFLAGS)

%.o: %.cpp
	$(CC) $< -o $@ -c $(CFLAGS)

dep: .dep.mk
.dep.mk: $(SRCS)
	echo '' > .dep.mk
	for SRC in $(patsubst %.c,,$^) ; do \
		$(NVCC) $(NVFLAGS) -M $$SRC | sed -e 's/^.*\://g' | (echo "$${SRC%%.*}.o: \\" && cat) >> .dep.mk ; \
	done

-include .dep.mk

