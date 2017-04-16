include ../lib/config.mk

LDFLAGS := -L"../lib" -lccv $(LDFLAGS)
CFLAGS := -O3 -Wall -I"../lib" -I"." $(CFLAGS)

SRCS := regression/defects.l0.1.tests.c unit/3rdparty.tests.c unit/io.tests.c unit/algebra.tests.c unit/memory.tests.c unit/convnet.tests.c unit/transform.tests.c unit/image_processing.tests.c unit/output.tests.c unit/nnc/while.tests.c unit/nnc/backward.tests.c unit/nnc/winograd.tests.c unit/nnc/tensor.tests.c unit/nnc/forward.tests.c unit/nnc/autograd.tests.c unit/nnc/tfb.tests.c unit/nnc/gradient.tests.c unit/nnc/transform.tests.c unit/nnc/graph.tests.c unit/nnc/autograd.vector.tests.c unit/nnc/symbolic.graph.tests.c unit/util.tests.c unit/basic.tests.c unit/numeric.tests.c int/nnc/cudnn.tests.c int/nnc/graph.vgg.d.tests.c int/nnc/symbolic.graph.vgg.d.tests.c

SRC_OBJS := $(patsubst %.c,%.o,$(SRCS))

include ../lib/scheme.mk

all.tests: all.tests.o $(SRC_OBJS) libccv.a
	$(CC) -o $@ all.tests.o $(SRC_OBJS) $(LDFLAGS)

all.tests.o: all.tests.c
	$(CC) $< -o $@ -c $(CFLAGS)

regression/defects.l0.1.tests.o: regression/defects.l0.1.tests.c
	$(CC) $< -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"regression"' -o $@ -c $(CFLAGS)

unit/3rdparty.tests.o: unit/3rdparty.tests.c
	$(CC) $< -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit"' -o $@ -c $(CFLAGS)

unit/io.tests.o: unit/io.tests.c
	$(CC) $< -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit"' -o $@ -c $(CFLAGS)

unit/algebra.tests.o: unit/algebra.tests.c
	$(CC) $< -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit"' -o $@ -c $(CFLAGS)

unit/memory.tests.o: unit/memory.tests.c
	$(CC) $< -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit"' -o $@ -c $(CFLAGS)

unit/convnet.tests.o: unit/convnet.tests.c
	$(CC) $< -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit"' -o $@ -c $(CFLAGS)

unit/transform.tests.o: unit/transform.tests.c
	$(CC) $< -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit"' -o $@ -c $(CFLAGS)

unit/image_processing.tests.o: unit/image_processing.tests.c
	$(CC) $< -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit"' -o $@ -c $(CFLAGS)

unit/output.tests.o: unit/output.tests.c
	$(CC) $< -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit"' -o $@ -c $(CFLAGS)

unit/nnc/while.tests.o: unit/nnc/while.tests.c
	$(CC) $< -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/backward.tests.o: unit/nnc/backward.tests.c
	$(CC) $< -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/winograd.tests.o: unit/nnc/winograd.tests.c
	$(CC) $< -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/tensor.tests.o: unit/nnc/tensor.tests.c
	$(CC) $< -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/forward.tests.o: unit/nnc/forward.tests.c
	$(CC) $< -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/autograd.tests.o: unit/nnc/autograd.tests.c
	$(CC) $< -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/tfb.tests.o: unit/nnc/tfb.tests.c
	$(CC) $< -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/gradient.tests.o: unit/nnc/gradient.tests.c
	$(CC) $< -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/transform.tests.o: unit/nnc/transform.tests.c
	$(CC) $< -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/graph.tests.o: unit/nnc/graph.tests.c
	$(CC) $< -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/autograd.vector.tests.o: unit/nnc/autograd.vector.tests.c
	$(CC) $< -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/symbolic.graph.tests.o: unit/nnc/symbolic.graph.tests.c
	$(CC) $< -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/util.tests.o: unit/util.tests.c
	$(CC) $< -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit"' -o $@ -c $(CFLAGS)

unit/basic.tests.o: unit/basic.tests.c
	$(CC) $< -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit"' -o $@ -c $(CFLAGS)

unit/numeric.tests.o: unit/numeric.tests.c
	$(CC) $< -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit"' -o $@ -c $(CFLAGS)

int/nnc/cudnn.tests.o: int/nnc/cudnn.tests.c
	$(CC) $< -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

int/nnc/graph.vgg.d.tests.o: int/nnc/graph.vgg.d.tests.c
	$(CC) $< -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

int/nnc/symbolic.graph.vgg.d.tests.o: int/nnc/symbolic.graph.vgg.d.tests.c
	$(CC) $< -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

libccv.a:
	${MAKE} -C ../lib
