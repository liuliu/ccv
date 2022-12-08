include ../lib/config.mk

export LSAN_OPTIONS=suppressions=known-leaks.txt
LDFLAGS := -L"../lib" -lccv $(LDFLAGS)
CFLAGS := -O3 -Wall -I"../lib" -I"." $(CFLAGS)

SRCS := unit/util.tests.c unit/basic.tests.c unit/memory.tests.c unit/transform.tests.c unit/image_processing.tests.c unit/3rdparty.tests.c unit/algebra.tests.c unit/io.tests.c unit/nnc/gradient.tests.c unit/nnc/upsample.tests.c unit/nnc/concat.tests.c unit/nnc/tensor.bind.tests.c unit/nnc/backward.tests.c unit/nnc/graph.tests.c unit/nnc/case_of.backward.tests.c unit/nnc/while.backward.tests.c unit/nnc/autograd.vector.tests.c unit/nnc/dropout.tests.c unit/nnc/custom.tests.c unit/nnc/reduce.tests.c unit/nnc/tfb.tests.c unit/nnc/batch.norm.tests.c unit/nnc/crossentropy.tests.c unit/nnc/cnnp.core.tests.c unit/nnc/symbolic.graph.tests.c unit/nnc/group.norm.tests.c unit/nnc/case_of.tests.c unit/nnc/micro.tests.c unit/nnc/compression.tests.c unit/nnc/transform.tests.c unit/nnc/dataframe.tests.c unit/nnc/gemm.tests.c unit/nnc/roi_align.tests.c unit/nnc/swish.tests.c unit/nnc/index.tests.c unit/nnc/minimize.tests.c unit/nnc/symbolic.graph.compile.tests.c unit/nnc/histogram.tests.c unit/nnc/autograd.tests.c unit/nnc/tensor.tests.c unit/nnc/rand.tests.c unit/nnc/while.tests.c unit/nnc/nms.tests.c unit/nnc/graph.io.tests.c unit/nnc/cblas.tests.c unit/nnc/simplify.tests.c unit/nnc/gelu.tests.c unit/nnc/numa.tests.c unit/nnc/loss.tests.c unit/nnc/tape.tests.c unit/nnc/dynamic.graph.tests.c unit/nnc/layer.norm.tests.c unit/nnc/parallel.tests.c unit/nnc/winograd.tests.c unit/nnc/dataframe.addons.tests.c unit/nnc/broadcast.tests.c unit/nnc/compare.tests.c unit/nnc/smooth_l1.tests.c unit/nnc/forward.tests.c unit/output.tests.c unit/convnet.tests.c unit/numeric.tests.c regression/defects.l0.1.tests.c int/nnc/cublas.tests.c int/nnc/mpsblas.tests.c int/nnc/upsample.tests.c int/nnc/concat.tests.c int/nnc/symbolic.graph.vgg.d.tests.c int/nnc/imdb.tests.c int/nnc/lstm.tests.c int/nnc/datatype.tests.c int/nnc/graph.vgg.d.tests.c int/nnc/reduce.tests.c int/nnc/leaky_relu.tests.c int/nnc/random.tests.c int/nnc/cnnp.core.tests.c int/nnc/compression.tests.c int/nnc/transform.tests.c int/nnc/roi_align.tests.c int/nnc/cudnn.tests.c int/nnc/swish.tests.c int/nnc/index.tests.c int/nnc/dense.net.tests.c int/nnc/cifar.tests.c int/nnc/rmsprop.tests.c int/nnc/sgd.tests.c int/nnc/nccl.tests.c int/nnc/nms.tests.c int/nnc/schedule.tests.c int/nnc/gelu.tests.c int/nnc/mpsdnn.tests.c int/nnc/loss.tests.c int/nnc/dynamic.graph.tests.c int/nnc/adam.tests.c int/nnc/parallel.tests.c int/nnc/compare.tests.c int/nnc/smooth_l1.tests.c int/nnc/lamb.tests.c

SRC_OBJS := $(patsubst %.c,%.o,$(SRCS))

include ../lib/scheme.mk

all.tests: all.tests.o $(SRC_OBJS) libccv.a
	$(CC) -o $@ all.tests.o $(SRC_OBJS) $(LDFLAGS)

all.tests.o: all.tests.c
	$(CC) $< -o $@ -c $(CFLAGS)

unit/util.tests.o: unit/util.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit"' -o $@ -c $(CFLAGS)

unit/basic.tests.o: unit/basic.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit"' -o $@ -c $(CFLAGS)

unit/memory.tests.o: unit/memory.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit"' -o $@ -c $(CFLAGS)

unit/transform.tests.o: unit/transform.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit"' -o $@ -c $(CFLAGS)

unit/image_processing.tests.o: unit/image_processing.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit"' -o $@ -c $(CFLAGS)

unit/3rdparty.tests.o: unit/3rdparty.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit"' -o $@ -c $(CFLAGS)

unit/algebra.tests.o: unit/algebra.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit"' -o $@ -c $(CFLAGS)

unit/io.tests.o: unit/io.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit"' -o $@ -c $(CFLAGS)

unit/nnc/gradient.tests.o: unit/nnc/gradient.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/upsample.tests.o: unit/nnc/upsample.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/concat.tests.o: unit/nnc/concat.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/tensor.bind.tests.o: unit/nnc/tensor.bind.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/backward.tests.o: unit/nnc/backward.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/graph.tests.o: unit/nnc/graph.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/case_of.backward.tests.o: unit/nnc/case_of.backward.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/while.backward.tests.o: unit/nnc/while.backward.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/autograd.vector.tests.o: unit/nnc/autograd.vector.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/dropout.tests.o: unit/nnc/dropout.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/custom.tests.o: unit/nnc/custom.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/reduce.tests.o: unit/nnc/reduce.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/tfb.tests.o: unit/nnc/tfb.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/batch.norm.tests.o: unit/nnc/batch.norm.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/crossentropy.tests.o: unit/nnc/crossentropy.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/cnnp.core.tests.o: unit/nnc/cnnp.core.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/symbolic.graph.tests.o: unit/nnc/symbolic.graph.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/group.norm.tests.o: unit/nnc/group.norm.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/case_of.tests.o: unit/nnc/case_of.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/micro.tests.o: unit/nnc/micro.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/compression.tests.o: unit/nnc/compression.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/transform.tests.o: unit/nnc/transform.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/dataframe.tests.o: unit/nnc/dataframe.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/gemm.tests.o: unit/nnc/gemm.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/roi_align.tests.o: unit/nnc/roi_align.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/swish.tests.o: unit/nnc/swish.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/index.tests.o: unit/nnc/index.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/minimize.tests.o: unit/nnc/minimize.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/symbolic.graph.compile.tests.o: unit/nnc/symbolic.graph.compile.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/histogram.tests.o: unit/nnc/histogram.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/autograd.tests.o: unit/nnc/autograd.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/tensor.tests.o: unit/nnc/tensor.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/rand.tests.o: unit/nnc/rand.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/while.tests.o: unit/nnc/while.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/nms.tests.o: unit/nnc/nms.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/graph.io.tests.o: unit/nnc/graph.io.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/cblas.tests.o: unit/nnc/cblas.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/simplify.tests.o: unit/nnc/simplify.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/gelu.tests.o: unit/nnc/gelu.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/numa.tests.o: unit/nnc/numa.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/loss.tests.o: unit/nnc/loss.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/tape.tests.o: unit/nnc/tape.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/dynamic.graph.tests.o: unit/nnc/dynamic.graph.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/layer.norm.tests.o: unit/nnc/layer.norm.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/parallel.tests.o: unit/nnc/parallel.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/winograd.tests.o: unit/nnc/winograd.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/dataframe.addons.tests.o: unit/nnc/dataframe.addons.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/broadcast.tests.o: unit/nnc/broadcast.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/compare.tests.o: unit/nnc/compare.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/smooth_l1.tests.o: unit/nnc/smooth_l1.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/nnc/forward.tests.o: unit/nnc/forward.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit/nnc"' -o $@ -c $(CFLAGS)

unit/output.tests.o: unit/output.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit"' -o $@ -c $(CFLAGS)

unit/convnet.tests.o: unit/convnet.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit"' -o $@ -c $(CFLAGS)

unit/numeric.tests.o: unit/numeric.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"unit"' -o $@ -c $(CFLAGS)

regression/defects.l0.1.tests.o: regression/defects.l0.1.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"regression"' -o $@ -c $(CFLAGS)

int/nnc/cublas.tests.o: int/nnc/cublas.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

int/nnc/mpsblas.tests.o: int/nnc/mpsblas.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

int/nnc/upsample.tests.o: int/nnc/upsample.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

int/nnc/concat.tests.o: int/nnc/concat.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

int/nnc/symbolic.graph.vgg.d.tests.o: int/nnc/symbolic.graph.vgg.d.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

int/nnc/imdb.tests.o: int/nnc/imdb.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

int/nnc/lstm.tests.o: int/nnc/lstm.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

int/nnc/datatype.tests.o: int/nnc/datatype.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

int/nnc/graph.vgg.d.tests.o: int/nnc/graph.vgg.d.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

int/nnc/reduce.tests.o: int/nnc/reduce.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

int/nnc/leaky_relu.tests.o: int/nnc/leaky_relu.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

int/nnc/random.tests.o: int/nnc/random.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

int/nnc/cnnp.core.tests.o: int/nnc/cnnp.core.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

int/nnc/compression.tests.o: int/nnc/compression.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

int/nnc/transform.tests.o: int/nnc/transform.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

int/nnc/roi_align.tests.o: int/nnc/roi_align.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

int/nnc/cudnn.tests.o: int/nnc/cudnn.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

int/nnc/swish.tests.o: int/nnc/swish.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

int/nnc/index.tests.o: int/nnc/index.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

int/nnc/dense.net.tests.o: int/nnc/dense.net.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

int/nnc/cifar.tests.o: int/nnc/cifar.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

int/nnc/rmsprop.tests.o: int/nnc/rmsprop.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

int/nnc/sgd.tests.o: int/nnc/sgd.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

int/nnc/nccl.tests.o: int/nnc/nccl.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

int/nnc/nms.tests.o: int/nnc/nms.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

int/nnc/schedule.tests.o: int/nnc/schedule.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

int/nnc/gelu.tests.o: int/nnc/gelu.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

int/nnc/mpsdnn.tests.o: int/nnc/mpsdnn.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

int/nnc/loss.tests.o: int/nnc/loss.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

int/nnc/dynamic.graph.tests.o: int/nnc/dynamic.graph.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

int/nnc/adam.tests.o: int/nnc/adam.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

int/nnc/parallel.tests.o: int/nnc/parallel.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

int/nnc/compare.tests.o: int/nnc/compare.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

int/nnc/smooth_l1.tests.o: int/nnc/smooth_l1.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

int/nnc/lamb.tests.o: int/nnc/lamb.tests.c
	$(CC) $< -D COVERAGE_TESTS -D CASE_DISABLE_MAIN -D CASE_TEST_DIR='"int/nnc"' -o $@ -c $(CFLAGS)

libccv.a:
	${MAKE} -C ../lib
