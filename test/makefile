CC = clang
LDFLAGS = -L"../lib" -pthread -static -lccv -ljpeg -lpng -lfftw3 -lz -lgsl -lblas -lm # -lgslcblas
CXXFLAGS = -O3 -msse2 -Wall -I"../lib"

TARGETS = sobel_test hog_test cache_test sparse_matrix_test io_test filter_test algebra_test compress_matrix_test atan2_test resample_test sample_down_test daisy_test daisy_kmeans kmeans array_test group_test slice_test flip_test minimize_test gc_test blur_test sift_test

all: $(TARGETS)

clean:
	rm *.o ../lib/*.o ../lib/libccv.a $(TARGETS)

$(TARGETS): %: %.o libccv.a
	$(CC) -o $@ $< $(LDFLAGS)

libccv.a:
	${MAKE} -C ../lib

%.o: %.c ../lib/ccv.h
	$(CC) $(INCFLAGS) $< -o $@ -c $(CXXFLAGS)
