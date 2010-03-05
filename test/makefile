CXX = clang
LINKFLAG = -L"../src" -pthread -static -lccv -ljpeg -lpng -lfftw3 -lz -lgsl -lblas -lm # -lgslcblas
CXXFLAGS = -O3 -msse2 -Wall -I"../src"

all: libccv.a sobel_test cache_test sparse_matrix_test io_test filter_test algebra_test compress_matrix_test

clean:
	rm *.o ../src/*.o ../src/libccv.a

kmeans: kmeans.o
	$(CXX) -o $@ $^ $(LINKFLAG)	

algebra_test: algebra_test.o
	$(CXX) -o $@ $^ $(LINKFLAG)	

sobel_test: sobel_test.o
	$(CXX) -o $@ $^ $(LINKFLAG)	

filter_test: filter_test.o
	$(CXX) -o $@ $^ $(LINKFLAG)	

cache_test: cache_test.o
	$(CXX) -o $@ $^ $(LINKFLAG)

io_test: io_test.o
	$(CXX) -o $@ $^ $(LINKFLAG)

sparse_matrix_test: sparse_matrix_test.o
	$(CXX) -o $@ $^ $(LINKFLAG)

compress_matrix_test: compress_matrix_test.o
	$(CXX) -o $@ $^ $(LINKFLAG)

libccv.a:
	${MAKE} -C ../src

.c.o:
	$(CXX) $(INCFLAGS) $< -o $@ -c $(CXXFLAGS)

.cpp.o:
	$(CXX) $(INCFLAGS) $< -o $@ -c $(CXXFLAGS)

.cc.o:
	$(CXX) $(INCFLAGS) $< -o $@ -c $(CXXFLAGS)
