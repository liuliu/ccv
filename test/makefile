CXX = clang
LINKFLAG = -L"../src" -pthread -static -lccv -ljpeg -lpng -lz -lm
CXXFLAGS = -O2 -msse2 -Wall -I"../src"

all: libccv.a cache_test sparse_matrix_test io_test

clean:
	rm *.o ../src/*.o ../src/libccv.a

cache_test: cache_test.o
	$(CXX) -o $@ $^ $(LINKFLAG)

io_test: io_test.o
	$(CXX) -o $@ $^ $(LINKFLAG)

sparse_matrix_test: sparse_matrix_test.o
	$(CXX) -o $@ $^ $(LINKFLAG)

libccv.a:
	${MAKE} -C ../src

.c.o:
	$(CXX) $(INCFLAGS) $< -o $@ -c $(CXXFLAGS)

.cpp.o:
	$(CXX) $(INCFLAGS) $< -o $@ -c $(CXXFLAGS)

.cc.o:
	$(CXX) $(INCFLAGS) $< -o $@ -c $(CXXFLAGS)
