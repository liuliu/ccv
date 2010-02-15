CXX = clang
LINKFLAG = -pthread
CXXFLAGS = -O2 -msse2 -Wall

all: cache_test sparse_matrix_test

clean:
	rm *.o ../*.o

cache_test: cache_test.o ../ccv_memory.o ../3rdparty/sha1.o
	$(CXX) -o $@ $^ $(LINKFLAG)

sparse_matrix_test: sparse_matrix_test.o ../ccv_memory.o ../ccv_util.o ../3rdparty/sha1.o
	$(CXX) -o $@ $^ $(LINKFLAG)

.c.o:
	$(CXX) $(INCFLAGS) $< -o $@ -c $(CXXFLAGS)

.cpp.o:
	$(CXX) $(INCFLAGS) $< -o $@ -c $(CXXFLAGS)

.cc.o:
	$(CXX) $(INCFLAGS) $< -o $@ -c $(CXXFLAGS)
