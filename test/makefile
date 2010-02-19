CXX = clang
LINKFLAG = -pthread -ljpeg -lpng
CXXFLAGS = -O2 -msse2 -Wall

all: cache_test sparse_matrix_test io_test

clean:
	rm *.o ../*.o

cache_test: cache_test.o ../libccv.a
	$(CXX) -o $@ $^ $(LINKFLAG)

io_test: io_test.o ../libccv.a
	$(CXX) -o $@ $^ $(LINKFLAG)

sparse_matrix_test: sparse_matrix_test.o ../libccv.a
	$(CXX) -o $@ $^ $(LINKFLAG)

../libccv.a:
	${MAKE} -C ../src

.c.o:
	$(CXX) $(INCFLAGS) $< -o $@ -c $(CXXFLAGS)

.cpp.o:
	$(CXX) $(INCFLAGS) $< -o $@ -c $(CXXFLAGS)

.cc.o:
	$(CXX) $(INCFLAGS) $< -o $@ -c $(CXXFLAGS)
