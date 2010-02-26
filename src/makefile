CXX = clang
CXXFLAGS = -O3 -msse2 -Wall

all: libccv.a

clean:
	rm *.o libccv.a

libccv.a: ccv_memory.o 3rdparty/sha1.o ccv_io.o ccv_numeric.o ccv_algebra.o ccv_util.o 3rdparty/sha1.o
	ar rcs $@ $^

.c.o:
	$(CXX) $(INCFLAGS) $< -o $@ -c $(CXXFLAGS)

.cpp.o:
	$(CXX) $(INCFLAGS) $< -o $@ -c $(CXXFLAGS)

.cc.o:
	$(CXX) $(INCFLAGS) $< -o $@ -c $(CXXFLAGS)

