OPENCV = /opt/opencv
CXX = g++
LINKFLAG = -pthread -lcv -lhighgui -lcvaux
INCFLAGS = -I"$(OPENCV)/include/opencv"
CXXFLAGS = -O2 --fast-math -msse2

all: filter_test sobel_test hog_test

clean:
	rm *.o

hog_test: hog_test.o
	$(CXX) -o $@ $^ $(LINKFLAG)

sobel_test: sobel_test.o
	$(CXX) -o $@ $^ $(LINKFLAG)

filter_test: filter_test.o
	$(CXX) -o $@ $^ $(LINKFLAG)

.c.o:
	$(CXX) $(INCFLAGS) $< -o $@ -c $(CXXFLAGS)

.cpp.o:
	$(CXX) $(INCFLAGS) $< -o $@ -c $(CXXFLAGS)

.cc.o:
	$(CXX) $(INCFLAGS) $< -o $@ -c $(CXXFLAGS)

