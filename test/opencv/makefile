OPENCV = /opt/opencv
CXX = g++
LINKFLAG = -pthread -lcv -lhighgui -lcvaux
INCFLAGS = -I"$(OPENCV)/include/opencv"
CXXFLAGS = -O2 --fast-math -msse2

TARGETS = filter_test sobel_test hog_test resample_test sample_down_test

all: $(TARGETS)

clean:
	rm *.o $(TARGETS)

hog_test: hog_test.o
	$(CXX) -o $@ $^ $(LINKFLAG)

sample_down_test: sample_down_test.o
	$(CXX) -o $@ $^ $(LINKFLAG)

resample_test: resample_test.o
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

