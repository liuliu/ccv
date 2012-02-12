CC = gcc
CXXFLAGS = -mno-cygwin -O3 -msse2 -Wall -std=c99 -fms-extensions -pedantic -I"../lib"
LDFLAGS= -L"../lib" -L"/strawberry/c/lib" -lccv -ljpeg -lpng -lz -lm -lws2_32

_EXE = .exe
o = .o
a = .a

RM_F = del

.SUFFIXES : .c .i $(o) .dll $(a) .exe .rc .res


TARGETS = bbffmt siftmatch bbfcreate bbfdetect swtdetect swtcreate dpmdetect convert

all: ../lib/libccv.a $(TARGETS)

clean:
	$(RM_F) *$o ../lib/*$o ../lib/3rdparty/*$o ../lib/libccv$a $(TARGETS)

$(TARGETS) : $@$_EXE

%.exe : %$o '../lib/libccv.a'
	$(CC) $< -o $@ $(LDFLAGS) 

#$(TARGETS) : $@$(_OBJ) '../lib/libccv.a'
#	echo exe
#	$(CC) $(LDFLAGS) -o $@$(_EXE) $<
	

../lib/libccv.a:
	+cd ..\lib && $(MAKE) libccv.a

%.o: %.c '../lib/ccv.h'
	$(CC) $(CXXFLAGS) -c $< -o $@ 
