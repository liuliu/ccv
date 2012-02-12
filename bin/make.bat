cd ..\lib
call make

cd /d %~dp0

set CC=gcc
set CXXFLAGS=-mno-cygwin -O3 -msse2 -Wall -std=c99 -fms-extensions -pedantic -I"../lib"
set LDFLAGS= -L"../lib" -lccv -ljpeg -lpng -lz -lm -I"../lib" -msse2

del *.exe
del *.o

%CC% siftmatch.c -o siftmatch.o -c %CXXFLAGS%
%CC% -o siftmatch.exe siftmatch.o %LDFLAGS%

siftmatch.exe ..\samples\scene.png ..\samples\book.png 