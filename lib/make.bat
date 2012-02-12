set CCFLAGS=-O3 -msse2 -Wall -std=c99 -fms-extensions -pedantic -DHAVE_LIBPNG -DHAVE_LIBJPEG -DWIN32

del *.o
del *.dll
del *.a

gcc %CCFLAGS% -c ccv_sift.c -o ccv_sgf.o
gcc %CCFLAGS% -c ccv_basic.c -o ccv_basic.o
gcc %CCFLAGS% -c ccv_algebra.c -o ccv_algebra.o
gcc %CCFLAGS% -c ccv_bbf.c -o ccv_bbf.o
gcc %CCFLAGS% -c ccv_cache.c -o ccv_cache.o
gcc %CCFLAGS% -c ccv_daisy.c -o ccv_daisy.o
gcc %CCFLAGS% -c ccv_dpm.c -o ccv_dpm.o
gcc %CCFLAGS% -c ccv_memory.c -o ccv_memory.o
gcc %CCFLAGS% -c ccv_util.c -o ccv_util.o
gcc %CCFLAGS% -c ccv_numeric.c -o ccv_numeric.o
gcc %CCFLAGS% -c ccv_io.c -o ccv_io.o
gcc %CCFLAGS% -c ccv_swt.c -o ccv_swt.o
gcc %CCFLAGS% -c 3rdparty/sha1.c -o 3rdparty/sha1.o
gcc -shared -o ccv.dll ccv_cache.o ccv_memory.o 3rdparty/sha1.o ccv_io.o ccv_numeric.o ccv_algebra.o ccv_util.o ccv_basic.o ccv_daisy.o ccv_sgf.o -lws2_32 -ljpeg -lpng -Wl,--output-def,ccv.def,--out-implib,libccv.a
@rem lib /machine:i386 /def:ccv.def
copy /y ccv.dll ..\bin
