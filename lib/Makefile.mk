CC = gcc
# CFLAGS = -Wall -D HAVE_LIBJPEG -D HAVE_CBLAS -D HAVE_GSL -D HAVE_LIBPNG -D HAVE_FFTW3
DEFINES		= -DWIN32 -D HAVE_LIBJPEG -D HAVE_LIBPNG
LIBFILES        = -lws2_32 -ljpeg -lpng
INCLUDES        = 
OPTIMIZE        = -O3 -msse2 -Wall -fms-extensions

CFLAGS		= $(INCLUDES) $(DEFINES) $(OPTIMIZE)
LINK32		= g++
LIB32		= ar rc
#AR = ar
LD = g++.exe
#RM_F = rm -f
RM_F = del

SO = dll
_EXE = .exe
o = .o
a = .a

LDFLAGS=$(LIBFILES) -Wl,--output-def,ccv.def,--out-lib,libccv.a

# Handy lists of source code files:
C_FILES  = ccv_algebra.c \
	ccv_basic.c \
	ccv_bbf.c \
	ccv_cache.c \
	ccv_daisy.c \
	ccv_dpm.c \
	ccv_io.c \
	io/_ccv_io_binary.c \
	io/_ccv_io_bmp.c \
	io/_ccv_io_libpng.c \
	io/_ccv_io_libjpeg.c \
	ccv_memory.c \
	ccv_numeric.c \
	ccv_sift.c \
	ccv_sparse_coding.c \
	ccv_swt.c \
	ccv_util.c \
	3rdparty/sha1.c
	
O_FILES  = ccv_algebra.o \
	ccv_basic.o \
	ccv_bbf.o \
	ccv_cache.o \
	ccv_daisy.o \
	ccv_dpm.o \
	ccv_io.o \
	ccv_memory.o \
	ccv_numeric.o \
	ccv_sift.o \
	ccv_sparse_coding.o \
	ccv_swt.o \
	ccv_util.o \
	3rdparty/sha1.o

H_FILES  = ccv.h 3rdparty/sha1.h

all: libccv.a

clean:
	$(RM_F) *.o 3rdparty/*.o libccv.a

libccv.a: $(O_FILES)
	$(AR) rcs $@ $<

ccv.$(SO): libccv.a
	gcc -shared -o ccv.$(SO) $(O_FILES) $(LDFLAGS)

.c$(o):
	$(CC) -c $(null,$(<:d) $(NULL) -I$(<:d)) $(CFLAGS) -o$@ $<

$(o).dll:
	$(LINK32) -o $@ $(BLINK_FLAGS) $< $(LIBFILES)
	$(IMPLIB) --input-def $(*B).def --output-lib $(*B).a $@
