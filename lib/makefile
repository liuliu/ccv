# Include config file if var are not set
ifneq ($(MAKECMDGOALS),clean)
  ifeq  "$(and $(CC),$(AR),$(CFLAGS))" ""
    include config.mk
  endif
endif 

# Configure output trace
VERBOSE?=0
ifeq ($(VERBOSE),0)
  SILENT=@
  TO_NULL=
else
  SILENT=
  TO_NULL=1>/dev/null
endif

# Configure color
COLOR?=1
ifeq ($(COLOR),1)
  SHELL = bash
  RED="\033[31m"
  PURPLE="\033[35m"
  WHITE="\033[0m"
endif

#CC += -faddress-sanitizer -fno-omit-frame-pointer
CFLAGS := -O3 -ffast-math -Wall $(CFLAGS)# -fprofile-arcs -ftest-coverage

.PHONY: all clean

all: libccv.a

clean:
	@echo -en $(RED)
	@echo CLEAN $(TO_NULL)
	$(SILENT)rm -f *.o 3rdparty/sha1/*.o 3rdparty/sfmt/*.o 3rdparty/kissfft/*.o 3rdparty/dsfmt/*.o libccv.a
	@echo -en $(WHITE)

libccv.a: ccv_cache.o ccv_memory.o 3rdparty/sha1/sha1.o 3rdparty/kissfft/kiss_fft.o 3rdparty/kissfft/kiss_fftnd.o 3rdparty/kissfft/kiss_fftr.o 3rdparty/kissfft/kiss_fftndr.o 3rdparty/kissfft/kissf_fft.o 3rdparty/kissfft/kissf_fftnd.o 3rdparty/kissfft/kissf_fftr.o 3rdparty/kissfft/kissf_fftndr.o 3rdparty/dsfmt/dSFMT.o 3rdparty/sfmt/SFMT.o ccv_io.o ccv_numeric.o ccv_algebra.o ccv_util.o ccv_basic.o ccv_resample.o ccv_transform.o ccv_classic.o ccv_daisy.o ccv_sift.o ccv_bbf.o ccv_mser.o ccv_swt.o ccv_dpm.o ccv_tld.o ccv_ferns.o ccv_icf.o ccv_convnet.o
	@echo -en $(PURPLE)
	@echo AR $@ $(TO_NULL)
	$(SILENT)$(AR) rcs $@ $^
	@echo -en $(WHITE)

ccv_io.o: ccv_io.c ccv.h ccv_internal.h io/*.c
	@echo CC $< $(TO_NULL)
	$(SILENT)$(CC) $< -o $@ -c $(CFLAGS)

%.o: %.c ccv.h ccv_internal.h
	@echo CC $< $(TO_NULL)
	$(SILENT)$(CC) $< -o $@ -c $(CFLAGS)
