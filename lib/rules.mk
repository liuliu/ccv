OBJECTS := \
	$(b)3rdparty/dsfmt/dSFMT.o \
	$(b)3rdparty/kissfft/kissf_fftnd.o \
	$(b)3rdparty/kissfft/kissf_fftndr.o \
	$(b)3rdparty/kissfft/kissf_fft.o \
	$(b)3rdparty/kissfft/kissf_fftr.o \
	$(b)3rdparty/kissfft/kiss_fftnd.o \
	$(b)3rdparty/kissfft/kiss_fftndr.o \
	$(b)3rdparty/kissfft/kiss_fft.o \
	$(b)3rdparty/kissfft/kiss_fftr.o \
	$(b)3rdparty/sfmt/SFMT.o \
	$(b)3rdparty/sha1/sha1.o \
	$(b)ccv_algebra.o \
	$(b)ccv_basic.o \
	$(b)ccv_bbf.o \
	$(b)ccv_cache.o \
	$(b)ccv_classic.o \
	$(b)ccv_daisy.o \
	$(b)ccv_dpm.o \
	$(b)ccv_ferns.o \
	$(b)ccv_io.o \
	$(b)ccv_memory.o \
	$(b)ccv_mser.o \
	$(b)ccv_numeric.o \
	$(b)ccv_resample.o \
	$(b)ccv_sift.o \
	$(b)ccv_swt.o \
	$(b)ccv_tld.o \
	$(b)ccv_transform.o \
	$(b)ccv_util.o \

TGT_DIR += \
	$(b)3rdparty/dsfmt/ \
	$(b)3rdparty/kissfft \
	$(b)3rdparty/sfmt \
	$(b)3rdparty/sha1 \

$(LIBCCV_PATH): $(OBJECTS)
