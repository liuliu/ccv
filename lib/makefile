include config.mk

CFLAGS := -O3 -ffast-math -Wall -I"." $(CFLAGS)
NVFLAGS := -O3 $(NVFLAGS)

SRCS := ccv_cache.c ccv_memory.c 3rdparty/siphash/siphash24.c 3rdparty/kissfft/kiss_fft.c 3rdparty/kissfft/kiss_fftnd.c 3rdparty/kissfft/kiss_fftr.c 3rdparty/kissfft/kiss_fftndr.c 3rdparty/kissfft/kissf_fft.c 3rdparty/kissfft/kissf_fftnd.c 3rdparty/kissfft/kissf_fftr.c 3rdparty/kissfft/kissf_fftndr.c 3rdparty/dsfmt/dSFMT.c 3rdparty/sfmt/SFMT.c 3rdparty/sqlite3/sqlite3.c ccv_io.c ccv_numeric.c ccv_algebra.c ccv_util.c ccv_basic.c ccv_image_processing.c ccv_resample.c ccv_transform.c ccv_classic.c ccv_daisy.c ccv_sift.c ccv_bbf.c ccv_mser.c ccv_swt.c ccv_dpm.c ccv_tld.c ccv_ferns.c ccv_icf.c ccv_scd.c ccv_convnet.c ccv_output.c

SRC_OBJS := $(patsubst %.c,%.o,$(SRCS))

CUDA_OBJS := $(patsubst %.c,%.o,$(patsubst %.cu,%.o,$(CUDA_SRCS)))

# Make nnc/libnnc.o a phony target therefore it will be triggered every time.
.PHONY: release all lib clean dep nnc/libnnc.o

release: all

include scheme.mk

all: lib ../samples/image-net-2012-vgg-d.sqlite3

lib: libccv.a

clean:
	${MAKE} -C ./nnc clean && rm -f $(SRC_OBJS) $(CUDA_OBJS) libccv.a

libccv.a: $(SRC_OBJS) $(CUDA_OBJS) nnc/libnnc.o
	$(AR) rcs $@ $^

nnc/libnnc.o:
	${MAKE} -C ./nnc

3rdparty/sqlite3/sqlite3.o: 3rdparty/sqlite3/sqlite3.c
	$(CC) $< -o $@ -c -O3 -D HAVE_USLEEP -D SQLITE_DQS=0 -D SQLITE_DEFAULT_CACHE_SIZE=128 -D SQLITE_DEFAULT_CKPTFULLFSYNC -D SQLITE_DEFAULT_JOURNAL_SIZE_LIMIT=32768 -D SQLITE_DEFAULT_PAGE_SIZE=4096 -D SQLITE_DEFAULT_SYNCHRONOUS=2 -D SQLITE_DEFAULT_WAL_SYNCHRONOUS=1 -D SQLITE_ENABLE_API_ARMOR -D SQLITE_ENABLE_COLUMN_METADATA -D SQLITE_ENABLE_DBSTAT_VTAB -D SQLITE_ENABLE_FTS3 -D SQLITE_ENABLE_FTS3_PARENTHESIS -D SQLITE_ENABLE_FTS3_TOKENIZER -D SQLITE_ENABLE_FTS4 -D SQLITE_ENABLE_FTS5 -D SQLITE_ENABLE_JSON1 -D SQLITE_ENABLE_MATH_FUNCTIONS -D SQLITE_ENABLE_PREUPDATE_HOOK -D SQLITE_ENABLE_RTREE -D SQLITE_ENABLE_SESSION -D SQLITE_ENABLE_SNAPSHOT -D SQLITE_ENABLE_STMT_SCANSTATUS -D SQLITE_ENABLE_UNKNOWN_SQL_FUNCTION -D SQLITE_ENABLE_UPDATE_DELETE_LIMIT -D SQLITE_HAS_CODEC_RESTRICTED -D SQLITE_HAVE_ISNAN -D SQLITE_MAX_LENGTH=2147483645 -D SQLITE_MAX_MMAP_SIZE=20971520 -D SQLITE_MAX_VARIABLE_NUMBER=500000 -D SQLITE_OMIT_AUTORESET -D SQLITE_OMIT_DEPRECATED -D SQLITE_OMIT_PROGRESS_CALLBACK -D SQLITE_OMIT_LOAD_EXTENSION -D SQLITE_STMTJRNL_SPILL=131072 -D SQLITE_SUBSTR_COMPATIBILITY -D SQLITE_THREADSAFE=2 -D SQLITE_USE_URI

../samples/image-net-2012-vgg-d.sqlite3:
	../samples/download-vgg-d-model.sh

%.o: %.c
	$(CC) $< -o $@ -c $(CFLAGS)

%.o: %.cu
	$(NVCC) $< -o $@ -c $(NVFLAGS)

dep: .dep.mk
.dep.mk: $(SRCS) $(CUDA_SRCS)
	echo '' > .dep.mk
	for SRC in $(patsubst %.cu,,$^) ; do \
		$(CC) $(CFLAGS) -MM $$SRC | sed -e 's/^.*\://g' | (echo "$${SRC%%.*}.o: \\" && cat) >> .dep.mk ; \
	done
ifneq ($(CUDA_SRCS),)
	for CUDA_SRC in $(patsubst %.c,,$^) ; do \
		$(NVCC) $(NVFLAGS) -M $$CUDA_SRC | sed -e 's/^.*\://g' | (echo "$${CUDA_SRC%%.*}.o: \\" && cat) >> .dep.mk ; \
	done
endif

-include .dep.mk
