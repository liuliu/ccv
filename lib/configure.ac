AC_INIT([libccv], [0.6])

AC_SUBST(DEFINE_MACROS, [""])
AC_SUBST(MKCFLAGS, [""])
AC_SUBST(MKLDFLAGS, ["-lm "])
AC_SUBST(CUDA_SRCS, ["cuda/cwc_convnet.cu cuda/cwc_convnet_ext.c cuda/convnet/cwc_convnet_convolutional.cu cuda/convnet/cwc_convnet_rnorm.cu cuda/convnet/cwc_convnet_pool.cu cuda/convnet/cwc_convnet_full_connect.cu"])
AC_SUBST(CUDA_COMPAT_LIB, ["gpu/libnnc-compat-cuda.o"])
AC_SUBST(CUDA_CMD_LIB, ["libnnc-cmd-cuda.o"])

# check if clang exists
AC_CHECK_PROG(CC, clang, clang)

# check if nvcc exists
AC_CHECK_PROG(NVCC, nvcc, nvcc)

# check for ARM NEON support
AC_MSG_CHECKING([neon])
AC_ARG_ENABLE(neon, [AS_HELP_STRING([--enable-neon], [optimize with NEON instruction set])], [neon_support=$enableval], [neon_support="no"])
if test "$neon_support" = yes; then
	AC_SUBST(DEFINE_MACROS, ["$DEFINE_MACROS-D HAVE_NEON "])
	AC_SUBST(MKCFLAGS, ["$MKCFLAGS-mfpu=neon -mfloat-abi=hard "])
	AC_MSG_RESULT(yes)
else
	AC_MSG_RESULT(no)
fi

AC_MSG_CHECKING([arch])
AC_ARG_WITH(arch, [AS_HELP_STRING([--with-arch], [specify a particular architecture to optimize on])], [arch_native=$withval], [arch_native="native"])
if test "$arch_native" = native; then
	AC_MSG_RESULT(native)
else
	AC_SUBST(MKCFLAGS, ["$MKCFLAGS-march=$arch_native -mtune=$arch_native "])
	AC_MSG_RESULT($arch_native)
fi

# check for blas
blas_ok=no
# check for Generic BLAS library
if test "$blas_ok" = no; then
	AC_CHECK_LIB(blas, sgemm, [blas_ok=yes; BLAS_LIBS="-lblas"])
fi
# check for ATLAS library
if test "$blas_ok" = no; then
	AC_CHECK_LIB(atlas, ATL_xerbla,
		[AC_CHECK_LIB(cblas, cblas_dgemm,
		[blas_ok=yes
		BLAS_LIBS="-lcblas -latlas"],
		[], [-latlas])])
fi
if test "$blas_ok" = yes; then
	AC_SUBST(DEFINE_MACROS, ["$DEFINE_MACROS-D HAVE_CBLAS "])
	AC_SUBST(MKLDFLAGS, ["$MKLDFLAGS$BLAS_LIBS "])
fi

AC_CHECK_LIB(rt, clock_gettime,
			[AC_SUBST(MKLDFLAGS, ["$MKLDFLAGS-lrt "])])

# check for libpng, libjpeg, fftw3, liblinear, Accelerate framework, avformat, avcodec, avutil, swscale
AC_CHECK_HEADER([png.h],
				[AC_SUBST(DEFINE_MACROS, ["$DEFINE_MACROS-D HAVE_LIBPNG "]) AC_SUBST(MKLDFLAGS, ["$MKLDFLAGS-lpng "])])
AC_CHECK_HEADER([jpeglib.h],
				[AC_SUBST(DEFINE_MACROS, ["$DEFINE_MACROS-D HAVE_LIBJPEG "]) AC_SUBST(MKLDFLAGS, ["$MKLDFLAGS-ljpeg "])])
AC_CHECK_HEADER([fftw3.h],
				[AC_SUBST(DEFINE_MACROS, ["$DEFINE_MACROS-D HAVE_FFTW3 "]) AC_SUBST(MKLDFLAGS, ["$MKLDFLAGS-lfftw3 -lfftw3f -lpthread "])])
AC_CHECK_HEADER([linear.h],
				[AC_SUBST(DEFINE_MACROS, ["$DEFINE_MACROS-D HAVE_LIBLINEAR "]) AC_SUBST(MKLDFLAGS, ["$MKLDFLAGS-llinear "])])
AC_CHECK_HEADER([tesseract/capi.h],
				[AC_SUBST(DEFINE_MACROS, ["$DEFINE_MACROS-D HAVE_TESSERACT "]) AC_SUBST(MKLDFLAGS, ["$MKLDFLAGS-ltesseract "])])
AC_CHECK_HEADER([Accelerate/Accelerate.h],
				[AC_SUBST(DEFINE_MACROS, ["$DEFINE_MACROS-D HAVE_ACCELERATE_FRAMEWORK "]) AC_SUBST(MKLDFLAGS, ["$MKLDFLAGS-framework Accelerate "])])
AC_CHECK_HEADER([libavcodec/avcodec.h],
				[AC_SUBST(DEFINE_MACROS, ["$DEFINE_MACROS-D HAVE_AVCODEC "]) AC_SUBST(MKLDFLAGS, ["$MKLDFLAGS-lavcodec "])])
AC_CHECK_HEADER([libavformat/avformat.h],
				[AC_SUBST(DEFINE_MACROS, ["$DEFINE_MACROS-D HAVE_AVFORMAT "]) AC_SUBST(MKLDFLAGS, ["$MKLDFLAGS-lavformat "])])
AC_CHECK_HEADER([libavutil/avutil.h],
				[AC_SUBST(DEFINE_MACROS, ["$DEFINE_MACROS-D HAVE_AVUTIL "]) AC_SUBST(MKLDFLAGS, ["$MKLDFLAGS-lavutil "])])
AC_CHECK_HEADER([libswscale/swscale.h],
				[AC_SUBST(DEFINE_MACROS, ["$DEFINE_MACROS-D HAVE_SWSCALE "]) AC_SUBST(MKLDFLAGS, ["$MKLDFLAGS-lswscale "])])
# try to find CPU parallel libraries, OpenMP or libdispatch, we will prefer OpenMP when possible.
AC_OPENMP
if test -n "${OPENMP_CFLAGS}"; then
	AC_SUBST(DEFINE_MACROS, ["$DEFINE_MACROS-D USE_OPENMP "])
	AC_SUBST(MKCFLAGS, ["$MKCFLAGS$OPENMP_CFLAGS "])
	AC_SUBST(MKLDFLAGS, ["$MKLDFLAGS$OPENMP_CFLAGS "])
fi
# only check dispatch if compiled with clang
if test $CC = clang; then
	AC_CHECK_HEADER([dispatch/dispatch.h],
					[AC_SUBST(DEFINE_MACROS, ["$DEFINE_MACROS-D USE_DISPATCH "]) AC_SUBST(MKCFLAGS, ["$MKCFLAGS-fblocks "])])
	AC_CHECK_LIB(dispatch, dispatch_apply, [AC_SUBST(MKLDFLAGS, ["$MKLDFLAGS-ldispatch -lBlocksRuntime "])])
fi
# check for SSE2 support only we don't enable NEON explicitly
if test "$neon_support" != yes; then
	AC_CHECK_HEADER(xmmintrin.h,
					[AC_SUBST(DEFINE_MACROS, ["$DEFINE_MACROS-D HAVE_SSE2 "]) AC_SUBST(MKCFLAGS, ["$MKCFLAGS-msse2 "])])
fi

# check for gsl, and I need to first check these two before I can check gsl
AC_CHECK_LIB(m, cos)
AC_CHECK_LIB(gslcblas, cblas_dgemm)
AC_CHECK_LIB(gsl, gsl_blas_dgemm,
			 [AC_SUBST(DEFINE_MACROS, ["$DEFINE_MACROS-D HAVE_GSL "]) AC_SUBST(MKLDFLAGS, ["$MKLDFLAGS-lgsl -lgslcblas "])])

# prepare for cuda
AC_MSG_CHECKING([cuda])
AC_ARG_WITH(cuda, [AS_HELP_STRING([--with-cuda], [CUDA installation [ARG=/usr/local/cuda]])], [cuda_prefix=$withval], [cuda_prefix="/usr/local/cuda"])
if [[ -d "$cuda_prefix" ]]; then
	AC_SUBST(DEFINE_MACROS, ["$DEFINE_MACROS-D HAVE_CUDA "])
	AC_SUBST(MKLDFLAGS, ["$MKLDFLAGS-lcuda -lcudart -lcublas "])
	AC_SUBST(CPPFLAGS, ["$CPPFLAGS-I$cuda_prefix/include "])
	AC_CHECK_HEADER([cudnn.h],
					[AC_SUBST(DEFINE_MACROS, ["$DEFINE_MACROS-D HAVE_CUDNN "]) AC_SUBST(MKLDFLAGS, ["$MKLDFLAGS-lcudnn "])])
	if [[ -d "$cuda_prefix/lib64" ]]; then
		AC_SUBST(MKLDFLAGS, ["$MKLDFLAGS-L$cuda_prefix/lib64 "])
		AC_MSG_RESULT([yes, x86_64])
	else
		AC_SUBST(MKLDFLAGS, ["$MKLDFLAGS-L$cuda_prefix/lib "])
		AC_MSG_RESULT([yes, i386])
	fi
else
	AC_SUBST(CUDA_SRCS, [""])
	AC_SUBST(CUDA_COMPAT_LIB, [""])
	AC_SUBST(CUDA_CMD_LIB, [""])
	AC_MSG_RESULT(no)
fi

# for display only, concatenate CFLAGS and DEFINE_MACROS
AC_SUBST(DISCFLAGS, ["$MKCFLAGS$DEFINE_MACROS"])

AC_SUBST(MKCFLAGS, ["$MKCFLAGS\$(DEFINE_MACROS)"])

AC_SUBST(DEFINE_MACROS, ["${DEFINE_MACROS#"${DEFINE_MACROS%%@<:@!@<:@:space:@:>@@:>@*}"}"]) # remove leading whitespaces
AC_SUBST(DEFINE_MACROS, ["${DEFINE_MACROS%"${DEFINE_MACROS@%:@@%:@*@<:@!@<:@:space:@:>@@:>@}"}"]) # remove trailing whitespaces
AC_SUBST(MKCFLAGS, ["${MKCFLAGS#"${MKCFLAGS%%@<:@!@<:@:space:@:>@@:>@*}"}"])
AC_SUBST(MKCFLAGS, ["${MKCFLAGS%"${MKCFLAGS@%:@@%:@*@<:@!@<:@:space:@:>@@:>@}"}"])
AC_SUBST(MKLDFLAGS, ["${MKLDFLAGS#"${MKLDFLAGS%%@<:@!@<:@:space:@:>@@:>@*}"}"])
AC_SUBST(MKLDFLAGS, ["${MKLDFLAGS%"${MKLDFLAGS@%:@@%:@*@<:@!@<:@:space:@:>@@:>@}"}"])
AC_SUBST(DISCFLAGS, ["${DISCFLAGS#"${DISCFLAGS%%@<:@!@<:@:space:@:>@@:>@*}"}"])
AC_SUBST(DISCFLAGS, ["${DISCFLAGS%"${DISCFLAGS@%:@@%:@*@<:@!@<:@:space:@:>@@:>@}"}"])

echo -e "\n \033@<:@1mUSE\033@<:@m: \033@<:@4m$CC\033@<:@m\n \033@<:@1mCOMPILE FLAGS\033@<:@m: \033@<:@4m$DISCFLAGS\033@<:@m\n \033@<:@1mLINK FLAGS\033@<:@m: \033@<:@4m$MKLDFLAGS\033@<:@m\n"

echo $MKLDFLAGS > .deps

AC_CONFIG_FILES(config.mk)

AC_OUTPUT
