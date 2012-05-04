#ifndef KISSF_FFT_H
#define KISSF_FFT_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 ATTENTION!
 If you would like a :
 -- a utility that will handle the caching of fft objects
 -- real-only (no imaginary time component ) FFT
 -- a multi-dimensional FFT
 -- a command-line utility to perform ffts
 -- a command-line utility to perform fast-convolution filtering

 Then see kfc.h kissf_fftr.h kissf_fftnd.h fftutil.c kissf_fastfir.c
  in the tools/ directory.
*/

#ifdef USE_SIMD
# include <xmmintrin.h>
# define kissf_fft_scalar __m128
#define KISSF_FFT_MALLOC(nbytes) _mm_malloc(nbytes,16)
#define KISSF_FFT_FREE _mm_free
#else	
#define KISSF_FFT_MALLOC malloc
#define KISSF_FFT_FREE free
#endif	


/*  default is double */
#define kissf_fft_scalar float

typedef struct {
    kissf_fft_scalar r;
    kissf_fft_scalar i;
}kissf_fft_cpx;

typedef struct kissf_fft_state* kissf_fft_cfg;

/* 
 *  kissf_fft_alloc
 *  
 *  Initialize a FFT (or IFFT) algorithm's cfg/state buffer.
 *
 *  typical usage:      kissf_fft_cfg mycfg=kissf_fft_alloc(1024,0,NULL,NULL);
 *
 *  The return value from fft_alloc is a cfg buffer used internally
 *  by the fft routine or NULL.
 *
 *  If lenmem is NULL, then kissf_fft_alloc will allocate a cfg buffer using malloc.
 *  The returned value should be free()d when done to avoid memory leaks.
 *  
 *  The state can be placed in a user supplied buffer 'mem':
 *  If lenmem is not NULL and mem is not NULL and *lenmem is large enough,
 *      then the function places the cfg in mem and the size used in *lenmem
 *      and returns mem.
 *  
 *  If lenmem is not NULL and ( mem is NULL or *lenmem is not large enough),
 *      then the function returns NULL and places the minimum cfg 
 *      buffer size in *lenmem.
 * */

kissf_fft_cfg kissf_fft_alloc(int nfft,int inverse_fft,void * mem,size_t * lenmem); 

/*
 * kissf_fft(cfg,in_out_buf)
 *
 * Perform an FFT on a complex input buffer.
 * for a forward FFT,
 * fin should be  f[0] , f[1] , ... ,f[nfft-1]
 * fout will be   F[0] , F[1] , ... ,F[nfft-1]
 * Note that each element is complex and can be accessed like
    f[k].r and f[k].i
 * */
void kissf_fft(kissf_fft_cfg cfg,const kissf_fft_cpx *fin,kissf_fft_cpx *fout);

/*
 A more generic version of the above function. It reads its input from every Nth sample.
 * */
void kissf_fft_stride(kissf_fft_cfg cfg,const kissf_fft_cpx *fin,kissf_fft_cpx *fout,int fin_stride);

/* If kissf_fft_alloc allocated a buffer, it is one contiguous 
   buffer and can be simply free()d when no longer needed*/
#define kissf_fft_free free

/*
 Cleans up some memory that gets managed internally. Not necessary to call, but it might clean up 
 your compiler output to call this before you exit.
*/
void kissf_fft_cleanup(void);
	

/*
 * Returns the smallest integer k, such that k>=n and k has only "fast" factors (2,3,5)
 */
int kissf_fft_next_fast_size(int n);

/* for real ffts, we need an even size */
#define kissf_fftr_next_fast_size_real(n) \
        (kissf_fft_next_fast_size( ((n)+1)>>1)<<1)

#ifdef __cplusplus
} 
#endif

#endif
