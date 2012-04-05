#ifndef KISSF_FTR_H
#define KISSF_FTR_H

#include "kissf_fft.h"
#ifdef __cplusplus
extern "C" {
#endif

    
/* 
 
 Real optimized version can save about 45% cpu time vs. complex fft of a real seq.

 
 
 */

typedef struct kissf_fftr_state *kissf_fftr_cfg;


kissf_fftr_cfg kissf_fftr_alloc(int nfft,int inverse_fft,void * mem, size_t * lenmem);
/*
 nfft must be even

 If you don't care to allocate space, use mem = lenmem = NULL 
*/


void kissf_fftr(kissf_fftr_cfg cfg,const kissf_fft_scalar *timedata,kissf_fft_cpx *freqdata);
/*
 input timedata has nfft scalar points
 output freqdata has nfft/2+1 complex points
*/

void kissf_fftri(kissf_fftr_cfg cfg,const kissf_fft_cpx *freqdata,kissf_fft_scalar *timedata);
/*
 input freqdata has  nfft/2+1 complex points
 output timedata has nfft scalar points
*/

#define kissf_fftr_free free

#ifdef __cplusplus
}
#endif
#endif
