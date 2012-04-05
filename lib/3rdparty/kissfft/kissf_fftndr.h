#ifndef KISSF_NDR_H
#define KISSF_NDR_H

#include "kissf_fft.h"
#include "kissf_fftr.h"
#include "kissf_fftnd.h"

#ifdef __cplusplus
extern "C" {
#endif
    
typedef struct kissf_fftndr_state *kissf_fftndr_cfg;


kissf_fftndr_cfg  kissf_fftndr_alloc(const int *dims,int ndims,int inverse_fft,void*mem,size_t*lenmem);
/*
 dims[0] must be even

 If you don't care to allocate space, use mem = lenmem = NULL 
*/


void kissf_fftndr(
        kissf_fftndr_cfg cfg,
        const kissf_fft_scalar *timedata,
        kissf_fft_cpx *freqdata);
/*
 input timedata has dims[0] X dims[1] X ... X  dims[ndims-1] scalar points
 output freqdata has dims[0] X dims[1] X ... X  dims[ndims-1]/2+1 complex points
*/

void kissf_fftndri(
        kissf_fftndr_cfg cfg,
        const kissf_fft_cpx *freqdata,
        kissf_fft_scalar *timedata);
/*
 input and output dimensions are the exact opposite of kissf_fftndr
*/


#define kissf_fftr_free free

#ifdef __cplusplus
}
#endif

#endif
