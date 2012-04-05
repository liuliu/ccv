#ifndef KISSF_FFTND_H
#define KISSF_FFTND_H

#include "kissf_fft.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kissf_fftnd_state * kissf_fftnd_cfg;
    
kissf_fftnd_cfg  kissf_fftnd_alloc(const int *dims,int ndims,int inverse_fft,void*mem,size_t*lenmem);
void kissf_fftnd(kissf_fftnd_cfg  cfg,const kissf_fft_cpx *fin,kissf_fft_cpx *fout);

#ifdef __cplusplus
}
#endif
#endif
