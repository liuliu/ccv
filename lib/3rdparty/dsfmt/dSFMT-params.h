#ifndef DSFMT_PARAMS_H
#define DSFMT_PARAMS_H

#include "dSFMT.h"

/*----------------------
  the parameters of DSFMT
  following definitions are in dSFMT-paramsXXXX.h file.
  ----------------------*/
/** the pick up position of the array.
#define DSFMT_POS1 122 
*/

/** the parameter of shift left as four 32-bit registers.
#define DSFMT_SL1 18
 */

/** the parameter of shift right as four 32-bit registers.
#define DSFMT_SR1 12
*/

/** A bitmask, used in the recursion.  These parameters are introduced
 * to break symmetry of SIMD.
#define DSFMT_MSK1 (uint64_t)0xdfffffefULL
#define DSFMT_MSK2 (uint64_t)0xddfecb7fULL
*/

/** These definitions are part of a 128-bit period certification vector.
#define DSFMT_PCV1	UINT64_C(0x00000001)
#define DSFMT_PCV2	UINT64_C(0x00000000)
*/

#define DSFMT_LOW_MASK  UINT64_C(0x000FFFFFFFFFFFFF)
#define DSFMT_HIGH_CONST UINT64_C(0x3FF0000000000000)
#define DSFMT_SR	12

/* for sse2 */
#if defined(HAVE_SSE2)
  #define SSE2_SHUFF 0x1b
#elif defined(HAVE_ALTIVEC)
  #if defined(__APPLE__)  /* For OSX */
    #define ALTI_SR (vector unsigned char)(4)
    #define ALTI_SR_PERM \
        (vector unsigned char)(15,0,1,2,3,4,5,6,15,8,9,10,11,12,13,14)
    #define ALTI_SR_MSK \
        (vector unsigned int)(0x000fffffU,0xffffffffU,0x000fffffU,0xffffffffU)
    #define ALTI_PERM \
        (vector unsigned char)(12,13,14,15,8,9,10,11,4,5,6,7,0,1,2,3)
  #else
    #define ALTI_SR      {4}
    #define ALTI_SR_PERM {15,0,1,2,3,4,5,6,15,8,9,10,11,12,13,14}
    #define ALTI_SR_MSK  {0x000fffffU,0xffffffffU,0x000fffffU,0xffffffffU}
    #define ALTI_PERM    {12,13,14,15,8,9,10,11,4,5,6,7,0,1,2,3}
  #endif
#endif

#include "dSFMT-params19937.h"

#endif /* DSFMT_PARAMS_H */
