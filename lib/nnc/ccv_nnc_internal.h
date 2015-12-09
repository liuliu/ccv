/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

/**********************************************************
 * CCV - Neural Network Collection
 **********************************************************/

#ifndef GUARD_ccv_nnc_internal_h
#define GUARD_ccv_nnc_internal_h

// This dim_for supports up to 4 dim, you can easily modify it to support more though, and the change is transparent to the rest of the program.

#define dim_for(__i, __c, __dim, block) \
	{ \
		int __i[__c]; \
		switch (__c) { \
			case 4: \
				for (__i[3] = 0; __i[3] < __dim[3]; __i[3]++) { \
					block(3); \
			case 3: \
				for (__i[2] = 0; __i[2] < __dim[2]; __i[2]++) { \
					block(2); \
			case 2: \
				for (__i[1] = 0; __i[1] < __dim[1]; __i[1]++) { \
					block(1); \
			case 1: \
				for (__i[0] = 0; __i[0] < __dim[0]; __i[0]++) { \
					block(0);

#define dim_endfor(__c, block) \
					block(0); \
				} \
				if (__c < 2) break; \
					block(1); \
				} \
				if (__c < 3) break; \
					block(2); \
				} \
				if (__c < 4) break; \
					block(3); \
				} \
		} \
	}

#endif
