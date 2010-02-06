#ifndef _GUARD_ccv_h_
#define _GUARD_ccv_h_

typedef struct {
	int sig;
	int rows;
	int cols;
	int type;
	union {
		char* ptr;
		int* i;
		float* fl;
		double* db;
	} data;
} ccv_mat_t;

#endif
