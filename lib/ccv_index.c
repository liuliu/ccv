#include "ccv.h"

typedef union {
	struct {
		uint64_t bitmap;
		uint64_t set;
	} branch;
	struct {
		uint64_t off;
		uint64_t sign;
	} terminal;
} ccv_cache_index_t;

typedef struct {
	ccv_cache_index_t origin;
	uint32_t rnum;
} ccv_cache_t;

static ccv_cache_index_t* ccv_cache_seek(ccv_cache_index_t* branch, uint8_t* key)
{
}

void ccv_cache_get()
{
}

void ccv_cache_put()
{
}

void ccv_cache_delete()
{
}
