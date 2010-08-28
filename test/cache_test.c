#include "ccv.h"

uint64_t uniqid()
{
	union {
		uint64_t u;
		uint8_t chr[8];
	} sign;
	int i;
	for (i = 0; i < 8; i++)
		sign.chr[i] = rand() & 0xff;
	return sign.u;
}

#define N (500000)

int main(int argc, char** argv)
{
	ccv_cache_t cache;
	ccv_cache_init(&cache);
	uint64_t sigs[N];
	void* mems[N];
	int i;
	for (i = 0; i < N; i++)
	{
		sigs[i] = uniqid();
		mems[i] = malloc(1);
		ccv_cache_put(&cache, sigs[i], mems[i]);
	}
	uint8_t deleted[N];
	for (i = 0; i < N; i++)
	{
		deleted[i] = 1;
		if (deleted[i])
			ccv_cache_delete(&cache, sigs[i]);
	}
	for (i = 0; i < N; i++)
	{
		deleted[i] = (rand() % 3 == 0);
		if (!deleted[i])
		{
			mems[i] = malloc(1);
			ccv_cache_put(&cache, sigs[i], mems[i]);
		}
	}
	for (i = 0; i < N; i++)
	{
		deleted[i] = (rand() % 3 == 0);
		if (deleted[i])
			ccv_cache_delete(&cache, sigs[i]);
		else {
			mems[i] = malloc(1);
			ccv_cache_put(&cache, sigs[i], mems[i]);
		}
	}
	for (i = 0; i < N; i++)
	{
		ccv_matrix_t* x = ccv_cache_get(&cache, sigs[i]);
		if (!deleted[i])
		{
			assert(x);
			assert(mems[i] == x);
		} else
	 		assert(x == 0);
	}
	ccv_cache_close(&cache);
	return 0;
}
