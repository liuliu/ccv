#include "ccv.h"
#include "case.h"

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

#define N (250000)

TEST_CASE("cache test")
{
	ccv_cache_t cache;
	ccv_cache_init(&cache, ccfree, N);
	uint64_t sigs[N];
	void* mems[N];
	int i;
	for (i = 0; i < N; i++)
	{
		sigs[i] = uniqid();
		mems[i] = ccmalloc(1);
		ccv_cache_put(&cache, sigs[i], mems[i], 1);
	 	REQUIRE_EQ(i, cache.size, "at %d should has cache size %d", i, i);
	}
	uint8_t deleted[N];
	for (i = 0; i < N; i++)
	{
		deleted[i] = 1;
		if (deleted[i])
			ccv_cache_delete(&cache, sigs[i]);
	 	REQUIRE_EQ(N - 1 - i, cache.size, "at %d should has cache size %d", i, N - 1 - i);
	}
	for (i = 0; i < N; i++)
	{
		deleted[i] = (rand() % 3 == 0);
		if (!deleted[i])
		{
			mems[i] = ccmalloc(1);
			ccv_cache_put(&cache, sigs[i], mems[i], 1);
		}
	}
	for (i = 0; i < N; i++)
	{
		deleted[i] = (rand() % 3 == 0);
		if (deleted[i])
			ccv_cache_delete(&cache, sigs[i]);
		else {
			mems[i] = ccmalloc(1);
			ccv_cache_put(&cache, sigs[i], mems[i], 1);
		}
	}
	for (i = 0; i < N; i++)
	{
		void* x = ccv_cache_get(&cache, sigs[i]);
		if (!deleted[i] && x) // x may be pull off the cache
		{
			REQUIRE_EQ((uint64_t)mems[i], (uint64_t)x, "value at %d should be consistent", i);
		} else
	 		REQUIRE_EQ(0, (uint64_t)x, "at %d should not exist", i);
	}
	ccv_cache_close(&cache);
}

TEST_CASE("garbage collector test")
{
	int i;
	// deliberately let only cache size fits 90% of data
	ccv_enable_cache(44 * N * 90 / 100);
	for (i = 0; i < N; i++)
	{
		ccv_dense_matrix_t* dmt = ccv_dense_matrix_new(1, 1, CCV_32S | CCV_C1, 0, 0);
		dmt->data.i[0] = i;
		dmt->sig = ccv_matrix_generate_signature((const char*)&i, 4, 0);
		dmt->type |= CCV_REUSABLE;
		ccv_matrix_free(dmt);
	}
	int percent = 0, total = 0;
	for (i = N - 1; i > N * 6 / 100; i--)
	{
		uint64_t sig = ccv_matrix_generate_signature((const char*)&i, 4, 0);
		ccv_dense_matrix_t* dmt = ccv_dense_matrix_new(1, 1, CCV_32S | CCV_C1, 0, sig);
		if (i == dmt->data.i[0])
			++percent;
		++total;
		ccv_matrix_free_immediately(dmt);
	}
	REQUIRE((double)percent / (double)total > 0.95, "the cache hit (%lf) should be greater than 95%%", (double)percent / (double)total);
	ccv_disable_cache();
}

#include "case_main.h"
