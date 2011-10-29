#include "ccv.h"

void ccv_cache_init(ccv_cache_t* cache, ccv_cache_index_free_f ffree, size_t up, uint32_t cnum, uint32_t wnum)
{
	cache->rnum = 0;
	cache->cnum = ccv_min(1 << 16, ccv_max(cnum, 4));
	cache->wnum = wnum;
	cache->inum = 0;
	cache->g = 0;
	cache->up = up;
	cache->size = 0;
	cache->ffree = ffree;
	while ((1 << cache->inum) < cache->cnum + 1)
		++(cache->inum);
	cache->way = (ccv_cache_index_t*)ccmalloc(cache->cnum * cache->wnum * sizeof(ccv_cache_index_t));
	memset(cache->way, 0, cache->cnum * cache->wnum * sizeof(ccv_cache_index_t));
}

void* ccv_cache_get(ccv_cache_t* cache, uint64_t sign)
{
	int i;
	ccv_cache_index_t* way = cache->way;
	for (i = 0; i < cache->wnum; i++)
	{
		uint32_t k = ((sign >> (cache->inum * i)) & ((1 << cache->inum) - 1)) % cache->cnum;
		if (way[k].sign == sign)
			return way[k].off;
		way += cache->cnum;
	}
	return 0;
}

// only call this function when the cache space is delpeted
static void ccv_cache_depleted(ccv_cache_t* cache)
{
	int i;
	ccv_cache_index_t* way = cache->way;
	for (i = 0; i < cache->cnum * cache->wnum; i++)
	{
		if (way[i].off)
		{
			cache->ffree(way[i].off);
			cache->size -= way[i].size;
			way[i].sign = 0;
			way[i].day = 0;
			way[i].off = 0;
			way[i].size = 0;
			--cache->rnum;
			if (cache->size <= cache->up)
				break;
		}
	}
}

int ccv_cache_put(ccv_cache_t* cache, uint64_t sign, void* x, size_t size)
{
	int i, c = -1;
	uint32_t g = ++cache->g;
	++cache->rnum;
	cache->size += size;
	ccv_cache_index_t* way = cache->way;
	uint64_t day = 0;
	uint64_t m = (1 << cache->inum) - 1;
	for (i = 0; i < cache->wnum; i++)
	{
		uint32_t k = (uint32_t)(((sign >> (cache->inum * i)) & m) % cache->cnum);
		if (way[k].sign == sign)
		{
			cache->size -= way[k].size;
			way[k].day = g;
			if (way[k].off && way[k].off != x)
				cache->ffree(way[k].off);
			way[k].off = x;
			way[k].sign = sign;
			way[k].size = size;
			if (cache->size > cache->up)
				ccv_cache_depleted(cache);
			return 1;
		} else if (c < 0 || way[k].day < day) {
			c = k + i * cache->cnum;
			day = way[k].day;
		}
		way += cache->cnum;
	}
	if (c >= 0)
	{
		way = cache->way + c;
		cache->size -= way->size;
		if (way->off && way->off != x)
			cache->ffree(way->off);
		way->off = x;
		way->sign = sign;
		way->day = g;
		way->size = size;
		if (cache->size > cache->up)
			ccv_cache_depleted(cache);
		return 0;
	}
	return -1;
}

void* ccv_cache_out(ccv_cache_t* cache, uint64_t sign)
{
	int i;
	ccv_cache_index_t* way = cache->way;
	for (i = 0; i < cache->wnum; i++)
	{
		uint32_t k = ((sign >> (cache->inum * i)) & ((1 << cache->inum) - 1)) % cache->cnum;
		if (way[k].sign == sign)
		{
			void* off = way[k].off;
			cache->size -= way[k].size;
			way[k].sign = 0;
			way[k].day = 0;
			way[k].off = 0;
			way[k].size = 0;
			--cache->rnum;
			return off;
		}
		way += cache->cnum;
	}
	return 0;
}

int ccv_cache_delete(ccv_cache_t* cache, uint64_t sign)
{
	int i;
	ccv_cache_index_t* way = cache->way;
	for (i = 0; i < cache->wnum; i++)
	{
		uint32_t k = ((sign >> (cache->inum * i)) & ((1 << cache->inum) - 1)) % cache->cnum;
		if (way[k].sign == sign)
		{
			if (way[k].off)
				cache->ffree(way[k].off);
			cache->size -= way[k].size;
			way[k].sign = 0;
			way[k].day = 0;
			way[k].off = 0;
			way[k].size = 0;
			--cache->rnum;
			return 1;
		}
		way += cache->cnum;
	}
	return 0;
}

void ccv_cache_cleanup(ccv_cache_t* cache)
{
	int i;
	ccv_cache_index_t* way = cache->way;
	for (i = 0; (i < cache->cnum * cache->wnum && cache->rnum > 0); i++)
	{
		if (way[i].off)
		{
			cache->ffree(way[i].off);
			way[i].sign = 0;
			way[i].day = 0;
			way[i].off = 0;
			way[i].size = 0;
			--cache->rnum;
		}
	}
	cache->g = 0;
	cache->size = 0;
}

void ccv_cache_close(ccv_cache_t* cache)
{
	int i;
	for (i = 0; (i < cache->cnum * cache->wnum && cache->rnum > 0); i++)
	{
		if (cache->way[i].off != 0)
		{
			cache->ffree(cache->way[i].off);
			--cache->rnum;
		}
	}
	ccfree(cache->way);
}
