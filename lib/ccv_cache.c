#include "ccv.h"
#include "ccv_internal.h"

#define CCV_GET_CACHE_TYPE(x) ((x) >> 60)
#define CCV_GET_TERMINAL_AGE(x) (((x) >> 32) & 0x0FFFFFFF)
#define CCV_GET_TERMINAL_SIZE(x) ((x) & 0xFFFFFFFF)
#define CCV_SET_TERMINAL_TYPE(x, y, z) (((uint64_t)(x) << 60) | ((uint64_t)(y) << 32) | (z))

void ccv_cache_init(ccv_cache_t* cache, size_t up, int cache_types, ccv_cache_index_free_f ffree, ...)
{
	cache->rnum = 0;
	cache->age = 0;
	cache->up = up;
	cache->size = 0;
	assert(cache_types > 0 && cache_types <= 16);
	va_list arguments;
	va_start(arguments, ffree);
	int i;
	cache->ffree[0] = ffree;
	for (i = 1; i < cache_types; i++)
		cache->ffree[i] = va_arg(arguments, ccv_cache_index_free_f);
	va_end(arguments);
	memset(&cache->origin, 0, sizeof(ccv_cache_index_t));
}

static int bits_in_16bits[0x1u << 16];
static int bits_in_16bits_init = 0;

static int sparse_bitcount(unsigned int n) {
	int count = 0;
	while (n) {
		count++;
		n &= (n - 1);
	}
	return count;
}

static void precomputed_16bits() {
	int i;
	for (i = 0; i < (0x1u << 16); i++)
		bits_in_16bits[i] = sparse_bitcount(i);
	bits_in_16bits_init = 1;
}

static uint32_t compute_bits(uint64_t m) {
	return (bits_in_16bits[m & 0xffff] + bits_in_16bits[(m >> 16) & 0xffff] +
			bits_in_16bits[(m >> 32) & 0xffff] + bits_in_16bits[(m >> 48) & 0xffff]);
}

/* update age along a path in the radix tree */
static void _ccv_cache_aging(ccv_cache_index_t* branch, uint64_t sign)
{
	if (!bits_in_16bits_init)
		precomputed_16bits();
	int i;
	uint64_t j = 63;
	ccv_cache_index_t* breadcrumb[10];
	for (i = 0; i < 10; i++)
	{
		breadcrumb[i] = branch;
		int leaf = branch->terminal.off & 0x1;
		int full = branch->terminal.off & 0x2;
		if (leaf)
		{
			break;
		} else {
			ccv_cache_index_t* set = (ccv_cache_index_t*)(branch->branch.set - (branch->branch.set & 0x3));
			int dice = (sign & j) >> (i * 6);
			if (full)
			{
				branch = set + dice;
			} else {
				uint64_t k = 1;
				k = k << dice;
				if (k & branch->branch.bitmap) {
					uint64_t m = (k - 1) & branch->branch.bitmap;
					branch = set + compute_bits(m);
				} else {
					break;
				}
			}
			j <<= 6;
		}
	}
	assert(i < 10);
	for (; i >= 0; i--)
	{
		branch = breadcrumb[i];
		int leaf = branch->terminal.off & 0x1;
		if (!leaf)
		{
			ccv_cache_index_t* set = (ccv_cache_index_t*)(branch->branch.set - (branch->branch.set & 0x3));
			uint32_t total = compute_bits(branch->branch.bitmap);
			uint32_t min_age = (set[0].terminal.off & 0x1) ? CCV_GET_TERMINAL_AGE(set[0].terminal.type) : set[0].branch.age;
			for (j = 1; j < total; j++)
			{
				uint32_t age = (set[j].terminal.off & 0x1) ? CCV_GET_TERMINAL_AGE(set[j].terminal.type) : set[j].branch.age;
				if (age < min_age)
					min_age = age;
			}
			branch->branch.age = min_age;
		}
	}
}

static ccv_cache_index_t* _ccv_cache_seek(ccv_cache_index_t* branch, uint64_t sign, int* depth)
{
	if (!bits_in_16bits_init)
		precomputed_16bits();
	int i;
	uint64_t j = 63;
	for (i = 0; i < 10; i++)
	{
		int leaf = branch->terminal.off & 0x1;
		int full = branch->terminal.off & 0x2;
		if (leaf)
		{
			if (depth)
				*depth = i;
			return branch;
		} else {
			ccv_cache_index_t* set = (ccv_cache_index_t*)(branch->branch.set - (branch->branch.set & 0x3));
			int dice = (sign & j) >> (i * 6);
			if (full)
			{
				branch = set + dice;
			} else {
				uint64_t k = 1;
				k = k << dice;
				if (k & branch->branch.bitmap) {
					uint64_t m = (k - 1) & branch->branch.bitmap;
					branch = set + compute_bits(m);
				} else {
					if (depth)
						*depth = i;
					return branch;
				}
			}
			j <<= 6;
		}
	}
	return 0;
}

void* ccv_cache_get(ccv_cache_t* cache, uint64_t sign, uint8_t* type)
{
	if (cache->rnum == 0)
		return 0;
	ccv_cache_index_t* branch = _ccv_cache_seek(&cache->origin, sign, 0);
	if (!branch)
		return 0;
	int leaf = branch->terminal.off & 0x1;
	if (!leaf)
		return 0;
	if (branch->terminal.sign != sign)
		return 0;
	if (type)
		*type = CCV_GET_CACHE_TYPE(branch->terminal.type);
	return (void*)(branch->terminal.off - (branch->terminal.off & 0x3));
}

// only call this function when the cache space is delpeted
static void _ccv_cache_lru(ccv_cache_t* cache)
{
	ccv_cache_index_t* branch = &cache->origin;
	int leaf = branch->terminal.off & 0x1;
	if (leaf)
	{
		void* result = (void*)(branch->terminal.off - (branch->terminal.off & 0x3));
		uint8_t type = CCV_GET_CACHE_TYPE(branch->terminal.type);
		if (result != 0)
		{
			assert(type >= 0 && type < 16);
			cache->ffree[type](result);
		}
		cache->rnum = 0;
		cache->size = 0;
		return;
	}
	uint32_t min_age = branch->branch.age;
	int i, j;
	for (i = 0; i < 10; i++)
	{
		ccv_cache_index_t* old_branch = branch;
		int leaf = branch->terminal.off & 0x1;
		if (leaf)
		{
			ccv_cache_delete(cache, branch->terminal.sign);
			break;
		} else {
			ccv_cache_index_t* set = (ccv_cache_index_t*)(branch->branch.set - (branch->branch.set & 0x3));
			uint32_t total = compute_bits(branch->branch.bitmap);
			for (j = 0; j < total; j++)
			{
				uint32_t age = (set[j].terminal.off & 0x1) ? CCV_GET_TERMINAL_AGE(set[j].terminal.type) : set[j].branch.age;
				assert(age >= min_age);
				if (age == min_age)
				{
					branch = set + j;
					break;
				}
			}
			assert(old_branch != branch);
		}
	}
	assert(i < 10);
}

static void _ccv_cache_depleted(ccv_cache_t* cache, size_t size)
{
	while (cache->size > size)
		_ccv_cache_lru(cache);
}

int ccv_cache_put(ccv_cache_t* cache, uint64_t sign, void* x, uint32_t size, uint8_t type)
{
	assert(((uint64_t)x & 0x3) == 0);
	if (size > cache->up)
		return -1;
	if (size + cache->size > cache->up)
		_ccv_cache_depleted(cache, cache->up - size);
	if (cache->rnum == 0)
	{
		cache->age = 1;
		cache->origin.terminal.off = (uint64_t)x | 0x1;
		cache->origin.terminal.sign = sign;
		cache->origin.terminal.type = CCV_SET_TERMINAL_TYPE(type, cache->age, size);
		cache->size = size;
		cache->rnum = 1;
		return 0;
	}
	++cache->age;
	int i, depth = -1;
	ccv_cache_index_t* branch = _ccv_cache_seek(&cache->origin, sign, &depth);
	if (!branch)
		return -1;
	int leaf = branch->terminal.off & 0x1;
	uint64_t on = 1;
	assert(depth >= 0);
	if (leaf)
	{
		if (sign == branch->terminal.sign)
		{
			cache->ffree[CCV_GET_CACHE_TYPE(branch->terminal.type)]((void*)(branch->terminal.off - (branch->terminal.off & 0x3)));
			branch->terminal.off = (uint64_t)x | 0x1;
			uint32_t old_size = CCV_GET_TERMINAL_SIZE(branch->terminal.type);
			cache->size = cache->size + size - old_size;
			branch->terminal.type = CCV_SET_TERMINAL_TYPE(type, cache->age, size);
			_ccv_cache_aging(&cache->origin, sign);
			return 1;
		} else {
			ccv_cache_index_t t = *branch;
			uint32_t age = CCV_GET_TERMINAL_AGE(branch->terminal.type);
			uint64_t j = 63;
			j = j << (depth * 6);
			int dice, udice;
			assert(depth < 10);
			for (i = depth; i < 10; i++)
			{
				dice = (t.terminal.sign & j) >> (i * 6);
				udice = (sign & j) >> (i * 6);
				if (dice == udice)
				{
					branch->branch.bitmap = on << dice;
					ccv_cache_index_t* set = (ccv_cache_index_t*)ccmalloc(sizeof(ccv_cache_index_t));
					assert(((uint64_t)set & 0x3) == 0);
					branch->branch.set = (uint64_t)set;
					branch->branch.age = age;
					branch = set;
				} else {
					break;
				}
				j <<= 6;
			}
			branch->branch.bitmap = (on << dice) | (on << udice);
			ccv_cache_index_t* set = (ccv_cache_index_t*)ccmalloc(sizeof(ccv_cache_index_t) * 2);
			assert(((uint64_t)set & 0x3) == 0);
			branch->branch.set = (uint64_t)set;
			branch->branch.age = age;
			int u = dice < udice;
			set[u].terminal.sign = sign;
			set[u].terminal.off = (uint64_t)x | 0x1;
			set[u].terminal.type = CCV_SET_TERMINAL_TYPE(type, cache->age, size);
			set[1 - u] = t;
		}
	} else {
		uint64_t k = 1, j = 63;
		k = k << ((sign & (j << (depth * 6))) >> (depth * 6));
		uint64_t m = (k - 1) & branch->branch.bitmap;
		uint32_t start = compute_bits(m);
		uint32_t total = compute_bits(branch->branch.bitmap);
		ccv_cache_index_t* set = (ccv_cache_index_t*)(branch->branch.set - (branch->branch.set & 0x3));
		set = (ccv_cache_index_t*)ccrealloc(set, sizeof(ccv_cache_index_t) * (total + 1));
		assert(((uint64_t)set & 0x3) == 0);
		for (i = total; i > start; i--)
			set[i] = set[i - 1];
		set[start].terminal.off = (uint64_t)x | 0x1;
		set[start].terminal.sign = sign;
		set[start].terminal.type = CCV_SET_TERMINAL_TYPE(type, cache->age, size);
		branch->branch.set = (uint64_t)set;
		branch->branch.bitmap |= k;
		if (total == 63)
			branch->branch.set |= 0x2;
	}
	cache->rnum++;
	cache->size += size;
	return 0;
}

static void _ccv_cache_cleanup(ccv_cache_index_t* branch)
{
	int leaf = branch->terminal.off & 0x1;
	if (!leaf)
	{
		int i;
		uint64_t total = compute_bits(branch->branch.bitmap);
		ccv_cache_index_t* set = (ccv_cache_index_t*)(branch->branch.set - (branch->branch.set & 0x3));
		for (i = 0; i < total; i++)
		{
			if (!(set[i].terminal.off & 0x1))
				_ccv_cache_cleanup(set + i);
		}
		ccfree(set);
	}
}

static void _ccv_cache_cleanup_and_free(ccv_cache_index_t* branch, ccv_cache_index_free_f ffree[])
{
	int leaf = branch->terminal.off & 0x1;
	if (!leaf)
	{
		int i;
		uint64_t total = compute_bits(branch->branch.bitmap);
		ccv_cache_index_t* set = (ccv_cache_index_t*)(branch->branch.set - (branch->branch.set & 0x3));
		for (i = 0; i < total; i++)
			_ccv_cache_cleanup_and_free(set + i, ffree);
		ccfree(set);
	} else {
		assert(CCV_GET_CACHE_TYPE(branch->terminal.type) >= 0 && CCV_GET_CACHE_TYPE(branch->terminal.type) < 16);
		ffree[CCV_GET_CACHE_TYPE(branch->terminal.type)]((void*)(branch->terminal.off - (branch->terminal.off & 0x3)));
	}
}

void* ccv_cache_out(ccv_cache_t* cache, uint64_t sign, uint8_t* type)
{
	if (!bits_in_16bits_init)
		precomputed_16bits();
	if (cache->rnum == 0)
		return 0;
	int i, found = 0, depth = -1;
	ccv_cache_index_t* parent = 0;
	ccv_cache_index_t* uncle = &cache->origin;
	ccv_cache_index_t* branch = &cache->origin;
	uint64_t j = 63;
	for (i = 0; i < 10; i++)
	{
		int leaf = branch->terminal.off & 0x1;
		int full = branch->terminal.off & 0x2;
		if (leaf)
		{
			found = 1;
			break;
		}
		if (parent != 0 && compute_bits(parent->branch.bitmap) > 1)
			uncle = branch;
		parent = branch;
		depth = i;
		ccv_cache_index_t* set = (ccv_cache_index_t*)(branch->branch.set - (branch->branch.set & 0x3));
		int dice = (sign & j) >> (i * 6);
		if (full)
		{
			branch = set + dice;
		} else {
			uint64_t k = 1;
			k = k << dice;
			if (k & branch->branch.bitmap)
			{
				uint64_t m = (k - 1) & branch->branch.bitmap;
				branch = set + compute_bits(m);
			} else {
				return 0;
			}
		}
		j <<= 6;
	}
	if (!found)
		return 0;
	int leaf = branch->terminal.off & 0x1;
	if (!leaf)
		return 0;
	if (branch->terminal.sign != sign)
		return 0;
	void* result = (void*)(branch->terminal.off - (branch->terminal.off & 0x3));
	if (type)
		*type = CCV_GET_CACHE_TYPE(branch->terminal.type);
	uint32_t size = CCV_GET_TERMINAL_SIZE(branch->terminal.type);
	if (branch != &cache->origin)
	{
		uint64_t k = 1, j = 63;
		int dice = (sign & (j << (depth * 6))) >> (depth * 6);
		k = k << dice;
		uint64_t m = (k - 1) & parent->branch.bitmap;
		uint32_t start = compute_bits(m);
		uint32_t total = compute_bits(parent->branch.bitmap);
		assert(total > 1);
		ccv_cache_index_t* set = (ccv_cache_index_t*)(parent->branch.set - (parent->branch.set & 0x3));
		if (total > 2 || (total == 2 && !(set[1 - start].terminal.off & 0x1)))
		{
			parent->branch.bitmap &= ~k;
			for (i = start + 1; i < total; i++)
				set[i - 1] = set[i];
			set = (ccv_cache_index_t*)ccrealloc(set, sizeof(ccv_cache_index_t) * (total - 1));
			parent->branch.set = (uint64_t)set;
		} else {
			ccv_cache_index_t t = set[1 - start];
			_ccv_cache_cleanup(uncle);
			*uncle = t;
		}
		_ccv_cache_aging(&cache->origin, sign);
	} else {
		// if I only have one item, reset age to 1
		cache->age = 1;
	}
	cache->rnum--;
	cache->size -= size;
	return result;
}

int ccv_cache_delete(ccv_cache_t* cache, uint64_t sign)
{
	uint8_t type = 0;
	void* result = ccv_cache_out(cache, sign, &type);
	if (result != 0)
	{
		assert(type >= 0 && type < 16);
		cache->ffree[type](result);
		return 0;
	}
	return -1;
}

void ccv_cache_cleanup(ccv_cache_t* cache)
{
	if (cache->rnum > 0)
	{
		_ccv_cache_cleanup_and_free(&cache->origin, cache->ffree);
		cache->size = 0;
		cache->age = 0;
		cache->rnum = 0;
		memset(&cache->origin, 0, sizeof(ccv_cache_index_t));
	}
}

void ccv_cache_close(ccv_cache_t* cache)
{
	// for radix-tree based cache, close/cleanup are the same (it is not the same for cuckoo based one,
	// because for cuckoo based one, it will free up space in close whereas only cleanup space in cleanup
	ccv_cache_cleanup(cache);
}
