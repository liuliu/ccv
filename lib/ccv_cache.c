#include "ccv.h"

void ccv_cache_init(ccv_cache_t* cache)
{
	memset(&cache->origin, 0, sizeof(ccv_cache_index_t));
	cache->rnum = 0;
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

static ccv_cache_index_t* __ccv_cache_seek(ccv_cache_index_t* branch, uint64_t sign, int* depth)
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

ccv_matrix_t* ccv_cache_get(ccv_cache_t* cache, uint64_t sign)
{
	ccv_cache_index_t* branch = __ccv_cache_seek(&cache->origin, sign, 0);
	if (!branch)
		return 0;
	int leaf = branch->terminal.off & 0x1;
	if (!leaf)
		return 0;
	if (branch->terminal.sign != sign)
		return 0;
	return (ccv_matrix_t*)(branch->terminal.off - (branch->terminal.off & 0x3));
}

int ccv_cache_put(ccv_cache_t* cache, uint64_t sign, ccv_matrix_t* x)
{
	assert(((uint64_t)x & 0x3) == 0);
	if (cache->rnum == 0)
	{
		cache->origin.terminal.off = (uint64_t)x | 0x1;
		cache->origin.terminal.sign = sign;
		cache->rnum = 1;
		return 0;
	}
	int i, depth = -1;
	ccv_cache_index_t* branch = __ccv_cache_seek(&cache->origin, sign, &depth);
	if (!branch)
		return -1;
	int leaf = branch->terminal.off & 0x1;
	uint64_t on = 1;
	assert(depth >= 0);
	if (leaf)
	{
		if (sign == branch->terminal.sign)
		{
			ccfree((void*)(branch->terminal.off - (branch->terminal.off & 0x3)));
			branch->terminal.off = (uint64_t)x | 0x1;
			return 1;
		} else {
			ccv_cache_index_t t = *branch;
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
			int u = dice < udice;
			set[u].terminal.sign = sign;
			set[u].terminal.off = (uint64_t)x | 0x1;
			set[1 - u] = t;
		}
	} else {
		uint64_t k = 1, j = 63;
		k = k << ((sign & (j << (depth * 6))) >> (depth * 6));
		uint64_t m = (k - 1) & branch->branch.bitmap;
		uint32_t start = compute_bits(m);
		uint32_t total = compute_bits(branch->branch.bitmap);
		ccv_cache_index_t* set = (ccv_cache_index_t*)(branch->branch.set - (branch->branch.set & 0x3));
		set = (ccv_cache_index_t*)realloc(set, sizeof(ccv_cache_index_t) * (total + 1));
		assert(((uint64_t)set & 0x3) == 0);
		for (i = total; i > start; i--)
			set[i] = set[i - 1];
		set[start].terminal.off = (uint64_t)x | 0x1;
		set[start].terminal.sign = sign;
		branch->branch.set = (uint64_t)set;
		branch->branch.bitmap |= k;
		if (total == 63)
			branch->branch.set |= 0x2;
	}
	cache->rnum++;
	return 0;
}

static void __ccv_cache_cleanup(ccv_cache_index_t* branch)
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
				__ccv_cache_cleanup(set + i);
		}
		ccfree(set);
	}
}

static void __ccv_cache_nuke(ccv_cache_index_t* branch)
{
	int leaf = branch->terminal.off & 0x1;
	if (!leaf)
	{
		int i;
		uint64_t total = compute_bits(branch->branch.bitmap);
		ccv_cache_index_t* set = (ccv_cache_index_t*)(branch->branch.set - (branch->branch.set & 0x3));
		for (i = 0; i < total; i++)
			__ccv_cache_nuke(set + i);
		ccfree(set);
	} else {
		ccfree((void*)(branch->terminal.off - (branch->terminal.off & 0x3)));
	}
}

ccv_matrix_t* ccv_cache_out(ccv_cache_t* cache, uint64_t sign)
{
	if (!bits_in_16bits_init)
		precomputed_16bits();
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
	ccv_matrix_t* result = (ccv_matrix_t*)(branch->terminal.off - (branch->terminal.off & 0x3));
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
			set = (ccv_cache_index_t*)realloc(set, sizeof(ccv_cache_index_t) * (total - 1));
			parent->branch.set = (uint64_t)set;
		} else {
			ccv_cache_index_t t = set[1 - start];
			__ccv_cache_cleanup(uncle);
			*uncle = t;
		}
		cache->rnum--;
	}
	return result;
}

int ccv_cache_delete(ccv_cache_t* cache, uint64_t sign)
{
	ccv_matrix_t* result = ccv_cache_out(cache, sign);
	if (result != 0)
	{
		ccfree(result);
		return 0;
	}
	return -1;
}

void ccv_cache_close(ccv_cache_t* cache)
{
	__ccv_cache_nuke(&cache->origin);
	cache->rnum = 0;
}
