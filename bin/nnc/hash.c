#include <ccv.h>
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <ctype.h>

static inline uint32_t rotl(uint32_t a)
{
	return a * 33;
	/*
	a = (a + 0x7ed55d16) + (a << 12);
	a = (a ^ 0xc761c23c) ^ (a >> 19);
	a = (a + 0x165667b1) + (a << 5);
	a = (a + 0xd3a2646c) ^ (a << 9);
	a = (a + 0xfd7046c5) + (a << 3);
	a = (a ^ 0xb55a4f09) ^ (a >> 16);
	return a;
	*/
}

typedef struct {
	uint32_t key;
	uint64_t value;
} ccv_hash_kv_t;

typedef struct {
	uint32_t size;
	uint32_t extra;
	uint32_t rnum;
	uint16_t* ifbits;
	ccv_hash_kv_t* ptr;
} ccv_hash_t;

ccv_hash_t* ccv_hash_new(void)
{
	ccv_hash_t* hash = (ccv_hash_t*)ccmalloc(sizeof(ccv_hash_t));
	hash->extra = 0;
	hash->rnum = 0;
	hash->size = 4;
	hash->ifbits = (uint16_t*)cccalloc(sizeof(uint16_t), hash->size);
	hash->ptr = (ccv_hash_kv_t*)ccmalloc(sizeof(ccv_hash_kv_t) * hash->size);
	return hash;
}

void ccv_hash_re(ccv_hash_t* hash)
{
	uint32_t size = hash->size + hash->extra;
	if (hash->extra)
		hash->extra = 0, hash->size = hash->size << 1;
	else
		hash->extra = hash->size >> 1;
	uint32_t new_size = hash->size + hash->extra;
	uint16_t* ifbits = hash->ifbits = (uint16_t*)ccrealloc(hash->ifbits, sizeof(uint16_t) * new_size);
	memset(hash->ifbits + size, 0, sizeof(uint16_t) * (new_size - size));
	ccv_hash_kv_t* ptr = hash->ptr = (ccv_hash_kv_t*)ccrealloc(hash->ptr, sizeof(ccv_hash_kv_t) * new_size);
	uint32_t i;
	uint16_t dirty_bit = hash->extra ? 0 : 0x80;
	uint16_t new_dirty_bit = hash->extra ? 0x80 : 0;
	for (i = 0; i < size; i++)
		if ((ifbits[i] & 0x0100) && (ifbits[i] & 0x80) == dirty_bit)
		{
			ifbits[i] = 0;
			// This item is from old hash table, need to find a new location for it.
			uint32_t key = ptr[i].key;
			uint64_t value = ptr[i].value;
			uint32_t hval = rotl(key);
			uint16_t bits = hval >> 25;
			uint32_t k = 0, idx = hval % (hash->size + hash->extra);
			for (; k < 128; ++idx, ++k)
			{
				if (idx >= new_size)
					idx = 0;
				if (!(ifbits[idx] & 0x0100))
					break;
				// I can place into this location, only catch is that I need to re-compute where to host it.
				int evict = ((ifbits[idx] & 0x80) == dirty_bit);
				uint32_t j = ifbits[idx] >> 9;
				if (k > j || evict)
				{
					uint16_t old_bits = ifbits[idx] & 0x7f;
					uint32_t old_key = ptr[idx].key;
					uint64_t old_value = ptr[idx].value;
					ifbits[idx] = 0x0100 | (k << 9) | new_dirty_bit | bits;
					ccv_hash_kv_t* ptr = hash->ptr;
					ptr[idx].key = key;
					ptr[idx].value = value;
					bits = old_bits;
					key = old_key;
					value = old_value;
					if (evict) // In this case, I cannot keep going with the idx, need to recompute idx as well restart k.
					{
						hval = rotl(key);
						k = 0, idx = hval % (hash->size + hash->extra);
					} else 
						k = j;
				}
			}
			assert(k < 128); // Otherwise I need to do rehash.
			ifbits[idx] = 0x0100 | (k << 9) | new_dirty_bit | bits;
			ptr[idx].key = key;
			ptr[idx].value = value;
		}
}

void ccv_hash_add(ccv_hash_t* hash, uint32_t key, uint64_t value)
{
	uint32_t size = hash->size + hash->extra;
	if (hash->rnum + 1 > size - (size + 8) / 10)
	{
		ccv_hash_re(hash);
		size = hash->size + hash->extra;
	}
	uint32_t hval = rotl(key);
	uint16_t bits = hval >> 25;
	uint16_t* ifbits = hash->ifbits;
	ccv_hash_kv_t* ptr = hash->ptr;
	uint16_t dirty_bit = hash->extra ? 0x80 : 0;
	uint32_t k = 0, idx = hval % (hash->size + hash->extra);
	for (; k < 128; ++idx, ++k)
	{
		if (idx >= size)
			idx = 0;
		if (!(ifbits[idx] & 0x0100))
			break;
		uint32_t j = ifbits[idx] >> 9;
		if (k > j)
		{
			uint16_t old_bits = ifbits[idx] & 0x7f;
			uint32_t old_key = ptr[idx].key;
			uint64_t old_value = ptr[idx].value;
			ifbits[idx] = 0x0100 | (k << 9) | dirty_bit | bits;
			ptr[idx].key = key;
			ptr[idx].value = value;
			k = j;
			bits = old_bits;
			key = old_key;
			value = old_value;
		}
	}
	assert(k < 128);
	hash->rnum = hash->rnum + 1;
	ifbits[idx] = 0x0100 | (k << 9) | dirty_bit | bits;
	ptr[idx].key = key;
	ptr[idx].value = value;
}

ccv_hash_kv_t* ccv_hash_find(ccv_hash_t* hash, uint32_t key)
{
	uint32_t size = hash->size + hash->extra;
	uint32_t hval = rotl(key);
	uint16_t* ifbits = hash->ifbits;
	uint32_t k = 0, idx = hval % (hash->size + hash->extra);
	uint16_t bits = hval >> 25;
	ccv_hash_kv_t* ptr = hash->ptr;
	for (; k < 128; ++idx, ++k)
	{
		if (idx >= size)
			idx = 0;
		if (!(ifbits[idx] & 0x0100) || k > (ifbits[idx] >> 9))
			break;
		if ((ifbits[idx] & 0x7f) == bits && ptr[idx].key == key)
			return ptr + idx;
	}
	return 0;
}

int main(int argc, char** argv)
{
	int i;
	ccv_hash_t* hash = ccv_hash_new();
	uint64_t elapsed = ccv_nnc_cmd_mono_time();
	for (i = 0; i < 100000000; i++)
		ccv_hash_add(hash, i, i + 1);
	elapsed = ccv_nnc_cmd_mono_time() - elapsed;
	printf("insert %lf ms\n", elapsed / 1000000.0);

	for (i = 1000; i < 1010; i++)
	{
		ccv_hash_kv_t* ptr = ccv_hash_find(hash, i);
		printf("%u %lu\n", ptr->key, ptr->value);
	}
	return 0;
}
