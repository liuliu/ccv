extern "C" {
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <ctype.h>
}

#include <unordered_map>

static inline uint32_t rotl(uint32_t a)
{
	return a * 33;
}

typedef struct {
	uint32_t key;
	uint64_t value;
} ccv_hash_kv_t;

typedef struct {
	uint32_t size;
	uint32_t extra;
	uint32_t rnum;
	uint8_t* ifbits;
	ccv_hash_kv_t* ptr;
} ccv_hash_t;

ccv_hash_t* ccv_hash_new(void)
{
	ccv_hash_t* hash = (ccv_hash_t*)ccmalloc(sizeof(ccv_hash_t));
	hash->extra = 0;
	hash->rnum = 0;
	hash->size = 4;
	hash->ifbits = (uint8_t*)cccalloc(sizeof(uint8_t), hash->size);
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
	uint8_t* ifbits = hash->ifbits = (uint8_t*)ccrealloc(hash->ifbits, sizeof(uint8_t) * new_size);
	memset(hash->ifbits + size, 0, sizeof(uint8_t) * (new_size - size));
	ccv_hash_kv_t* ptr = hash->ptr = (ccv_hash_kv_t*)ccrealloc(hash->ptr, sizeof(ccv_hash_kv_t) * new_size);
	uint32_t i;
	uint8_t dirty_bit = hash->extra ? 0 : 0x40;
	uint8_t new_dirty_bit = hash->extra ? 0x40 : 0;
	for (i = 0; i < size; i++)
		if ((ifbits[i] & 0x80) && (ifbits[i] & 0x40) == dirty_bit)
		{
			ifbits[i] = 0;
			// This item is from old hash table, need to find a new location for it.
			uint32_t key = ptr[i].key;
			uint64_t value = ptr[i].value;
			uint32_t hval = rotl(key);
			uint32_t k = 0, idx = hval % (hash->size + hash->extra);
			for (; k < 64; ++idx, ++k)
			{
				if (idx >= new_size)
					idx = 0;
				if (!(ifbits[idx] & 0x80))
					break;
				// I can place into this location, only catch is that I need to re-compute where to host it.
				int evict = ((ifbits[idx] & 0x40) == dirty_bit);
				uint32_t j = ifbits[idx] & 0x3f;
				if (k > j || evict)
				{
					uint32_t old_key = ptr[idx].key;
					uint64_t old_value = ptr[idx].value;
					ifbits[idx] = 0x80 | k | new_dirty_bit;
					ccv_hash_kv_t* ptr = hash->ptr;
					ptr[idx].key = key;
					ptr[idx].value = value;
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
			assert(k < 64); // Otherwise I need to do rehash.
			ifbits[idx] = 0x80 | k | new_dirty_bit;
			ptr[idx].key = key;
			ptr[idx].value = value;
		}
}

void ccv_hash_add(ccv_hash_t* hash, uint32_t key, uint64_t value)
{
	uint32_t size = hash->size + hash->extra;
	if (hash->rnum + 1 > size - (size + 9) / 10)
	{
		ccv_hash_re(hash);
		size = hash->size + hash->extra;
	}
	uint32_t hval = rotl(key);
	uint8_t* ifbits = hash->ifbits;
	ccv_hash_kv_t* ptr = hash->ptr;
	uint8_t dirty_bit = hash->extra ? 0x40 : 0;
	uint32_t k = 0, idx = hval % (hash->size + hash->extra);
	for (; k < 64; ++idx, ++k)
	{
		if (idx >= size)
			idx = 0;
		if (!(ifbits[idx] & 0x80))
			break;
		uint32_t j = ifbits[idx] & 0x3f;
		if (k > j)
		{
			uint32_t old_key = ptr[idx].key;
			uint64_t old_value = ptr[idx].value;
			ifbits[idx] = 0x80 | k | dirty_bit;
			ptr[idx].key = key;
			ptr[idx].value = value;
			k = j;
			key = old_key;
			value = old_value;
		}
	}
	assert(k < 64);
	hash->rnum = hash->rnum + 1;
	ifbits[idx] = 0x80 | k | dirty_bit;
	ptr[idx].key = key;
	ptr[idx].value = value;
}

ccv_hash_kv_t* ccv_hash_find(ccv_hash_t* hash, uint32_t key)
{
	uint32_t size = hash->size + hash->extra;
	uint32_t hval = rotl(key);
	uint8_t* ifbits = hash->ifbits;
	uint32_t k = 0, idx = hval % (hash->size + hash->extra);
	ccv_hash_kv_t* ptr = hash->ptr;
	for (; k < 64; ++idx, ++k)
	{
		if (idx >= size)
			idx = 0;
		if (!(ifbits[idx] & 0x80) || k > (ifbits[idx] & 0x3f))
			break;
		if (ptr[idx].key == key)
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

	elapsed = ccv_nnc_cmd_mono_time();
	std::unordered_map<uint32_t, uint64_t> hashmap;
	for (i = 0; i < 100000000; i++)
		hashmap[i] = i + 1;
	elapsed = ccv_nnc_cmd_mono_time() - elapsed;
	printf("insert unordered map %lf ms\n", elapsed / 1000000.0);

	for (i = 100000; i < 100010; i++)
	{
		ccv_hash_kv_t* ptr = ccv_hash_find(hash, i);
		printf("%u %lu\n", ptr->key, ptr->value);
	}
	return 0;
}
