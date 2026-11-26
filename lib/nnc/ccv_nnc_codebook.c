#include "ccv_nnc_codebook.h"
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include "ccv_nnc_trellis_builtin_table.inc"

typedef struct {
	int entries;
	int flags;
	uint32_t crc32;
	int8_t* pairs;
} ccv_nnc_codebook_t;

static pthread_mutex_t codebook_mutex = PTHREAD_MUTEX_INITIALIZER;
static ccv_nnc_codebook_t codebooks[CCV_NNC_CODEBOOK_MAX_SLOTS];
static int codebook_count = 0; // Slots 1 ... codebook_count are registered.

static uint32_t crc_table[256];
static pthread_once_t crc_once = PTHREAD_ONCE_INIT;

static void _ccv_nnc_codebook_build_crc_table(void)
{
	uint32_t i;
	for (i = 0; i < 256; i++)
	{
		uint32_t c = i;
		int k;
		for (k = 0; k < 8; k++)
			c = (c & 1) ? 0xedb88320u ^ (c >> 1) : c >> 1;
		crc_table[i] = c;
	}
}

uint32_t ccv_nnc_codebook_crc32(const void* data, const size_t size)
{
	pthread_once(&crc_once, _ccv_nnc_codebook_build_crc_table);
	const uint8_t* const p = (const uint8_t*)data;
	uint32_t crc = 0xffffffffu;
	size_t i;
	for (i = 0; i < size; i++)
		crc = crc_table[(crc ^ p[i]) & 0xff] ^ (crc >> 8);
	return crc ^ 0xffffffffu;
}

int ccv_nnc_codebook_register_with_flags(const int8_t* const pairs, const int entries, const int flags)
{
	if (flags & ~CCV_NNC_CODEBOOK_FOLD)
		return -1;
	if (ccv_nnc_codebook_log2_entries(entries) < 0)
		return -1;
	const size_t size = (size_t)entries * 2;
	size_t i;
	for (i = 0; i < size; i++)
		if (pairs[i] == -128 || ((flags & CCV_NNC_CODEBOOK_FOLD) && pairs[i] < 0))
			return -1;
	if ((flags & CCV_NNC_CODEBOOK_FOLD) && entries != 512)
		return -1;
	const uint32_t crc = ccv_nnc_codebook_crc32(pairs, size);
	pthread_mutex_lock(&codebook_mutex);
	int slot;
	for (slot = 1; slot <= codebook_count; slot++)
		if (codebooks[slot].entries == entries && codebooks[slot].flags == flags && codebooks[slot].crc32 == crc && memcmp(codebooks[slot].pairs, pairs, size) == 0)
		{
			pthread_mutex_unlock(&codebook_mutex);
			return slot;
		}
	// The inline slot (255) is never registered.
	if (codebook_count + 1 >= CCV_NNC_CODEBOOK_INLINE_SLOT)
	{
		pthread_mutex_unlock(&codebook_mutex);
		return -1;
	}
	int8_t* const copy = (int8_t*)malloc(size);
	memcpy(copy, pairs, size);
	slot = codebook_count + 1;
	codebooks[slot].entries = entries;
	codebooks[slot].flags = flags;
	codebooks[slot].crc32 = crc;
	codebooks[slot].pairs = copy;
	// Publish after the entry is complete; readers of a slot they were handed never see it half-written.
	__atomic_store_n(&codebook_count, slot, __ATOMIC_RELEASE);
	pthread_mutex_unlock(&codebook_mutex);
	return slot;
}

int ccv_nnc_codebook_register(const int8_t* const pairs, const int entries)
{
	return ccv_nnc_codebook_register_with_flags(pairs, entries, 0);
}

int ccv_nnc_codebook_flags(const int slot)
{
	if (slot == 0)
		return 0;
	if (slot < 0 || slot > __atomic_load_n(&codebook_count, __ATOMIC_ACQUIRE))
		return -1;
	return codebooks[slot].flags;
}

int ccv_nnc_codebook_fold_slot(const int slot)
{
	if (slot <= 0 || ccv_nnc_codebook_flags(slot) != 0)
		return -1;
	int entries;
	const int8_t* const pairs = ccv_nnc_codebook_get(slot, &entries, 0);
	return pairs ? ccv_nnc_codebook_register_with_flags(pairs, entries, CCV_NNC_CODEBOOK_FOLD) : -1;
}

static uint32_t builtin_crc = 0;
static pthread_once_t builtin_once = PTHREAD_ONCE_INIT;

static void _ccv_nnc_codebook_builtin_crc(void)
{
	builtin_crc = ccv_nnc_codebook_crc32(ccv_nnc_trellis_builtin_table, sizeof(ccv_nnc_trellis_builtin_table));
}

const int8_t* ccv_nnc_codebook_get(const int slot, int* const entries, uint32_t* const crc32)
{
	if (slot == 0)
	{
		pthread_once(&builtin_once, _ccv_nnc_codebook_builtin_crc);
		if (entries)
			*entries = 512;
		if (crc32)
			*crc32 = builtin_crc;
		return ccv_nnc_trellis_builtin_table;
	}
	if (slot < 0 || slot > __atomic_load_n(&codebook_count, __ATOMIC_ACQUIRE))
		return 0;
	if (entries)
		*entries = codebooks[slot].entries;
	if (crc32)
		*crc32 = codebooks[slot].crc32;
	return codebooks[slot].pairs;
}

static uint32_t _ccv_nnc_codebook_u32(const uint8_t* const p)
{
	return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

int ccv_nnc_codebook_register_serialized(const void* const data, const size_t size)
{
	const uint8_t* const p = (const uint8_t*)data;
	if (!p || size < CCV_NNC_CODEBOOK_HEADER_SIZE || memcmp(p, CCV_NNC_CODEBOOK_MAGIC, 8) != 0)
		return -1;
	const uint32_t entries = _ccv_nnc_codebook_u32(p + 8);
	if (ccv_nnc_codebook_log2_entries((int)entries) < 0 || size != CCV_NNC_CODEBOOK_HEADER_SIZE + (size_t)entries * 2)
		return -1;
	if (ccv_nnc_codebook_crc32(p + CCV_NNC_CODEBOOK_HEADER_SIZE, (size_t)entries * 2) != _ccv_nnc_codebook_u32(p + 12))
		return -1;
	return ccv_nnc_codebook_register((const int8_t*)(p + CCV_NNC_CODEBOOK_HEADER_SIZE), (int)entries);
}

size_t ccv_nnc_codebook_serialize(const int slot, void* const data, const size_t size)
{
	int entries;
	uint32_t crc;
	const int8_t* const pairs = ccv_nnc_codebook_get(slot, &entries, &crc);
	const size_t total = CCV_NNC_CODEBOOK_HEADER_SIZE + (size_t)entries * 2;
	if (!pairs || size < total)
		return 0;
	uint8_t* const p = (uint8_t*)data;
	memcpy(p, CCV_NNC_CODEBOOK_MAGIC, 8);
	int i;
	for (i = 0; i < 4; i++)
	{
		p[8 + i] = (uint8_t)((uint32_t)entries >> (8 * i));
		p[12 + i] = (uint8_t)(crc >> (8 * i));
	}
	memcpy(p + CCV_NNC_CODEBOOK_HEADER_SIZE, pairs, (size_t)entries * 2);
	return total;
}

int ccv_nnc_codebook_resolve_inline(const void* const data, const int reserved)
{
	return ccv_nnc_codebook_resolve_inline_with_flags(data, reserved, 0);
}

int ccv_nnc_codebook_resolve_inline_with_flags(const void* const data, const int reserved, const int flags)
{
	const int entries = ccv_nnc_codebook_inline_entries(reserved);
	if (!entries)
		return reserved;
	if (ccv_nnc_codebook_slot_of(reserved) != CCV_NNC_CODEBOOK_INLINE_SLOT || !data)
		return -1;
	const uint8_t* const p = (const uint8_t*)data;
	const size_t size = (size_t)entries * 2;
	if (memcmp(p, CCV_NNC_CODEBOOK_MAGIC, 8) != 0 || _ccv_nnc_codebook_u32(p + 8) != (uint32_t)entries)
		return -1;
	size_t i;
	for (i = CCV_NNC_CODEBOOK_HEADER_SIZE + size; i < 128 + size; i++)
		if (p[i])
			return -1;
	const int8_t* const pairs = (const int8_t*)(p + CCV_NNC_CODEBOOK_HEADER_SIZE);
	const uint32_t crc = _ccv_nnc_codebook_u32(p + 12);
	// Most calls name a table that is registered already: match it by content and its stored crc32, without computing
	// one or taking the lock (slots 1 ... codebook_count are never written again once published).
	const int count = __atomic_load_n(&codebook_count, __ATOMIC_ACQUIRE);
	int slot;
	for (slot = 1; slot <= count; slot++)
		if (codebooks[slot].entries == entries && codebooks[slot].flags == flags && codebooks[slot].crc32 == crc && memcmp(codebooks[slot].pairs, pairs, size) == 0)
			break;
	if (slot > count)
	{
		// The stored crc32 must be the table's; register_with_flags computes it and checks the values.
		if (ccv_nnc_codebook_crc32(pairs, size) != crc)
			return -1;
		slot = ccv_nnc_codebook_register_with_flags(pairs, entries, flags);
	}
	if (slot < 0)
		return -1;
	return ccv_nnc_codebook_with_slot(reserved & ~CCV_NNC_CODEBOOK_INLINE_MASK, slot);
}
