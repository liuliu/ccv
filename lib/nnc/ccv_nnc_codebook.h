/* Pair codebooks for the trellis row-wise format TMCQ.
 *
 * A codebook is P int8 pairs, interleaved like ccv_nnc_trellis_builtin_table, with P = 512, 1024 or 2048; pair k is
 * (table[2 k], table[2 k + 1]) and a 15-bit trellis window cw selects k = ((cw * 0x83DCD12D) mod 2^32) >> (32 - log2 P).
 *
 * Codebooks live in a process-wide registry of slots. Slot 0 is the built-in table (512 pairs) and is always
 * present; registering a table returns a slot >= 1, and registering the same content again returns the same slot.
 * A tensor refers to its codebook through bits 16 - 23 of its reserved field (CCV_NNC_CODEBOOK_SLOT_SHIFT); in a
 * checkpoint those bits hold the file-local id of a __ccv_codebook__/<id> tensor, and reading the weight maps it to the
 * slot of that table. Old files have zero there, which is the built-in table.
 *
 * Serialized codebook (the data of a __ccv_codebook__/<id> tensor): 16 bytes "CCVCBOOK", uint32 entries, uint32 crc32
 * (zlib crc32 of the 2 P bytes that follow), then the 2 P int8 values, little-endian.
 *
 * Inline codebook (bits 24 - 25 of reserved, CCV_NNC_CODEBOOK_INLINE_SHIFT, with slot CCV_NNC_CODEBOOK_INLINE_SLOT):
 * the tensor data starts with the serialized table and zeros up to 128 + 2 P bytes, and the rest is the tensor
 * without it. No table is registered in the inline slot. Kernels still take a slot: a backend registers the inline
 * table (ccv_nnc_codebook_resolve_inline) and hands the kernel the data after it with the returned reserved field. */
#ifndef GUARD_ccv_nnc_codebook_h
#define GUARD_ccv_nnc_codebook_h

#include <stddef.h>
#include <stdint.h>

#define CCV_NNC_CODEBOOK_SLOT_SHIFT (16)
#define CCV_NNC_CODEBOOK_SLOT_MASK (0xff << CCV_NNC_CODEBOOK_SLOT_SHIFT)
#define CCV_NNC_CODEBOOK_MAX_SLOTS (256)
#define CCV_NNC_CODEBOOK_HEADER_SIZE (16)
#define CCV_NNC_CODEBOOK_MAGIC "CCVCBOOK" // 8 bytes, the start of a serialized codebook.
#define CCV_NNC_CODEBOOK_TENSOR_PREFIX "__ccv_codebook__/"
#define CCV_NNC_CODEBOOK_INLINE_SHIFT (24)
#define CCV_NNC_CODEBOOK_INLINE_MASK (0x3 << CCV_NNC_CODEBOOK_INLINE_SHIFT)
#define CCV_NNC_CODEBOOK_INLINE_SLOT (255) // Registered slots are 1 ... 254.

/* The codebook slot recorded in a tensor's reserved field. */
static inline int ccv_nnc_codebook_slot_of(const int reserved)
{
	return (reserved >> CCV_NNC_CODEBOOK_SLOT_SHIFT) & 0xff;
}

/* reserved with its codebook slot replaced by slot. */
static inline int ccv_nnc_codebook_with_slot(const int reserved, const int slot)
{
	return (reserved & ~CCV_NNC_CODEBOOK_SLOT_MASK) | ((slot & 0xff) << CCV_NNC_CODEBOOK_SLOT_SHIFT);
}

/* log2 of the entry count, 9 / 10 / 11, or -1 if entries is not 512 / 1024 / 2048. */
static inline int ccv_nnc_codebook_log2_entries(const int entries)
{
	return entries == 512 ? 9 : entries == 1024 ? 10 : entries == 2048 ? 11 : -1;
}

/* The pair count of the inline codebook named in reserved (512, 1024 or 2048), 0 if there is none. */
static inline int ccv_nnc_codebook_inline_entries(const int reserved)
{
	const int code = (reserved & CCV_NNC_CODEBOOK_INLINE_MASK) >> CCV_NNC_CODEBOOK_INLINE_SHIFT;
	return code ? 256 << code : 0;
}

/* zlib-compatible CRC-32 of size bytes. */
uint32_t ccv_nnc_codebook_crc32(const void* data, const size_t size);

/* Registers a table of entries pairs (2 * entries int8 values, copied). Returns its slot (>= 1, the same slot for the
 * same content), or -1 if entries is not 512 / 1024 / 2048, a value is -128, or the registry is full. */
int ccv_nnc_codebook_register(const int8_t* const pairs, const int entries);

/* The pairs of slot (slot 0 is the built-in table), with the entry count in *entries and the crc32 of the 2 * entries bytes
 * in *crc32 (either may be NULL). NULL if the slot is not registered. The memory stays valid for the process. */
const int8_t* ccv_nnc_codebook_get(const int slot, int* const entries, uint32_t* const crc32);

/* Parses a serialized codebook of size bytes and registers it. Returns the slot, or -1 if the magic, entry count, size,
 * crc32 or values are invalid. */
int ccv_nnc_codebook_register_serialized(const void* const data, const size_t size);

/* Writes the serialized form of slot into data (CCV_NNC_CODEBOOK_HEADER_SIZE + 2 * entries bytes, size checked).
 * Returns the number of bytes written, or 0 if the slot is unknown or size is too small. */
size_t ccv_nnc_codebook_serialize(const int slot, void* const data, const size_t size);

/* The reserved field of the data after the inline codebook of a TMCQ tensor whose data starts at data (CPU memory, at
 * least 128 + 2 P bytes): the inline bits cleared and the slot of the registered table in bits 16 - 23. reserved
 * unchanged if it has no inline codebook; -1 if the slot is not CCV_NNC_CODEBOOK_INLINE_SLOT, or the magic, entry
 * count (must be the P of reserved), crc32, zero padding or values are invalid, or the registry is full. A table
 * registered before is found by its content without computing its crc32, so this is cheap to call per command. */
int ccv_nnc_codebook_resolve_inline(const void* const data, const int reserved);

#endif
