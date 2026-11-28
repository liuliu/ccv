/* FTQ tile schedule (see CCV_NNC_QX_8I_ROWWISE_FTQ_N16), shared by the C codec (ccv_nnc_easy.h) and the Metal
 * descriptors (C++). A row_count x row_length tensor (row_count % 16 == 0, row_length % 16 == 0) has
 * tiles = (row_count / 16) (row_length / 16) tiles of 16 x 16 weights. At rate n every tile has exactly 16 n bits
 * (high_bits = 16 n, drop_count = 0), the code fills tiles * 2 n bytes after the 32-byte header, and the FP16 scales
 * start at align128(32 + tiles * 2 n) of the blob. With n even every tile starts on a 32-bit word of the code, for
 * any row_length % 16 == 0 (row_length % 256 is not needed). */
#ifndef CCV_NNC_FTQ_SCHEDULE_INC
#define CCV_NNC_FTQ_SCHEDULE_INC

#include <stdint.h>

typedef struct {
	uint64_t tile_count;
	uint32_t high_bits;
	uint64_t drop_count;
	uint64_t code_bits;
	uint64_t scale_offset;
} ccv_nnc_8i_rowwise_x_ftq_schedule_t;

/* The 16 x 16 tiles of a row_count x row_length TMCQ / FTQ tensor: (row_count / 16) tile rows of row_length / 16
 * tiles each. */
static inline uint64_t ccv_nnc_8i_rowwise_x_tile_count(const uint64_t row_count, const uint64_t row_length)
{
	return (row_count / 16) * (row_length / 16);
}

static inline ccv_nnc_8i_rowwise_x_ftq_schedule_t ccv_nnc_8i_rowwise_x_ftq_schedule_of_rate(const uint64_t n, const uint64_t row_count, const uint64_t row_length)
{
	const uint64_t tiles = ccv_nnc_8i_rowwise_x_tile_count(row_count, row_length);
	ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule;
	schedule.tile_count = tiles;
	schedule.high_bits = (uint32_t)(16 * n);
	schedule.drop_count = 0;
	schedule.code_bits = tiles * 16 * n;
	schedule.scale_offset = (32 + tiles * 2 * n + 127) & ~(uint64_t)127;
	return schedule;
}

#endif
