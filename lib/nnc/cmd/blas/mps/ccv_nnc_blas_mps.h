#ifndef GUARD_ccv_nnc_blas_mps_h
#define GUARD_ccv_nnc_blas_mps_h

// [M, 1, D]op [1, H, 1] -> [M, H, D]. Return the channel-weight input bit.
static inline uint8_t _ccv_nnc_mps_channel_broadcast(const ccv_nnc_tensor_param_t a, const ccv_nnc_tensor_param_t b, const ccv_nnc_tensor_param_t c)
{
	if (ccv_nnc_tensor_nd(a.dim) != 3 || ccv_nnc_tensor_nd(b.dim) != 3 || ccv_nnc_tensor_nd(c.dim) != 3)
		return 0;
	if (a.dim[0] == c.dim[0] && a.dim[1] == 1 && a.dim[2] == c.dim[2] &&
		b.dim[0] == 1 && b.dim[1] == c.dim[1] && b.dim[2] == 1)
		return 2;
	if (b.dim[0] == c.dim[0] && b.dim[1] == 1 && b.dim[2] == c.dim[2] &&
		a.dim[0] == 1 && a.dim[1] == c.dim[1] && a.dim[2] == 1)
		return 1;
	return 0;
}

// A contiguous row vector can be reused across every row of a full-sized input.
// Reject overlapping writes to the vector and shifted overlap with the full input.
static inline uint8_t _ccv_nnc_mps_row_broadcast(const ccv_nnc_tensor_view_t* const a, const ccv_nnc_tensor_view_t* const b, const ccv_nnc_tensor_view_t* const c)
{
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
	const int c_nd = ccv_nnc_tensor_nd(c->info.dim);
	if (c_nd < 2 || a_nd < 1 || b_nd < 1 || a_nd > c_nd || b_nd > c_nd ||
		!CCV_IS_TENSOR_CONTIGUOUS(a) || !CCV_IS_TENSOR_CONTIGUOUS(b) || !CCV_IS_TENSOR_CONTIGUOUS(c))
		return 0;
	const int row_length = c->info.dim[c_nd - 1];
	// Preserve the existing scalar and elementwise paths when no row repeats.
	if (row_length <= 1 || ccv_nnc_tensor_count(c->info) <= row_length)
		return 0;
	uint8_t row_broadcast = 0;
	if (memcmp(a->info.dim, c->info.dim, sizeof(c->info.dim)) == 0 &&
		b->info.dim[b_nd - 1] == row_length && ccv_nnc_tensor_count(b->info) == row_length)
		row_broadcast = 2;
	else if (memcmp(b->info.dim, c->info.dim, sizeof(c->info.dim)) == 0 &&
		a->info.dim[a_nd - 1] == row_length && ccv_nnc_tensor_count(a->info) == row_length)
		row_broadcast = 1;
	if (!row_broadcast)
		return 0;
	const ccv_nnc_tensor_t* const inputs[] = { (const ccv_nnc_tensor_t*)a, (const ccv_nnc_tensor_t*)b };
	const size_t c_size = ccv_nnc_tensor_data_size_without_padding(c->info);
	int i;
	for (i = 0; i < 2; i++)
		if (mpgetbuffer(inputs[i]) == mpgetbuffer((const ccv_nnc_tensor_t*)c))
		{
			const size_t input_size = ccv_nnc_tensor_data_size_without_padding(inputs[i]->info);
			const int overlap = inputs[i]->dataof < c->dataof + c_size && c->dataof < inputs[i]->dataof + input_size;
			if (overlap && ((row_broadcast & (1u << i)) || inputs[i]->dataof != c->dataof))
				return 0;
		}
	return row_broadcast;
}

#endif
