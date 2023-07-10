#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#ifdef USE_OPENMP
#include <omp.h>
#endif
#ifdef USE_DISPATCH
#include <dispatch/dispatch.h>
#endif

#include "../_ccv_nnc_cpu_ref.h"

void _ccv_nnc_ewsum_forw_cpu_ref_f32(ccv_nnc_tensor_view_t* const* const inputs, const int input_size, ccv_nnc_tensor_view_t* const* const outputs, const int output_size)
{
	if (input_size == 1 && output_size == 1)
	{
		_ccv_nnc_tensor_transfer_cpu_ref_f32(inputs[0], outputs[0]);
		return;
	}
	// Assuming this is float 32.
	int dim[CCV_NNC_MAX_DIM_ALLOC];
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
	int cstride[CCV_NNC_MAX_DIM_ALLOC];
	int x, z;
	int k = 0;
	// Bad, I promised this can be inplace operation. Need to first find out if there are share the same pointer first.
	for (z = 1; z < input_size; z++)
	{
		ccv_nnc_tensor_view_t* c = outputs[0];
		ccv_nnc_tensor_view_t* a = inputs[z];
		if (c->data.f32 == a->data.f32)
		{
			k = z;
			break;
		}
	}
	for (z = 0; z < input_size - 1; z++)
	{
		ccv_nnc_tensor_view_t* c = outputs[0];
		ccv_nnc_tensor_view_t* a = z > 0 ? c : inputs[k];
		ccv_nnc_tensor_view_t* b = z >= k ? inputs[z + 1] : inputs[z];
		assert(ccv_nnc_tensor_nd(a->info.dim) <= CCV_NNC_MAX_DIM + 2);
		assert(ccv_nnc_tensor_nd(b->info.dim) <= CCV_NNC_MAX_DIM + 2);
		assert(ccv_nnc_tensor_nd(c->info.dim) <= CCV_NNC_MAX_DIM + 2);
		ccv_nnc_tensor_view_get_dim(a, dim);
		assert(ccv_nnc_tensor_view_check_dim(b, dim));
		assert(ccv_nnc_tensor_view_check_dim(c, dim));
		if (!CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(b) && !CCV_IS_TENSOR_VIEW(c))
		{
			// Super optimal case, just do one for-loop for sum.
			const int tensor_count = ccv_nnc_tensor_count(a->info);
			for (x = 0; x < tensor_count; x++)
				c->data.f32[x] = a->data.f32[x] + b->data.f32[x];
			continue;
		}
		assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
		ccv_nnc_tensor_view_get_stride(a, astride);
		ccv_nnc_tensor_view_get_stride(b, bstride);
		ccv_nnc_tensor_view_get_stride(c, cstride);
		int i[CCV_NNC_MAX_DIM + 2];
		float* const ap = a->data.f32;
		float* const bp = b->data.f32;
		float* const cp = c->data.f32;
		const int count = dim[2] * dim[3];
		if (astride[2] == dim[3] && bstride[2] == dim[3] && cstride[2] == dim[3] && astride[3] == 1 && bstride[3] == 1 && cstride[3] == 1)
		{
			// Special casing if the ainc[3] is the same as dim[3] (do memcpy for the last two dim)
			for (i[0] = 0; i[0] < dim[0]; i[0]++)
			{
				float* ap0 = ap + i[0] * astride[0];
				float* bp0 = bp + i[0] * bstride[0];
				float* cp0 = cp + i[0] * cstride[0];
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					for (x = 0; x < count; x++)
						cp0[x] = ap0[x] + bp0[x];
					ap0 += astride[1];
					bp0 += bstride[1];
					cp0 += cstride[1];
				}
			}
			continue;
		}
		// Non-optimal case, need to do skip copy.
		for (i[0] = 0; i[0] < dim[0]; i[0]++)
		{
			float* const ap0 = ap + i[0] * astride[0];
			float* const bp0 = bp + i[0] * bstride[0];
			float* const cp0 = cp + i[0] * cstride[0];
			for (i[1] = 0; i[1] < dim[1]; i[1]++)
			{
				float* ap1 = ap0 + i[1] * astride[1];
				float* bp1 = bp0 + i[1] * bstride[1];
				float* cp1 = cp0 + i[1] * cstride[1];
				for (i[2] = 0; i[2] < dim[2]; i[2]++)
				{
					for (x = 0; x < dim[3]; x++)
						cp1[x * cstride[3]] = ap1[x * astride[3]] + bp1[x * bstride[3]];
					ap1 += astride[2];
					bp1 += bstride[2];
					cp1 += cstride[2];
				}
			}
		}
	}
}

void _ccv_nnc_ewsum_forw_cpu_ref_i32(ccv_nnc_tensor_view_t* const* const inputs, const int input_size, ccv_nnc_tensor_view_t* const* const outputs, const int output_size)
{
	if (input_size == 1 && output_size == 1)
	{
		_ccv_nnc_tensor_transfer_cpu_ref_f32(inputs[0], outputs[0]);
		return;
	}
	// Assuming this is float 32.
	int dim[CCV_NNC_MAX_DIM_ALLOC];
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
	int cstride[CCV_NNC_MAX_DIM_ALLOC];
	int x, z;
	int k = 0;
	// Bad, I promised this can be inplace operation. Need to first find out if there are share the same pointer first.
	for (z = 1; z < input_size; z++)
	{
		ccv_nnc_tensor_view_t* c = outputs[0];
		ccv_nnc_tensor_view_t* a = inputs[z];
		if (c->data.f32 == a->data.f32)
		{
			k = z;
			break;
		}
	}
	for (z = 0; z < input_size - 1; z++)
	{
		ccv_nnc_tensor_view_t* c = outputs[0];
		ccv_nnc_tensor_view_t* a = z > 0 ? c : inputs[k];
		ccv_nnc_tensor_view_t* b = z >= k ? inputs[z + 1] : inputs[z];
		assert(ccv_nnc_tensor_nd(a->info.dim) <= CCV_NNC_MAX_DIM + 2);
		assert(ccv_nnc_tensor_nd(b->info.dim) <= CCV_NNC_MAX_DIM + 2);
		assert(ccv_nnc_tensor_nd(c->info.dim) <= CCV_NNC_MAX_DIM + 2);
		ccv_nnc_tensor_view_get_dim(a, dim);
		assert(ccv_nnc_tensor_view_check_dim(b, dim));
		assert(ccv_nnc_tensor_view_check_dim(c, dim));
		if (!CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(b) && !CCV_IS_TENSOR_VIEW(c))
		{
			// Super optimal case, just do one for-loop for sum.
			const int tensor_count = ccv_nnc_tensor_count(a->info);
			for (x = 0; x < tensor_count; x++)
				c->data.f32[x] = a->data.f32[x] + b->data.f32[x];
			continue;
		}
		assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
		ccv_nnc_tensor_view_get_stride(a, astride);
		ccv_nnc_tensor_view_get_stride(b, bstride);
		ccv_nnc_tensor_view_get_stride(c, cstride);
		int i[CCV_NNC_MAX_DIM + 2];
		int* const ap = a->data.i32;
		int* const bp = b->data.i32;
		int* const cp = c->data.i32;
		const int count = dim[2] * dim[3];
		if (astride[2] == dim[3] && bstride[2] == dim[3] && cstride[2] == dim[3] && astride[3] == 1 && bstride[3] == 1 && cstride[3] == 1)
		{
			// Special casing if the ainc[3] is the same as dim[3] (do memcpy for the last two dim)
			for (i[0] = 0; i[0] < dim[0]; i[0]++)
			{
				int* ap0 = ap + i[0] * astride[0];
				int* bp0 = bp + i[0] * bstride[0];
				int* cp0 = cp + i[0] * cstride[0];
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					for (x = 0; x < count; x++)
						cp0[x] = ap0[x] + bp0[x];
					ap0 += astride[1];
					bp0 += bstride[1];
					cp0 += cstride[1];
				}
			}
			continue;
		}
		// Non-optimal case, need to do skip copy.
		for (i[0] = 0; i[0] < dim[0]; i[0]++)
		{
			int* const ap0 = ap + i[0] * astride[0];
			int* const bp0 = bp + i[0] * bstride[0];
			int* const cp0 = cp + i[0] * cstride[0];
			for (i[1] = 0; i[1] < dim[1]; i[1]++)
			{
				int* ap1 = ap0 + i[1] * astride[1];
				int* bp1 = bp0 + i[1] * bstride[1];
				int* cp1 = cp0 + i[1] * cstride[1];
				for (i[2] = 0; i[2] < dim[2]; i[2]++)
				{
					for (x = 0; x < dim[3]; x++)
						cp1[x * cstride[3]] = ap1[x * astride[3]] + bp1[x * bstride[3]];
					ap1 += astride[2];
					bp1 += bstride[2];
					cp1 += cstride[2];
				}
			}
		}
	}
}

static int _ccv_nnc_ewsum_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	if (outputs[0]->info.datatype == CCV_32S)
		_ccv_nnc_ewsum_forw_cpu_ref_i32((ccv_nnc_tensor_view_t**)inputs, input_size, (ccv_nnc_tensor_view_t**)outputs, output_size);
	else
		_ccv_nnc_ewsum_forw_cpu_ref_f32((ccv_nnc_tensor_view_t**)inputs, input_size, (ccv_nnc_tensor_view_t**)outputs, output_size);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_ewsum_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	// D[x + y + z, x] = 1
	int i;
	if (inputs[0] == 0)
	{
		// Set them to 1.
		for (i = 0; i < output_size; i++)
			if (outputs[i])
				_ccv_nnc_tensor_set_cpu_ref_f32((ccv_nnc_tensor_view_t*)outputs[i], 1);
	} else {
		// Copy over the gradient (If they are not pointing to the same tensor already).
		for (i = 0; i < output_size; i++)
			if (outputs[i] && inputs[0]->data.f32 != outputs[i]->data.f32)
				_ccv_nnc_tensor_transfer_cpu_ref_f32((ccv_nnc_tensor_view_t*)inputs[0], (ccv_nnc_tensor_view_t*)outputs[i]);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

void _ccv_nnc_ewprod_forw_cpu_ref(ccv_nnc_tensor_view_t* const* const inputs, const int input_size, ccv_nnc_tensor_view_t* const* const outputs, const int output_size)
{
	if (input_size == 1 && output_size == 1)
	{
		_ccv_nnc_tensor_transfer_cpu_ref_f32(inputs[0], outputs[0]);
		return;
	}
	// Assuming this is float 32.
	int dim[CCV_NNC_MAX_DIM_ALLOC];
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
	int cstride[CCV_NNC_MAX_DIM_ALLOC];
	int x, z;
	int k = 0;
	// Bad, I promised this can be inplace operation. Need to first find out if there are share the same pointer first.
	for (z = 1; z < input_size; z++)
	{
		ccv_nnc_tensor_view_t* c = outputs[0];
		ccv_nnc_tensor_view_t* a = inputs[z];
		if (c->data.f32 == a->data.f32)
		{
			k = z;
			break;
		}
	}
	for (z = 0; z < input_size - 1; z++)
	{
		ccv_nnc_tensor_view_t* c = outputs[0];
		ccv_nnc_tensor_view_t* a = z > 0 ? c : inputs[k];
		ccv_nnc_tensor_view_t* b = z >= k ? inputs[z + 1] : inputs[z];
		assert(ccv_nnc_tensor_nd(a->info.dim) <= CCV_NNC_MAX_DIM + 2);
		assert(ccv_nnc_tensor_nd(b->info.dim) <= CCV_NNC_MAX_DIM + 2);
		assert(ccv_nnc_tensor_nd(c->info.dim) <= CCV_NNC_MAX_DIM + 2);
		ccv_nnc_tensor_view_get_dim(a, dim);
		assert(ccv_nnc_tensor_view_check_dim(b, dim));
		assert(ccv_nnc_tensor_view_check_dim(c, dim));
		if (!CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(b) && !CCV_IS_TENSOR_VIEW(c))
		{
			// Super optimal case, just do one for-loop for sum.
			const int tensor_count = ccv_nnc_tensor_count(a->info);
			for (x = 0; x < tensor_count; x++)
				c->data.f32[x] = a->data.f32[x] * b->data.f32[x];
			continue;
		}
		assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
		ccv_nnc_tensor_view_get_stride(a, astride);
		ccv_nnc_tensor_view_get_stride(b, bstride);
		ccv_nnc_tensor_view_get_stride(c, cstride);
		int i[CCV_NNC_MAX_DIM + 2];
		float* const ap = a->data.f32;
		float* const bp = b->data.f32;
		float* const cp = c->data.f32;
		const int count = dim[2] * dim[3];
		if (astride[2] == dim[3] && bstride[2] == dim[3] && cstride[2] == dim[3])
		{
			// Special casing if the ainc[3] is the same as dim[3]
			for (i[0] = 0; i[0] < dim[0]; i[0]++)
			{
				float* ap0 = ap + i[0] * astride[0];
				float* bp0 = bp + i[0] * bstride[0];
				float* cp0 = cp + i[0] * cstride[0];
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					for (x = 0; x < count; x++)
						cp0[x] = ap0[x] * bp0[x];
					ap0 += astride[1];
					bp0 += bstride[1];
					cp0 += cstride[1];
				}
			}
			continue;
		}
		// Non-optimal case, need to do skip copy.
		for (i[0] = 0; i[0] < dim[0]; i[0]++)
		{
			float* const ap0 = ap + i[0] * astride[0];
			float* const bp0 = bp + i[0] * bstride[0];
			float* const cp0 = cp + i[0] * cstride[0];
			for (i[1] = 0; i[1] < dim[1]; i[1]++)
			{
				float* ap1 = ap0 + i[1] * astride[1];
				float* bp1 = bp0 + i[1] * bstride[1];
				float* cp1 = cp0 + i[1] * cstride[1];
				for (i[2] = 0; i[2] < dim[2]; i[2]++)
				{
					for (x = 0; x < dim[3]; x++)
						cp1[x] = ap1[x] * bp1[x];
					ap1 += astride[2];
					bp1 += bstride[2];
					cp1 += cstride[2];
				}
			}
		}
	}
}

static int _ccv_nnc_ewprod_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	_ccv_nnc_ewprod_forw_cpu_ref((ccv_nnc_tensor_view_t**)inputs, input_size, (ccv_nnc_tensor_view_t**)outputs, output_size);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_ewprod_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	// D[x * y * z, x] = y * z
	// Assuming this is float 32.
	int dim[CCV_NNC_MAX_DIM_ALLOC];
	int gstride[CCV_NNC_MAX_DIM_ALLOC];
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
	int hstride[CCV_NNC_MAX_DIM_ALLOC];
	int x, z;
	ccv_nnc_tensor_view_t* g = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)inputs[output_size + 1];
	if (g == 0)
	{
		assert(ccv_nnc_tensor_nd(b->info.dim) <= CCV_NNC_MAX_DIM + 2);
		ccv_nnc_tensor_view_get_dim(b, dim);
		ccv_nnc_tensor_view_get_stride(b, bstride);
		for (z = 0; z < output_size; z++)
		{
			ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[z + 1];
			ccv_nnc_tensor_view_t* h = (ccv_nnc_tensor_view_t*)outputs[z];
			assert(ccv_nnc_tensor_nd(a->info.dim) <= CCV_NNC_MAX_DIM + 2);
			assert(ccv_nnc_tensor_nd(h->info.dim) <= CCV_NNC_MAX_DIM + 2);
			assert(ccv_nnc_tensor_view_check_dim(a, dim));
			assert(ccv_nnc_tensor_view_check_dim(h, dim));
			ccv_nnc_tensor_view_get_stride(a, astride);
			ccv_nnc_tensor_view_get_stride(h, hstride);
			if (!CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(b) && !CCV_IS_TENSOR_VIEW(h))
			{
				// Super optimal case, just do one for-loop for sum.
				const int tensor_count = ccv_nnc_tensor_count(b->info);
				for (x = 0; x < tensor_count; x++)
					h->data.f32[x] = b->data.f32[x] / a->data.f32[x];
				continue;
			}
			assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
			int i[CCV_NNC_MAX_DIM + 2];
			float* const ap = a->data.f32;
			float* const bp = b->data.f32;
			float* const hp = h->data.f32;
			const int count = dim[2] * dim[3];
			if (astride[2] == dim[3] && bstride[2] == dim[3] && hstride[2] == dim[3])
			{
				// Special casing if the ainc[3] is the same as dim[3]
				for (i[0] = 0; i[0] < dim[0]; i[0]++)
				{
					float* ap0 = ap + i[0] * astride[0];
					float* bp0 = bp + i[0] * bstride[0];
					float* hp0 = hp + i[0] * hstride[0];
					for (i[1] = 0; i[1] < dim[1]; i[1]++)
					{
						for (x = 0; x < count; x++)
							hp0[x] = bp0[x] / ap0[x];
						ap0 += astride[1];
						bp0 += bstride[1];
						hp0 += hstride[1];
					}
				}
				continue;
			}
			// Non-optimal case, need to do skip copy.
			for (i[0] = 0; i[0] < dim[0]; i[0]++)
			{
				float* const ap0 = ap + i[0] * astride[0];
				float* const bp0 = bp + i[0] * bstride[0];
				float* const hp0 = hp + i[0] * hstride[0];
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					float* ap1 = ap0 + i[1] * astride[1];
					float* bp1 = bp0 + i[1] * bstride[1];
					float* hp1 = hp0 + i[1] * hstride[1];
					for (i[2] = 0; i[2] < dim[2]; i[2]++)
					{
						for (x = 0; x < dim[3]; x++)
							hp1[x] = bp1[x] / ap1[x];
						ap1 += astride[2];
						bp1 += bstride[2];
						hp1 += hstride[2];
					}
				}
			}
		}
	} else {
		assert(ccv_nnc_tensor_nd(g->info.dim) <= CCV_NNC_MAX_DIM + 2);
		assert(ccv_nnc_tensor_nd(b->info.dim) <= CCV_NNC_MAX_DIM + 2);
		ccv_nnc_tensor_view_get_dim(b, dim);
		assert(ccv_nnc_tensor_view_check_dim(g, dim));
		ccv_nnc_tensor_view_get_stride(b, bstride);
		ccv_nnc_tensor_view_get_stride(g, gstride);
		for (z = 0; z < output_size; z++)
		{
			ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[z + 1];
			ccv_nnc_tensor_view_t* h = (ccv_nnc_tensor_view_t*)outputs[z];
			assert(ccv_nnc_tensor_nd(a->info.dim) <= CCV_NNC_MAX_DIM + 2);
			assert(ccv_nnc_tensor_nd(h->info.dim) <= CCV_NNC_MAX_DIM + 2);
			assert(ccv_nnc_tensor_view_check_dim(a, dim));
			assert(ccv_nnc_tensor_view_check_dim(h, dim));
			ccv_nnc_tensor_view_get_stride(a, astride);
			ccv_nnc_tensor_view_get_stride(h, hstride);
			if (!CCV_IS_TENSOR_VIEW(g) && !CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(b) && !CCV_IS_TENSOR_VIEW(h))
			{
				// Super optimal case, just do one for-loop for sum.
				const int tensor_count = ccv_nnc_tensor_count(g->info);
				for (x = 0; x < tensor_count; x++)
					h->data.f32[x] = g->data.f32[x] * b->data.f32[x] / a->data.f32[x];
				continue;
			}
			assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
			int i[CCV_NNC_MAX_DIM + 2];
			float* const gp = g->data.f32;
			float* const ap = a->data.f32;
			float* const bp = b->data.f32;
			float* const hp = h->data.f32;
			const int count = dim[2] * dim[3];
			if (gstride[2] == dim[3] && astride[2] == dim[3] && bstride[2] == dim[3] && hstride[2] == dim[3])
			{
				// Special casing if the ainc[3] is the same as dim[3]
				for (i[0] = 0; i[0] < dim[0]; i[0]++)
				{
					float* gp0 = gp + i[0] * gstride[0];
					float* ap0 = ap + i[0] * astride[0];
					float* bp0 = bp + i[0] * bstride[0];
					float* hp0 = hp + i[0] * hstride[0];
					for (i[1] = 0; i[1] < dim[1]; i[1]++)
					{
						for (x = 0; x < count; x++)
							hp0[x] = gp0[x] * bp0[x] / ap0[x];
						gp0 += gstride[1];
						ap0 += astride[1];
						bp0 += bstride[1];
						hp0 += hstride[1];
					}
				}
				continue;
			}
			// Non-optimal case, need to do skip copy.
			for (i[0] = 0; i[0] < dim[0]; i[0]++)
			{
				float* const gp0 = gp + i[0] * gstride[0];
				float* const ap0 = ap + i[0] * astride[0];
				float* const bp0 = bp + i[0] * bstride[0];
				float* const hp0 = hp + i[0] * hstride[0];
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					float* gp1 = gp0 + i[1] * gstride[1];
					float* ap1 = ap0 + i[1] * astride[1];
					float* bp1 = bp0 + i[1] * bstride[1];
					float* hp1 = hp0 + i[1] * hstride[1];
					for (i[2] = 0; i[2] < dim[2]; i[2]++)
					{
						for (x = 0; x < dim[3]; x++)
							hp1[x] = gp1[x] * bp1[x] / ap1[x];
						gp1 += gstride[2];
						ap1 += astride[2];
						bp1 += bstride[2];
						hp1 += hstride[2];
					}
				}
			}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static void _ccv_nnc_ewdiv_forw_cpu_ref(const float p, ccv_nnc_tensor_view_t* const a, ccv_nnc_tensor_view_t* const b, ccv_nnc_tensor_view_t* const c)
{
	// Assuming this is float 32.
	int dim[CCV_NNC_MAX_DIM_ALLOC];
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
	int cstride[CCV_NNC_MAX_DIM_ALLOC];
	if (a == 0) // Take 0 as all ones tensor.
	{
		assert(ccv_nnc_tensor_nd(b->info.dim) <= CCV_NNC_MAX_DIM + 2);
		assert(ccv_nnc_tensor_nd(c->info.dim) <= CCV_NNC_MAX_DIM + 2);
		ccv_nnc_tensor_view_get_dim(b, dim);
		assert(ccv_nnc_tensor_view_check_dim(c, dim));
		int x;
		if (!CCV_IS_TENSOR_VIEW(b) && !CCV_IS_TENSOR_VIEW(c))
		{
			// Super optimal case, just do one for-loop for sum.
			const int tensor_count = ccv_nnc_tensor_count(b->info);
			for (x = 0; x < tensor_count; x++)
				c->data.f32[x] = p / b->data.f32[x];
			return;
		}
		assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
		ccv_nnc_tensor_view_get_stride(b, bstride);
		ccv_nnc_tensor_view_get_stride(c, cstride);
		int i[CCV_NNC_MAX_DIM + 2];
		float* const bp = b->data.f32;
		float* const cp = c->data.f32;
		const int count = dim[2] * dim[3];
		if (bstride[2] == dim[3] && cstride[2] == dim[3])
		{
			// Special casing if the ainc[3] is the same as dim[3]
			for (i[0] = 0; i[0] < dim[0]; i[0]++)
			{
				float* bp0 = bp + i[0] * bstride[0];
				float* cp0 = cp + i[0] * cstride[0];
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					for (x = 0; x < count; x++)
						cp0[x] = p / bp0[x];
					bp0 += bstride[1];
					cp0 += cstride[1];
				}
			}
			return;
		}
		// Non-optimal case, need to do skip copy.
		for (i[0] = 0; i[0] < dim[0]; i[0]++)
		{
			float* const bp0 = bp + i[0] * bstride[0];
			float* const cp0 = cp + i[0] * cstride[0];
			for (i[1] = 0; i[1] < dim[1]; i[1]++)
			{
				float* bp1 = bp0 + i[1] * bstride[1];
				float* cp1 = cp0 + i[1] * cstride[1];
				for (i[2] = 0; i[2] < dim[2]; i[2]++)
				{
					for (x = 0; x < dim[3]; x++)
						cp1[x] = p / bp1[x];
					bp1 += bstride[2];
					cp1 += cstride[2];
				}
			}
		}
	} else {
		assert(ccv_nnc_tensor_nd(a->info.dim) <= CCV_NNC_MAX_DIM + 2);
		assert(ccv_nnc_tensor_nd(b->info.dim) <= CCV_NNC_MAX_DIM + 2);
		assert(ccv_nnc_tensor_nd(c->info.dim) <= CCV_NNC_MAX_DIM + 2);
		ccv_nnc_tensor_view_get_dim(a, dim);
		assert(ccv_nnc_tensor_view_check_dim(b, dim));
		assert(ccv_nnc_tensor_view_check_dim(c, dim));
		int x;
		if (!CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(b) && !CCV_IS_TENSOR_VIEW(c))
		{
			// Super optimal case, just do one for-loop for sum.
			const int tensor_count = ccv_nnc_tensor_count(a->info);
			for (x = 0; x < tensor_count; x++)
				c->data.f32[x] = p * a->data.f32[x] / b->data.f32[x];
			return;
		}
		assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
		ccv_nnc_tensor_view_get_stride(a, astride);
		ccv_nnc_tensor_view_get_stride(b, bstride);
		ccv_nnc_tensor_view_get_stride(c, cstride);
		int i[CCV_NNC_MAX_DIM + 2];
		float* const ap = a->data.f32;
		float* const bp = b->data.f32;
		float* const cp = c->data.f32;
		const int count = dim[2] * dim[3];
		if (astride[2] == dim[3] && bstride[2] == dim[3] && cstride[2] == dim[3])
		{
			// Special casing if the ainc[3] is the same as dim[3]
			for (i[0] = 0; i[0] < dim[0]; i[0]++)
			{
				float* ap0 = ap + i[0] * astride[0];
				float* bp0 = bp + i[0] * bstride[0];
				float* cp0 = cp + i[0] * cstride[0];
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					for (x = 0; x < count; x++)
						cp0[x] = p * ap0[x] / bp0[x];
					ap0 += astride[1];
					bp0 += bstride[1];
					cp0 += cstride[1];
				}
			}
			return;
		}
		// Non-optimal case, need to do skip copy.
		for (i[0] = 0; i[0] < dim[0]; i[0]++)
		{
			float* const ap0 = ap + i[0] * astride[0];
			float* const bp0 = bp + i[0] * bstride[0];
			float* const cp0 = cp + i[0] * cstride[0];
			for (i[1] = 0; i[1] < dim[1]; i[1]++)
			{
				float* ap1 = ap0 + i[1] * astride[1];
				float* bp1 = bp0 + i[1] * bstride[1];
				float* cp1 = cp0 + i[1] * cstride[1];
				for (i[2] = 0; i[2] < dim[2]; i[2]++)
				{
					for (x = 0; x < dim[3]; x++)
						cp1[x] = p * ap1[x] / bp1[x];
					ap1 += astride[2];
					bp1 += bstride[2];
					cp1 += cstride[2];
				}
			}
		}
	}
}

static int _ccv_nnc_ewdiv_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	_ccv_nnc_ewdiv_forw_cpu_ref(1, (ccv_nnc_tensor_view_t*)inputs[0], (ccv_nnc_tensor_view_t*)inputs[1], (ccv_nnc_tensor_view_t*)outputs[0]);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_ewdiv_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	// D[x / y, x] = 1 / y, D[x / y, y] = -x / y^2
	if (output_size == 1 || outputs[1] == 0)
	{
		// When we only need D[x / y, x]
		_ccv_nnc_ewdiv_forw_cpu_ref(1, (ccv_nnc_tensor_view_t*)inputs[0], (ccv_nnc_tensor_view_t*)inputs[2], (ccv_nnc_tensor_view_t*)outputs[0]);
		return CCV_NNC_EXEC_SUCCESS;
	}
	int dim[CCV_NNC_MAX_DIM_ALLOC];
	int gstride[CCV_NNC_MAX_DIM_ALLOC];
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
	int cstride[CCV_NNC_MAX_DIM_ALLOC];
	int hastride[CCV_NNC_MAX_DIM_ALLOC];
	int hbstride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_t* g = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)inputs[2];
	ccv_nnc_tensor_view_t* c = (ccv_nnc_tensor_view_t*)inputs[3];
	ccv_nnc_tensor_view_t* ha = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* hb = (ccv_nnc_tensor_view_t*)outputs[1];
	if (g == 0)
	{
		assert(ccv_nnc_tensor_nd(b->info.dim) <= CCV_NNC_MAX_DIM + 2);
		assert(ccv_nnc_tensor_nd(c->info.dim) <= CCV_NNC_MAX_DIM + 2);
		assert(ccv_nnc_tensor_nd(hb->info.dim) <= CCV_NNC_MAX_DIM + 2);
		ccv_nnc_tensor_view_get_dim(b, dim);
		assert(ccv_nnc_tensor_view_check_dim(c, dim));
		assert(ccv_nnc_tensor_view_check_dim(hb, dim));
		if (ha)
		{
			assert(ccv_nnc_tensor_nd(ha->info.dim) <= CCV_NNC_MAX_DIM + 2);
			assert(ccv_nnc_tensor_view_check_dim(ha, dim));
		}
		int x;
		if (!CCV_IS_TENSOR_VIEW(b) && !CCV_IS_TENSOR_VIEW(c) && (ha == 0 || !CCV_IS_TENSOR_VIEW(ha)) && !CCV_IS_TENSOR_VIEW(hb))
		{
			// Super optimal case, just do one for-loop for sum.
			const int tensor_count = ccv_nnc_tensor_count(b->info);
			if (ha == 0)
			{
				for (x = 0; x < tensor_count; x++)
				{
					const float v = 1 / b->data.f32[x];
					hb->data.f32[x] = -c->data.f32[x] * v;
				}
			} else {
				for (x = 0; x < tensor_count; x++)
				{
					const float v = 1 / b->data.f32[x];
					ha->data.f32[x] = v;
					hb->data.f32[x] = -c->data.f32[x] * v;
				}
			}
			return CCV_NNC_EXEC_SUCCESS;
		}
		assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
		ccv_nnc_tensor_view_get_stride(b, bstride);
		ccv_nnc_tensor_view_get_stride(c, cstride);
		ccv_nnc_tensor_view_get_stride(hb, hbstride);
		int i[CCV_NNC_MAX_DIM + 2];
		float* const bp = b->data.f32;
		float* const cp = c->data.f32;
		float* const hbp = hb->data.f32;
		const int count = dim[2] * dim[3];
		if (ha == 0)
		{
			if (bstride[2] == dim[3] && cstride[2] == dim[3] && hbstride[2] == dim[3])
			{
				// Special casing if the ainc[3] is the same as dim[3]
				for (i[0] = 0; i[0] < dim[0]; i[0]++)
				{
					float* bp0 = bp + i[0] * bstride[0];
					float* cp0 = cp + i[0] * cstride[0];
					float* hbp0 = hbp + i[0] * hbstride[0];
					for (i[1] = 0; i[1] < dim[1]; i[1]++)
					{
						for (x = 0; x < count; x++)
						{
							const float v = 1 / bp0[x];
							hbp0[x] = -cp0[x] * v;
						}
						bp0 += bstride[1];
						cp0 += cstride[1];
						hbp0 += hbstride[1];
					}
				}
				return CCV_NNC_EXEC_SUCCESS;
			}
			// Non-optimal case, need to do skip copy.
			for (i[0] = 0; i[0] < dim[0]; i[0]++)
			{
				float* const bp0 = bp + i[0] * bstride[0];
				float* const cp0 = cp + i[0] * cstride[0];
				float* const hbp0 = hbp + i[0] * hbstride[0];
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					float* bp1 = bp0 + i[1] * bstride[1];
					float* cp1 = cp0 + i[1] * cstride[1];
					float* hbp1 = hbp0 + i[1] * hbstride[1];
					for (i[2] = 0; i[2] < dim[2]; i[2]++)
					{
						for (x = 0; x < dim[3]; x++)
						{
							const float v = 1 / bp1[x];
							hbp1[x] = -cp1[x] * v;
						}
						bp1 += bstride[2];
						cp1 += cstride[2];
						hbp1 += hbstride[2];
					}
				}
			}
		} else {
			float* const hap = ha->data.f32;
			ccv_nnc_tensor_view_get_stride(ha, hastride);
			if (bstride[2] == dim[3] && cstride[2] == dim[3] && hastride[2] == dim[3] && hbstride[2] == dim[3])
			{
				// Special casing if the ainc[3] is the same as dim[3]
				for (i[0] = 0; i[0] < dim[0]; i[0]++)
				{
					float* bp0 = bp + i[0] * bstride[0];
					float* cp0 = cp + i[0] * cstride[0];
					float* hap0 = hap + i[0] * hastride[0];
					float* hbp0 = hbp + i[0] * hbstride[0];
					for (i[1] = 0; i[1] < dim[1]; i[1]++)
					{
						for (x = 0; x < count; x++)
						{
							const float v = 1 / bp0[x];
							hap0[x] = v;
							hbp0[x] = -cp0[x] * v;
						}
						bp0 += bstride[1];
						cp0 += cstride[1];
						hap0 += hastride[1];
						hbp0 += hbstride[1];
					}
				}
				return CCV_NNC_EXEC_SUCCESS;
			}
			// Non-optimal case, need to do skip copy.
			for (i[0] = 0; i[0] < dim[0]; i[0]++)
			{
				float* const bp0 = bp + i[0] * bstride[0];
				float* const cp0 = cp + i[0] * cstride[0];
				float* const hap0 = hap + i[0] * hastride[0];
				float* const hbp0 = hbp + i[0] * hbstride[0];
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					float* bp1 = bp0 + i[1] * bstride[1];
					float* cp1 = cp0 + i[1] * cstride[1];
					float* hap1 = hap0 + i[1] * hastride[1];
					float* hbp1 = hbp0 + i[1] * hbstride[1];
					for (i[2] = 0; i[2] < dim[2]; i[2]++)
					{
						for (x = 0; x < dim[3]; x++)
						{
							const float v = 1 / bp1[x];
							hap1[x] = v;
							hbp1[x] = -cp1[x] * v;
						}
						bp1 += bstride[2];
						cp1 += cstride[2];
						hap1 += hastride[2];
						hbp1 += hbstride[2];
					}
				}
			}
		}
	} else {
		assert(ccv_nnc_tensor_nd(g->info.dim) <= CCV_NNC_MAX_DIM + 2);
		assert(ccv_nnc_tensor_nd(b->info.dim) <= CCV_NNC_MAX_DIM + 2);
		assert(ccv_nnc_tensor_nd(c->info.dim) <= CCV_NNC_MAX_DIM + 2);
		assert(ccv_nnc_tensor_nd(hb->info.dim) <= CCV_NNC_MAX_DIM + 2);
		ccv_nnc_tensor_view_get_dim(b, dim);
		assert(ccv_nnc_tensor_view_check_dim(g, dim));
		assert(ccv_nnc_tensor_view_check_dim(c, dim));
		assert(ccv_nnc_tensor_view_check_dim(hb, dim));
		if (ha)
		{
			assert(ccv_nnc_tensor_nd(ha->info.dim) <= CCV_NNC_MAX_DIM + 2);
			assert(ccv_nnc_tensor_view_check_dim(ha, dim));
		}
		int x;
		if (!CCV_IS_TENSOR_VIEW(g) && !CCV_IS_TENSOR_VIEW(b) && !CCV_IS_TENSOR_VIEW(c) && (ha == 0 || !CCV_IS_TENSOR_VIEW(ha)) && !CCV_IS_TENSOR_VIEW(hb))
		{
			// Super optimal case, just do one for-loop for sum.
			const int tensor_count = ccv_nnc_tensor_count(g->info);
			if (ha == 0)
			{
				for (x = 0; x < tensor_count; x++)
				{
					const float v = g->data.f32[x] / b->data.f32[x];
					hb->data.f32[x] = -c->data.f32[x] * v;
				}
			} else {
				for (x = 0; x < tensor_count; x++)
				{
					const float v = g->data.f32[x] / b->data.f32[x];
					ha->data.f32[x] = v;
					hb->data.f32[x] = -c->data.f32[x] * v;
				}
			}
			return CCV_NNC_EXEC_SUCCESS;
		}
		assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
		ccv_nnc_tensor_view_get_stride(g, gstride);
		ccv_nnc_tensor_view_get_stride(b, bstride);
		ccv_nnc_tensor_view_get_stride(c, cstride);
		ccv_nnc_tensor_view_get_stride(hb, hbstride);
		int i[CCV_NNC_MAX_DIM + 2];
		float* const gp = g->data.f32;
		float* const bp = b->data.f32;
		float* const cp = c->data.f32;
		float* const hbp = hb->data.f32;
		const int count = dim[2] * dim[3];
		if (ha == 0)
		{
			if (gstride[2] == dim[3] && bstride[2] == dim[3] && cstride[2] == dim[3] && hbstride[2] == dim[3])
			{
				// Special casing if the ainc[3] is the same as dim[3]
				for (i[0] = 0; i[0] < dim[0]; i[0]++)
				{
					float* gp0 = gp + i[0] * gstride[0];
					float* bp0 = bp + i[0] * bstride[0];
					float* cp0 = cp + i[0] * cstride[0];
					float* hbp0 = hbp + i[0] * hbstride[0];
					for (i[1] = 0; i[1] < dim[1]; i[1]++)
					{
						for (x = 0; x < count; x++)
						{
							const float v = gp0[x] / bp0[x];
							hbp0[x] = -cp0[x] * v;
						}
						gp0 += gstride[1];
						bp0 += bstride[1];
						cp0 += cstride[1];
						hbp0 += hbstride[1];
					}
				}
				return CCV_NNC_EXEC_SUCCESS;
			}
			// Non-optimal case, need to do skip copy.
			for (i[0] = 0; i[0] < dim[0]; i[0]++)
			{
				float* const gp0 = gp + i[0] * gstride[0];
				float* const bp0 = bp + i[0] * bstride[0];
				float* const cp0 = cp + i[0] * cstride[0];
				float* const hbp0 = hbp + i[0] * hbstride[0];
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					float* gp1 = gp0 + i[1] * gstride[1];
					float* bp1 = bp0 + i[1] * bstride[1];
					float* cp1 = cp0 + i[1] * cstride[1];
					float* hbp1 = hbp0 + i[1] * hbstride[1];
					for (i[2] = 0; i[2] < dim[2]; i[2]++)
					{
						for (x = 0; x < dim[3]; x++)
						{
							const float v = gp1[x] / bp1[x];
							hbp1[x] = -cp1[x] * v;
						}
						gp1 += gstride[2];
						bp1 += bstride[2];
						cp1 += cstride[2];
						hbp1 += hbstride[2];
					}
				}
			}
		} else {
			ccv_nnc_tensor_view_get_stride(ha, hastride);
			float* const hap = ha->data.f32;
			if (gstride[2] == dim[3] && bstride[2] == dim[3] && cstride[2] == dim[3] && hastride[2] == dim[3] && hbstride[2] == dim[3])
			{
				// Special casing if the ainc[3] is the same as dim[3]
				for (i[0] = 0; i[0] < dim[0]; i[0]++)
				{
					float* gp0 = gp + i[0] * gstride[0];
					float* bp0 = bp + i[0] * bstride[0];
					float* cp0 = cp + i[0] * cstride[0];
					float* hap0 = hap + i[0] * hastride[0];
					float* hbp0 = hbp + i[0] * hbstride[0];
					for (i[1] = 0; i[1] < dim[1]; i[1]++)
					{
						for (x = 0; x < count; x++)
						{
							const float v = gp0[x] / bp0[x];
							hap0[x] = v;
							hbp0[x] = -cp0[x] * v;
						}
						gp0 += gstride[1];
						bp0 += bstride[1];
						cp0 += cstride[1];
						hap0 += hastride[1];
						hbp0 += hbstride[1];
					}
				}
				return CCV_NNC_EXEC_SUCCESS;
			}
			// Non-optimal case, need to do skip copy.
			for (i[0] = 0; i[0] < dim[0]; i[0]++)
			{
				float* const gp0 = gp + i[0] * gstride[0];
				float* const bp0 = bp + i[0] * bstride[0];
				float* const cp0 = cp + i[0] * cstride[0];
				float* const hap0 = hap + i[0] * hastride[0];
				float* const hbp0 = hbp + i[0] * hbstride[0];
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					float* gp1 = gp0 + i[1] * gstride[1];
					float* bp1 = bp0 + i[1] * bstride[1];
					float* cp1 = cp0 + i[1] * cstride[1];
					float* hap1 = hap0 + i[1] * hastride[1];
					float* hbp1 = hbp0 + i[1] * hbstride[1];
					for (i[2] = 0; i[2] < dim[2]; i[2]++)
					{
						for (x = 0; x < dim[3]; x++)
						{
							const float v = gp1[x] / bp1[x];
							hap1[x] = v;
							hbp1[x] = -cp1[x] * v;
						}
						gp1 += gstride[2];
						bp1 += bstride[2];
						cp1 += cstride[2];
						hap1 += hastride[2];
						hbp1 += hbstride[2];
					}
				}
			}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_ewexp_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	// Assuming this is float 32.
	int dim[CCV_NNC_MAX_DIM_ALLOC];
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(ccv_nnc_tensor_nd(a->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(b->info.dim) <= CCV_NNC_MAX_DIM + 2);
	ccv_nnc_tensor_view_get_dim(a, dim);
	assert(ccv_nnc_tensor_view_check_dim(b, dim));
	int x;
	if (!CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(b))
	{
		// Super optimal case, just do one for-loop for sum.
		const int tensor_count = ccv_nnc_tensor_count(a->info);
		for (x = 0; x < tensor_count; x++)
			b->data.f32[x] = exp(a->data.f32[x]);
		return CCV_NNC_EXEC_SUCCESS;
	}
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	ccv_nnc_tensor_view_get_stride(a, astride);
	ccv_nnc_tensor_view_get_stride(b, bstride);
	int i[CCV_NNC_MAX_DIM + 2];
	float* const ap = a->data.f32;
	float* const bp = b->data.f32;
	const int count = dim[2] * dim[3];
	if (astride[2] == dim[3] && bstride[2] == dim[3])
	{
		// Special casing if the ainc[3] is the same as dim[3]
		for (i[0] = 0; i[0] < dim[0]; i[0]++)
		{
			float* ap0 = ap + i[0] * astride[0];
			float* bp0 = bp + i[0] * bstride[0];
			for (i[1] = 0; i[1] < dim[1]; i[1]++)
			{
				for (x = 0; x < count; x++)
					bp0[x] = exp(ap0[x]);
				ap0 += astride[1];
				bp0 += bstride[1];
			}
		}
		return CCV_NNC_EXEC_SUCCESS;
	}
	// Non-optimal case, need to do skip copy.
	for (i[0] = 0; i[0] < dim[0]; i[0]++)
	{
		float* const ap0 = ap + i[0] * astride[0];
		float* const bp0 = bp + i[0] * bstride[0];
		for (i[1] = 0; i[1] < dim[1]; i[1]++)
		{
			float* ap1 = ap0 + i[1] * astride[1];
			float* bp1 = bp0 + i[1] * bstride[1];
			for (i[2] = 0; i[2] < dim[2]; i[2]++)
			{
				for (x = 0; x < dim[3]; x++)
					bp1[x] = exp(ap1[x]);
				ap1 += astride[2];
				bp1 += bstride[2];
			}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_ewexp_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	// D[Exp[x], x] = Exp[x]
	if (inputs[0] == 0)
		_ccv_nnc_tensor_transfer_cpu_ref_f32((ccv_nnc_tensor_view_t*)inputs[2], (ccv_nnc_tensor_view_t*)outputs[0]);
	else
		_ccv_nnc_ewprod_forw_cpu_ref((ccv_nnc_tensor_view_t*[]){
			(ccv_nnc_tensor_view_t*)inputs[0], (ccv_nnc_tensor_view_t*)inputs[2]
		}, 2, (ccv_nnc_tensor_view_t**)outputs, output_size);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_ewlog_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	// Assuming this is float 32.
	int dim[CCV_NNC_MAX_DIM_ALLOC];
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(ccv_nnc_tensor_nd(a->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(b->info.dim) <= CCV_NNC_MAX_DIM + 2);
	ccv_nnc_tensor_view_get_dim(a, dim);
	assert(ccv_nnc_tensor_view_check_dim(b, dim));
	int x;
	if (!CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(b))
	{
		// Super optimal case, just do one for-loop for sum.
		const int tensor_count = ccv_nnc_tensor_count(a->info);
		for (x = 0; x < tensor_count; x++)
			b->data.f32[x] = log(a->data.f32[x]);
		return CCV_NNC_EXEC_SUCCESS;
	}
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	ccv_nnc_tensor_view_get_stride(a, astride);
	ccv_nnc_tensor_view_get_stride(b, bstride);
	int i[CCV_NNC_MAX_DIM + 2];
	float* const ap = a->data.f32;
	float* const bp = b->data.f32;
	const int count = dim[2] * dim[3];
	if (astride[2] == dim[3] && bstride[2] == dim[3])
	{
		// Special casing if the ainc[3] is the same as dim[3]
		for (i[0] = 0; i[0] < dim[0]; i[0]++)
		{
			float* ap0 = ap + i[0] * astride[0];
			float* bp0 = bp + i[0] * bstride[0];
			for (i[1] = 0; i[1] < dim[1]; i[1]++)
			{
				for (x = 0; x < count; x++)
					bp0[x] = log(ap0[x]);
				ap0 += astride[1];
				bp0 += bstride[1];
			}
		}
		return CCV_NNC_EXEC_SUCCESS;
	}
	// Non-optimal case, need to do skip copy.
	for (i[0] = 0; i[0] < dim[0]; i[0]++)
	{
		float* const ap0 = ap + i[0] * astride[0];
		float* const bp0 = bp + i[0] * bstride[0];
		for (i[1] = 0; i[1] < dim[1]; i[1]++)
		{
			float* ap1 = ap0 + i[1] * astride[1];
			float* bp1 = bp0 + i[1] * bstride[1];
			for (i[2] = 0; i[2] < dim[2]; i[2]++)
			{
				for (x = 0; x < dim[3]; x++)
					bp1[x] = log(ap1[x]);
				ap1 += astride[2];
				bp1 += bstride[2];
			}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_ewlog_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	// D[Log[x], x] = 1 / x
	_ccv_nnc_ewdiv_forw_cpu_ref(1, (ccv_nnc_tensor_view_t*)inputs[0], (ccv_nnc_tensor_view_t*)inputs[1], (ccv_nnc_tensor_view_t*)outputs[0]);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_ewsqrt_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	// Assuming this is float 32.
	int dim[CCV_NNC_MAX_DIM_ALLOC];
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(ccv_nnc_tensor_nd(a->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(b->info.dim) <= CCV_NNC_MAX_DIM + 2);
	ccv_nnc_tensor_view_get_dim(a, dim);
	assert(ccv_nnc_tensor_view_check_dim(b, dim));
	int x;
	if (!CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(b))
	{
		// Super optimal case, just do one for-loop for sum.
		const int tensor_count = ccv_nnc_tensor_count(a->info);
		for (x = 0; x < tensor_count; x++)
			b->data.f32[x] = sqrt(a->data.f32[x]);
		return CCV_NNC_EXEC_SUCCESS;
	}
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	ccv_nnc_tensor_view_get_stride(a, astride);
	ccv_nnc_tensor_view_get_stride(b, bstride);
	int i[CCV_NNC_MAX_DIM + 2];
	float* const ap = a->data.f32;
	float* const bp = b->data.f32;
	const int count = dim[2] * dim[3];
	if (astride[2] == dim[3] && bstride[2] == dim[3])
	{
		// Special casing if the ainc[3] is the same as dim[3]
		for (i[0] = 0; i[0] < dim[0]; i[0]++)
		{
			float* ap0 = ap + i[0] * astride[0];
			float* bp0 = bp + i[0] * bstride[0];
			for (i[1] = 0; i[1] < dim[1]; i[1]++)
			{
				for (x = 0; x < count; x++)
					bp0[x] = sqrt(ap0[x]);
				ap0 += astride[1];
				bp0 += bstride[1];
			}
		}
		return CCV_NNC_EXEC_SUCCESS;
	}
	// Non-optimal case, need to do skip copy.
	for (i[0] = 0; i[0] < dim[0]; i[0]++)
	{
		float* const ap0 = ap + i[0] * astride[0];
		float* const bp0 = bp + i[0] * bstride[0];
		for (i[1] = 0; i[1] < dim[1]; i[1]++)
		{
			float* ap1 = ap0 + i[1] * astride[1];
			float* bp1 = bp0 + i[1] * bstride[1];
			for (i[2] = 0; i[2] < dim[2]; i[2]++)
			{
				for (x = 0; x < dim[3]; x++)
					bp1[x] = sqrt(ap1[x]);
				ap1 += astride[2];
				bp1 += bstride[2];
			}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_ewsqrt_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	// D[Sqrt[x], x] = 0.5 / Sqrt[x]
	_ccv_nnc_ewdiv_forw_cpu_ref(0.5, (ccv_nnc_tensor_view_t*)inputs[0], (ccv_nnc_tensor_view_t*)inputs[2], (ccv_nnc_tensor_view_t*)outputs[0]);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_clamp_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	// Assuming this is float 32.
	int dim[CCV_NNC_MAX_DIM_ALLOC];
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(ccv_nnc_tensor_nd(a->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(b->info.dim) <= CCV_NNC_MAX_DIM + 2);
	ccv_nnc_tensor_view_get_dim(a, dim);
	assert(ccv_nnc_tensor_view_check_dim(b, dim));
	int x;
	const float min = cmd.info.clamp.min;
	const float max = cmd.info.clamp.max;
	assert(!isnan(min) || !isnan(max));
	if (!CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(b))
	{
		// Super optimal case, just do one for-loop for sum.
		const int tensor_count = ccv_nnc_tensor_count(a->info);
		if (isnan(min))
		{
			for (x = 0; x < tensor_count; x++)
				b->data.f32[x] = ccv_min(a->data.f32[x], max);
		} else if (isnan(max)) {
			for (x = 0; x < tensor_count; x++)
				b->data.f32[x] = ccv_max(a->data.f32[x], min);
		} else {
			for (x = 0; x < tensor_count; x++)
				b->data.f32[x] = ccv_clamp(a->data.f32[x], min, max);
		}
		return CCV_NNC_EXEC_SUCCESS;
	}
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	ccv_nnc_tensor_view_get_stride(a, astride);
	ccv_nnc_tensor_view_get_stride(b, bstride);
	int i[CCV_NNC_MAX_DIM + 2];
	float* const ap = a->data.f32;
	float* const bp = b->data.f32;
	const int count = dim[2] * dim[3];
	if (isnan(min))
	{
		if (astride[2] == dim[3] && bstride[2] == dim[3])
		{
			// Special casing if the ainc[3] is the same as dim[3]
			for (i[0] = 0; i[0] < dim[0]; i[0]++)
			{
				float* ap0 = ap + i[0] * astride[0];
				float* bp0 = bp + i[0] * bstride[0];
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					for (x = 0; x < count; x++)
						bp0[x] = ccv_min(ap0[x], max);
					ap0 += astride[1];
					bp0 += bstride[1];
				}
			}
			return CCV_NNC_EXEC_SUCCESS;
		}
		// Non-optimal case, need to do skip copy.
		for (i[0] = 0; i[0] < dim[0]; i[0]++)
		{
			float* const ap0 = ap + i[0] * astride[0];
			float* const bp0 = bp + i[0] * bstride[0];
			for (i[1] = 0; i[1] < dim[1]; i[1]++)
			{
				float* ap1 = ap0 + i[1] * astride[1];
				float* bp1 = bp0 + i[1] * bstride[1];
				for (i[2] = 0; i[2] < dim[2]; i[2]++)
				{
					for (x = 0; x < dim[3]; x++)
						bp1[x] = ccv_min(ap1[x], max);
					ap1 += astride[2];
					bp1 += bstride[2];
				}
			}
		}
	} else if (isnan(max)) {
		if (astride[2] == dim[3] && bstride[2] == dim[3])
		{
			// Special casing if the ainc[3] is the same as dim[3]
			for (i[0] = 0; i[0] < dim[0]; i[0]++)
			{
				float* ap0 = ap + i[0] * astride[0];
				float* bp0 = bp + i[0] * bstride[0];
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					for (x = 0; x < count; x++)
						bp0[x] = ccv_max(ap0[x], min);
					ap0 += astride[1];
					bp0 += bstride[1];
				}
			}
			return CCV_NNC_EXEC_SUCCESS;
		}
		// Non-optimal case, need to do skip copy.
		for (i[0] = 0; i[0] < dim[0]; i[0]++)
		{
			float* const ap0 = ap + i[0] * astride[0];
			float* const bp0 = bp + i[0] * bstride[0];
			for (i[1] = 0; i[1] < dim[1]; i[1]++)
			{
				float* ap1 = ap0 + i[1] * astride[1];
				float* bp1 = bp0 + i[1] * bstride[1];
				for (i[2] = 0; i[2] < dim[2]; i[2]++)
				{
					for (x = 0; x < dim[3]; x++)
						bp1[x] = ccv_max(ap1[x], min);
					ap1 += astride[2];
					bp1 += bstride[2];
				}
			}
		}
	} else {
		if (astride[2] == dim[3] && bstride[2] == dim[3])
		{
			// Special casing if the ainc[3] is the same as dim[3]
			for (i[0] = 0; i[0] < dim[0]; i[0]++)
			{
				float* ap0 = ap + i[0] * astride[0];
				float* bp0 = bp + i[0] * bstride[0];
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					for (x = 0; x < count; x++)
						bp0[x] = ccv_clamp(ap0[x], min, max);
					ap0 += astride[1];
					bp0 += bstride[1];
				}
			}
			return CCV_NNC_EXEC_SUCCESS;
		}
		// Non-optimal case, need to do skip copy.
		for (i[0] = 0; i[0] < dim[0]; i[0]++)
		{
			float* const ap0 = ap + i[0] * astride[0];
			float* const bp0 = bp + i[0] * bstride[0];
			for (i[1] = 0; i[1] < dim[1]; i[1]++)
			{
				float* ap1 = ap0 + i[1] * astride[1];
				float* bp1 = bp0 + i[1] * bstride[1];
				for (i[2] = 0; i[2] < dim[2]; i[2]++)
				{
					for (x = 0; x < dim[3]; x++)
						bp1[x] = ccv_clamp(ap1[x], min, max);
					ap1 += astride[2];
					bp1 += bstride[2];
				}
			}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_clamp_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 3);
	const ccv_nnc_tensor_view_t* g = (ccv_nnc_tensor_view_t*)inputs[0]; // gradient
	const ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)inputs[2];
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* h = (ccv_nnc_tensor_view_t*)outputs[0];
	// Assuming this is float 32.
	int dim[CCV_NNC_MAX_DIM_ALLOC];
	int hstride[CCV_NNC_MAX_DIM_ALLOC];
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
	assert(ccv_nnc_tensor_nd(h->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(b->info.dim) <= CCV_NNC_MAX_DIM + 2);
	ccv_nnc_tensor_view_get_dim(g, dim);
	ccv_nnc_tensor_view_get_dim(h, dim);
	assert(ccv_nnc_tensor_view_check_dim(b, dim));
	int x;
	const float min = cmd.info.clamp.min;
	const float max = cmd.info.clamp.max;
	assert(!isnan(min) || !isnan(max));
	if (g)
	{
		if (!CCV_IS_TENSOR_VIEW(g) && !CCV_IS_TENSOR_VIEW(h) && !CCV_IS_TENSOR_VIEW(b))
		{
			// Super optimal case, just do one for-loop for sum.
			const int tensor_count = ccv_nnc_tensor_count(g->info);
			if (isnan(min))
			{
				for (x = 0; x < tensor_count; x++)
					h->data.f32[x] = b->data.f32[x] >= max ? 0 : g->data.f32[x];
			} else if (isnan(max)) {
				for (x = 0; x < tensor_count; x++)
					h->data.f32[x] = b->data.f32[x] <= min ? 0 : g->data.f32[x];
			} else {
				for (x = 0; x < tensor_count; x++)
					h->data.f32[x] = (b->data.f32[x] >= max || b->data.f32[x] <= min) ? 0 : g->data.f32[x];
			}
			return CCV_NNC_EXEC_SUCCESS;
		}
		int gstride[CCV_NNC_MAX_DIM_ALLOC];
		assert(ccv_nnc_tensor_nd(g->info.dim) <= CCV_NNC_MAX_DIM + 2);
		assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
		ccv_nnc_tensor_view_get_stride(g, gstride);
		ccv_nnc_tensor_view_get_stride(b, bstride);
		ccv_nnc_tensor_view_get_stride(h, hstride);
		int i[CCV_NNC_MAX_DIM + 2];
		float* const gp = g->data.f32;
		float* const bp = b->data.f32;
		float* const hp = h->data.f32;
		const int count = dim[2] * dim[3];
		const float min = cmd.info.clamp.min;
		const float max = cmd.info.clamp.max;
		assert(!isnan(min) || !isnan(max));
		if (isnan(min))
		{
			if (gstride[2] == dim[3] && bstride[2] == dim[3] && hstride[2] == dim[3])
			{
				// Special casing if the ginc[3] is the same as dim[3]
				for (i[0] = 0; i[0] < dim[0]; i[0]++)
				{
					float* gp0 = gp + i[0] * gstride[0];
					float* bp0 = bp + i[0] * bstride[0];
					float* hp0 = hp + i[0] * hstride[0];
					for (i[1] = 0; i[1] < dim[1]; i[1]++)
					{
						for (x = 0; x < count; x++)
							hp0[x] = bp0[x] >= max ? 0 : gp0[x];
						gp0 += gstride[1];
						bp0 += bstride[1];
						hp0 += hstride[1];
					}
				}
				return CCV_NNC_EXEC_SUCCESS;
			}
			// Non-optimal case, need to do skip copy.
			for (i[0] = 0; i[0] < dim[0]; i[0]++)
			{
				float* const gp0 = gp + i[0] * gstride[0];
				float* const bp0 = bp + i[0] * bstride[0];
				float* const hp0 = hp + i[0] * hstride[0];
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					float* gp1 = gp0 + i[1] * gstride[1];
					float* bp1 = bp0 + i[1] * bstride[1];
					float* hp1 = hp0 + i[1] * hstride[1];
					for (i[2] = 0; i[2] < dim[2]; i[2]++)
					{
						for (x = 0; x < dim[3]; x++)
							hp1[x] = bp1[x] >= max ? 0 : gp1[x];
						gp1 += gstride[2];
						bp1 += bstride[2];
						hp1 += hstride[2];
					}
				}
			}
		} else if (isnan(max)) {
			if (gstride[2] == dim[3] && bstride[2] == dim[3] && hstride[2] == dim[3])
			{
				// Special casing if the ginc[3] is the same as dim[3]
				for (i[0] = 0; i[0] < dim[0]; i[0]++)
				{
					float* gp0 = gp + i[0] * gstride[0];
					float* bp0 = bp + i[0] * bstride[0];
					float* hp0 = hp + i[0] * hstride[0];
					for (i[1] = 0; i[1] < dim[1]; i[1]++)
					{
						for (x = 0; x < count; x++)
							hp0[x] = bp0[x] <= min ? 0 : gp0[x];
						gp0 += gstride[1];
						bp0 += bstride[1];
						hp0 += hstride[1];
					}
				}
				return CCV_NNC_EXEC_SUCCESS;
			}
			// Non-optimal case, need to do skip copy.
			for (i[0] = 0; i[0] < dim[0]; i[0]++)
			{
				float* const gp0 = gp + i[0] * gstride[0];
				float* const bp0 = bp + i[0] * bstride[0];
				float* const hp0 = hp + i[0] * hstride[0];
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					float* gp1 = gp0 + i[1] * gstride[1];
					float* bp1 = bp0 + i[1] * bstride[1];
					float* hp1 = hp0 + i[1] * hstride[1];
					for (i[2] = 0; i[2] < dim[2]; i[2]++)
					{
						for (x = 0; x < dim[3]; x++)
							hp1[x] = bp1[x] <= min ? 0 : gp1[x];
						gp1 += gstride[2];
						bp1 += bstride[2];
						hp1 += hstride[2];
					}
				}
			}
		} else {
			if (gstride[2] == dim[3] && bstride[2] == dim[3] && hstride[2] == dim[3])
			{
				// Special casing if the ginc[3] is the same as dim[3]
				for (i[0] = 0; i[0] < dim[0]; i[0]++)
				{
					float* gp0 = gp + i[0] * gstride[0];
					float* bp0 = bp + i[0] * bstride[0];
					float* hp0 = hp + i[0] * hstride[0];
					for (i[1] = 0; i[1] < dim[1]; i[1]++)
					{
						for (x = 0; x < count; x++)
							hp0[x] = (bp0[x] >= max || bp0[x] <= min) ? 0 : gp0[x];
						gp0 += gstride[1];
						bp0 += bstride[1];
						hp0 += hstride[1];
					}
				}
				return CCV_NNC_EXEC_SUCCESS;
			}
			// Non-optimal case, need to do skip copy.
			for (i[0] = 0; i[0] < dim[0]; i[0]++)
			{
				float* const gp0 = gp + i[0] * gstride[0];
				float* const bp0 = bp + i[0] * bstride[0];
				float* const hp0 = hp + i[0] * hstride[0];
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					float* gp1 = gp0 + i[1] * gstride[1];
					float* bp1 = bp0 + i[1] * bstride[1];
					float* hp1 = hp0 + i[1] * hstride[1];
					for (i[2] = 0; i[2] < dim[2]; i[2]++)
					{
						for (x = 0; x < dim[3]; x++)
							hp1[x] = (bp1[x] >= max || bp1[x] <= min) ? 0 : gp1[x];
						gp1 += gstride[2];
						bp1 += bstride[2];
						hp1 += hstride[2];
					}
				}
			}
		}
	} else {
		if (!CCV_IS_TENSOR_VIEW(h) && !CCV_IS_TENSOR_VIEW(b))
		{
			// Super optimal case, just do one for-loop for sum.
			const int tensor_count = ccv_nnc_tensor_count(h->info);
			if (isnan(min))
			{
				for (x = 0; x < tensor_count; x++)
					h->data.f32[x] = b->data.f32[x] >= max ? 0 : 1;
			} else if (isnan(max)) {
				for (x = 0; x < tensor_count; x++)
					h->data.f32[x] = b->data.f32[x] <= min ? 0 : 1;
			} else {
				for (x = 0; x < tensor_count; x++)
					h->data.f32[x] = (b->data.f32[x] >= max || b->data.f32[x] <= min) ? 0 : 1;
			}
			return CCV_NNC_EXEC_SUCCESS;
		}
		assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
		ccv_nnc_tensor_view_get_stride(b, bstride);
		ccv_nnc_tensor_view_get_stride(h, hstride);
		int i[CCV_NNC_MAX_DIM + 2];
		float* const bp = b->data.f32;
		float* const hp = h->data.f32;
		const int count = dim[2] * dim[3];
		const float min = cmd.info.clamp.min;
		const float max = cmd.info.clamp.max;
		assert(!isnan(min) || !isnan(max));
		if (isnan(min))
		{
			if (bstride[2] == dim[3] && hstride[2] == dim[3])
			{
				// Special casing if the binc[3] is the same as dim[3]
				for (i[0] = 0; i[0] < dim[0]; i[0]++)
				{
					float* bp0 = bp + i[0] * bstride[0];
					float* hp0 = hp + i[0] * hstride[0];
					for (i[1] = 0; i[1] < dim[1]; i[1]++)
					{
						for (x = 0; x < count; x++)
							hp0[x] = bp0[x] >= max ? 0 : 1;
						bp0 += bstride[1];
						hp0 += hstride[1];
					}
				}
				return CCV_NNC_EXEC_SUCCESS;
			}
			// Non-optimal case, need to do skip copy.
			for (i[0] = 0; i[0] < dim[0]; i[0]++)
			{
				float* const bp0 = bp + i[0] * bstride[0];
				float* const hp0 = hp + i[0] * hstride[0];
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					float* bp1 = bp0 + i[1] * bstride[1];
					float* hp1 = hp0 + i[1] * hstride[1];
					for (i[2] = 0; i[2] < dim[2]; i[2]++)
					{
						for (x = 0; x < dim[3]; x++)
							hp1[x] = bp1[x] >= max ? 0 : 1;
						bp1 += bstride[2];
						hp1 += hstride[2];
					}
				}
			}
		} else if (isnan(max)) {
			if (bstride[2] == dim[3] && hstride[2] == dim[3])
			{
				// Special casing if the binc[3] is the same as dim[3]
				for (i[0] = 0; i[0] < dim[0]; i[0]++)
				{
					float* bp0 = bp + i[0] * bstride[0];
					float* hp0 = hp + i[0] * hstride[0];
					for (i[1] = 0; i[1] < dim[1]; i[1]++)
					{
						for (x = 0; x < count; x++)
							hp0[x] = bp0[x] <= min ? 0 : 1;
						bp0 += bstride[1];
						hp0 += hstride[1];
					}
				}
				return CCV_NNC_EXEC_SUCCESS;
			}
			// Non-optimal case, need to do skip copy.
			for (i[0] = 0; i[0] < dim[0]; i[0]++)
			{
				float* const bp0 = bp + i[0] * bstride[0];
				float* const hp0 = hp + i[0] * hstride[0];
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					float* bp1 = bp0 + i[1] * bstride[1];
					float* hp1 = hp0 + i[1] * hstride[1];
					for (i[2] = 0; i[2] < dim[2]; i[2]++)
					{
						for (x = 0; x < dim[3]; x++)
							hp1[x] = bp1[x] <= min ? 0 : 1;
						bp1 += bstride[2];
						hp1 += hstride[2];
					}
				}
			}
		} else {
			if (bstride[2] == dim[3] && hstride[2] == dim[3])
			{
				// Special casing if the binc[3] is the same as dim[3]
				for (i[0] = 0; i[0] < dim[0]; i[0]++)
				{
					float* bp0 = bp + i[0] * bstride[0];
					float* hp0 = hp + i[0] * hstride[0];
					for (i[1] = 0; i[1] < dim[1]; i[1]++)
					{
						for (x = 0; x < count; x++)
							hp0[x] = (bp0[x] >= max || bp0[x] <= min) ? 0 : 1;
						bp0 += bstride[1];
						hp0 += hstride[1];
					}
				}
				return CCV_NNC_EXEC_SUCCESS;
			}
			// Non-optimal case, need to do skip copy.
			for (i[0] = 0; i[0] < dim[0]; i[0]++)
			{
				float* const bp0 = bp + i[0] * bstride[0];
				float* const hp0 = hp + i[0] * hstride[0];
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					float* bp1 = bp0 + i[1] * bstride[1];
					float* hp1 = hp0 + i[1] * hstride[1];
					for (i[2] = 0; i[2] < dim[2]; i[2]++)
					{
						for (x = 0; x < dim[3]; x++)
							hp1[x] = (bp1[x] >= max || bp1[x] <= min) ? 0 : 1;
						bp1 += bstride[2];
						hp1 += hstride[2];
					}
				}
			}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_EWSUM_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_ewsum_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_EWSUM_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_ewsum_back;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_EWPROD_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_ewprod_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_EWPROD_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_ewprod_back;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_EWDIV_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_ewdiv_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_EWDIV_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_ewdiv_back;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_EWEXP_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_ewexp_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_EWEXP_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_ewexp_back;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_EWLOG_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_ewlog_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_EWLOG_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_ewlog_back;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_EWSQRT_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_ewsqrt_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_EWSQRT_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_ewsqrt_back;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_CLAMP_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_clamp_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_CLAMP_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_clamp_back;
}
