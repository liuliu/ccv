#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#ifdef USE_OPENMP
#include <omp.h>
#endif
#ifdef USE_DISPATCH
#include <dispatch/dispatch.h>
#endif

#include "../_ccv_nnc_cpu_ref.h"

void _ccv_nnc_ewsum_forw_cpu_ref(ccv_nnc_tensor_view_t* const* const inputs, const int input_size, ccv_nnc_tensor_view_t* const* const outputs, const int output_size)
{
	if (input_size == 1 && output_size == 1)
	{
		_ccv_nnc_tensor_transfer_cpu_ref(inputs[0], outputs[0]);
		return;
	}
	// Assuming this is float 32.
	int dim[CCV_NNC_MAX_DIM + 2];
	int ainc[CCV_NNC_MAX_DIM + 2];
	int binc[CCV_NNC_MAX_DIM + 2];
	int cinc[CCV_NNC_MAX_DIM + 2];
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
		assert(a->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		assert(c->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
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
		ccv_nnc_tensor_view_get_inc(a, ainc);
		ccv_nnc_tensor_view_get_inc(b, binc);
		ccv_nnc_tensor_view_get_inc(c, cinc);
		int i[CCV_NNC_MAX_DIM + 2];
		float* ap = a->data.f32;
		float* bp = b->data.f32;
		float* cp = c->data.f32;
		const int count = dim[2] * dim[3];
		if (ainc[3] == dim[3] && binc[3] == dim[3] && cinc[3] == dim[3])
		{
			// Special casing if the ainc[3] is the same as dim[3] (do memcpy for the last two dim)
			for (i[0] = 0; i[0] < dim[0]; i[0]++)
			{
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					for (x = 0; x < count; x++)
						cp[x] = ap[x] + bp[x];
					ap += ainc[2] * ainc[3];
					bp += binc[2] * binc[3];
					cp += cinc[2] * cinc[3];
				}
				ap += (ainc[1] - dim[1]) * ainc[2] * ainc[0];
				bp += (binc[1] - dim[1]) * binc[2] * binc[3];
				cp += (cinc[1] - dim[1]) * cinc[2] * cinc[3];
			}
			continue;
		}
		// Non-optimal case, need to do skip copy.
		for (i[0] = 0; i[0] < dim[0]; i[0]++)
		{
			for (i[1] = 0; i[1] < dim[1]; i[1]++)
			{
				for (i[2] = 0; i[2] < dim[2]; i[2]++)
				{
					for (x = 0; x < dim[3]; x++)
						cp[x] = ap[x] + bp[x];
					ap += ainc[3];
					bp += binc[3];
					cp += cinc[3];
				}
				ap += (ainc[2] - dim[2]) * ainc[3];
				bp += (binc[2] - dim[2]) * binc[3];
				cp += (cinc[2] - dim[2]) * cinc[3];
			}
			ap += (ainc[1] - dim[1]) * ainc[2] * ainc[3];
			bp += (binc[1] - dim[1]) * binc[2] * binc[3];
			cp += (cinc[1] - dim[1]) * cinc[2] * cinc[3];
		}
	}
}

static int _ccv_nnc_ewsum_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	_ccv_nnc_ewsum_forw_cpu_ref((ccv_nnc_tensor_view_t**)inputs, input_size, (ccv_nnc_tensor_view_t**)outputs, output_size);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_ewsum_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	// D[x + y + z, x] = 1
	int i;
	if (inputs[0] == 0)
	{
		// Set them to 1.
		for (i = 0; i < output_size; i++)
			if (outputs[i])
				_ccv_nnc_tensor_set_cpu_ref((ccv_nnc_tensor_view_t*)outputs[i], 1);
	} else {
		// Copy over the gradient (If they are not pointing to the same tensor already).
		for (i = 0; i < output_size; i++)
			if (inputs[0] != outputs[i] && outputs[i])
				_ccv_nnc_tensor_transfer_cpu_ref((ccv_nnc_tensor_view_t*)inputs[0], (ccv_nnc_tensor_view_t*)outputs[i]);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

void _ccv_nnc_ewprod_forw_cpu_ref(ccv_nnc_tensor_view_t* const* const inputs, const int input_size, ccv_nnc_tensor_view_t* const* const outputs, const int output_size)
{
	if (input_size == 1 && output_size == 1)
	{
		_ccv_nnc_tensor_transfer_cpu_ref(inputs[0], outputs[0]);
		return;
	}
	// Assuming this is float 32.
	int dim[CCV_NNC_MAX_DIM + 2];
	int ainc[CCV_NNC_MAX_DIM + 2];
	int binc[CCV_NNC_MAX_DIM + 2];
	int cinc[CCV_NNC_MAX_DIM + 2];
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
		assert(a->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		assert(c->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
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
		ccv_nnc_tensor_view_get_inc(a, ainc);
		ccv_nnc_tensor_view_get_inc(b, binc);
		ccv_nnc_tensor_view_get_inc(c, cinc);
		int i[CCV_NNC_MAX_DIM + 2];
		float* ap = a->data.f32;
		float* bp = b->data.f32;
		float* cp = c->data.f32;
		const int count = dim[2] * dim[3];
		if (ainc[3] == dim[3] && binc[3] == dim[3] && cinc[3] == dim[3])
		{
			// Special casing if the ainc[3] is the same as dim[3]
			for (i[0] = 0; i[0] < dim[0]; i[0]++)
			{
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					for (x = 0; x < count; x++)
						cp[x] = ap[x] * bp[x];
					ap += ainc[2] * ainc[3];
					bp += binc[2] * binc[3];
					cp += cinc[2] * cinc[3];
				}
				ap += (ainc[1] - dim[1]) * ainc[2] * ainc[3];
				bp += (binc[1] - dim[1]) * binc[2] * binc[3];
				cp += (cinc[1] - dim[1]) * cinc[2] * cinc[3];
			}
			continue;
		}
		// Non-optimal case, need to do skip copy.
		for (i[0] = 0; i[0] < dim[0]; i[0]++)
		{
			for (i[1] = 0; i[1] < dim[1]; i[1]++)
			{
				for (i[2] = 0; i[2] < dim[2]; i[2]++)
				{
					for (x = 0; x < dim[3]; x++)
						cp[x] = ap[x] * bp[x];
					ap += ainc[3];
					bp += binc[3];
					cp += cinc[3];
				}
				ap += (ainc[2] - dim[2]) * ainc[3];
				bp += (binc[2] - dim[2]) * binc[3];
				cp += (cinc[2] - dim[2]) * cinc[3];
			}
			ap += (ainc[1] - dim[1]) * ainc[2] * ainc[3];
			bp += (binc[1] - dim[1]) * binc[2] * binc[3];
			cp += (cinc[1] - dim[1]) * cinc[2] * cinc[3];
		}
	}
}

static int _ccv_nnc_ewprod_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	_ccv_nnc_ewprod_forw_cpu_ref((ccv_nnc_tensor_view_t**)inputs, input_size, (ccv_nnc_tensor_view_t**)outputs, output_size);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_ewprod_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	// D[x * y * z, x] = y * z
	// Assuming this is float 32.
	int dim[CCV_NNC_MAX_DIM + 2];
	int ginc[CCV_NNC_MAX_DIM + 2];
	int ainc[CCV_NNC_MAX_DIM + 2];
	int binc[CCV_NNC_MAX_DIM + 2];
	int hinc[CCV_NNC_MAX_DIM + 2];
	int x, z;
	ccv_nnc_tensor_view_t* g = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)inputs[output_size + 1];
	if (g == 0)
	{
		assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		ccv_nnc_tensor_view_get_dim(b, dim);
		ccv_nnc_tensor_view_get_inc(b, binc);
		for (z = 0; z < output_size; z++)
		{
			ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[z + 1];
			ccv_nnc_tensor_view_t* h = (ccv_nnc_tensor_view_t*)outputs[z];
			assert(a->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
			assert(h->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
			assert(ccv_nnc_tensor_view_check_dim(a, dim));
			assert(ccv_nnc_tensor_view_check_dim(h, dim));
			ccv_nnc_tensor_view_get_inc(a, ainc);
			ccv_nnc_tensor_view_get_inc(h, hinc);
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
			float* ap = a->data.f32;
			float* bp = b->data.f32;
			float* hp = h->data.f32;
			const int count = dim[2] * dim[3];
			if (ainc[3] == dim[3] && binc[3] == dim[3] && hinc[3] == dim[3])
			{
				// Special casing if the ainc[3] is the same as dim[3]
				for (i[0] = 0; i[0] < dim[0]; i[0]++)
				{
					for (i[1] = 0; i[1] < dim[1]; i[1]++)
					{
						for (x = 0; x < count; x++)
							hp[x] = bp[x] / ap[x];
						ap += ainc[2] * ainc[3];
						bp += binc[2] * binc[3];
						hp += hinc[2] * hinc[3];
					}
					ap += (ainc[1] - dim[1]) * ainc[2] * ainc[3];
					bp += (binc[1] - dim[1]) * binc[2] * binc[3];
					hp += (hinc[1] - dim[1]) * hinc[2] * hinc[3];
				}
				continue;
			}
			// Non-optimal case, need to do skip copy.
			for (i[0] = 0; i[0] < dim[0]; i[0]++)
			{
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					for (i[2] = 0; i[2] < dim[2]; i[2]++)
					{
						for (x = 0; x < dim[3]; x++)
							hp[x] = bp[x] / ap[x];
						ap += ainc[3];
						bp += binc[3];
						hp += hinc[3];
					}
					ap += (ainc[2] - dim[2]) * ainc[3];
					bp += (binc[2] - dim[2]) * binc[3];
					hp += (hinc[2] - dim[2]) * hinc[3];
				}
				ap += (ainc[1] - dim[1]) * ainc[2] * ainc[3];
				bp += (binc[1] - dim[1]) * binc[2] * binc[3];
				hp += (hinc[1] - dim[1]) * hinc[2] * hinc[3];
			}
		}
	} else {
		assert(g->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		ccv_nnc_tensor_view_get_dim(b, dim);
		assert(ccv_nnc_tensor_view_check_dim(g, dim));
		ccv_nnc_tensor_view_get_inc(b, binc);
		ccv_nnc_tensor_view_get_inc(g, ginc);
		for (z = 0; z < output_size; z++)
		{
			ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[z + 1];
			ccv_nnc_tensor_view_t* h = (ccv_nnc_tensor_view_t*)outputs[z];
			assert(a->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
			assert(h->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
			assert(ccv_nnc_tensor_view_check_dim(a, dim));
			assert(ccv_nnc_tensor_view_check_dim(h, dim));
			ccv_nnc_tensor_view_get_inc(a, ainc);
			ccv_nnc_tensor_view_get_inc(h, hinc);
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
			float* gp = g->data.f32;
			float* ap = a->data.f32;
			float* bp = b->data.f32;
			float* hp = h->data.f32;
			const int count = dim[2] * dim[3];
			if (ginc[3] == dim[3] && ainc[3] == dim[3] && binc[3] == dim[3] && hinc[3] == dim[3])
			{
				// Special casing if the ainc[3] is the same as dim[3]
				for (i[0] = 0; i[0] < dim[0]; i[0]++)
				{
					for (i[1] = 0; i[1] < dim[1]; i[1]++)
					{
						for (x = 0; x < count; x++)
							hp[x] = gp[x] * bp[x] / ap[x];
						gp += ginc[2] * ginc[3];
						ap += ainc[2] * ainc[3];
						bp += binc[2] * binc[3];
						hp += hinc[2] * hinc[3];
					}
					gp += (ginc[1] - dim[1]) * ginc[2] * ginc[3];
					ap += (ainc[1] - dim[1]) * ainc[2] * ainc[3];
					bp += (binc[1] - dim[1]) * binc[2] * binc[3];
					hp += (hinc[1] - dim[1]) * hinc[2] * hinc[3];
				}
				continue;
			}
			// Non-optimal case, need to do skip copy.
			for (i[0] = 0; i[0] < dim[0]; i[0]++)
			{
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					for (i[2] = 0; i[2] < dim[2]; i[2]++)
					{
						for (x = 0; x < dim[3]; x++)
							hp[x] = gp[x] * bp[x] / ap[x];
						gp += ginc[3];
						ap += ainc[3];
						bp += binc[3];
						hp += hinc[3];
					}
					gp += (ginc[2] - dim[2]) * ginc[3];
					ap += (ainc[2] - dim[2]) * ainc[3];
					bp += (binc[2] - dim[2]) * binc[3];
					hp += (hinc[2] - dim[2]) * hinc[3];
				}
				gp += (ginc[1] - dim[1]) * ginc[2] * ginc[3];
				ap += (ainc[1] - dim[1]) * ainc[2] * ainc[3];
				bp += (binc[1] - dim[1]) * binc[2] * binc[3];
				hp += (hinc[1] - dim[1]) * hinc[2] * hinc[3];
			}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static void _ccv_nnc_ewdiv_forw_cpu_ref(const float p, ccv_nnc_tensor_view_t* const a, ccv_nnc_tensor_view_t* const b, ccv_nnc_tensor_view_t* const c)
{
	// Assuming this is float 32.
	int dim[CCV_NNC_MAX_DIM + 2];
	int ainc[CCV_NNC_MAX_DIM + 2];
	int binc[CCV_NNC_MAX_DIM + 2];
	int cinc[CCV_NNC_MAX_DIM + 2];
	if (a == 0) // Take 0 as all ones tensor.
	{
		assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		assert(c->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
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
		ccv_nnc_tensor_view_get_inc(b, binc);
		ccv_nnc_tensor_view_get_inc(c, cinc);
		int i[CCV_NNC_MAX_DIM + 2];
		float* bp = b->data.f32;
		float* cp = c->data.f32;
		const int count = dim[2] * dim[3];
		if (binc[3] == dim[3] && cinc[3] == dim[3])
		{
			// Special casing if the ainc[3] is the same as dim[3]
			for (i[0] = 0; i[0] < dim[0]; i[0]++)
			{
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					for (x = 0; x < count; x++)
						cp[x] = p / bp[x];
					bp += binc[2] * binc[3];
					cp += cinc[2] * cinc[3];
				}
				bp += (binc[1] - dim[1]) * binc[2] * binc[3];
				cp += (cinc[1] - dim[1]) * cinc[2] * cinc[3];
			}
			return;
		}
		// Non-optimal case, need to do skip copy.
		for (i[0] = 0; i[0] < dim[0]; i[0]++)
		{
			for (i[1] = 0; i[1] < dim[1]; i[1]++)
			{
				for (i[2] = 0; i[2] < dim[2]; i[2]++)
				{
					for (x = 0; x < dim[3]; x++)
						cp[x] = p / bp[x];
					bp += binc[3];
					cp += cinc[3];
				}
				bp += (binc[2] - dim[2]) * binc[3];
				cp += (cinc[2] - dim[2]) * cinc[3];
			}
			bp += (binc[1] - dim[1]) * binc[2] * binc[3];
			cp += (cinc[1] - dim[1]) * cinc[2] * cinc[3];
		}
	} else {
		assert(a->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		assert(c->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
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
		ccv_nnc_tensor_view_get_inc(a, ainc);
		ccv_nnc_tensor_view_get_inc(b, binc);
		ccv_nnc_tensor_view_get_inc(c, cinc);
		int i[CCV_NNC_MAX_DIM + 2];
		float* ap = a->data.f32;
		float* bp = b->data.f32;
		float* cp = c->data.f32;
		const int count = dim[2] * dim[3];
		if (ainc[3] == dim[3] && binc[3] == dim[3] && cinc[3] == dim[3])
		{
			// Special casing if the ainc[3] is the same as dim[3]
			for (i[0] = 0; i[0] < dim[0]; i[0]++)
			{
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					for (x = 0; x < count; x++)
						cp[x] = p * ap[x] / bp[x];
					ap += ainc[2] * ainc[3];
					bp += binc[2] * binc[3];
					cp += cinc[2] * cinc[3];
				}
				ap += (ainc[1] - dim[1]) * ainc[2] * ainc[3];
				bp += (binc[1] - dim[1]) * binc[2] * binc[3];
				cp += (cinc[1] - dim[1]) * cinc[2] * cinc[3];
			}
			return;
		}
		// Non-optimal case, need to do skip copy.
		for (i[0] = 0; i[0] < dim[0]; i[0]++)
		{
			for (i[1] = 0; i[1] < dim[1]; i[1]++)
			{
				for (i[2] = 0; i[2] < dim[2]; i[2]++)
				{
					for (x = 0; x < dim[3]; x++)
						cp[x] = p * ap[x] / bp[x];
					ap += ainc[3];
					bp += binc[3];
					cp += cinc[3];
				}
				ap += (ainc[2] - dim[2]) * ainc[3];
				bp += (binc[2] - dim[2]) * binc[3];
				cp += (cinc[2] - dim[2]) * cinc[3];
			}
			ap += (ainc[1] - dim[1]) * ainc[2] * ainc[3];
			bp += (binc[1] - dim[1]) * binc[2] * binc[3];
			cp += (cinc[1] - dim[1]) * cinc[2] * cinc[3];
		}
	}
}

static int _ccv_nnc_ewdiv_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	_ccv_nnc_ewdiv_forw_cpu_ref(1, (ccv_nnc_tensor_view_t*)inputs[0], (ccv_nnc_tensor_view_t*)inputs[1], (ccv_nnc_tensor_view_t*)outputs[0]);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_ewdiv_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	// D[x / y, x] = 1 / y, D[x / y, y] = -x / y^2
	if (output_size == 1 || outputs[1] == 0)
	{
		// When we only need D[x / y, y]
		_ccv_nnc_ewdiv_forw_cpu_ref(1, (ccv_nnc_tensor_view_t*)inputs[0], (ccv_nnc_tensor_view_t*)inputs[2], (ccv_nnc_tensor_view_t*)outputs[0]);
		return CCV_NNC_EXEC_SUCCESS;
	}
	int dim[CCV_NNC_MAX_DIM + 2];
	int ginc[CCV_NNC_MAX_DIM + 2];
	int binc[CCV_NNC_MAX_DIM + 2];
	int cinc[CCV_NNC_MAX_DIM + 2];
	int hainc[CCV_NNC_MAX_DIM + 2];
	int hbinc[CCV_NNC_MAX_DIM + 2];
	ccv_nnc_tensor_view_t* g = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)inputs[2];
	ccv_nnc_tensor_view_t* c = (ccv_nnc_tensor_view_t*)inputs[3];
	ccv_nnc_tensor_view_t* ha = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* hb = (ccv_nnc_tensor_view_t*)outputs[1];
	if (g == 0)
	{
		assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		assert(c->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		assert(hb->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		ccv_nnc_tensor_view_get_dim(b, dim);
		assert(ccv_nnc_tensor_view_check_dim(c, dim));
		assert(ccv_nnc_tensor_view_check_dim(hb, dim));
		if (ha)
		{
			assert(ha->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
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
		ccv_nnc_tensor_view_get_inc(b, binc);
		ccv_nnc_tensor_view_get_inc(c, cinc);
		ccv_nnc_tensor_view_get_inc(hb, hbinc);
		int i[CCV_NNC_MAX_DIM + 2];
		float* bp = b->data.f32;
		float* cp = c->data.f32;
		float* hbp = hb->data.f32;
		const int count = dim[2] * dim[3];
		if (ha == 0)
		{
			if (binc[3] == dim[3] && cinc[3] == dim[3] && hbinc[3] == dim[3])
			{
				// Special casing if the ainc[3] is the same as dim[3]
				for (i[0] = 0; i[0] < dim[0]; i[0]++)
				{
					for (i[1] = 0; i[1] < dim[1]; i[1]++)
					{
						for (x = 0; x < count; x++)
						{
							const float v = 1 / bp[x];
							hbp[x] = -cp[x] * v;
						}
						bp += binc[2] * binc[3];
						cp += cinc[2] * cinc[3];
						hbp += hbinc[2] * hbinc[3];
					}
					bp += (binc[1] - dim[1]) * binc[2] * binc[3];
					cp += (cinc[1] - dim[1]) * cinc[2] * cinc[3];
					hbp += (hbinc[1] - dim[1]) * hbinc[2] * hbinc[3];
				}
				return CCV_NNC_EXEC_SUCCESS;
			}
			// Non-optimal case, need to do skip copy.
			for (i[0] = 0; i[0] < dim[0]; i[0]++)
			{
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					for (i[2] = 0; i[2] < dim[2]; i[2]++)
					{
						for (x = 0; x < dim[3]; x++)
						{
							const float v = 1 / bp[x];
							hbp[x] = -cp[x] * v;
						}
						bp += binc[3];
						cp += cinc[3];
						hbp += hbinc[3];
					}
					bp += (binc[2] - dim[2]) * binc[3];
					cp += (cinc[2] - dim[2]) * cinc[3];
					hbp += (hbinc[2] - dim[2]) * hbinc[3];
				}
				bp += (binc[1] - dim[1]) * binc[2] * binc[3];
				cp += (cinc[1] - dim[1]) * cinc[2] * cinc[3];
				hbp += (hbinc[1] - dim[1]) * hbinc[2] * hbinc[3];
			}
		} else {
			float* hap = ha->data.f32;
			ccv_nnc_tensor_view_get_inc(ha, hainc);
			if (binc[3] == dim[3] && cinc[3] == dim[3] && hainc[3] == dim[3] && hbinc[3] == dim[3])
			{
				// Special casing if the ainc[3] is the same as dim[3]
				for (i[0] = 0; i[0] < dim[0]; i[0]++)
				{
					for (i[1] = 0; i[1] < dim[1]; i[1]++)
					{
						for (x = 0; x < count; x++)
						{
							const float v = 1 / bp[x];
							hap[x] = v;
							hbp[x] = -cp[x] * v;
						}
						bp += binc[2] * binc[3];
						cp += cinc[2] * cinc[3];
						hap += hainc[2] * hainc[3];
						hbp += hbinc[2] * hbinc[3];
					}
					bp += (binc[1] - dim[1]) * binc[2] * binc[3];
					cp += (cinc[1] - dim[1]) * cinc[2] * cinc[3];
					hap += (hainc[1] - dim[1]) * hainc[2] * hainc[3];
					hbp += (hbinc[1] - dim[1]) * hbinc[2] * hbinc[3];
				}
				return CCV_NNC_EXEC_SUCCESS;
			}
			// Non-optimal case, need to do skip copy.
			for (i[0] = 0; i[0] < dim[0]; i[0]++)
			{
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					for (i[2] = 0; i[2] < dim[2]; i[2]++)
					{
						for (x = 0; x < dim[3]; x++)
						{
							const float v = 1 / bp[x];
							hap[x] = v;
							hbp[x] = -cp[x] * v;
						}
						bp += binc[3];
						cp += cinc[3];
						hap += hainc[3];
						hbp += hbinc[3];
					}
					bp += (binc[2] - dim[2]) * binc[3];
					cp += (cinc[2] - dim[2]) * cinc[3];
					hap += (hainc[2] - dim[2]) * hainc[3];
					hbp += (hbinc[2] - dim[2]) * hbinc[3];
				}
				bp += (binc[1] - dim[1]) * binc[2] * binc[3];
				cp += (cinc[1] - dim[1]) * cinc[2] * cinc[3];
				hap += (hainc[1] - dim[1]) * hainc[2] * hainc[3];
				hbp += (hbinc[1] - dim[1]) * hbinc[2] * hbinc[3];
			}
		}
	} else {
		assert(g->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		assert(c->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		assert(hb->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		ccv_nnc_tensor_view_get_dim(b, dim);
		assert(ccv_nnc_tensor_view_check_dim(g, dim));
		assert(ccv_nnc_tensor_view_check_dim(c, dim));
		assert(ccv_nnc_tensor_view_check_dim(hb, dim));
		if (ha)
		{
			assert(ha->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
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
		ccv_nnc_tensor_view_get_inc(g, ginc);
		ccv_nnc_tensor_view_get_inc(b, binc);
		ccv_nnc_tensor_view_get_inc(c, cinc);
		ccv_nnc_tensor_view_get_inc(hb, hbinc);
		int i[CCV_NNC_MAX_DIM + 2];
		float* gp = g->data.f32;
		float* bp = b->data.f32;
		float* cp = c->data.f32;
		float* hbp = hb->data.f32;
		const int count = dim[2] * dim[3];
		if (ha == 0)
		{
			if (ginc[3] == dim[3] && binc[3] == dim[3] && cinc[3] == dim[3] && hbinc[3] == dim[3])
			{
				// Special casing if the ainc[3] is the same as dim[3]
				for (i[0] = 0; i[0] < dim[0]; i[0]++)
				{
					for (i[1] = 0; i[1] < dim[1]; i[1]++)
					{
						for (x = 0; x < count; x++)
						{
							const float v = gp[x] / bp[x];
							hbp[x] = -cp[x] * v;
						}
						gp += ginc[2] * ginc[3];
						bp += binc[2] * binc[3];
						cp += cinc[2] * cinc[3];
						hbp += hbinc[2] * hbinc[3];
					}
					gp += (ginc[1] - dim[1]) * ginc[2] * ginc[3];
					bp += (binc[1] - dim[1]) * binc[2] * binc[3];
					cp += (cinc[1] - dim[1]) * cinc[2] * cinc[3];
					hbp += (hbinc[1] - dim[1]) * hbinc[2] * hbinc[3];
				}
				return CCV_NNC_EXEC_SUCCESS;
			}
			// Non-optimal case, need to do skip copy.
			for (i[0] = 0; i[0] < dim[0]; i[0]++)
			{
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					for (i[2] = 0; i[2] < dim[2]; i[2]++)
					{
						for (x = 0; x < dim[3]; x++)
						{
							const float v = gp[x] / bp[x];
							hbp[x] = -cp[x] * v;
						}
						gp += ginc[3];
						bp += binc[3];
						cp += cinc[3];
						hbp += hbinc[3];
					}
					gp += (ginc[2] - dim[2]) * ginc[3];
					bp += (binc[2] - dim[2]) * binc[3];
					cp += (cinc[2] - dim[2]) * cinc[3];
					hbp += (hbinc[2] - dim[2]) * hbinc[3];
				}
				gp += (ginc[1] - dim[1]) * ginc[2] * ginc[3];
				bp += (binc[1] - dim[1]) * binc[2] * binc[3];
				cp += (cinc[1] - dim[1]) * cinc[2] * cinc[3];
				hbp += (hbinc[1] - dim[1]) * hbinc[2] * hbinc[3];
			}
		} else {
			ccv_nnc_tensor_view_get_inc(ha, hainc);
			float* hap = ha->data.f32;
			if (ginc[3] == dim[3] && binc[3] == dim[3] && cinc[3] == dim[3] && hainc[3] == dim[3] && hbinc[3] == dim[3])
			{
				// Special casing if the ainc[3] is the same as dim[3]
				for (i[0] = 0; i[0] < dim[0]; i[0]++)
				{
					for (i[1] = 0; i[1] < dim[1]; i[1]++)
					{
						for (x = 0; x < count; x++)
						{
							const float v = gp[x] / bp[x];
							hap[x] = v;
							hbp[x] = -cp[x] * v;
						}
						gp += ginc[2] * ginc[3];
						bp += binc[2] * binc[3];
						cp += cinc[2] * cinc[3];
						hap += hainc[2] * hainc[3];
						hbp += hbinc[2] * hbinc[3];
					}
					gp += (ginc[1] - dim[1]) * ginc[2] * ginc[3];
					bp += (binc[1] - dim[1]) * binc[2] * binc[3];
					cp += (cinc[1] - dim[1]) * cinc[2] * cinc[3];
					hap += (hainc[1] - dim[1]) * hainc[2] * hainc[3];
					hbp += (hbinc[1] - dim[1]) * hbinc[2] * hbinc[3];
				}
				return CCV_NNC_EXEC_SUCCESS;
			}
			// Non-optimal case, need to do skip copy.
			for (i[0] = 0; i[0] < dim[0]; i[0]++)
			{
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					for (i[2] = 0; i[2] < dim[2]; i[2]++)
					{
						for (x = 0; x < dim[3]; x++)
						{
							const float v = gp[x] / bp[x];
							hap[x] = v;
							hbp[x] = -cp[x] * v;
						}
						gp += ginc[3];
						bp += binc[3];
						cp += cinc[3];
						hap += hainc[3];
						hbp += hbinc[3];
					}
					gp += (ginc[2] - dim[2]) * ginc[3];
					bp += (binc[2] - dim[2]) * binc[3];
					cp += (cinc[2] - dim[2]) * cinc[3];
					hap += (hainc[2] - dim[2]) * hainc[3];
					hbp += (hbinc[2] - dim[2]) * hbinc[3];
				}
				gp += (ginc[1] - dim[1]) * ginc[2] * ginc[3];
				bp += (binc[1] - dim[1]) * binc[2] * binc[3];
				cp += (cinc[1] - dim[1]) * cinc[2] * cinc[3];
				hap += (hainc[1] - dim[1]) * hainc[2] * hainc[3];
				hbp += (hbinc[1] - dim[1]) * hbinc[2] * hbinc[3];
			}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_ewexp_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	// Assuming this is float 32.
	int dim[CCV_NNC_MAX_DIM + 2];
	int ainc[CCV_NNC_MAX_DIM + 2];
	int binc[CCV_NNC_MAX_DIM + 2];
	ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(a->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
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
	ccv_nnc_tensor_view_get_inc(a, ainc);
	ccv_nnc_tensor_view_get_inc(b, binc);
	int i[CCV_NNC_MAX_DIM + 2];
	float* ap = a->data.f32;
	float* bp = b->data.f32;
	const int count = dim[2] * dim[3];
	if (ainc[3] == dim[3] && binc[3] == dim[3])
	{
		// Special casing if the ainc[3] is the same as dim[3]
		for (i[0] = 0; i[0] < dim[0]; i[0]++)
		{
			for (i[1] = 0; i[1] < dim[1]; i[1]++)
			{
				for (x = 0; x < count; x++)
					bp[x] = exp(ap[x]);
				ap += ainc[2] * ainc[3];
				bp += binc[2] * binc[3];
			}
			ap += (ainc[1] - dim[1]) * ainc[2] * ainc[3];
			bp += (binc[1] - dim[1]) * binc[2] * binc[3];
		}
		return CCV_NNC_EXEC_SUCCESS;
	}
	// Non-optimal case, need to do skip copy.
	for (i[0] = 0; i[0] < dim[0]; i[0]++)
	{
		for (i[1] = 0; i[1] < dim[1]; i[1]++)
		{
			for (i[2] = 0; i[2] < dim[2]; i[2]++)
			{
				for (x = 0; x < dim[3]; x++)
					bp[x] = exp(ap[x]);
				ap += ainc[3];
				bp += binc[3];
			}
			ap += (ainc[2] - dim[2]) * ainc[3];
			bp += (binc[2] - dim[2]) * binc[3];
		}
		ap += (ainc[1] - dim[1]) * ainc[2] * ainc[3];
		bp += (binc[1] - dim[1]) * binc[2] * binc[3];
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_ewexp_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	// D[Exp[x], x] = Exp[x]
	if (inputs[0] == 0)
		_ccv_nnc_tensor_transfer_cpu_ref((ccv_nnc_tensor_view_t*)inputs[2], (ccv_nnc_tensor_view_t*)outputs[0]);
	else
		_ccv_nnc_ewprod_forw_cpu_ref((ccv_nnc_tensor_view_t*[]){
			(ccv_nnc_tensor_view_t*)inputs[0], (ccv_nnc_tensor_view_t*)inputs[2]
		}, 2, (ccv_nnc_tensor_view_t**)outputs, output_size);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_ewlog_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	// Assuming this is float 32.
	int dim[CCV_NNC_MAX_DIM + 2];
	int ainc[CCV_NNC_MAX_DIM + 2];
	int binc[CCV_NNC_MAX_DIM + 2];
	ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(a->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
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
	ccv_nnc_tensor_view_get_inc(a, ainc);
	ccv_nnc_tensor_view_get_inc(b, binc);
	int i[CCV_NNC_MAX_DIM + 2];
	float* ap = a->data.f32;
	float* bp = b->data.f32;
	const int count = dim[2] * dim[3];
	if (ainc[3] == dim[3] && binc[3] == dim[3])
	{
		// Special casing if the ainc[3] is the same as dim[3]
		for (i[0] = 0; i[0] < dim[0]; i[0]++)
		{
			for (i[1] = 0; i[1] < dim[1]; i[1]++)
			{
				for (x = 0; x < count; x++)
					bp[x] = log(ap[x]);
				ap += ainc[2] * ainc[3];
				bp += binc[2] * binc[3];
			}
			ap += (ainc[1] - dim[1]) * ainc[2] * ainc[3];
			bp += (binc[1] - dim[1]) * binc[2] * binc[3];
		}
		return CCV_NNC_EXEC_SUCCESS;
	}
	// Non-optimal case, need to do skip copy.
	for (i[0] = 0; i[0] < dim[0]; i[0]++)
	{
		for (i[1] = 0; i[1] < dim[1]; i[1]++)
		{
			for (i[2] = 0; i[2] < dim[2]; i[2]++)
			{
				for (x = 0; x < dim[3]; x++)
					bp[x] = log(ap[x]);
				ap += ainc[3];
				bp += binc[3];
			}
			ap += (ainc[2] - dim[2]) * ainc[3];
			bp += (binc[2] - dim[2]) * binc[3];
		}
		ap += (ainc[1] - dim[1]) * ainc[2] * ainc[3];
		bp += (binc[1] - dim[1]) * binc[2] * binc[3];
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_ewlog_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	// D[Log[x], x] = 1 / x
	_ccv_nnc_ewdiv_forw_cpu_ref(1, (ccv_nnc_tensor_view_t*)inputs[0], (ccv_nnc_tensor_view_t*)inputs[1], (ccv_nnc_tensor_view_t*)outputs[0]);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_ewsqrt_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	// Assuming this is float 32.
	int dim[CCV_NNC_MAX_DIM + 2];
	int ainc[CCV_NNC_MAX_DIM + 2];
	int binc[CCV_NNC_MAX_DIM + 2];
	ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(a->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
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
	ccv_nnc_tensor_view_get_inc(a, ainc);
	ccv_nnc_tensor_view_get_inc(b, binc);
	int i[CCV_NNC_MAX_DIM + 2];
	float* ap = a->data.f32;
	float* bp = b->data.f32;
	const int count = dim[2] * dim[3];
	if (ainc[3] == dim[3] && binc[3] == dim[3])
	{
		// Special casing if the ainc[3] is the same as dim[3]
		for (i[0] = 0; i[0] < dim[0]; i[0]++)
		{
			for (i[1] = 0; i[1] < dim[1]; i[1]++)
			{
				for (x = 0; x < count; x++)
					bp[x] = sqrt(ap[x]);
				ap += ainc[2] * ainc[3];
				bp += binc[2] * binc[3];
			}
			ap += (ainc[1] - dim[1]) * ainc[2] * ainc[3];
			bp += (binc[1] - dim[1]) * binc[2] * binc[3];
		}
		return CCV_NNC_EXEC_SUCCESS;
	}
	// Non-optimal case, need to do skip copy.
	for (i[0] = 0; i[0] < dim[0]; i[0]++)
	{
		for (i[1] = 0; i[1] < dim[1]; i[1]++)
		{
			for (i[2] = 0; i[2] < dim[2]; i[2]++)
			{
				for (x = 0; x < dim[3]; x++)
					bp[x] = sqrt(ap[x]);
				ap += ainc[3];
				bp += binc[3];
			}
			ap += (ainc[2] - dim[2]) * ainc[3];
			bp += (binc[2] - dim[2]) * binc[3];
		}
		ap += (ainc[1] - dim[1]) * ainc[2] * ainc[3];
		bp += (binc[1] - dim[1]) * binc[2] * binc[3];
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_ewsqrt_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	// D[Sqrt[x], x] = 0.5 / Sqrt[x]
	_ccv_nnc_ewdiv_forw_cpu_ref(0.5, (ccv_nnc_tensor_view_t*)inputs[0], (ccv_nnc_tensor_view_t*)inputs[2], (ccv_nnc_tensor_view_t*)outputs[0]);
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
