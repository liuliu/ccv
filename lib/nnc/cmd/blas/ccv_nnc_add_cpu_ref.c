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

// Shared methods.
#include "../_ccv_nnc_cpu_ref.h"

void _ccv_nnc_add_forw_cpu_ref(const float p, const float q, ccv_nnc_tensor_view_t* const a, ccv_nnc_tensor_view_t* const b, ccv_nnc_tensor_view_t* const c)
{
	if (b == 0)
	{
		// It cannot be set otherwise we have trouble.
		assert(q == 0);
		if (p == 1)
		{
			_ccv_nnc_tensor_transfer_cpu_ref_f32(a, c);
			return;
		} else if (p == 0) {
			ccv_nnc_tensor_zero(c);
			return;
		}
		// Assuming this is float 32.
		int dim[CCV_NNC_MAX_DIM_ALLOC];
		int astride[CCV_NNC_MAX_DIM_ALLOC];
		int cstride[CCV_NNC_MAX_DIM_ALLOC];
		assert(ccv_nnc_tensor_nd(a->info.dim) <= CCV_NNC_MAX_DIM + 2);
		assert(ccv_nnc_tensor_nd(c->info.dim) <= CCV_NNC_MAX_DIM + 2);
		ccv_nnc_tensor_view_get_dim(a, dim);
		assert(ccv_nnc_tensor_view_check_dim(c, dim));
		int x;
		if (!CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(c))
		{
			// Super optimal case, just do one for-loop for sum.
			const int tensor_count = ccv_nnc_tensor_count(a->info);
			for (x = 0; x < tensor_count; x++)
				c->data.f32[x] = p * a->data.f32[x];
			return;
		}
		assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
		ccv_nnc_tensor_view_get_stride(a, astride);
		ccv_nnc_tensor_view_get_stride(c, cstride);
		int i[CCV_NNC_MAX_DIM + 2];
		float* const ap = a->data.f32;
		float* const cp = c->data.f32;
		const int count = dim[2] * dim[3];
		if (astride[2] == dim[3] && cstride[2] == dim[3] && astride[3] == 1 && cstride[3] == 1)
		{
			// Special casing if the ainc[3] is the same as dim[3]
			for (i[0] = 0; i[0] < dim[0]; i[0]++)
			{
				float* ap0 = ap + i[0] * astride[0];
				float* cp0 = cp + i[0] * cstride[0];
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					for (x = 0; x < count; x++)
						cp0[x] = p * ap0[x];
					ap0 += astride[1];
					cp0 += cstride[1];
				}
			}
			return;
		}
		// Non-optimal case, need to do skip copy.
		for (i[0] = 0; i[0] < dim[0]; i[0]++)
		{
			float* const ap0 = ap + i[0] * astride[0];
			float* const cp0 = cp + i[0] * cstride[0];
			for (i[1] = 0; i[1] < dim[1]; i[1]++)
			{
				float* ap1 = ap0 + i[1] * astride[1];
				float* cp1 = cp0 + i[1] * cstride[1];
				for (i[2] = 0; i[2] < dim[2]; i[2]++)
				{
					for (x = 0; x < dim[3]; x++)
						cp1[x * cstride[3]] = p * ap1[x * astride[3]];
					ap1 += astride[2];
					cp1 += cstride[2];
				}
			}
		}
		return;
	}
	int cdim[CCV_NNC_MAX_DIM_ALLOC];
	assert(ccv_nnc_tensor_nd(a->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(b->info.dim) <= CCV_NNC_MAX_DIM + 2);
	ccv_nnc_tensor_view_get_dim(a, cdim); // Fill in cdim first.
	ccv_nnc_tensor_view_get_broadcast_dim(b, cdim);
	assert(ccv_nnc_tensor_view_check_broadcast_dim(a, cdim));
	assert(ccv_nnc_tensor_view_check_broadcast_dim(b, cdim));
	const int a_check_dim = ccv_nnc_tensor_view_check_dim(a, cdim);
	const int b_check_dim = ccv_nnc_tensor_view_check_dim(b, cdim);
	if (p == 1 && q == 1 && a_check_dim && b_check_dim)
	{
		_ccv_nnc_ewsum_forw_cpu_ref_f32((ccv_nnc_tensor_view_t*[]){
			a, b
		}, 2, &c, 1);
		return;
	} else if (p == 1 && q == 0 && a_check_dim) {
		_ccv_nnc_tensor_transfer_cpu_ref_f32(a, c);
		return;
	} else if (p == 0 && q == 1 && b_check_dim) {
		_ccv_nnc_tensor_transfer_cpu_ref_f32(b, c);
		return;
	} else if (p == 0 && q == 0) {
		ccv_nnc_tensor_zero(c);
		return;
	}
	// Assuming this is float 32.
	int adim[CCV_NNC_MAX_DIM_ALLOC];
	int bdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(a, adim);
	ccv_nnc_tensor_view_get_dim(b, bdim);
	assert(ccv_nnc_tensor_nd(c->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_view_check_dim(c, cdim));
	int x;
	if (!CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(b) && !CCV_IS_TENSOR_VIEW(c) && a_check_dim && b_check_dim)
	{
		const int tensor_count = ccv_nnc_tensor_count(a->info);
		// Super optimal case, just do one for-loop for sum.
		for (x = 0; x < tensor_count; x++)
			c->data.f32[x] = p * a->data.f32[x] + q * b->data.f32[x];
		return;
	}
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
	int cstride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_stride(a, astride);
	ccv_nnc_tensor_view_get_stride(b, bstride);
	ccv_nnc_tensor_view_get_stride(c, cstride);
	int i[CCV_NNC_MAX_DIM + 2];
	float* const ap = a->data.f32;
	float* const bp = b->data.f32;
	float* const cp = c->data.f32;
	const int count = cdim[2] * cdim[3];
	if (astride[2] == cdim[3] && bstride[2] == cdim[3] && cstride[2] == cdim[3] && adim[2] == cdim[2] && bdim[2] == cdim[2] && astride[3] == 1 && bstride[3] == 1)
	{
		// Special casing if the ainc[3] is the same as dim[3]
		for (i[0] = 0; i[0] < cdim[0]; i[0]++)
		{
			float* const ap0 = adim[0] == 1 ? ap : ap + i[0] * astride[0];
			float* const bp0 = bdim[0] == 1 ? bp : bp + i[0] * bstride[0];
			float* cp0 = cp + i[0] * cstride[0];
			for (i[1] = 0; i[1] < cdim[1]; i[1]++)
			{
				float* const ap1 = adim[1] == 1 ? ap0 : ap0 + i[1] * astride[1];
				float* const bp1 = bdim[1] == 1 ? bp0 : bp0 + i[1] * bstride[1];
				for (x = 0; x < count; x++)
					cp0[x] = p * ap1[x] + q * bp1[x];
				cp0 += cstride[1];
			}
		}
		return;
	}
	// Non-optimal case, need to do skip copy and handle broadcasting.
	for (i[0] = 0; i[0] < cdim[0]; i[0]++)
	{
		float* const ap0 = adim[0] == 1 ? ap : ap + i[0] * astride[0];
		float* const bp0 = bdim[0] == 1 ? bp : bp + i[0] * bstride[0];
		float* const cp0 = cp + i[0] * cstride[0];
		for (i[1] = 0; i[1] < cdim[1]; i[1]++)
		{
			float* const ap1 = adim[1] == 1 ? ap0 : ap0 + i[1] * astride[1];
			float* const bp1 = bdim[1] == 1 ? bp0 : bp0 + i[1] * bstride[1];
			float* cp1 = cp0 + i[1] * cstride[1];
			for (i[2] = 0; i[2] < cdim[2]; i[2]++)
			{
				float* const ap2 = adim[2] == 1 ? ap1 : ap1 + i[2] * astride[2];
				float* const bp2 = bdim[2] == 1 ? bp1 : bp1 + i[2] * bstride[2];
				if (adim[3] == 1)
					for (x = 0; x < cdim[3]; x++)
						cp1[x] = p * ap2[0] + q * bp2[x * bstride[3]];
				else if (bdim[3] == 1)
					for (x = 0; x < cdim[3]; x++)
						cp1[x] = p * ap2[x * astride[3]] + q * bp2[0];
				else
					for (x = 0; x < cdim[3]; x++)
						cp1[x] = p * ap2[x * astride[3]] + q * bp2[x * bstride[3]];
				cp1 += cstride[2];
			}
		}
	}
}

static int _ccv_nnc_add_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 2);
	_ccv_nnc_add_forw_cpu_ref(cmd.info.blas.a[0], cmd.info.blas.a[1], (ccv_nnc_tensor_view_t*)inputs[0], (ccv_nnc_tensor_view_t*)inputs[1], (ccv_nnc_tensor_view_t*)outputs[0]);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_add_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	if (inputs[0] == 0)
	{
		if (outputs[0])
			_ccv_nnc_tensor_set_cpu_ref_f32((ccv_nnc_tensor_view_t*)outputs[0], cmd.info.blas.a[0]);
		if (output_size > 1 && outputs[1])
			_ccv_nnc_tensor_set_cpu_ref_f32((ccv_nnc_tensor_view_t*)outputs[1], cmd.info.blas.a[1]);
		return CCV_NNC_EXEC_SUCCESS;
	}
	int gdim[CCV_NNC_MAX_DIM_ALLOC];
	int gstride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_t* const g = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_get_dim(g, gdim);
	ccv_nnc_tensor_view_get_stride(g, gstride);
	if (outputs[0])
	{
		ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)outputs[0];
		if (ccv_nnc_tensor_view_check_dim(a, gdim))
			_ccv_nnc_add_forw_cpu_ref(cmd.info.blas.a[0], 0, (ccv_nnc_tensor_view_t*)inputs[0], 0, (ccv_nnc_tensor_view_t*)outputs[0]);
		else {
			assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
			const float p = cmd.info.blas.a[0];
			int adim[CCV_NNC_MAX_DIM_ALLOC];
			int astride[CCV_NNC_MAX_DIM_ALLOC];
			ccv_nnc_tensor_view_get_dim(a, adim);
			ccv_nnc_tensor_view_get_stride(a, astride);
			int i[CCV_NNC_MAX_DIM + 2];
			int x;
			float* const ap = a->data.f32;
			float* const gp = g->data.f32;
			// zeroing out so that we can accumulate.
			ccv_nnc_tensor_zero(a);
			// Non-optimal case, need to do skip copy and handle broadcasting.
			for (i[0] = 0; i[0] < gdim[0]; i[0]++)
			{
				float* const ap0 = adim[0] == 1 ? ap : ap + i[0] * astride[0];
				float* const gp0 = gp + i[0] * gstride[0];
				for (i[1] = 0; i[1] < gdim[1]; i[1]++)
				{
					float* const ap1 = adim[1] == 1 ? ap0 : ap0 + i[1] * astride[1];
					float* gp1 = gp0 + i[1] * gstride[1];
					for (i[2] = 0; i[2] < gdim[2]; i[2]++)
					{
						float* const ap2 = adim[2] == 1 ? ap1 : ap1 + i[2] * astride[2];
						if (adim[3] == 1)
							for (x = 0; x < gdim[3]; x++)
								ap2[0] += p * gp1[x];
						else
							for (x = 0; x < gdim[3]; x++)
								ap2[x] += p * gp1[x];
						gp1 += gstride[2];
					}
				}
			}
		}
	}
	if (output_size > 1 && outputs[1])
	{
		ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)outputs[1];
		if (ccv_nnc_tensor_view_check_dim(a, gdim))
			_ccv_nnc_add_forw_cpu_ref(cmd.info.blas.a[1], 0, (ccv_nnc_tensor_view_t*)inputs[0], 0, (ccv_nnc_tensor_view_t*)outputs[1]);
		else {
			assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
			const float p = cmd.info.blas.a[1];
			int adim[CCV_NNC_MAX_DIM_ALLOC];
			int astride[CCV_NNC_MAX_DIM_ALLOC];
			ccv_nnc_tensor_view_get_dim(a, adim);
			ccv_nnc_tensor_view_get_stride(a, astride);
			int i[CCV_NNC_MAX_DIM + 2];
			int x;
			float* const ap = a->data.f32;
			float* const gp = g->data.f32;
			// zeroing out so that we can accumulate.
			ccv_nnc_tensor_zero(a);
			// Non-optimal case, need to do skip copy and handle broadcasting.
			for (i[0] = 0; i[0] < gdim[0]; i[0]++)
			{
				float* const ap0 = adim[0] == 1 ? ap : ap + i[0] * astride[0];
				float* const gp0 = gp + i[0] * gstride[0];
				for (i[1] = 0; i[1] < gdim[1]; i[1]++)
				{
					float* const ap1 = adim[1] == 1 ? ap0 : ap0 + i[1] * astride[1];
					float* gp1 = gp0 + i[1] * gstride[1];
					for (i[2] = 0; i[2] < gdim[2]; i[2]++)
					{
						float* const ap2 = adim[2] == 1 ? ap1 : ap1 + i[2] * astride[2];
						if (adim[3] == 1)
							for (x = 0; x < gdim[3]; x++)
								ap2[0] += p * gp1[x];
						else
							for (x = 0; x < gdim[3]; x++)
								ap2[x] += p * gp1[x];
						gp1 += gstride[2];
					}
				}
			}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_ADD_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_add_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_ADD_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_add_back;
}
