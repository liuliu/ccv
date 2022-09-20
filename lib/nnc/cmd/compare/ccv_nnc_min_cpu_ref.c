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

static int _ccv_nnc_min_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)inputs[1];
	ccv_nnc_tensor_view_t* const c = (ccv_nnc_tensor_view_t*)outputs[0];
	// Assuming this is float 32.
	int dim[CCV_NNC_MAX_DIM_ALLOC];
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
	int cstride[CCV_NNC_MAX_DIM_ALLOC];
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
			c->data.f32[x] = ccv_min(a->data.f32[x], b->data.f32[x]);
		return CCV_NNC_EXEC_SUCCESS;
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
					cp0[x] = ccv_min(ap0[x], bp0[x]);
				ap0 += astride[1];
				bp0 += bstride[1];
				cp0 += cstride[1];
			}
		}
		return CCV_NNC_EXEC_SUCCESS;
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
					cp1[x] = ccv_min(ap1[x], bp1[x]);
				ap1 += astride[2];
				bp1 += bstride[2];
				cp1 += cstride[2];
			}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_min_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	ccv_nnc_tensor_view_t* const g = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[1];
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)inputs[2];
	ccv_nnc_tensor_view_t* const ha = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* const hb = (ccv_nnc_tensor_view_t*)outputs[1];
	// Assuming this is float 32.
	int dim[CCV_NNC_MAX_DIM_ALLOC];
	int gstride[CCV_NNC_MAX_DIM_ALLOC];
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
	int hastride[CCV_NNC_MAX_DIM_ALLOC];
	int hbstride[CCV_NNC_MAX_DIM_ALLOC];
	assert(ccv_nnc_tensor_nd(a->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(b->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(ha->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(hb->info.dim) <= CCV_NNC_MAX_DIM + 2);
	ccv_nnc_tensor_view_get_dim(a, dim);
	assert(ccv_nnc_tensor_view_check_dim(b, dim));
	assert(ccv_nnc_tensor_view_check_dim(ha, dim));
	assert(ccv_nnc_tensor_view_check_dim(hb, dim));
	if (g)
	{
		assert(ccv_nnc_tensor_nd(g->info.dim) <= CCV_NNC_MAX_DIM + 2);
		assert(ccv_nnc_tensor_view_check_dim(g, dim));
		int x;
		if (!CCV_IS_TENSOR_VIEW(g) && !CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(b) && !CCV_IS_TENSOR_VIEW(ha) && !CCV_IS_TENSOR_VIEW(hb))
		{
			// Super optimal case, just do one for-loop for sum.
			const int tensor_count = ccv_nnc_tensor_count(a->info);
			for (x = 0; x < tensor_count; x++)
				if (a->data.f32[x] < b->data.f32[x])
				{
					ha->data.f32[x] = g->data.f32[x];
					hb->data.f32[x] = 0;
				} else if (a->data.f32[x] > b->data.f32[x]) {
					hb->data.f32[x] = g->data.f32[x];
					ha->data.f32[x] = 0;
				} else
					ha->data.f32[x] = hb->data.f32[x] = g->data.f32[x];
			return CCV_NNC_EXEC_SUCCESS;
		}
		assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
		ccv_nnc_tensor_view_get_stride(g, gstride);
		ccv_nnc_tensor_view_get_stride(a, astride);
		ccv_nnc_tensor_view_get_stride(b, bstride);
		ccv_nnc_tensor_view_get_stride(ha, hastride);
		ccv_nnc_tensor_view_get_stride(hb, hbstride);
		int i[CCV_NNC_MAX_DIM + 2];
		float* const gp = g->data.f32;
		float* const ap = a->data.f32;
		float* const bp = b->data.f32;
		float* const hap = ha->data.f32;
		float* const hbp = hb->data.f32;
		const int count = dim[2] * dim[3];
		if (astride[2] == dim[3] && bstride[2] == dim[3] && hastride[2] == dim[3] && hbstride[2] == dim[3])
		{
			// Special casing if the ainc[3] is the same as dim[3]
			for (i[0] = 0; i[0] < dim[0]; i[0]++)
			{
				float* gp0 = gp + i[0] * gstride[0];
				float* ap0 = ap + i[0] * astride[0];
				float* bp0 = bp + i[0] * bstride[0];
				float* hap0 = hap + i[0] * hastride[0];
				float* hbp0 = hbp + i[0] * hbstride[0];
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					for (x = 0; x < count; x++)
						if (ap0[x] < bp0[x]) {
							hap0[x] = gp0[x];
							hbp0[x] = 0;
						} else if (ap0[x] > bp0[x]) {
							hbp0[x] = gp0[x];
							hap0[x] = 0;
						} else
							hap0[x] = hbp0[x] = gp0[x];
					gp0 += gstride[1];
					ap0 += astride[1];
					bp0 += bstride[1];
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
			float* const ap0 = ap + i[0] * astride[0];
			float* const bp0 = bp + i[0] * bstride[0];
			float* const hap0 = hap + i[0] * hastride[0];
			float* const hbp0 = hbp + i[0] * hbstride[0];
			for (i[1] = 0; i[1] < dim[1]; i[1]++)
			{
				float* gp1 = gp0 + i[1] * gstride[1];
				float* ap1 = ap0 + i[1] * astride[1];
				float* bp1 = bp0 + i[1] * bstride[1];
				float* hap1 = hap0 + i[1] * hastride[1];
				float* hbp1 = hbp0 + i[1] * hbstride[1];
				for (i[2] = 0; i[2] < dim[2]; i[2]++)
				{
					for (x = 0; x < dim[3]; x++)
						if (ap1[x] < bp1[x]) {
							hap1[x] = gp1[x];
							hbp1[x] = 0;
						} else if (ap1[x] > bp1[x]) {
							hbp1[x] = gp1[x];
							hap1[x] = 0;
						} else
							hap1[x] = hbp1[x] = gp1[x];
					gp1 += gstride[2];
					ap1 += astride[2];
					bp1 += bstride[2];
					hap1 += hastride[2];
					hbp1 += hbstride[2];
				}
			}
		}
	} else {
		int x;
		if (!CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(b) && !CCV_IS_TENSOR_VIEW(ha) && !CCV_IS_TENSOR_VIEW(hb))
		{
			// Super optimal case, just do one for-loop for sum.
			const int tensor_count = ccv_nnc_tensor_count(a->info);
			for (x = 0; x < tensor_count; x++)
				if (a->data.f32[x] < b->data.f32[x]) {
					ha->data.f32[x] = 1;
					hb->data.f32[x] = 0;
				} else if (a->data.f32[x] > b->data.f32[x]) {
					ha->data.f32[x] = 0;
					hb->data.f32[x] = 1;
				} else
					ha->data.f32[x] = hb->data.f32[x] = 1;
			return CCV_NNC_EXEC_SUCCESS;
		}
		assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
		ccv_nnc_tensor_view_get_stride(a, astride);
		ccv_nnc_tensor_view_get_stride(b, bstride);
		ccv_nnc_tensor_view_get_stride(ha, hastride);
		ccv_nnc_tensor_view_get_stride(hb, hbstride);
		int i[CCV_NNC_MAX_DIM + 2];
		float* const ap = a->data.f32;
		float* const bp = b->data.f32;
		float* const hap = ha->data.f32;
		float* const hbp = hb->data.f32;
		const int count = dim[2] * dim[3];
		if (astride[2] == dim[3] && bstride[2] == dim[3] && hastride[2] == dim[3] && hbstride[2] == dim[3])
		{
			// Special casing if the ainc[3] is the same as dim[3]
			for (i[0] = 0; i[0] < dim[0]; i[0]++)
			{
				float* ap0 = ap + i[0] * astride[0];
				float* bp0 = bp + i[0] * bstride[0];
				float* hap0 = hap + i[0] * hastride[0];
				float* hbp0 = hbp + i[0] * hbstride[0];
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					for (x = 0; x < count; x++)
						if (ap0[x] < bp0[x]) {
							hap0[x] = 1;
							hbp0[x] = 0;
						} else if (ap0[x] > bp0[x]) {
							hap0[x] = 0;
							hbp0[x] = 1;
						} else
							hap0[x] = hbp0[x] = 1;
					ap0 += astride[1];
					bp0 += bstride[1];
					hap0 += hastride[1];
					hbp0 += hbstride[1];
				}
			}
			return CCV_NNC_EXEC_SUCCESS;
		}
		// Non-optimal case, need to do skip copy.
		for (i[0] = 0; i[0] < dim[0]; i[0]++)
		{
			float* const ap0 = ap + i[0] * astride[0];
			float* const bp0 = bp + i[0] * bstride[0];
			float* const hap0 = hap + i[0] * hastride[0];
			float* const hbp0 = hbp + i[0] * hbstride[0];
			for (i[1] = 0; i[1] < dim[1]; i[1]++)
			{
				float* ap1 = ap0 + i[1] * astride[1];
				float* bp1 = bp0 + i[1] * bstride[1];
				float* hap1 = hap0 + i[1] * hastride[1];
				float* hbp1 = hbp0 + i[1] * hbstride[1];
				for (i[2] = 0; i[2] < dim[2]; i[2]++)
				{
					for (x = 0; x < dim[3]; x++)
						if (ap1[x] < bp1[x]) {
							hap1[x] = 1;
							hbp1[x] = 0;
						} else if (ap1[x] > bp1[x]) {
							hap1[x] = 0;
							hbp1[x] = 1;
						} else
							hap1[x] = hbp1[x] = 1;
					ap1 += astride[2];
					bp1 += bstride[2];
					hap1 += hastride[2];
					hbp1 += hbstride[2];
				}
			}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_MIN_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_min_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_MIN_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_min_back;
}
