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
	int ainc[CCV_NNC_MAX_DIM_ALLOC];
	int binc[CCV_NNC_MAX_DIM_ALLOC];
	int cinc[CCV_NNC_MAX_DIM_ALLOC];
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
			c->data.f32[x] = ccv_min(a->data.f32[x], b->data.f32[x]);
		return CCV_NNC_EXEC_SUCCESS;
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
					cp[x] = ccv_min(ap[x], bp[x]);
				ap += ainc[2] * ainc[3];
				bp += binc[2] * binc[3];
				cp += cinc[2] * cinc[3];
			}
			ap += (ainc[1] - dim[1]) * ainc[2] * ainc[3];
			bp += (binc[1] - dim[1]) * binc[2] * binc[3];
			cp += (cinc[1] - dim[1]) * cinc[2] * cinc[3];
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
					cp[x] = ccv_min(ap[x], bp[x]);
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
	int ginc[CCV_NNC_MAX_DIM_ALLOC];
	int ainc[CCV_NNC_MAX_DIM_ALLOC];
	int binc[CCV_NNC_MAX_DIM_ALLOC];
	int hainc[CCV_NNC_MAX_DIM_ALLOC];
	int hbinc[CCV_NNC_MAX_DIM_ALLOC];
	assert(a->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	assert(ha->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	assert(hb->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	ccv_nnc_tensor_view_get_dim(a, dim);
	assert(ccv_nnc_tensor_view_check_dim(b, dim));
	assert(ccv_nnc_tensor_view_check_dim(ha, dim));
	assert(ccv_nnc_tensor_view_check_dim(hb, dim));
	if (g)
	{
		assert(g->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		assert(ccv_nnc_tensor_view_check_dim(g, dim));
		int x;
		if (!CCV_IS_TENSOR_VIEW(g) && !CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(b) && !CCV_IS_TENSOR_VIEW(ha) && !CCV_IS_TENSOR_VIEW(hb))
		{
			// Super optimal case, just do one for-loop for sum.
			const int tensor_count = ccv_nnc_tensor_count(a->info);
			for (x = 0; x < tensor_count; x++)
				if (a->data.f32[x] < b->data.f32[x])
					ha->data.f32[x] = g->data.f32[x];
				else if (a->data.f32[x] > b->data.f32[x])
					hb->data.f32[x] = g->data.f32[x];
				else
					ha->data.f32[x] = hb->data.f32[x] = g->data.f32[x];
			return CCV_NNC_EXEC_SUCCESS;
		}
		assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
		ccv_nnc_tensor_view_get_inc(g, ginc);
		ccv_nnc_tensor_view_get_inc(a, ainc);
		ccv_nnc_tensor_view_get_inc(b, binc);
		ccv_nnc_tensor_view_get_inc(ha, hainc);
		ccv_nnc_tensor_view_get_inc(hb, hbinc);
		int i[CCV_NNC_MAX_DIM + 2];
		float* gp = g->data.f32;
		float* ap = a->data.f32;
		float* bp = b->data.f32;
		float* hap = ha->data.f32;
		float* hbp = hb->data.f32;
		const int count = dim[2] * dim[3];
		if (ainc[3] == dim[3] && binc[3] == dim[3] && hainc[3] == dim[3] && hbinc[3] == dim[3])
		{
			// Special casing if the ainc[3] is the same as dim[3]
			for (i[0] = 0; i[0] < dim[0]; i[0]++)
			{
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					for (x = 0; x < count; x++)
						if (ap[x] < bp[x])
							hap[x] = gp[x];
						else if (ap[x] > bp[x])
							hbp[x] = gp[x];
						else
							hap[x] = hbp[x] = gp[x];
					gp += ginc[2] * ginc[3];
					ap += ainc[2] * ainc[3];
					bp += binc[2] * binc[3];
					hap += hainc[2] * hainc[3];
					hbp += hbinc[2] * hbinc[3];
				}
				gp += (ginc[1] - dim[1]) * ginc[2] * ginc[3];
				ap += (ainc[1] - dim[1]) * ainc[2] * ainc[3];
				bp += (binc[1] - dim[1]) * binc[2] * binc[3];
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
						if (ap[x] < bp[x])
							hap[x] = gp[x];
						else if (ap[x] > bp[x])
							hbp[x] = gp[x];
						else
							hap[x] = hbp[x] = gp[x];
					gp += ginc[3];
					ap += ainc[3];
					bp += binc[3];
					hap += hainc[3];
					hbp += hbinc[3];
				}
				gp += (ginc[2] - dim[2]) * ginc[3];
				ap += (ainc[2] - dim[2]) * ainc[3];
				bp += (binc[2] - dim[2]) * binc[3];
				hap += (hainc[2] - dim[2]) * hainc[3];
				hbp += (hbinc[2] - dim[2]) * hbinc[3];
			}
			gp += (ginc[1] - dim[1]) * ginc[2] * ginc[3];
			ap += (ainc[1] - dim[1]) * ainc[2] * ainc[3];
			bp += (binc[1] - dim[1]) * binc[2] * binc[3];
			hap += (hainc[1] - dim[1]) * hainc[2] * hainc[3];
			hbp += (hbinc[1] - dim[1]) * hbinc[2] * hbinc[3];
		}
	} else {
		int x;
		if (!CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(b) && !CCV_IS_TENSOR_VIEW(ha) && !CCV_IS_TENSOR_VIEW(hb))
		{
			// Super optimal case, just do one for-loop for sum.
			const int tensor_count = ccv_nnc_tensor_count(a->info);
			for (x = 0; x < tensor_count; x++)
				if (a->data.f32[x] < b->data.f32[x])
					ha->data.f32[x] = 1;
				else if (a->data.f32[x] > b->data.f32[x])
					hb->data.f32[x] = 1;
				else
					ha->data.f32[x] = hb->data.f32[x] = 1;
			return CCV_NNC_EXEC_SUCCESS;
		}
		assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
		ccv_nnc_tensor_view_get_inc(a, ainc);
		ccv_nnc_tensor_view_get_inc(b, binc);
		ccv_nnc_tensor_view_get_inc(ha, hainc);
		ccv_nnc_tensor_view_get_inc(hb, hbinc);
		int i[CCV_NNC_MAX_DIM + 2];
		float* ap = a->data.f32;
		float* bp = b->data.f32;
		float* hap = ha->data.f32;
		float* hbp = hb->data.f32;
		const int count = dim[2] * dim[3];
		if (ainc[3] == dim[3] && binc[3] == dim[3] && hainc[3] == dim[3] && hbinc[3] == dim[3])
		{
			// Special casing if the ainc[3] is the same as dim[3]
			for (i[0] = 0; i[0] < dim[0]; i[0]++)
			{
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					for (x = 0; x < count; x++)
						if (ap[x] < bp[x])
							hap[x] = 1;
						else if (ap[x] > bp[x])
							hbp[x] = 1;
						else
							hap[x] = hbp[x] = 1;
					ap += ainc[2] * ainc[3];
					bp += binc[2] * binc[3];
					hap += hainc[2] * hainc[3];
					hbp += hbinc[2] * hbinc[3];
				}
				ap += (ainc[1] - dim[1]) * ainc[2] * ainc[3];
				bp += (binc[1] - dim[1]) * binc[2] * binc[3];
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
						if (ap[x] < bp[x])
							hap[x] = 1;
						else if (ap[x] > bp[x])
							hbp[x] = 1;
						else
							hap[x] = hbp[x] = 1;
					ap += ainc[3];
					bp += binc[3];
					hap += hainc[3];
					hbp += hbinc[3];
				}
				ap += (ainc[2] - dim[2]) * ainc[3];
				bp += (binc[2] - dim[2]) * binc[3];
				hap += (hainc[2] - dim[2]) * hainc[3];
				hbp += (hbinc[2] - dim[2]) * hbinc[3];
			}
			ap += (ainc[1] - dim[1]) * ainc[2] * ainc[3];
			bp += (binc[1] - dim[1]) * binc[2] * binc[3];
			hap += (hainc[1] - dim[1]) * hainc[2] * hainc[3];
			hbp += (hbinc[1] - dim[1]) * hbinc[2] * hbinc[3];
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
