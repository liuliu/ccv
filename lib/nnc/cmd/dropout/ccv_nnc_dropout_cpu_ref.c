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
#include <3rdparty/dsfmt/dSFMT.h>

// Shared methods.
#include "../_ccv_nnc_cpu_ref.h"

static int _ccv_nnc_dropout_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	const float p = cmd.info.dropout.p;
	const float inv_p = 1. / (1. - p);
	assert(output_size >= 2);
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
	const int tensor_count = ccv_nnc_tensor_count(inputs[0]->info);
	uint8_t* const maskdata = outputs[1]->data.u8;
	uint8_t* maskp = maskdata + (tensor_count - 1);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, (uint32_t)a->data.i32[0]);
	for (; maskp >= maskdata; --maskp)
		*maskp = (dsfmt_genrand_open_close(&dsfmt) <= p);
	int x;
	if (!CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(b))
	{
		// Super optimal case, just do one for-loop for sum.
		for (x = 0; x < tensor_count; x++)
			b->data.f32[x] = maskdata[x] ? 0 : a->data.f32[x] * inv_p;
		return CCV_NNC_EXEC_SUCCESS;
	}
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	ccv_nnc_tensor_view_get_inc(a, ainc);
	ccv_nnc_tensor_view_get_inc(b, binc);
	int i[CCV_NNC_MAX_DIM + 2];
	float* ap = a->data.f32;
	float* bp = b->data.f32;
	const int count = dim[2] * dim[3];
	maskp = maskdata;
	if (ainc[3] == dim[3] && binc[3] == dim[3])
	{
		// Special casing if the ainc[3] is the same as dim[3]
		for (i[0] = 0; i[0] < dim[0]; i[0]++)
		{
			for (i[1] = 0; i[1] < dim[1]; i[1]++)
			{
				for (x = 0; x < count; x++)
					bp[x] = maskp[x] ? 0 : ap[x] * inv_p;
				ap += ainc[2] * ainc[3];
				bp += binc[2] * binc[3];
				maskp += count;
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
					bp[x] = maskp[x] ? 0 : ap[x] * inv_p;
				maskp += dim[3];
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

static int _ccv_nnc_dropout_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 5);
	const float p = cmd.info.dropout.p;
	const float inv_p = 1. / (1. - p);
	uint8_t* const maskdata = inputs[4]->data.u8;
	// Assuming this is float 32.
	int dim[CCV_NNC_MAX_DIM + 2];
	int ginc[CCV_NNC_MAX_DIM + 2];
	int hinc[CCV_NNC_MAX_DIM + 2];
	ccv_nnc_tensor_view_t* g = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* h = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(g->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	assert(h->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	ccv_nnc_tensor_view_get_dim(g, dim);
	assert(ccv_nnc_tensor_view_check_dim(h, dim));
	int x;
	if (!CCV_IS_TENSOR_VIEW(g) && !CCV_IS_TENSOR_VIEW(h))
	{
		// Super optimal case, just do one for-loop for sum.
		const int tensor_count = ccv_nnc_tensor_count(inputs[0]->info);
		for (x = 0; x < tensor_count; x++)
			h->data.f32[x] = maskdata[x] ? 0 : g->data.f32[x] * inv_p;
		return CCV_NNC_EXEC_SUCCESS;
	}
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	ccv_nnc_tensor_view_get_inc(g, ginc);
	ccv_nnc_tensor_view_get_inc(h, hinc);
	int i[CCV_NNC_MAX_DIM + 2];
	float* gp = g->data.f32;
	float* hp = h->data.f32;
	const int count = dim[2] * dim[3];
	uint8_t* maskp = maskdata;
	if (ginc[3] == dim[3] && hinc[3] == dim[3])
	{
		// Special casing if the ginc[3] is the same as dim[3]
		for (i[0] = 0; i[0] < dim[0]; i[0]++)
		{
			for (i[1] = 0; i[1] < dim[1]; i[1]++)
			{
				for (x = 0; x < count; x++)
					hp[x] = maskp[x] ? 0 : gp[x] * inv_p;
				gp += ginc[2] * ginc[3];
				hp += hinc[2] * hinc[3];
				maskp += count;
			}
			gp += (ginc[1] - dim[1]) * ginc[2] * ginc[3];
			hp += (hinc[1] - dim[1]) * hinc[2] * hinc[3];
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
					hp[x] = maskp[x] ? 0 : gp[x] * inv_p;
				maskp += dim[3];
				gp += ginc[3];
				hp += hinc[3];
			}
			gp += (ginc[2] - dim[2]) * ginc[3];
			hp += (hinc[2] - dim[2]) * hinc[3];
		}
		gp += (ginc[1] - dim[1]) * ginc[2] * ginc[3];
		hp += (hinc[1] - dim[1]) * hinc[2] * hinc[3];
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_DROPOUT_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_dropout_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_DROPOUT_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_dropout_back;
}
