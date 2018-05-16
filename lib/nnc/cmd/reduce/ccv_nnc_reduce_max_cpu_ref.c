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

// Shared methods.
#include "../_ccv_nnc_cpu_ref.h"

static int _ccv_nnc_reduce_max_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(a->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	// Assuming this is float 32.
	int adim[CCV_NNC_MAX_DIM + 2];
	int bdim[CCV_NNC_MAX_DIM + 2];
	ccv_nnc_tensor_view_get_dim(a, adim);
	ccv_nnc_tensor_view_get_dim(b, bdim);
	assert(ccv_nnc_tensor_view_check_broadcast_dim(b, adim));
	int ainc[CCV_NNC_MAX_DIM + 2];
	int binc[CCV_NNC_MAX_DIM + 2];
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	ccv_nnc_tensor_view_get_inc(a, ainc);
	ccv_nnc_tensor_view_get_inc(b, binc);
	int i[CCV_NNC_MAX_DIM + 2];
	int x;
	_ccv_nnc_tensor_set_cpu_ref(b, -FLT_MAX);
	float* ap = a->data.f32;
	float* const bp = b->data.f32;
	// Non-optimal case, need to do skip if needed.
	for (i[0] = 0; i[0] < adim[0]; i[0]++)
	{
		float* const bp0 = bdim[0] == 1 ? bp : bp + i[0] * binc[1] * binc[2] * binc[3];
		for (i[1] = 0; i[1] < adim[1]; i[1]++)
		{
			float* const bp1 = bdim[1] == 1 ? bp0 : bp0 + i[1] * binc[2] * binc[3];
			for (i[2] = 0; i[2] < adim[2]; i[2]++)
			{
				float* const bp2 = bdim[2] == 1 ? bp1 : bp1 + i[2] * binc[3];
				if (bdim[3] == 1)
				{
					for (x = 0; x < adim[3]; x++)
						if (ap[x] > bp2[0])
							bp2[0] = ap[x];
				} else {
					for (x = 0; x < adim[3]; x++)
						if (ap[x] > bp2[x])
							bp2[x] = ap[x];
				}
				ap += ainc[3];
			}
			ap += (ainc[2] - adim[2]) * ainc[3];
		}
		ap += (ainc[1] - adim[1]) * ainc[2] * ainc[3];
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_reduce_max_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	if (inputs[0] == 0)
	{
		ccv_nnc_tensor_view_t* const h = (ccv_nnc_tensor_view_t*)outputs[0];
		ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[1];
		ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)inputs[2];
		assert(h->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		assert(a->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		// Assuming this is float 32.
		int hdim[CCV_NNC_MAX_DIM + 2];
		int bdim[CCV_NNC_MAX_DIM + 2];
		ccv_nnc_tensor_view_get_dim(h, hdim);
		ccv_nnc_tensor_view_get_dim(b, bdim);
		assert(ccv_nnc_tensor_view_check_broadcast_dim(b, hdim));
		assert(ccv_nnc_tensor_view_check_dim(a, hdim));
		int hinc[CCV_NNC_MAX_DIM + 2];
		int ainc[CCV_NNC_MAX_DIM + 2];
		int binc[CCV_NNC_MAX_DIM + 2];
		assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
		ccv_nnc_tensor_view_get_inc(h, hinc);
		ccv_nnc_tensor_view_get_inc(a, ainc);
		ccv_nnc_tensor_view_get_inc(b, binc);
		int i[CCV_NNC_MAX_DIM + 2];
		int x;
		float* hp = h->data.f32;
		float* ap = a->data.f32;
		float* const bp = b->data.f32;
		ccv_nnc_tensor_zero(h);
		// Non-optimal case, need to do skip if needed.
		for (i[0] = 0; i[0] < hdim[0]; i[0]++)
		{
			float* const bp0 = bdim[0] == 1 ? bp : bp + i[0] * binc[1] * binc[2] * binc[3];
			for (i[1] = 0; i[1] < hdim[1]; i[1]++)
			{
				float* const bp1 = bdim[1] == 1 ? bp0 : bp0 + i[1] * binc[2] * binc[3];
				for (i[2] = 0; i[2] < hdim[2]; i[2]++)
				{
					float* const bp2 = bdim[2] == 1 ? bp1 : bp1 + i[2] * binc[3];
					if (bdim[3] == 1)
					{
						for (x = 0; x < hdim[3]; x++)
							if (ap[x] == bp2[0])
								hp[x] = 1;
					} else {
						for (x = 0; x < hdim[3]; x++)
							if (ap[x] == bp2[x])
								hp[x] = 1;
					}
					hp += hinc[3];
					ap += ainc[3];
				}
				hp += (hinc[2] - hdim[2]) * hinc[3];
				ap += (ainc[2] - hdim[2]) * ainc[3];
			}
			hp += (hinc[1] - hdim[1]) * hinc[2] * hinc[3];
			ap += (ainc[1] - hdim[1]) * ainc[2] * ainc[3];
		}
		return CCV_NNC_EXEC_SUCCESS;
	}
	ccv_nnc_tensor_view_t* const h = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* const g = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[1];
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)inputs[2];
	assert(h->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	assert(g->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	assert(a->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	// Assuming this is float 32.
	int hdim[CCV_NNC_MAX_DIM + 2];
	int gdim[CCV_NNC_MAX_DIM + 2];
	ccv_nnc_tensor_view_get_dim(h, hdim);
	ccv_nnc_tensor_view_get_dim(g, gdim);
	assert(ccv_nnc_tensor_view_check_broadcast_dim(g, hdim));
	assert(ccv_nnc_tensor_view_check_dim(a, hdim));
	assert(ccv_nnc_tensor_view_check_dim(b, gdim));
	int hinc[CCV_NNC_MAX_DIM + 2];
	int ginc[CCV_NNC_MAX_DIM + 2];
	int ainc[CCV_NNC_MAX_DIM + 2];
	int binc[CCV_NNC_MAX_DIM + 2];
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	ccv_nnc_tensor_view_get_inc(h, hinc);
	ccv_nnc_tensor_view_get_inc(g, ginc);
	ccv_nnc_tensor_view_get_inc(a, ainc);
	ccv_nnc_tensor_view_get_inc(b, binc);
	int i[CCV_NNC_MAX_DIM + 2];
	int x;
	float* hp = h->data.f32;
	float* const gp = g->data.f32;
	float* ap = a->data.f32;
	float* const bp = b->data.f32;
	ccv_nnc_tensor_zero(h);
	// Non-optimal case, need to do skip if needed.
	for (i[0] = 0; i[0] < hdim[0]; i[0]++)
	{
		float* const gp0 = gdim[0] == 1 ? gp : gp + i[0] * ginc[1] * ginc[2] * ginc[3];
		float* const bp0 = gdim[0] == 1 ? bp : bp + i[0] * binc[1] * binc[2] * binc[3];
		for (i[1] = 0; i[1] < hdim[1]; i[1]++)
		{
			float* const gp1 = gdim[1] == 1 ? gp0 : gp0 + i[1] * ginc[2] * ginc[3];
			float* const bp1 = gdim[1] == 1 ? bp0 : bp0 + i[1] * binc[2] * binc[3];
			for (i[2] = 0; i[2] < hdim[2]; i[2]++)
			{
				float* const gp2 = gdim[2] == 1 ? gp1 : gp1 + i[2] * ginc[3];
				float* const bp2 = gdim[2] == 1 ? bp1 : bp1 + i[2] * binc[3];
				if (gdim[3] == 1)
				{
					for (x = 0; x < hdim[3]; x++)
						if (ap[x] == bp2[0])
							hp[x] = gp2[0];
				} else {
					for (x = 0; x < hdim[3]; x++)
						if (ap[x] == bp2[x])
							hp[x] = gp2[x];
				}
				hp += hinc[3];
				ap += ainc[3];
			}
			hp += (hinc[2] - hdim[2]) * hinc[3];
			ap += (ainc[2] - hdim[2]) * ainc[3];
		}
		hp += (hinc[1] - hdim[1]) * hinc[2] * hinc[3];
		ap += (ainc[1] - hdim[1]) * ainc[2] * ainc[3];
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_REDUCE_MAX_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_reduce_max_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_REDUCE_MAX_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_reduce_max_back;
}
