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

void _ccv_nnc_tensor_transfer_cpu_ref(const ccv_nnc_tensor_view_t* const a, ccv_nnc_tensor_view_t* const b)
{
	// Assuming this is float 32.
	int dim[CCV_NNC_MAX_DIM + 2];
	int ainc[CCV_NNC_MAX_DIM + 2];
	int binc[CCV_NNC_MAX_DIM + 2];
	int x;
	assert(a->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
	{
		assert(ccv_max(1, a->info.dim[x]) == ccv_max(1, b->info.dim[x]));
		dim[x] = ccv_max(1, a->info.dim[x]);
		ainc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(a) ? a->inc[x] : a->info.dim[x]);
		binc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(b) ? b->inc[x] : b->info.dim[x]);
	}
	if (!CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(b))
	{
		// Super optimal case, just do memcpy.
		memcpy(b->data.f32, a->data.f32, ccv_nnc_tensor_count(a->info) * sizeof(float));
		return;
	}
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	int i[CCV_NNC_MAX_DIM + 2];
	float* ap = a->data.f32;
	float* bp = b->data.f32;
	if (ainc[0] == dim[0] && binc[0] == dim[0])
	{
		// Special casing if the ainc[0] is the same as dim[0] (do memcpy for the last two dim)
		for (i[3] = 0; i[3] < dim[3]; i[3]++)
		{
			for (i[2] = 0; i[2] < dim[2]; i[2]++)
			{
				memcpy(bp, ap, dim[1] * dim[0] * sizeof(float));
				ap += ainc[1] * ainc[0];
				bp += binc[1] * binc[0];
			}
			ap += (ainc[2] - dim[2]) * ainc[1] * ainc[0];
			bp += (binc[2] - dim[2]) * binc[1] * binc[0];
		}
		return;
	}
	// Non-optimal case, need to do skip copy.
	for (i[3] = 0; i[3] < dim[3]; i[3]++)
	{
		for (i[2] = 0; i[2] < dim[2]; i[2]++)
		{
			for (i[1] = 0; i[1] < dim[1]; i[1]++)
			{
				memcpy(bp, ap, dim[0] * sizeof(float));
				ap += ainc[0];
				bp += binc[0];
			}
			ap += (ainc[1] - dim[1]) * ainc[0];
			bp += (binc[1] - dim[1]) * binc[0];
		}
		ap += (ainc[2] - dim[2]) * ainc[1] * ainc[0];
		bp += (binc[2] - dim[2]) * binc[1] * binc[0];
	}
}

void _ccv_nnc_tensor_set_cpu_ref(ccv_nnc_tensor_view_t* const a, const float b)
{
	// Assuming this is float 32.
	int dim[CCV_NNC_MAX_DIM + 2];
	int ainc[CCV_NNC_MAX_DIM + 2];
	assert(a->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	int x;
	for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
	{
		dim[x] = ccv_max(1, a->info.dim[x]);
		ainc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(a) ? a->inc[x] : a->info.dim[x]);
	}
	if (!CCV_IS_TENSOR_VIEW(a))
	{
		// Super optimal case, just do one for-loop for sum.
		const int tensor_count = ccv_nnc_tensor_count(a->info);
		for (x = 0; x < tensor_count; x++)
			a->data.f32[x] = b;
		return;
	}
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	int i[CCV_NNC_MAX_DIM + 2];
	float* ap = a->data.f32;
	const int count = dim[1] * dim[0];
	if (ainc[0] == dim[0])
	{
		// Special casing if the ainc[0] is the same as dim[0]
		for (i[3] = 0; i[3] < dim[3]; i[3]++)
		{
			for (i[2] = 0; i[2] < dim[2]; i[2]++)
			{
				for (x = 0; x < count; x++)
					ap[x] = b;
				ap += ainc[1] * ainc[0];
			}
			ap += (ainc[2] - dim[2]) * ainc[1] * ainc[0];
		}
		return;
	}
	// Non-optimal case, need to do skip copy.
	for (i[3] = 0; i[3] < dim[3]; i[3]++)
	{
		for (i[2] = 0; i[2] < dim[2]; i[2]++)
		{
			for (i[1] = 0; i[1] < dim[1]; i[1]++)
			{
				for (x = 0; x < dim[0]; x++)
					ap[x] = b;
				ap += ainc[0];
			}
			ap += (ainc[1] - dim[1]) * ainc[0];
		}
		ap += (ainc[2] - dim[2]) * ainc[1] * ainc[0];
	}
}

static int _ccv_nnc_data_transfer(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	assert(output_size == input_size);
	int i;
	for (i = 0; i < input_size; i++)
	{
		const ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[i];
		ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[i];
		if (a != b) // Only do transfer if these are two different tensors.
			_ccv_nnc_tensor_transfer_cpu_ref(a, b);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_data_transfer;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_DATA_TRANSFER_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_data_transfer;
}

static int _ccv_nnc_set(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	int i;
	if (cmd.info.blas.a[0] == 0)
		for (i = 0; i < output_size; i++)
			ccv_nnc_tensor_zero(outputs[i]);
	else
		for (i = 0; i < output_size; i++)
			_ccv_nnc_tensor_set_cpu_ref((ccv_nnc_tensor_view_t*)outputs[i], cmd.info.blas.a[0]);
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_set;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SET_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_set;
}

static void _ccv_nnc_tensor_nhwc_nchw(const ccv_nnc_tensor_view_t* a, ccv_nnc_tensor_view_t* b)
{
	// Assuming this is float 32.
	int ainc[CCV_NNC_MAX_DIM + 2];
	int binc[CCV_NNC_MAX_DIM + 2];
	int k;
	// In case it is Toll-free bridged matrix object (NHWC format is possible).
	assert(a->info.dim[CCV_NNC_MAX_DIM + 2] == 0 || a->info.dim[CCV_NNC_MAX_DIM + 1] == 0);
	assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	for (k = 0; k < CCV_NNC_MAX_DIM + 2; k++)
	{
		ainc[k] = ccv_max(1, CCV_IS_TENSOR_VIEW(a) ? a->inc[k] : a->info.dim[k]);
		binc[k] = ccv_max(1, CCV_IS_TENSOR_VIEW(b) ? b->inc[k] : b->info.dim[k]);
	}
	// Comparing N
	assert(ccv_max(1, a->info.dim[CCV_NNC_MAX_DIM + 1]) == ccv_max(1, b->info.dim[CCV_NNC_MAX_DIM + 1]));
	const int n = ccv_max(1, a->info.dim[CCV_NNC_MAX_DIM + 1]);
	// Comparing C
	assert(a->info.dim[0] == b->info.dim[CCV_NNC_MAX_DIM]);
	const int c = a->info.dim[0];
	// Comparing HW
	int hw[CCV_NNC_MAX_DIM];
	for (k = 0; k < CCV_NNC_MAX_DIM; k++)
	{
		assert(a->info.dim[k + 1] == b->info.dim[k]);
		hw[k] = a->info.dim[k + 1];
	}
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	int i[CCV_NNC_MAX_DIM + 2];
	float* ap = a->data.f32;
	float* bp = b->data.f32;
	// Non-optimal case, need to do skip copy.
	for (i[3] = 0; i[3] < n; i[3]++)
	{
		for (i[0] = 0; i[0] < c; i[0]++)
		{
			float* apu = ap + i[0];
			for (i[2] = 0; i[2] < hw[1]; i[2]++)
			{
				for (i[1] = 0; i[1] < hw[0]; i[1]++)
					bp[i[1]] = apu[i[1] * ainc[0]];
				apu += ainc[0] * ainc[1];
				bp += binc[0];
			}
			bp += (binc[1] - hw[1]) * binc[0];
		}
		ap += ainc[2] * ainc[1] * ainc[0];
		bp += (binc[2] - c) * binc[1] * binc[0];
	}
}

static void _ccv_nnc_tensor_nchw_nhwc(const ccv_nnc_tensor_view_t* a, ccv_nnc_tensor_view_t* b)
{
	// Assuming this is float 32.
	int ainc[CCV_NNC_MAX_DIM + 2];
	int binc[CCV_NNC_MAX_DIM + 2];
	int k;
	// In case it is Toll-free bridged matrix object (NHWC format is possible).
	assert(a->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0 || b->info.dim[CCV_NNC_MAX_DIM + 1] == 0);
	for (k = 0; k < CCV_NNC_MAX_DIM + 2; k++)
	{
		ainc[k] = ccv_max(1, CCV_IS_TENSOR_VIEW(a) ? a->inc[k] : a->info.dim[k]);
		binc[k] = ccv_max(1, CCV_IS_TENSOR_VIEW(b) ? b->inc[k] : b->info.dim[k]);
	}
	// Comparing N
	assert(ccv_max(1, a->info.dim[CCV_NNC_MAX_DIM + 1]) == ccv_max(1, b->info.dim[CCV_NNC_MAX_DIM + 1]));
	const int n = ccv_max(1, a->info.dim[CCV_NNC_MAX_DIM + 1]);
	// Comparing C
	assert(a->info.dim[CCV_NNC_MAX_DIM] == b->info.dim[0]);
	const int c = a->info.dim[CCV_NNC_MAX_DIM];
	// Comparing HW
	int hw[CCV_NNC_MAX_DIM];
	for (k = 0; k < CCV_NNC_MAX_DIM; k++)
	{
		assert(a->info.dim[k] == b->info.dim[k + 1]);
		hw[k] = a->info.dim[k];
	}
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	int i[CCV_NNC_MAX_DIM + 2];
	float* ap = a->data.f32;
	float* bp = b->data.f32;
	// Non-optimal case, need to do skip copy.
	for (i[3] = 0; i[3] < n; i[3]++)
	{
		for (i[0] = 0; i[0] < c; i[0]++)
		{
			float* bpu = bp + i[0];
			for (i[2] = 0; i[2] < hw[1]; i[2]++)
			{
				for (i[1] = 0; i[1] < hw[0]; i[1]++)
					bpu[i[1] * binc[0]] = ap[i[1]];
				ap += ainc[0];
				bpu += binc[0] * binc[1];
			}
			ap += (ainc[1] - hw[1]) * ainc[0];
		}
		ap += (ainc[2] - c) * ainc[1] * ainc[0];
		bp += binc[2] * binc[1] * binc[0];
	}
}

static int _ccv_nnc_format_transform(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	assert(output_size == input_size);
	int i;
	for (i = 0; i < input_size; i++)
	{
		const ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[i];
		ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[i];
		assert(a != b); // Cannot do inplace transform.
		if (a->info.format == b->info.format) {
			// If it is the same, just do a normal data transfer.
			_ccv_nnc_tensor_transfer_cpu_ref(a, b);
		} else if (a->info.format == CCV_TENSOR_FORMAT_NHWC && b->info.format == CCV_TENSOR_FORMAT_NCHW) {
			_ccv_nnc_tensor_nhwc_nchw(a, b);
		} else if (a->info.format == CCV_TENSOR_FORMAT_NHWC && b->info.format == CCV_TENSOR_FORMAT_CHWN) {
		} else if (a->info.format == CCV_TENSOR_FORMAT_NCHW && b->info.format == CCV_TENSOR_FORMAT_NHWC) {
			_ccv_nnc_tensor_nchw_nhwc(a, b);
		} else if (a->info.format == CCV_TENSOR_FORMAT_NCHW && b->info.format == CCV_TENSOR_FORMAT_CHWN) {
		} else if (a->info.format == CCV_TENSOR_FORMAT_CHWN && b->info.format == CCV_TENSOR_FORMAT_NHWC) {
		} else if (a->info.format == CCV_TENSOR_FORMAT_CHWN && b->info.format == CCV_TENSOR_FORMAT_NCHW) {
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_FORMAT_TRANSFORM_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_format_transform;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_FORMAT_TRANSFORM_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_format_transform;
}
