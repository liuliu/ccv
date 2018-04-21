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
	assert(a->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	if (!CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(b))
	{
		// Super optimal case, just do memcpy.
		memcpy(b->data.f32, a->data.f32, ccv_nnc_tensor_count(a->info) * sizeof(float));
		return;
	}
	ccv_nnc_tensor_view_get_dim(a, dim);
	assert(ccv_nnc_tensor_view_check_dim(b, dim));
	ccv_nnc_tensor_view_get_inc(a, ainc);
	ccv_nnc_tensor_view_get_inc(b, binc);
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	int i[CCV_NNC_MAX_DIM + 2];
	float* ap = a->data.f32;
	float* bp = b->data.f32;
	if (ainc[3] == dim[3] && binc[3] == dim[3])
	{
		// Special casing if the ainc[3] is the same as dim[3] (do memcpy for the last two dim)
		for (i[0] = 0; i[0] < dim[0]; i[0]++)
		{
			for (i[1] = 0; i[1] < dim[1]; i[1]++)
			{
				memcpy(bp, ap, dim[2] * dim[3] * sizeof(float));
				ap += ainc[2] * ainc[3];
				bp += binc[2] * binc[3];
			}
			ap += (ainc[1] - dim[1]) * ainc[2] * ainc[3];
			bp += (binc[1] - dim[1]) * binc[2] * binc[3];
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
				memcpy(bp, ap, dim[3] * sizeof(float));
				ap += ainc[3];
				bp += binc[3];
			}
			ap += (ainc[2] - dim[2]) * ainc[3];
			bp += (binc[2] - dim[2]) * binc[3];
		}
		ap += (ainc[1] - dim[1]) * ainc[2] * ainc[3];
		bp += (binc[1] - dim[1]) * binc[2] * binc[3];
	}
}

void _ccv_nnc_tensor_set_cpu_ref(ccv_nnc_tensor_view_t* const a, const float b)
{
	// Assuming this is float 32.
	int dim[CCV_NNC_MAX_DIM + 2];
	int ainc[CCV_NNC_MAX_DIM + 2];
	assert(a->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	int x;
	if (!CCV_IS_TENSOR_VIEW(a))
	{
		// Super optimal case, just do one for-loop for sum.
		const int tensor_count = ccv_nnc_tensor_count(a->info);
		for (x = 0; x < tensor_count; x++)
			a->data.f32[x] = b;
		return;
	}
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	ccv_nnc_tensor_view_get_dim(a, dim);
	ccv_nnc_tensor_view_get_inc(a, ainc);
	int i[CCV_NNC_MAX_DIM + 2];
	float* ap = a->data.f32;
	const int count = dim[2] * dim[3];
	if (ainc[3] == dim[3])
	{
		// Special casing if the ainc[3] is the same as dim[3]
		for (i[0] = 0; i[0] < dim[0]; i[0]++)
		{
			for (i[1] = 0; i[1] < dim[1]; i[1]++)
			{
				for (x = 0; x < count; x++)
					ap[x] = b;
				ap += ainc[2] * ainc[3];
			}
			ap += (ainc[1] - dim[1]) * ainc[2] * ainc[3];
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
					ap[x] = b;
				ap += ainc[3];
			}
			ap += (ainc[2] - dim[2]) * ainc[3];
		}
		ap += (ainc[1] - dim[1]) * ainc[2] * ainc[3];
	}
}

static int _ccv_nnc_data_transfer(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	assert(output_size <= input_size);
	int i;
	for (i = 0; i < output_size; i++)
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

static int _ccv_nnc_set_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
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

static int _ccv_nnc_set_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	int i;
	for (i = 0; i < output_size; i++)
		ccv_nnc_tensor_zero(outputs[i]);
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_set_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SET_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_set_back;
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
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
	const int a_offset = CCV_NNC_MAX_DIM + 2 - a_nd;
	assert(a_offset == 0 || a_offset == 1);
	const int b_offset = CCV_NNC_MAX_DIM + 2 - b_nd;
	assert(b_offset == 0 || b_offset == 1);
	ccv_nnc_tensor_view_get_inc(a, ainc);
	ccv_nnc_tensor_view_get_inc(b, binc);
	// Comparing N
	assert((a_offset == 0 ? a->info.dim[0] : 1) == (b_offset == 0 ? b->info.dim[0] : 1));
	const int n = (a_offset == 0 ? a->info.dim[0] : 1);
	// Comparing C
	assert(a->info.dim[a_nd - 1] == b->info.dim[1 - b_offset]);
	const int c = a->info.dim[a_nd - 1];
	// Comparing HW
	int hw[CCV_NNC_MAX_DIM];
	for (k = 0; k < CCV_NNC_MAX_DIM; k++)
	{
		assert(a->info.dim[k + 1 - a_offset] == b->info.dim[k + 2 - b_offset]);
		hw[k] = a->info.dim[k + 1 - a_offset];
	}
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	int i[CCV_NNC_MAX_DIM + 2];
	float* ap = a->data.f32;
	float* bp = b->data.f32;
	// Non-optimal case, need to do skip copy.
	for (i[0] = 0; i[0] < n; i[0]++)
	{
		for (i[3] = 0; i[3] < c; i[3]++)
		{
			float* apu = ap + i[3];
			for (i[1] = 0; i[1] < hw[0]; i[1]++)
			{
				for (i[2] = 0; i[2] < hw[1]; i[2]++)
					bp[i[2]] = apu[i[2] * ainc[3]];
				apu += ainc[2] * ainc[3];
				bp += binc[3];
			}
			bp += (binc[2] - hw[0]) * binc[3];
		}
		ap += ainc[1] * ainc[2] * ainc[3];
		bp += (binc[1] - c) * binc[2] * binc[3];
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
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
	const int a_offset = CCV_NNC_MAX_DIM + 2 - a_nd;
	assert(a_offset == 0 || a_offset == 1);
	const int b_offset = CCV_NNC_MAX_DIM + 2 - b_nd;
	assert(b_offset == 0 || b_offset == 1);
	ccv_nnc_tensor_view_get_inc(a, ainc);
	ccv_nnc_tensor_view_get_inc(b, binc);
	// Comparing N
	assert((a_offset == 0 ? a->info.dim[0] : 1) == (b_offset == 0 ? b->info.dim[0] : 1));
	const int n = (a_offset == 0 ? a->info.dim[0] : 1);
	// Comparing C
	assert(a->info.dim[1 - a_offset] == b->info.dim[b_nd - 1]);
	const int c = a->info.dim[1 - a_offset];
	// Comparing HW
	int hw[CCV_NNC_MAX_DIM];
	for (k = 0; k < CCV_NNC_MAX_DIM; k++)
	{
		assert(a->info.dim[k + 2 - a_offset] == b->info.dim[k + 1 - b_offset]);
		hw[k] = a->info.dim[k + 2 - a_offset];
	}
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	int i[CCV_NNC_MAX_DIM + 2];
	float* ap = a->data.f32;
	float* bp = b->data.f32;
	// Non-optimal case, need to do skip copy.
	for (i[0] = 0; i[0] < n; i[0]++)
	{
		for (i[3] = 0; i[3] < c; i[3]++)
		{
			float* bpu = bp + i[3];
			for (i[1] = 0; i[1] < hw[0]; i[1]++)
			{
				for (i[2] = 0; i[2] < hw[1]; i[2]++)
					bpu[i[2] * binc[3]] = ap[i[2]];
				ap += ainc[3];
				bpu += binc[2] * binc[3];
			}
			ap += (ainc[2] - hw[0]) * ainc[3];
		}
		ap += (ainc[1] - c) * ainc[2] * ainc[3];
		bp += binc[1] * binc[2] * binc[3];
	}
}

static int _ccv_nnc_format_transform(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	assert(output_size <= input_size);
	int i;
	for (i = 0; i < output_size; i++)
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
