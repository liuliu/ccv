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

static int _ccv_nnc_axpy_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	if (input_size == 1 || inputs[1] == 0)
	{
		// It cannot be set otherwise we have trouble.
		assert(cmd.info.blas.a[1] == 0);
		if (cmd.info.blas.a[0] == 1)
		{
			_ccv_nnc_tensor_transfer_cpu_ref((ccv_nnc_tensor_view_t*)inputs[0], (ccv_nnc_tensor_view_t*)outputs[0]);
			return CCV_NNC_EXEC_SUCCESS;
		} else if (cmd.info.blas.a[0] == 0) {
			ccv_nnc_tensor_zero(outputs[0]);
			return CCV_NNC_EXEC_SUCCESS;
		}
		// Assuming this is float 32.
		int dim[CCV_NNC_MAX_DIM + 2];
		int ainc[CCV_NNC_MAX_DIM + 2];
		int binc[CCV_NNC_MAX_DIM + 2];
		ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[0];
		ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[0];
		assert(a->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		const float p = cmd.info.blas.a[0];
		ccv_nnc_tensor_view_get_dim(a, dim);
		ccv_nnc_tensor_view_check_dim(b, dim);
		int x;
		if (!CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(b))
		{
			// Super optimal case, just do one for-loop for sum.
			const int tensor_count = ccv_nnc_tensor_count(a->info);
			for (x = 0; x < tensor_count; x++)
				b->data.f32[x] = p * a->data.f32[x];
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
						bp[x] = p * ap[x];
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
						bp[x] = p * ap[x];
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
	if (cmd.info.blas.a[0] == 1 && cmd.info.blas.a[1] == 1)
	{
		ccv_nnc_cmd_t forw_cmd = cmd;
		forw_cmd.cmd = CCV_NNC_EWSUM_FORWARD;
		return _ccv_nnc_ewsum_forw_cpu_ref(cmd, hint, flags, inputs, input_size, outputs, output_size, stream_context);
	} else if (cmd.info.blas.a[0] == 1 && cmd.info.blas.a[1] == 0) {
		_ccv_nnc_tensor_transfer_cpu_ref((const ccv_nnc_tensor_view_t*)inputs[0], (ccv_nnc_tensor_view_t*)outputs[0]);
		return CCV_NNC_EXEC_SUCCESS;
	} else if (cmd.info.blas.a[0] == 0 && cmd.info.blas.a[1] == 1) {
		_ccv_nnc_tensor_transfer_cpu_ref((const ccv_nnc_tensor_view_t*)inputs[1], (ccv_nnc_tensor_view_t*)outputs[0]);
		return CCV_NNC_EXEC_SUCCESS;
	} else if (cmd.info.blas.a[0] == 0 && cmd.info.blas.a[1] == 0) {
		ccv_nnc_tensor_zero(outputs[0]);
		return CCV_NNC_EXEC_SUCCESS;
	}
	// Assuming this is float 32.
	int dim[CCV_NNC_MAX_DIM + 2];
	int ainc[CCV_NNC_MAX_DIM + 2];
	int binc[CCV_NNC_MAX_DIM + 2];
	int cinc[CCV_NNC_MAX_DIM + 2];
	ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)inputs[1];
	ccv_nnc_tensor_view_t* c = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(a->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	assert(c->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	const float p = cmd.info.blas.a[0];
	const float q = cmd.info.blas.a[1];
	ccv_nnc_tensor_view_get_dim(a, dim);
	ccv_nnc_tensor_view_check_dim(b, dim);
	ccv_nnc_tensor_view_check_dim(c, dim);
	int x;
	if (!CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(b) && !CCV_IS_TENSOR_VIEW(c))
	{
		// Super optimal case, just do one for-loop for sum.
		const int tensor_count = ccv_nnc_tensor_count(a->info);
		for (x = 0; x < tensor_count; x++)
			c->data.f32[x] = p * a->data.f32[x] + q * b->data.f32[x];
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
					cp[x] = p * ap[x] + q * bp[x];
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
					cp[x] = p * ap[x] + q * bp[x];
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

static int _ccv_nnc_axpy_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	if (inputs[0] == 0)
	{
		if (outputs[0])
			_ccv_nnc_tensor_set_cpu_ref((ccv_nnc_tensor_view_t*)outputs[0], cmd.info.blas.a[0]);
		if (output_size > 1 && outputs[1])
			_ccv_nnc_tensor_set_cpu_ref((ccv_nnc_tensor_view_t*)outputs[1], cmd.info.blas.a[1]);
	} else {
		ccv_nnc_cmd_t forw_cmd = cmd;
		forw_cmd.cmd = CCV_NNC_AXPY_FORWARD;
		memset(forw_cmd.info.blas.a, 0, sizeof(forw_cmd.info.blas.a));
		if (outputs[0])
		{
			forw_cmd.info.blas.a[0] = cmd.info.blas.a[0];
			_ccv_nnc_axpy_forw(forw_cmd, hint, flags, inputs, 1, outputs, 1, stream_context);
		}
		if (output_size > 1 && outputs[1])
		{
			forw_cmd.info.blas.a[0] = cmd.info.blas.a[1];
			_ccv_nnc_axpy_forw(forw_cmd, hint, flags, inputs, 1, outputs + 1, 1, stream_context);
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_AXPY_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_axpy_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_AXPY_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_axpy_back;
}
