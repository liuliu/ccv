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

static int _ccv_nnc_swiglu_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 3);
	assert(output_size == 1);
	const ccv_nnc_tensor_t* const a = inputs[0];
	const ccv_nnc_tensor_t* const gate_w = inputs[1];
	const ccv_nnc_tensor_t* const up_w = inputs[2];
	ccv_nnc_tensor_t* const output = outputs[0];
	if (!CCV_IS_TENSOR_CONTIGUOUS(a) || !CCV_IS_TENSOR_CONTIGUOUS(gate_w) ||
		!CCV_IS_TENSOR_CONTIGUOUS(up_w) || !CCV_IS_TENSOR_CONTIGUOUS(output))
		return CCV_NNC_EXEC_INVALID;
	if (a->info.datatype != CCV_32F || gate_w->info.datatype != CCV_32F ||
		up_w->info.datatype != CCV_32F || output->info.datatype != CCV_32F)
		return CCV_NNC_EXEC_INVALID;
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	const int gate_w_nd = ccv_nnc_tensor_nd(gate_w->info.dim);
	const int up_w_nd = ccv_nnc_tensor_nd(up_w->info.dim);
	const int output_nd = ccv_nnc_tensor_nd(output->info.dim);
	if (a_nd < 1 || gate_w_nd != 2 || up_w_nd != 2 || output_nd != a_nd)
		return CCV_NNC_EXEC_INVALID;
	const int k = a->info.dim[a_nd - 1];
	const int n = gate_w->info.dim[gate_w_nd - 2];
	if (k <= 0 || n <= 0 || gate_w->info.dim[gate_w_nd - 1] != k ||
		up_w->info.dim[up_w_nd - 2] != n || up_w->info.dim[up_w_nd - 1] != k ||
		output->info.dim[output_nd - 1] != n || output->info.format != a->info.format)
		return CCV_NNC_EXEC_INVALID;
	int i;
	for (i = 0; i < a_nd - 1; i++)
		if (output->info.dim[i] != a->info.dim[i])
			return CCV_NNC_EXEC_INVALID;
	const size_t rows = ccv_nnc_tensor_count(a->info) / k;
	if (ccv_nnc_tensor_count(gate_w->info) != (size_t)n * k ||
		ccv_nnc_tensor_count(up_w->info) != (size_t)n * k ||
		ccv_nnc_tensor_count(output->info) != rows * n)
		return CCV_NNC_EXEC_INVALID;
	const float limit = cmd.info.swiglu.clamp;
	const int clamp_enabled = limit > 0;
	parallel_for(row, rows) {
		const float* const activation = a->data.f32 + row * k;
		float* const output_row = output->data.f32 + row * n;
		int column;
		for (column = 0; column < n; column++)
		{
			const float* const gate_row = gate_w->data.f32 + (size_t)column * k;
			const float* const up_row = up_w->data.f32 + (size_t)column * k;
			float gate = 0;
			float up = 0;
			int inner;
			for (inner = 0; inner < k; inner++)
			{
				gate += activation[inner] * gate_row[inner];
				up += activation[inner] * up_row[inner];
			}
			if (clamp_enabled)
			{
				gate = ccv_min(gate, limit);
				up = ccv_min(ccv_max(up, -limit), limit);
			}
			output_row[column] = up * gate / (1 + expf(-gate));
		}
	} parallel_endfor
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_swiglu_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_INVALID;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SWIGLU_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_swiglu_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SWIGLU_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_swiglu_back;
}
