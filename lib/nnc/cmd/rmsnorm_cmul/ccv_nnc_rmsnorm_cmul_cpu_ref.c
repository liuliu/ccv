#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#include <math.h>

static int _ccv_nnc_rmsnorm_cmul_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 2);
	assert(output_size == 1);
	const ccv_nnc_tensor_t* const a = inputs[0];
	const ccv_nnc_tensor_t* const rotation = inputs[1];
	ccv_nnc_tensor_t* const b = outputs[0];
	assert(a->info.datatype == CCV_32F);
	assert(rotation->info.datatype == CCV_32F);
	assert(b->info.datatype == CCV_32F);
	assert(CCV_IS_TENSOR_CONTIGUOUS(a));
	assert(CCV_IS_TENSOR_CONTIGUOUS(rotation));
	assert(CCV_IS_TENSOR_CONTIGUOUS(b));
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	const int rotation_nd = ccv_nnc_tensor_nd(rotation->info.dim);
	const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
	assert(a_nd >= 1);
	assert(rotation_nd >= 1);
	assert(b_nd == a_nd);
	assert(cmd.info.rmsnorm_cmul.count == 1);
	assert(cmd.info.rmsnorm_cmul.axis[0] >= 0 && cmd.info.rmsnorm_cmul.axis[0] < a_nd);
	assert(cmd.info.rmsnorm_cmul.axis[0] == a_nd - 1);
	const int column_count = a->info.dim[a_nd - 1];
	assert(column_count > 0 && column_count % 2 == 0);
	assert(rotation->info.dim[rotation_nd - 1] == column_count);
	int i;
	for (i = 0; i < a_nd; i++)
		assert(b->info.dim[i] == a->info.dim[i]);
	const int rotation_axis_offset = a_nd - rotation_nd;
	for (i = 0; i < rotation_nd - 1; i++)
	{
		const int a_axis = i + rotation_axis_offset;
		if (a_axis < 0)
		{
			assert(rotation->info.dim[i] == 1);
		} else {
			assert(rotation->info.dim[i] == 1 || rotation->info.dim[i] == a->info.dim[a_axis]);
		}
	}
	const size_t count = ccv_nnc_tensor_count(a->info);
	assert(ccv_nnc_tensor_count(b->info) == count);
	const size_t row_count = count / column_count;
	const float* const ap = a->data.f32;
	const float* const rp = rotation->data.f32;
	float* const bp = b->data.f32;
	for (size_t row = 0; row < row_count; row++)
	{
		const size_t offset = row * column_count;
		size_t a_row_stride = row_count;
		size_t rotation_row = 0;
		for (i = 0; i < a_nd - 1; i++)
		{
			a_row_stride /= a->info.dim[i];
			const int a_index = (row / a_row_stride) % a->info.dim[i];
			const int rotation_axis = i - rotation_axis_offset;
			if (rotation_axis >= 0)
			{
				rotation_row *= rotation->info.dim[rotation_axis];
				if (rotation->info.dim[rotation_axis] != 1)
					rotation_row += a_index;
			}
		}
		const size_t rotation_offset = rotation_row * column_count;
		float square_sum = 0;
		for (i = 0; i < column_count; i++)
			square_sum += ap[offset + i] * ap[offset + i];
		const float inv_rms = 1.0f / sqrtf(square_sum / column_count + cmd.info.rmsnorm_cmul.epsilon);
		for (i = 0; i < column_count; i += 2)
		{
			const float real = ap[offset + i] * inv_rms;
			const float imag = ap[offset + i + 1] * inv_rms;
			const float rotation_real = rp[rotation_offset + i];
			const float rotation_imag = rp[rotation_offset + i + 1];
			bp[offset + i] = real * rotation_real - imag * rotation_imag;
			bp[offset + i + 1] = real * rotation_imag + imag * rotation_real;
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_RMSNORM_CMUL_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_rmsnorm_cmul_forw;
}
