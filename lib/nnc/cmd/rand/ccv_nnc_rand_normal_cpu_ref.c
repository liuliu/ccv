#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#include "3rdparty/dsfmt/dSFMT.h"
#ifdef USE_OPENMP
#include <omp.h>
#endif
#ifdef USE_DISPATCH
#include <dispatch/dispatch.h>
#endif

static int _ccv_nnc_random_normal(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(output_size == 1);
	ccv_nnc_tensor_t* const a = outputs[0];
	assert(!CCV_IS_TENSOR_VIEW(a));
	const int count = ccv_nnc_tensor_count(a->info);
	int i;
	float* const ap = a->data.f32;
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, ccv_nnc_stream_context_genrand_uint32(stream_context));
	const float std = cmd.info.blas.a[0];
	const float mean = cmd.info.blas.a[1];
	for (i = 0; i < count + 1; i += 2)
	{
		const double r0 = dsfmt_genrand_open_open(&dsfmt);
		const double r1 = dsfmt_genrand_open_open(&dsfmt);
		const float mag = std * sqrt(-2.0 * log(r0));
		const float z0  = mag * cos(CCV_PI * 2 * r1) + mean;
		const float z1  = mag * sin(CCV_PI * 2 * r1) + mean;
		ap[i] = z0;
		if (i + 1 < count)
			ap[i + 1] = z1;
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_RANDOM_NORMAL_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_random_normal;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_RANDOM_NORMAL_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_random_normal;
}
