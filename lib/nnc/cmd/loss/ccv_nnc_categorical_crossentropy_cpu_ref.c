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

static int _ccv_nnc_categorical_crossentropy_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 2);
	const ccv_nnc_tensor_t* a = inputs[0];
	assert(!CCV_IS_TENSOR_VIEW(a));
	const ccv_nnc_tensor_t* b = inputs[1];
	assert(!CCV_IS_TENSOR_VIEW(b));
	assert(output_size == 1);
	ccv_nnc_tensor_t* c = outputs[0];
	assert(!CCV_IS_TENSOR_VIEW(c));
	const int axis_count = ccv_nnc_tensor_nd(a->info.dim);
	const int batch_size = axis_count < 2 ? 1 : a->info.dim[0];
	const int count = ccv_nnc_tensor_count(a->info) / batch_size;
	int i;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && b->info.dim[i] > 0; i++)
		{ assert(b->info.dim[i] == c->info.dim[i]); }
	if (b->info.datatype == CCV_32F)
	{
		parallel_for(i, batch_size) {
			const int label = (int)(b->data.f32[i] + 0.5);
			assert(label >= 0 && label < count);
			const float p = a->data.f32[i * count + label];
			c->data.f32[i] = -logf(p);
		} parallel_endfor
	} else if (b->info.datatype == CCV_32S) {
		parallel_for(i, batch_size) {
			const int label = b->data.i32[i];
			assert(label >= 0 && label < count);
			const float p = a->data.f32[i * count + label];
			c->data.f32[i] = -logf(p);
		} parallel_endfor
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_categorical_crossentropy_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 3);
	assert(output_size >= 1);
	const ccv_nnc_tensor_t* g = inputs[0];
	assert(!g || !CCV_IS_TENSOR_VIEW(g));
	const ccv_nnc_tensor_t* a = inputs[1];
	assert(!CCV_IS_TENSOR_VIEW(a));
	const ccv_nnc_tensor_t* b = inputs[2];
	assert(!CCV_IS_TENSOR_VIEW(b));
	ccv_nnc_tensor_t* h = outputs[0];
	assert(!CCV_IS_TENSOR_VIEW(h));
	const int axis_count = ccv_nnc_tensor_nd(a->info.dim);
	const int batch_size = axis_count < 2 ? 1 : a->info.dim[0];
	const int count = ccv_nnc_tensor_count(a->info) / batch_size;
	int i;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && a->info.dim[i] > 0; i++)
		{ assert(a->info.dim[i] == h->info.dim[i]); }
	if (g)
	{
		if (b->info.datatype == CCV_32F)
		{
			parallel_for(i, batch_size) {
				int j;
				const float gp = g->data.f32[i];
				const int label = (int)(b->data.f32[i] + 0.5);
				float* const hp = h->data.f32 + i * count;
				for (j = 0; j < count; j++)
					hp[j] = 0;
				const float p = a->data.f32[i * count + label];
				hp[label] = -gp / p;
			} parallel_endfor
		} else if (b->info.datatype == CCV_32S) {
			parallel_for(i, batch_size) {
				int j;
				const float gp = g->data.f32[i];
				const int label = b->data.i32[i];
				float* const hp = h->data.f32 + i * count;
				for (j = 0; j < count; j++)
					hp[j] = 0;
				const float p = a->data.f32[i * count + label];
				hp[label] = -gp / p;
			} parallel_endfor
		}
	} else {
		if (b->info.datatype == CCV_32F)
		{
			parallel_for(i, batch_size) {
				int j;
				const int label = (int)(b->data.f32[i] + 0.5);
				float* const hp = h->data.f32 + i * count;
				for (j = 0; j < count; j++)
					hp[j] = 0;
				const float p = a->data.f32[i * count + label];
				hp[label] = -1. / p;
			} parallel_endfor
		} else if (b->info.datatype == CCV_32S) {
			parallel_for(i, batch_size) {
				int j;
				const int label = b->data.i32[i];
				float* const hp = h->data.f32 + i * count;
				for (j = 0; j < count; j++)
					hp[j] = 0;
				const float p = a->data.f32[i * count + label];
				hp[label] = -1. / p;
			} parallel_endfor
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_CATEGORICAL_CROSSENTROPY_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_categorical_crossentropy_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_CATEGORICAL_CROSSENTROPY_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_categorical_crossentropy_back;
}
