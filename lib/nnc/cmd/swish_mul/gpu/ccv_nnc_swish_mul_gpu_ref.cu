extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

static int _ccv_nnc_swish_mul_datatype_supported(const int datatype)
{
	return datatype == CCV_32F || datatype == CCV_16F || datatype == CCV_16BF;
}

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_swish_mul_forw_kernel(const size_t count, const float beta, const float scale, const NUM1* const a, const NUM2* const b, NUM1* const c)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		const float value = (float)a[i];
		const float gate = (float)b[i];
		const float sigmoid = 1.f / (1.f + expf(-beta * gate));
		c[i] = (NUM1)(scale * value * gate * sigmoid);
	}
}

template<typename NUM1, typename NUM2>
static void _ccv_nnc_swish_mul_forw_launch(const size_t count, const float beta, const float scale, const ccv_nnc_tensor_t* const a, const ccv_nnc_tensor_t* const b, ccv_nnc_tensor_t* const c, cudaStream_t stream)
{
	_ccv_nnc_swish_mul_forw_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, beta, scale, (const NUM1*)a->data.u8, (const NUM2*)b->data.u8, (NUM1*)c->data.u8);
}

template<typename NUM1>
static int _ccv_nnc_swish_mul_forw_gate(const size_t count, const float beta, const float scale, const ccv_nnc_tensor_t* const a, const ccv_nnc_tensor_t* const b, ccv_nnc_tensor_t* const c, cudaStream_t stream)
{
	if (b->info.datatype == CCV_32F)
		_ccv_nnc_swish_mul_forw_launch<NUM1, float>(count, beta, scale, a, b, c, stream);
	else if (b->info.datatype == CCV_16F)
		_ccv_nnc_swish_mul_forw_launch<NUM1, __half>(count, beta, scale, a, b, c, stream);
	else if (b->info.datatype == CCV_16BF)
		_ccv_nnc_swish_mul_forw_launch<NUM1, __nv_bfloat16>(count, beta, scale, a, b, c, stream);
	else
		return CCV_NNC_EXEC_INVALID;
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_swish_mul_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 2);
	assert(output_size == 1);
	const ccv_nnc_tensor_t* const a = inputs[0];
	const ccv_nnc_tensor_t* const b = inputs[1];
	ccv_nnc_tensor_t* const c = outputs[0];
	if (a->info.dim[0] == 0 || b->info.dim[0] == 0 || c->info.dim[0] == 0)
		return CCV_NNC_EXEC_INVALID;
	if (!CCV_IS_TENSOR_CONTIGUOUS(a) || !CCV_IS_TENSOR_CONTIGUOUS(b) || !CCV_IS_TENSOR_CONTIGUOUS(c))
		return CCV_NNC_EXEC_INVALID;
	if (c->info.datatype != a->info.datatype ||
		!_ccv_nnc_swish_mul_datatype_supported(a->info.datatype) ||
		!_ccv_nnc_swish_mul_datatype_supported(b->info.datatype))
		return CCV_NNC_EXEC_INVALID;
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	if (a_nd != ccv_nnc_tensor_nd(b->info.dim) ||
		a_nd != ccv_nnc_tensor_nd(c->info.dim))
		return CCV_NNC_EXEC_INVALID;
	int i;
	for (i = 0; i < a_nd; i++)
		if (a->info.dim[i] != b->info.dim[i] || a->info.dim[i] != c->info.dim[i])
			return CCV_NNC_EXEC_INVALID;
	const size_t count = ccv_nnc_tensor_count(a->info);
	const float beta = cmd.info.swish_mul.beta;
	const float scale = cmd.info.swish_mul.scale;
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	int status;
	if (a->info.datatype == CCV_32F)
		status = _ccv_nnc_swish_mul_forw_gate<float>(count, beta, scale, a, b, c, stream);
	else if (a->info.datatype == CCV_16F)
		status = _ccv_nnc_swish_mul_forw_gate<__half>(count, beta, scale, a, b, c, stream);
	else
		status = _ccv_nnc_swish_mul_forw_gate<__nv_bfloat16>(count, beta, scale, a, b, c, stream);
	if (status != CCV_NNC_EXEC_SUCCESS)
		return status;
	CUDA_ENFORCE(cudaGetLastError());
	return CCV_NNC_EXEC_SUCCESS;
}

template<typename NUM1, typename NUM2, typename NUM3>
__global__ void _ccv_nnc_swish_mul_dvalue_kernel(const size_t count, const float beta, const float scale, const NUM1* const g, const NUM2* const b, NUM3* const da)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		const float gate = (float)b[i];
		const float sigmoid = 1.f / (1.f + expf(-beta * gate));
		da[i] = (NUM3)(scale * (float)g[i] * gate * sigmoid);
	}
}

template<typename NUM1, typename NUM2, typename NUM3>
static void _ccv_nnc_swish_mul_dvalue_launch(const size_t count, const float beta, const float scale, const ccv_nnc_tensor_t* const g, const ccv_nnc_tensor_t* const b, ccv_nnc_tensor_t* const da, cudaStream_t stream)
{
	_ccv_nnc_swish_mul_dvalue_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, beta, scale, (const NUM1*)g->data.u8, (const NUM2*)b->data.u8, (NUM3*)da->data.u8);
}

template<typename NUM1, typename NUM2>
static int _ccv_nnc_swish_mul_dvalue_output(const size_t count, const float beta, const float scale, const ccv_nnc_tensor_t* const g, const ccv_nnc_tensor_t* const b, ccv_nnc_tensor_t* const da, cudaStream_t stream)
{
	if (da->info.datatype == CCV_32F)
		_ccv_nnc_swish_mul_dvalue_launch<NUM1, NUM2, float>(count, beta, scale, g, b, da, stream);
	else if (da->info.datatype == CCV_16F)
		_ccv_nnc_swish_mul_dvalue_launch<NUM1, NUM2, __half>(count, beta, scale, g, b, da, stream);
	else if (da->info.datatype == CCV_16BF)
		_ccv_nnc_swish_mul_dvalue_launch<NUM1, NUM2, __nv_bfloat16>(count, beta, scale, g, b, da, stream);
	else
		return CCV_NNC_EXEC_INVALID;
	return CCV_NNC_EXEC_SUCCESS;
}

template<typename NUM1>
static int _ccv_nnc_swish_mul_dvalue_gate(const size_t count, const float beta, const float scale, const ccv_nnc_tensor_t* const g, const ccv_nnc_tensor_t* const b, ccv_nnc_tensor_t* const da, cudaStream_t stream)
{
	if (b->info.datatype == CCV_32F)
		return _ccv_nnc_swish_mul_dvalue_output<NUM1, float>(count, beta, scale, g, b, da, stream);
	if (b->info.datatype == CCV_16F)
		return _ccv_nnc_swish_mul_dvalue_output<NUM1, __half>(count, beta, scale, g, b, da, stream);
	if (b->info.datatype == CCV_16BF)
		return _ccv_nnc_swish_mul_dvalue_output<NUM1, __nv_bfloat16>(count, beta, scale, g, b, da, stream);
	return CCV_NNC_EXEC_INVALID;
}

static int _ccv_nnc_swish_mul_dvalue(const size_t count, const float beta, const float scale, const ccv_nnc_tensor_t* const g, const ccv_nnc_tensor_t* const b, ccv_nnc_tensor_t* const da, cudaStream_t stream)
{
	if (g->info.datatype == CCV_32F)
		return _ccv_nnc_swish_mul_dvalue_gate<float>(count, beta, scale, g, b, da, stream);
	if (g->info.datatype == CCV_16F)
		return _ccv_nnc_swish_mul_dvalue_gate<__half>(count, beta, scale, g, b, da, stream);
	if (g->info.datatype == CCV_16BF)
		return _ccv_nnc_swish_mul_dvalue_gate<__nv_bfloat16>(count, beta, scale, g, b, da, stream);
	return CCV_NNC_EXEC_INVALID;
}

template<typename NUM1, typename NUM2, typename NUM3>
__global__ void _ccv_nnc_swish_mul_dgate_kernel(const size_t count, const float beta, const float scale, const NUM1* const g, const NUM2* const a, const NUM3* const b, NUM3* const db)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		const float value = (float)a[i];
		const float gate = (float)b[i];
		const float sigmoid = 1.f / (1.f + expf(-beta * gate));
		const float swish_gradient = sigmoid + beta * gate * sigmoid * (1.f - sigmoid);
		db[i] = (NUM3)(scale * (float)g[i] * value * swish_gradient);
	}
}

template<typename NUM1, typename NUM2, typename NUM3>
static void _ccv_nnc_swish_mul_dgate_launch(const size_t count, const float beta, const float scale, const ccv_nnc_tensor_t* const g, const ccv_nnc_tensor_t* const a, const ccv_nnc_tensor_t* const b, ccv_nnc_tensor_t* const db, cudaStream_t stream)
{
	_ccv_nnc_swish_mul_dgate_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, beta, scale, (const NUM1*)g->data.u8, (const NUM2*)a->data.u8, (const NUM3*)b->data.u8, (NUM3*)db->data.u8);
}

template<typename NUM1, typename NUM2>
static int _ccv_nnc_swish_mul_dgate_gate(const size_t count, const float beta, const float scale, const ccv_nnc_tensor_t* const g, const ccv_nnc_tensor_t* const a, const ccv_nnc_tensor_t* const b, ccv_nnc_tensor_t* const db, cudaStream_t stream)
{
	if (b->info.datatype == CCV_32F)
		_ccv_nnc_swish_mul_dgate_launch<NUM1, NUM2, float>(count, beta, scale, g, a, b, db, stream);
	else if (b->info.datatype == CCV_16F)
		_ccv_nnc_swish_mul_dgate_launch<NUM1, NUM2, __half>(count, beta, scale, g, a, b, db, stream);
	else if (b->info.datatype == CCV_16BF)
		_ccv_nnc_swish_mul_dgate_launch<NUM1, NUM2, __nv_bfloat16>(count, beta, scale, g, a, b, db, stream);
	else
		return CCV_NNC_EXEC_INVALID;
	return CCV_NNC_EXEC_SUCCESS;
}

template<typename NUM1>
static int _ccv_nnc_swish_mul_dgate_value(const size_t count, const float beta, const float scale, const ccv_nnc_tensor_t* const g, const ccv_nnc_tensor_t* const a, const ccv_nnc_tensor_t* const b, ccv_nnc_tensor_t* const db, cudaStream_t stream)
{
	if (a->info.datatype == CCV_32F)
		return _ccv_nnc_swish_mul_dgate_gate<NUM1, float>(count, beta, scale, g, a, b, db, stream);
	if (a->info.datatype == CCV_16F)
		return _ccv_nnc_swish_mul_dgate_gate<NUM1, __half>(count, beta, scale, g, a, b, db, stream);
	if (a->info.datatype == CCV_16BF)
		return _ccv_nnc_swish_mul_dgate_gate<NUM1, __nv_bfloat16>(count, beta, scale, g, a, b, db, stream);
	return CCV_NNC_EXEC_INVALID;
}

static int _ccv_nnc_swish_mul_dgate(const size_t count, const float beta, const float scale, const ccv_nnc_tensor_t* const g, const ccv_nnc_tensor_t* const a, const ccv_nnc_tensor_t* const b, ccv_nnc_tensor_t* const db, cudaStream_t stream)
{
	if (g->info.datatype == CCV_32F)
		return _ccv_nnc_swish_mul_dgate_value<float>(count, beta, scale, g, a, b, db, stream);
	if (g->info.datatype == CCV_16F)
		return _ccv_nnc_swish_mul_dgate_value<__half>(count, beta, scale, g, a, b, db, stream);
	if (g->info.datatype == CCV_16BF)
		return _ccv_nnc_swish_mul_dgate_value<__nv_bfloat16>(count, beta, scale, g, a, b, db, stream);
	return CCV_NNC_EXEC_INVALID;
}

static int _ccv_nnc_swish_mul_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 3);
	assert(output_size >= 1);
	const ccv_nnc_tensor_t* const g = inputs[0]; // gradient
	const ccv_nnc_tensor_t* const a = inputs[1]; // value
	const ccv_nnc_tensor_t* const b = inputs[2]; // gate
	ccv_nnc_tensor_t* const da = outputs[0];
	ccv_nnc_tensor_t* const db = output_size > 1 ? outputs[1] : 0;
	if (!g || !b || (!da && !db) || (db && !a))
		return CCV_NNC_EXEC_INVALID;
	if (g->info.dim[0] == 0 || b->info.dim[0] == 0 ||
		(a && a->info.dim[0] == 0) ||
		(da && da->info.dim[0] == 0) ||
		(db && db->info.dim[0] == 0))
		return CCV_NNC_EXEC_INVALID;
	if (!CCV_IS_TENSOR_CONTIGUOUS(g) || !CCV_IS_TENSOR_CONTIGUOUS(b) ||
		(a && !CCV_IS_TENSOR_CONTIGUOUS(a)) ||
		(da && !CCV_IS_TENSOR_CONTIGUOUS(da)) ||
		(db && !CCV_IS_TENSOR_CONTIGUOUS(db)))
		return CCV_NNC_EXEC_INVALID;
	if (!_ccv_nnc_swish_mul_datatype_supported(g->info.datatype) ||
		!_ccv_nnc_swish_mul_datatype_supported(b->info.datatype) ||
		(a && !_ccv_nnc_swish_mul_datatype_supported(a->info.datatype)) ||
		(da && !_ccv_nnc_swish_mul_datatype_supported(da->info.datatype)) ||
		(db && !_ccv_nnc_swish_mul_datatype_supported(db->info.datatype)))
		return CCV_NNC_EXEC_INVALID;
	if ((a && da && da->info.datatype != a->info.datatype) ||
		(db && db->info.datatype != b->info.datatype))
		return CCV_NNC_EXEC_INVALID;
	const int g_nd = ccv_nnc_tensor_nd(g->info.dim);
	if (g_nd != ccv_nnc_tensor_nd(b->info.dim) ||
		(a && g_nd != ccv_nnc_tensor_nd(a->info.dim)) ||
		(da && g_nd != ccv_nnc_tensor_nd(da->info.dim)) ||
		(db && g_nd != ccv_nnc_tensor_nd(db->info.dim)))
		return CCV_NNC_EXEC_INVALID;
	int i;
	for (i = 0; i < g_nd; i++)
		if (g->info.dim[i] != b->info.dim[i] ||
			(a && g->info.dim[i] != a->info.dim[i]) ||
			(da && g->info.dim[i] != da->info.dim[i]) ||
			(db && g->info.dim[i] != db->info.dim[i]))
			return CCV_NNC_EXEC_INVALID;
	const size_t count = ccv_nnc_tensor_count(g->info);
	const float beta = cmd.info.swish_mul.beta;
	const float scale = cmd.info.swish_mul.scale;
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	if (da)
	{
		const int status = _ccv_nnc_swish_mul_dvalue(count, beta, scale, g, b, da, stream);
		if (status != CCV_NNC_EXEC_SUCCESS)
			return status;
	}
	if (db)
	{
		const int status = _ccv_nnc_swish_mul_dgate(count, beta, scale, g, a, b, db, stream);
		if (status != CCV_NNC_EXEC_SUCCESS)
			return status;
	}
	CUDA_ENFORCE(cudaGetLastError());
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SWISH_MUL_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_swish_mul_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SWISH_MUL_BACKWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_swish_mul_back;
}
