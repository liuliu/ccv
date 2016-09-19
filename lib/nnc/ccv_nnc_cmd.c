#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#ifdef HAVE_CUDA
#include "gpu/ccv_nnc_compat.h"
#endif
#include <time.h>
#include <sys/time.h>

#ifdef __MACH__
#include <mach/mach.h>
#include <mach/mach_time.h>
#endif

typedef void(*ccv_nnc_init_f)(ccv_nnc_cmd_api_t cmd_api[]);

typedef struct {
	int backend;
	const char* name;
	size_t name_size;
	ccv_nnc_init_f init;
} ccv_nnc_init_t;

#define CCV_NNC_INIT_DECL(init_func) extern void (init_func)(ccv_nnc_cmd_api_t cmd_api[])
#define CCV_NNC_INIT_MAP_BEGIN() static ccv_nnc_init_t init_map[] = {
#define CCV_NNC_INIT_MAP(_name, _init) { .backend = _name, .name = #_name, .name_size = sizeof(#_name), .init = _init, },
#define CCV_NNC_INIT_MAP_END() };

#define CCV_NNC_INIT_EXEC(name, init_func) do { \
		(init_func)(command_api_decls[name]); \
	} while (0)

// Below is extracted from source code: `./nnc-init.rb cpu` or `./nnc-init.rb cpu gpu`
#ifdef HAVE_CUDA // Include the header for both CPU and GPU
#include "ccv_nnc_init.inc"
#else // Otherwise only include for CPU.
#include "cpu/ccv_nnc_init.inc"
#endif
// Above should be automatic generated.

static ccv_nnc_compute_attr_t compute_attrs[CCV_NNC_COMPUTE_COUNT];

// We support up to 4 bit patterns.
#define CCV_NNC_ATTR_BIT_PATTERNS(_v1, _v2, _v3, _v4, _v5, _v6, _v7, _v8, ...) \
	{ \
		.input = (_v1), \
		.output = (_v2) \
	}, \
	{ \
		.input = (_v3), \
		.output = (_v4) \
	}, \
	{ \
		.input = (_v5), \
		.output = (_v6) \
	}, \
	{ \
		.input = (_v7), \
		.output = (_v8) \
	} \

#define CCV_NNC_ATTR_DEF_X(_cmd, _attrs, ...) { \
		const ccv_nnc_compute_attr_t attr = { \
			.attrs = _attrs, \
			.bit_patterns = { \
				CCV_NNC_ATTR_BIT_PATTERNS(__VA_ARGS__, 0, 0, 0, 0, 0, 0, 0, 0) \
			} \
		}; \
		compute_attrs[_cmd] = attr; \
	}

// Only allow even number of parameters.
#define CCV_NNC_ATTR_DEF_SEL(_0, _1, _2, _3, _4, _5, _6, _7, _8, _FX, ...) _FX

#define CCV_NNC_ATTR_DEF(_cmd, _attrs, ...) CCV_NNC_ATTR_DEF_SEL(CCV_NNC_ATTR_DEF_NOT_ALLOWED, ##__VA_ARGS__, CCV_NNC_ATTR_DEF_X, CCV_NNC_ATTR_DEF_NOT_ALLOWED, CCV_NNC_ATTR_DEF_X, CCV_NNC_ATTR_DEF_NOT_ALLOWED, CCV_NNC_ATTR_DEF_X, CCV_NNC_ATTR_DEF_NOT_ALLOWED, CCV_NNC_ATTR_DEF_X, CCV_NNC_ATTR_DEF_NOT_ALLOWED, CCV_NNC_ATTR_DEF_NOT_ALLOWED)(_cmd, _attrs, __VA_ARGS__)

static ccv_nnc_cmd_api_t cmd_api_decls[CCV_NNC_BACKEND_COUNT][CCV_NNC_COMPUTE_COUNT];

void ccv_nnc_init(void)
{
	int i;
	int count = sizeof(init_map) / sizeof(ccv_nnc_init_t);
	// Init dynamic dispatch table.
	for (i = 0; i < count; i++)
		init_map[i].init(cmd_api_decls[init_map[i].backend]);
#include "ccv_nnc_attr.inc"
}

int ccv_nnc_cmd_backend(const char* name)
{
	int i;
	int count = sizeof(init_map) / sizeof(ccv_nnc_init_t);
	// Do simple linear scan across the dynamic dispatch table.
	for (i = 0; i < count; i++)
		if (strncmp(init_map[i].name, name, init_map[i].name_size) == 0)
			return init_map[i].backend;
	return -1;
}

const ccv_nnc_cmd_param_t ccv_nnc_cmd_auto = {{{0}}};

int ccv_nnc_is_cmd_auto(const ccv_nnc_cmd_param_t params)
{
	return (memcmp(&params, &ccv_nnc_cmd_auto, sizeof(ccv_nnc_cmd_param_t)) == 0);
}

int ccv_nnc_cmd_is_forward(const ccv_nnc_cmd_t cmd)
{
	assert(cmd.compute >= 0);
	assert(cmd.compute < CCV_NNC_COMPUTE_COUNT);
	switch (cmd.compute)
	{
		case CCV_NNC_COMPUTE_CUSTOM:
		case CCV_NNC_COMPUTE_NOOP:
			return 0;
		default:
			return !(cmd.compute & 0x1); // If it is even, it is forward
	}
}

int ccv_nnc_cmd_is_backward(const ccv_nnc_cmd_t cmd)
{
	assert(cmd.compute >= 0);
	assert(cmd.compute < CCV_NNC_COMPUTE_COUNT);
	switch (cmd.compute)
	{
		case CCV_NNC_COMPUTE_CUSTOM:
		case CCV_NNC_COMPUTE_NOOP:
			return 0;
		default:
			return !!(cmd.compute & 0x1); // If it is odd, it is backward
	}
}

ccv_nnc_cmd_t ccv_nnc_cmd(const int compute, ccv_nnc_cmd_exec_f exec, const ccv_nnc_cmd_param_t params, const int flags)
{
	ccv_nnc_cmd_t cmd;
	cmd.info = params;
	// Default to CPU ref implementation if the type is CPU memory, otherwise use GPU ref.
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert((compute == CCV_NNC_COMPUTE_CUSTOM && exec) || (compute != CCV_NNC_COMPUTE_CUSTOM && !exec));
	cmd.compute = compute;
	cmd.algorithm = -1; // This is default.
	cmd.exec = exec;
	return cmd;
}

const ccv_nnc_hint_t ccv_nnc_no_hint = {{{0}}};

int ccv_nnc_is_no_hint(const ccv_nnc_hint_t hint)
{
	return (memcmp(&hint, &ccv_nnc_no_hint, sizeof(ccv_nnc_hint_t)) == 0);
}

int ccv_nnc_hint_verify(const ccv_nnc_hint_t hint, const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t a, const ccv_nnc_tensor_param_t b)
{
	int i;
	assert(a.format == b.format);
	const int hw = (a.format == CCV_TENSOR_FORMAT_CHWN || a.format == CCV_TENSOR_FORMAT_NHWC) ? 1 : 0;
	for (i = hw; i < CCV_NNC_MAX_DIM + hw; i++)
	{
		if ((hint.border.begin[i] + hint.border.end[i] + a.dim[i] - cmd.size.dim[i]) % hint.stride.dim[i] != 0)
			return -1;
		int expected = (hint.border.begin[i] + hint.border.end[i] + a.dim[i] - cmd.size.dim[i]) / hint.stride.dim[i] + 1;
		if (expected != b.dim[i])
			return -1;
	}
	return 0;
}

ccv_nnc_hint_t ccv_nnc_hint_auto(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t a, const ccv_nnc_tensor_param_t b)
{
	int i;
	assert(a.format == b.format);
	const int hw = (a.format == CCV_TENSOR_FORMAT_CHWN || a.format == CCV_TENSOR_FORMAT_NHWC) ? 1 : 0;
	for (i = hw; i < CCV_NNC_MAX_DIM + hw; i++)
		if (!a.dim[i] || !b.dim[i]) // If one of the dim is zero, we cannot auto the hint, return no hint.
			return ccv_nnc_no_hint;
	ccv_nnc_hint_t hint_auto = {
		.stride = {
			.dim = {0}
		},
		.border = {
			.begin = {0},
			.end = {0}
		}
	};
	// 0-dim is reserved for channels
	for (i = hw; i < CCV_NNC_MAX_DIM + hw; i++)
	{
		// This is guessed by having a stride that will approximately match the scale.
		int stride = (a.dim[i] + b.dim[i] / 2) / b.dim[i];
		hint_auto.stride.dim[i] = stride;
		int border = (b.dim[i] - 1) * stride - a.dim[i] + cmd.size.dim[i];
		hint_auto.border.begin[i] = (border + 1) / 2; // Always prefer to have more padding in the beginning, this matches CUDNN behavior.
		hint_auto.border.end[i] = border - hint_auto.border.begin[i];
	}
	return hint_auto;
}

static void _ccv_nnc_hint_tensor_dim_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_tensor_param_t a, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* b)
{
	int i;
	assert(a.format == b->format);
	const int hw = (a.format == CCV_TENSOR_FORMAT_CHWN || a.format == CCV_TENSOR_FORMAT_NHWC) ? 1 : 0;
	for (i = hw; i < CCV_NNC_MAX_DIM + hw; i++)
	{
		int stride = ccv_max(1, hint.stride.dim[i]);
		b->dim[i] = (a.dim[i] + hint.border.begin[i] + hint.border.end[i] - cmd.info.size.dim[i]) / stride + 1;
	}
}

static void _ccv_nnc_hint_tensor_dim_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_tensor_param_t a, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* b)
{
	int i;
	assert(a.format == b->format);
	const int hw = (a.format == CCV_TENSOR_FORMAT_CHWN || a.format == CCV_TENSOR_FORMAT_NHWC) ? 1 : 0;
	for (i = hw; i < CCV_NNC_MAX_DIM + hw; i++)
	{
		int stride = ccv_max(1, hint.stride.dim[i]);
		b->dim[i] = (a.dim[i] - 1) * stride - hint.border.begin[i] - hint.border.end[i] + cmd.info.size.dim[i];
	}
}

void ccv_nnc_hint_tensor_auto(const ccv_nnc_cmd_t cmd, const ccv_nnc_tensor_param_t* inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* outputs, const int output_size)
{
	// zero out the parameters
	const ccv_nnc_tensor_param_t z = {
		.type = 0,
		.format = 0,
		.dim = {0}
	};
	int i;
	for (i = 0; i < output_size; i++)
		outputs[i] = z; // Reset the outputs.
	switch (cmd.compute)
	{
		// For neural networks
		case CCV_NNC_COMPUTE_CONVOLUTION_FORWARD: {
			assert(output_size == 1);
			outputs[0].type = inputs[0].type;
			outputs[0].format = inputs[0].format;
			// Get the channel output from the weight matrix.
			int count = ccv_nnc_tensor_get_n(inputs[1]);
			assert(count == cmd.info.convolution.count);
			assert(count == inputs[2].dim[0]); // from the bias matrix.
			ccv_nnc_tensor_set_c(outputs, count);
			_ccv_nnc_hint_tensor_dim_forw(cmd, inputs[0], hint, outputs);
			break;
		}
		case CCV_NNC_COMPUTE_GEMM_FORWARD: {
			assert(output_size == 1);
			outputs[0].type = inputs[0].type;
			outputs[0].format = inputs[0].format;
			outputs[0].dim[1] = inputs[0].dim[1]; // batch size.
			outputs[0].dim[0] = inputs[1].dim[1]; // from the weight matrix.
			assert(inputs[1].dim[1] == cmd.info.blas.count);
			assert(inputs[1].dim[1] == inputs[2].dim[0]); // from the bias matrix.
			break;
		}
		case CCV_NNC_COMPUTE_MAX_POOL_FORWARD:
		case CCV_NNC_COMPUTE_AVERAGE_POOL_FORWARD: {
			assert(output_size == 1);
			outputs[0].type = inputs[0].type;
			outputs[0].format = inputs[0].format;
			// Get channels from the original input.
			int count = ccv_nnc_tensor_get_c(inputs[0]);
			ccv_nnc_tensor_set_c(outputs, count);
			_ccv_nnc_hint_tensor_dim_forw(cmd, inputs[0], hint, outputs);
			break;
		}
		case CCV_NNC_COMPUTE_SOFTMAX_FORWARD:
		case CCV_NNC_COMPUTE_BATCH_NORM_FORWARD:
		case CCV_NNC_COMPUTE_RELU_FORWARD:
		// BLAS
		case CCV_NNC_COMPUTE_AXPY_FORWARD:
		// Element-wise computation
		case CCV_NNC_COMPUTE_EWSUM_FORWARD:
		case CCV_NNC_COMPUTE_EWPROD_FORWARD:
		case CCV_NNC_COMPUTE_EWDIV_FORWARD:
		case CCV_NNC_COMPUTE_EWEXP_FORWARD:
		case CCV_NNC_COMPUTE_EWLOG_FORWARD: {
			assert(output_size == 1);
			// All above have 1 output, therefore, it just copy from the first input.
			outputs[0] = inputs[0];
			break;
		}
		// For neural networks
		case CCV_NNC_COMPUTE_CONVOLUTION_BACKWARD:
		case CCV_NNC_COMPUTE_GEMM_BACKWARD: {
			// For both cases, we just copy from inputs.
			assert(output_size < input_size);
			for (i = 0; i < output_size; i++)
				outputs[i] = inputs[i + 1];
			break;
		}
		case CCV_NNC_COMPUTE_MAX_POOL_BACKWARD:
		case CCV_NNC_COMPUTE_AVERAGE_POOL_BACKWARD: {
			assert(output_size == 1);
			outputs[0].type = inputs[0].type;
			outputs[0].format = inputs[0].format;
			// Get channels from the original input.
			int count = ccv_nnc_tensor_get_c(inputs[0]);
			ccv_nnc_tensor_set_c(outputs, count);
			_ccv_nnc_hint_tensor_dim_back(cmd, inputs[0], hint, outputs);
			break;
		}
		case CCV_NNC_COMPUTE_SOFTMAX_BACKWARD:
		case CCV_NNC_COMPUTE_BATCH_NORM_BACKWARD:
		case CCV_NNC_COMPUTE_RELU_BACKWARD:
		// BLAS
		case CCV_NNC_COMPUTE_AXPY_BACKWARD:
		// Element-wise computation
		case CCV_NNC_COMPUTE_EWSUM_BACKWARD:
		case CCV_NNC_COMPUTE_EWPROD_BACKWARD:
		case CCV_NNC_COMPUTE_EWDIV_BACKWARD:
		case CCV_NNC_COMPUTE_EWEXP_BACKWARD:
		case CCV_NNC_COMPUTE_EWLOG_BACKWARD: {
			assert(input_size == 1);
			// All above have 1 input, therefore, outputs just copy from the input.
			for (i = 0; i < output_size; i++)
				outputs[i] = inputs[0];
			break;
		}
		// Other transforms
		case CCV_NNC_COMPUTE_DATA_TRANSFER_FORWARD:
		case CCV_NNC_COMPUTE_DATA_TRANSFER_BACKWARD:
		case CCV_NNC_COMPUTE_FORMAT_TRANSFORM_FORWARD:
		case CCV_NNC_COMPUTE_FORMAT_TRANSFORM_BACKWARD: {
			assert(output_size == input_size);
			for (i = 0; i < input_size; i++)
				outputs[i] = inputs[i];
			break;
		}
	}
}

// This returns absolute time.
uint64_t ccv_nnc_cmd_mono_time(void)
{
#ifdef __MACH__
	return mach_absolute_time();
#else
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
#endif
}

#define AUTO_TUNE_TRIAL_SIZE (3)

ccv_nnc_cmd_t ccv_nnc_cmd_autotune(const ccv_nnc_cmd_t cmd, const size_t max_workspace_size, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context)
{
	// This is a custom compute kernel, no need to autotune.
	if (cmd.compute == CCV_NNC_COMPUTE_CUSTOM)
		return cmd;
	int i, j, k;
	// Go through all the backends that supports the same type of memory input / output tensors support.
	int tensor_memory = 0;
	for (i = 0; i < input_size; i++)
		tensor_memory |= inputs[i]->info.type;
	for (i = 0; i < output_size; i++)
		tensor_memory |= outputs[i]->info.type;
	// In this case, we cannot determine the type of the tensor, skip auto-tune.
	if (!tensor_memory)
		return cmd;
	// Otherwise, we are good to go.
	ccv_nnc_cmd_t tuned_cmd = cmd;
	int64_t best_measured = -1;
	// We need to have trial loop through all the data.
	for (k = 0; k < AUTO_TUNE_TRIAL_SIZE; k++)
	{
		for (i = 0; i < CCV_NNC_BACKEND_COUNT; i++)
		{
			// We have the exec kernel, and support all the tensor memory types.
			ccv_nnc_cmd_api_t api_decl = cmd_api_decls[i][cmd.compute];
			if (api_decl.exec &&
				(api_decl.tensor_memory & tensor_memory) == tensor_memory)
			{
				ccv_nnc_cmd_t candid_cmd = cmd;
				candid_cmd.backend = i;
				// If a given API exist an autotune function, use that to pick the top algorithm.
				if (api_decl.autotune)
				{
					// Assuming k == 0 is sufficient, and we can skip.
					if (k > 0)
						continue;
					candid_cmd.algorithm = api_decl.autotune(candid_cmd, max_workspace_size, hint, flags, inputs, input_size, outputs, output_size, stream_context);
					uint64_t elapsed = ccv_nnc_cmd_mono_time();
					// Ready to run.
					int status = ccv_nnc_cmd_exec(candid_cmd, hint, flags, inputs, input_size, outputs, output_size, stream_context);
					ccv_nnc_stream_context_wait(stream_context);
					elapsed = ccv_nnc_cmd_mono_time() - elapsed;
					if (status == CCV_NNC_EXEC_SUCCESS &&
						(best_measured == -1 || elapsed < best_measured))
					{
						best_measured = elapsed;
						tuned_cmd = candid_cmd;
					}
				} else {
					// Otherwise loop over the existing algorithms and pick the top one.
					for (j = 0; j < api_decl.algorithms; j++)
					{
						candid_cmd.algorithm = j;
						uint64_t elapsed = ccv_nnc_cmd_mono_time();
						// Ready to run.
						int status = ccv_nnc_cmd_exec(candid_cmd, hint, flags, inputs, input_size, outputs, output_size, stream_context);
						elapsed = ccv_nnc_cmd_mono_time() - elapsed;
						if (status == CCV_NNC_EXEC_SUCCESS &&
							(best_measured == -1 || elapsed < best_measured))
						{
							best_measured = elapsed;
							tuned_cmd = candid_cmd;
						}
					}
				}
			}
		}
	}
	return tuned_cmd;
}

int ccv_nnc_cmd_exec(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context)
{
	assert(cmd.backend < CCV_NNC_BACKEND_COUNT);
	assert(cmd.compute < CCV_NNC_COMPUTE_COUNT);
	// If it is no-op, return as if succeed already.
	if (cmd.compute == CCV_NNC_COMPUTE_NOOP)
		return 0;
	// If it is a custom command, just apply it directly.
	if (cmd.compute == CCV_NNC_COMPUTE_CUSTOM)
		return cmd.exec(cmd, hint, flags, inputs, input_size, outputs, output_size, stream_context);
	ccv_nnc_cmd_api_t api_decl = cmd_api_decls[cmd.backend][cmd.compute];
	if (!api_decl.exec)
		return CCV_NNC_EXEC_NO_KERNEL;
	int i, j;
	for (i = 0; i < input_size; i++)
	{
		assert(api_decl.tensor_formats & inputs[i]->info.format);
	}
	for (i = 0; i < output_size; i++)
	{
		assert(api_decl.tensor_formats & outputs[i]->info.format);
	}
	// bit_patterns is a c-array, we can size it out.
	// To verify if the input matches the suggested pattern.
	int find = 0;
	ccv_nnc_compute_attr_t compute_attr = compute_attrs[cmd.compute];
	for (i = 0; i < sizeof(compute_attr.bit_patterns) / sizeof(compute_attr.bit_patterns[0]) && !find; i++)
	{
		if (!compute_attr.bit_patterns[i].input && !compute_attr.bit_patterns[i].output)
			break;
		uint64_t v = compute_attr.bit_patterns[i].input;
		int expected_input_size;
		if (v > 0)
		{
			expected_input_size = 1;
			while (v >>= 1)
				++expected_input_size;
		} else
			expected_input_size = 0;
		v = compute_attr.bit_patterns[i].output;
		int expected_output_size;
		if (v > 0)
		{
			expected_output_size = 1;
			while (v >>= 1)
				++expected_output_size;
		} else
			expected_output_size = 0;
		if (expected_input_size <= input_size && expected_output_size <= output_size)
		{
			find = 1;
			v = compute_attr.bit_patterns[i].input;
			for (j = 0; j < expected_input_size && find; j++)
				if (((1u << j) & v) && !inputs[j]) // If this parameter is required but the tensor is empty, we cannot match this pattern, skip.
					find = 0;
			v = compute_attr.bit_patterns[i].output;
			for (j = 0; j < expected_output_size && find; j++)
				if (((1u << j) & v) && !outputs[j]) // If this parameter is required but the tensor is empty, we cannot match this pattern, skip.
					find = 0;
		}
	}
	if (!find)
		return CCV_NNC_EXEC_INVALID; // Return invalid input.
	// Everything is out, call the underlying implementation.
	return api_decl.exec(cmd, hint, flags, inputs, input_size, outputs, output_size, stream_context);
}

int ccv_nnc_cmd_attr(const ccv_nnc_cmd_t cmd, const int flags)
{
	assert(cmd.backend < CCV_NNC_BACKEND_COUNT);
	assert(cmd.compute < CCV_NNC_COMPUTE_COUNT);
	// If it is a custom command, just apply it directly.
	assert(cmd.compute != CCV_NNC_COMPUTE_CUSTOM);
	ccv_nnc_compute_attr_t compute_attr = compute_attrs[cmd.compute];
	return !!(compute_attr.attrs & flags);
}

struct ccv_nnc_stream_context_s {
	int type;
	// Left for implementation yet, the CPU support for stream context.
};

ccv_nnc_stream_context_t* ccv_nnc_stream_context_new(int type)
{
	ccv_nnc_stream_context_t* stream_context = (ccv_nnc_stream_context_t*)ccmalloc(sizeof(ccv_nnc_stream_context_t));
	stream_context->type = type;
#ifdef HAVE_CUDA
	if (CCV_STREAM_GET_CONTEXT(type) == CCV_STREAM_CONTEXT_GPU)
		stream_context = ccv_nnc_init_stream_context(stream_context);
#endif
	return stream_context;
}

void ccv_nnc_stream_context_wait(const ccv_nnc_stream_context_t* stream_context)
{
	if (!stream_context)
		return;
#ifdef HAVE_CUDA
	if (CCV_STREAM_GET_CONTEXT(stream_context->type) == CCV_STREAM_CONTEXT_GPU)
		ccv_nnc_synchronize_stream_context(stream_context);
#endif
}

void ccv_nnc_stream_context_free(ccv_nnc_stream_context_t* stream_context)
{
#ifdef HAVE_CUDA
	if (CCV_STREAM_GET_CONTEXT(stream_context->type) == CCV_STREAM_CONTEXT_GPU)
		ccv_nnc_deinit_stream_context(stream_context);
#endif
	ccfree(stream_context);
}
