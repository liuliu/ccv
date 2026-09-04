#include "ccv_nnc.h"
#include "ccv_nnc_internal.h"
#include "3rdparty/khash/khash.h"
#include "ccv_nnc_easy.h"
#ifdef HAVE_CUDA
#include "gpu/ccv_nnc_compat.h"
#elif defined(HAVE_MPS)
#include "mps/ccv_nnc_mps.h"
#endif
#include <time.h>
#include <sys/time.h>

typedef struct {
	const uint32_t cmd;
	const char* name;
	ccv_nnc_cmd_registry_t registry;
	ccv_nnc_cmd_backend_registry_t backends[CCV_NNC_BACKEND_COUNT];
} ccv_nnc_cmd_init_t;

typedef struct {
	const uint32_t backend;
	const char* name;
} ccv_nnc_cmd_backend_init_t;

// The generated code configures command and its mapping.
#include "cmd/ccv_nnc_cmd.inc"

void ccv_nnc_init(void)
{
	_ccv_nnc_cmd_init();
}

static uint64_t _ccv_nnc_flags = 0;

uint64_t ccv_nnc_flags(void)
{
	return _ccv_nnc_flags;
}

void ccv_nnc_enable_flag(uint64_t flag)
{
	_ccv_nnc_flags |= flag;
}

void ccv_nnc_disable_flag(uint64_t flag)
{
	_ccv_nnc_flags &= ~flag;
}

const char* ccv_nnc_cmd_name(const uint32_t cmd)
{
	switch (cmd)
	{
		case CCV_NNC_NOOP:
			return "CCV_NNC_NOOP";
		case CCV_NNC_CUSTOM_FORWARD:
			return "CCV_NNC_CUSTOM_FORWARD";
		case CCV_NNC_CUSTOM_BACKWARD:
			return "CCV_NNC_CUSTOM_BACKWARD";
		case CCV_NNC_GRAPH_FORWARD:
			return "CCV_NNC_GRAPH_FORWARD";
		case CCV_NNC_GRAPH_BACKWARD:
			return "CCV_NNC_GRAPH_BACKWARD";
	}
	const int idx = _ccv_nnc_cmd_ph(cmd);
	assert(idx >= 0);
	assert(idx < sizeof(init_map) / sizeof(init_map[0]));
	return init_map[idx].name;
}

const char* ccv_nnc_cmd_backend_name(const uint32_t backend)
{
	if (backend == CCV_NNC_NO_BACKEND)
		return "CCV_NNC_NO_BACKEND";
	const int idx = _ccv_nnc_cmd_backend_ph(backend);
	assert(idx >= 0);
	assert(idx < CCV_NNC_BACKEND_COUNT);
	return backend_init_map[idx].name;
}

const ccv_nnc_cmd_param_t ccv_nnc_cmd_auto = {};

int ccv_nnc_is_cmd_auto(const ccv_nnc_cmd_param_t params)
{
	return (memcmp(&params, &ccv_nnc_cmd_auto, sizeof(ccv_nnc_cmd_param_t)) == 0);
}

int ccv_nnc_cmd_is_forward(const ccv_nnc_cmd_t cmd)
{
	switch (cmd.cmd)
	{
		case CCV_NNC_NOOP:
			return 0;
		case CCV_NNC_CUSTOM_FORWARD:
		case CCV_NNC_CUSTOM_BACKWARD:
		case CCV_NNC_GRAPH_FORWARD:
		case CCV_NNC_GRAPH_BACKWARD:
		default:
			return !(cmd.cmd & 0x1); // If it is even, it is forward
	}
}

int ccv_nnc_cmd_is_backward(const ccv_nnc_cmd_t cmd)
{
	switch (cmd.cmd)
	{
		case CCV_NNC_NOOP:
			return 0;
		case CCV_NNC_CUSTOM_FORWARD:
		case CCV_NNC_CUSTOM_BACKWARD:
		case CCV_NNC_GRAPH_FORWARD:
		case CCV_NNC_GRAPH_BACKWARD:
		default:
			return !!(cmd.cmd & 0x1); // If it is odd, it is backward
	}
}

int ccv_nnc_cmd_ok(const uint32_t cmd, const uint32_t backend)
{
	// If it is a custom command, a no op, or a graph op, there is no backend to check.
	if (cmd == CCV_NNC_NOOP ||
		cmd == CCV_NNC_GRAPH_FORWARD || cmd == CCV_NNC_GRAPH_BACKWARD ||
		cmd == CCV_NNC_CUSTOM_FORWARD || cmd == CCV_NNC_CUSTOM_BACKWARD)
		return 1;
	const int cmd_idx = _ccv_nnc_cmd_ph(cmd);
	const int backend_idx = _ccv_nnc_cmd_backend_ph(backend);
	assert(cmd_idx >= 0 && cmd_idx < sizeof(init_map) / sizeof(init_map[0]));
	assert(backend_idx >= 0 && backend_idx < CCV_NNC_BACKEND_COUNT);
	const ccv_nnc_cmd_backend_registry_t api_registry = init_map[cmd_idx].backends[backend_idx];
	// Check if the execution function exists or not.
	return !!api_registry.exec;
}

ccv_nnc_cmd_t ccv_nnc_cmd(const uint32_t _cmd, ccv_nnc_cmd_vtab_t* const isa, const ccv_nnc_cmd_param_t params, const int flags)
{
	ccv_nnc_cmd_t cmd;
	cmd.info = params;
	cmd.backend = CCV_NNC_NO_BACKEND;
	assert((_cmd == CCV_NNC_CUSTOM_FORWARD && isa) || (_cmd != CCV_NNC_CUSTOM_FORWARD && !isa));
	cmd.cmd = _cmd;
	cmd.algorithm = -1; // This is default.
	cmd.isa = isa;
	cmd.data = 0;
	return cmd;
}

const ccv_nnc_hint_t ccv_nnc_no_hint = {};

int ccv_nnc_is_no_hint(const ccv_nnc_hint_t hint)
{
	return (memcmp(&hint, &ccv_nnc_no_hint, sizeof(ccv_nnc_hint_t)) == 0);
}

int ccv_nnc_hint_verify(const ccv_nnc_hint_t hint, const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t a, const ccv_nnc_tensor_param_t b)
{
	int i;
	assert(a.format == b.format);
	const int nd = ccv_nnc_tensor_nd(a.dim);
	const int size_nd = ccv_max(2, ccv_nnc_tensor_nd(cmd.size.dim) - 1);
	assert(size_nd == 2 || size_nd == 3); // Support 3D convolution.
	assert(nd == size_nd + 1 || nd == size_nd + 2);
	int hw;
	if ((a.format == CCV_TENSOR_FORMAT_CHWN) ||
		(a.format == CCV_TENSOR_FORMAT_NHWC && nd == size_nd + 1))
		hw = 0;
	else if ((a.format == CCV_TENSOR_FORMAT_NHWC && nd == size_nd + 2) ||
			 (a.format == CCV_TENSOR_FORMAT_NCHW && nd == size_nd + 1))
		hw = 1;
	else if (a.format == CCV_TENSOR_FORMAT_NCHW && nd == size_nd + 2)
		hw = 2;
	else
		assert(0 && "unknown format");
	for (i = 0; i < size_nd; i++)
	{
		if ((hint.border.begin[i] + hint.border.end[i] + a.dim[i + hw] - cmd.size.dim[i]) % hint.stride.dim[i] != 0)
			return -1;
		int expected = (hint.border.begin[i] + hint.border.end[i] + a.dim[i + hw] - cmd.size.dim[i]) / hint.stride.dim[i] + 1;
		if (expected != b.dim[i + hw])
			return -1;
	}
	return 0;
}

ccv_nnc_hint_t ccv_nnc_hint_auto(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t a, const ccv_nnc_tensor_param_t b)
{
	int i;
	if (a.format != b.format)
		return ccv_nnc_no_hint;
	assert(a.format == b.format);
	const int a_nd = ccv_nnc_tensor_nd(a.dim);
	const int b_nd = ccv_nnc_tensor_nd(b.dim);
	const int size_nd = ccv_max(2, ccv_nnc_tensor_nd(cmd.size.dim) - 1);
	assert(size_nd == 2 || size_nd == 3); // Support 3D convolution.
	// Is not auto hint deducible dimensions.
	if (a_nd != b_nd || (a_nd != size_nd + 1 && a_nd != size_nd + 2))
		return ccv_nnc_no_hint;
	int hw;
	if ((a.format == CCV_TENSOR_FORMAT_CHWN) ||
		(a.format == CCV_TENSOR_FORMAT_NHWC && a_nd == size_nd + 1))
		hw = 0;
	else if ((a.format == CCV_TENSOR_FORMAT_NHWC && a_nd == size_nd + 2) ||
			 (a.format == CCV_TENSOR_FORMAT_NCHW && a_nd == size_nd + 1))
		hw = 1;
	else if (a.format == CCV_TENSOR_FORMAT_NCHW && a_nd == size_nd + 2)
		hw = 2;
	else
		assert(0 && "unknown format");
	ccv_nnc_hint_t hint_auto = {};
	// 0-dim is reserved for channels
	for (i = 0; i < size_nd; i++)
	{
		// Cannot have one of the dim is zero, we cannot auto the hint, return no hint.
		assert(a.dim[i + hw] && b.dim[i + hw]);
		// This is guessed by having a stride that will approximately match the scale.
		int stride = (a.dim[i + hw] + b.dim[i + hw] / 2) / b.dim[i + hw];
		hint_auto.stride.dim[i] = stride;
		int border = (b.dim[i + hw] - 1) * stride - a.dim[i + hw] + cmd.size.dim[i];
		hint_auto.border.begin[i] = (border + 1) / 2; // Always prefer to have more padding in the beginning, this matches CUDNN behavior.
		hint_auto.border.end[i] = border - hint_auto.border.begin[i];
	}
	return hint_auto;
}

void ccv_nnc_hint_tensor_auto_forward_from_inputs(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	int i;
	assert(output_size <= input_size);
	for (i = 0; i < output_size; i++)
		outputs[i] = inputs[i];
}

void ccv_nnc_hint_tensor_auto_backward_from_gradient(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	int i;
	for (i = 0; i < output_size; i++)
		outputs[i] = inputs[0];
}

void ccv_nnc_hint_tensor_auto_backward_from_inputs(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	int i;
	assert(output_size < input_size);
	for (i = 0; i < output_size; i++)
		outputs[i] = inputs[i + 1];
}

void ccv_nnc_hint_tensor_auto_backward_from_gradient_and_inputs(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	int i;
	outputs[0] = inputs[0];
	assert(output_size < input_size);
	for (i = 1; i < output_size; i++)
		outputs[i] = inputs[i + 1];
}

void ccv_nnc_hint_tensor_auto(const ccv_nnc_cmd_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	// zero out the parameters
	const ccv_nnc_tensor_param_t z = {};
	int i;
	for (i = 0; i < output_size; i++)
		outputs[i] = z; // Reset the outputs.
	// Cannot handle these situations.
	if (cmd.cmd == CCV_NNC_NOOP || cmd.cmd == CCV_NNC_CUSTOM_BACKWARD || cmd.cmd == CCV_NNC_GRAPH_FORWARD || cmd.cmd == CCV_NNC_GRAPH_BACKWARD)
		return;
	if (cmd.cmd == CCV_NNC_CUSTOM_FORWARD)
	{
		if (cmd.isa->tensor_auto)
			cmd.isa->tensor_auto(cmd, inputs, input_size, hint, outputs, output_size);
		return;
	}
	const int cmd_idx = _ccv_nnc_cmd_ph(cmd.cmd);
	const ccv_nnc_cmd_registry_t registry = init_map[cmd_idx].registry;
	if (registry.tensor_auto)
		registry.tensor_auto(cmd.info, inputs, input_size, hint, outputs, output_size);
	else if (ccv_nnc_cmd_is_forward(cmd)) // For forward, the default auto is forward_from_inputs
		ccv_nnc_hint_tensor_auto_forward_from_inputs(cmd.info, inputs, input_size, hint, outputs, output_size);
	else // For backward, the default auto is backward_from_inputs
		ccv_nnc_hint_tensor_auto_backward_from_inputs(cmd.info, inputs, input_size, hint, outputs, output_size);
}

int ccv_nnc_cmd_allow_inplace(const ccv_nnc_cmd_t cmd, const int input_idx, const int input_size, const int output_idx, const int output_size)
{
	if (cmd.cmd == CCV_NNC_NOOP || cmd.cmd == CCV_NNC_CUSTOM_FORWARD || cmd.cmd == CCV_NNC_CUSTOM_BACKWARD || cmd.cmd == CCV_NNC_GRAPH_FORWARD || cmd.cmd == CCV_NNC_GRAPH_BACKWARD)
		return 0;
	const int cmd_idx = _ccv_nnc_cmd_ph(cmd.cmd);
	const ccv_nnc_cmd_registry_t registry = init_map[cmd_idx].registry;
	if (registry.allow_inplace)
		return registry.allow_inplace(cmd.info, input_idx, input_size, output_idx, output_size);
	return 0;
}

int ccv_nnc_cmd_enforce_inplace(const ccv_nnc_cmd_t cmd, const int input_idx, const int input_size, const int output_idx, const int output_size)
{
	if (cmd.cmd == CCV_NNC_NOOP || cmd.cmd == CCV_NNC_CUSTOM_FORWARD || cmd.cmd == CCV_NNC_CUSTOM_BACKWARD || cmd.cmd == CCV_NNC_GRAPH_FORWARD || cmd.cmd == CCV_NNC_GRAPH_BACKWARD)
		return 0;
	const int cmd_idx = _ccv_nnc_cmd_ph(cmd.cmd);
	const ccv_nnc_cmd_registry_t registry = init_map[cmd_idx].registry;
	if (registry.enforce_inplace)
		return registry.enforce_inplace(cmd.info, input_idx, input_size, output_idx, output_size);
	return 0;
}

// This returns absolute time.
uint64_t ccv_nnc_cmd_mono_time(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

uint32_t ccv_nnc_cmd_find_backend(const ccv_nnc_cmd_t cmd, const int tensor_memory, const int tensor_formats, const int tensor_datatypes)
{
	if (cmd.cmd == CCV_NNC_NOOP ||
		cmd.cmd == CCV_NNC_GRAPH_FORWARD || cmd.cmd == CCV_NNC_GRAPH_BACKWARD ||
		cmd.cmd == CCV_NNC_CUSTOM_FORWARD || cmd.cmd == CCV_NNC_CUSTOM_BACKWARD)
		return cmd.backend;
	const int cmd_idx = _ccv_nnc_cmd_ph(cmd.cmd);
	assert(cmd_idx >= 0 && cmd_idx < sizeof(init_map) / sizeof(init_map[0]));
	assert(tensor_memory != 0 && tensor_formats != 0 && tensor_datatypes != 0);
	int i;
	for (i = 0; i < CCV_NNC_BACKEND_COUNT; i++)
	{
		const ccv_nnc_cmd_backend_registry_t api_registry = init_map[cmd_idx].backends[i];
		// We have the exec kernel, and support all the tensor memory types.
		if (api_registry.exec &&
			(api_registry.tensor_memory & tensor_memory) == tensor_memory &&
			(api_registry.tensor_formats & tensor_formats) == tensor_formats &&
			(api_registry.tensor_datatypes & tensor_datatypes) == tensor_datatypes)
			return backend_init_map[i].backend;
	}
	return cmd.backend;
}

#define AUTO_TUNE_TRIAL_SIZE (3)

static void _ccv_nnc_cmd_set_device_id(ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
#ifdef HAVE_CUDA
	if (!stream_context)
	{
		int device_id;
		if (ccv_nnc_device_ids_for_io(inputs, input_size, outputs, output_size, CCV_TENSOR_GPU_MEMORY, &device_id, 1) > 0)
			cudevice(device_id);
	}
#endif
}

typedef struct {
	int format;
	int datatype;
	int nd;
	off_t dataof;
	int dim[CCV_NNC_MAX_DIM_ALLOC];
	int stride[CCV_NNC_MAX_DIM_ALLOC];
} ccv_nnc_cmd_autotune_tensor_shape_t;

typedef struct {
	uint32_t cmd;
	ccv_nnc_cmd_param_t params;
	ccv_nnc_hint_t hint;
	int flags;
	int input_size;
	int output_size;
	size_t workspace_size;
	ccv_nnc_cmd_autotune_tensor_shape_t* inputs;
	ccv_nnc_cmd_autotune_tensor_shape_t* outputs;
} ccv_nnc_cmd_autotune_key_t;

static CCV_WARN_UNUSED(ccv_nnc_cmd_autotune_key_t) ccv_nnc_cmd_autotune_key_new(const ccv_nnc_cmd_t cmd, const size_t workspace_size, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size)
{
	ccv_nnc_cmd_autotune_key_t key = {
		.cmd = cmd.cmd,
		.params = cmd.info,
		.hint = hint,
		.workspace_size = workspace_size,
		.inputs = 0,
		.input_size = 0,
		.outputs = 0,
		.output_size = 0
	};
	if (input_size == 0 && output_size == 0)
		return key;
	assert(input_size >= 0 && output_size >= 0);
	key.input_size = input_size;
	key.output_size = output_size;
	key.inputs = (ccv_nnc_cmd_autotune_tensor_shape_t*)ccmalloc(sizeof(ccv_nnc_cmd_autotune_tensor_shape_t) * (input_size + output_size));
	key.outputs = key.inputs + input_size;
	int i, j;
	for (i = 0; i < input_size; i++)
	{
		memset(key.inputs[i].dim, 0, sizeof(key.inputs[i].dim));
		memset(key.inputs[i].stride, 0, sizeof(key.inputs[i].stride));
		if (!inputs[i])
		{
			key.inputs[i].format = 0;
			key.inputs[i].datatype = 0;
			key.inputs[i].dataof = 0;
			key.inputs[i].nd = 0;
			continue;
		}
		key.inputs[i].format = inputs[i]->info.format;
		key.inputs[i].datatype = inputs[i]->info.datatype;
		key.inputs[i].dataof = inputs[i]->dataof;
		const int nd = key.inputs[i].nd = ccv_nnc_tensor_nd(inputs[i]->info.dim);
		for (j = 0; j < nd; j++)
			key.inputs[i].dim[j] = inputs[i]->info.dim[j];
		if (CCV_IS_TENSOR_VIEW(inputs[i]))
			for (j = 0; j < nd; j++)
				key.inputs[i].stride[j] = ((ccv_nnc_tensor_view_t*)inputs[i])->stride[j];
	}
	for (i = 0; i < output_size; i++)
	{
		memset(key.outputs[i].dim, 0, sizeof(key.outputs[i].dim));
		memset(key.outputs[i].stride, 0, sizeof(key.outputs[i].stride));
		if (!outputs[i])
		{
			key.outputs[i].format = 0;
			key.outputs[i].datatype = 0;
			key.outputs[i].dataof = 0;
			key.outputs[i].nd = 0;
			continue;
		}
		key.outputs[i].format = outputs[i]->info.format;
		key.outputs[i].datatype = outputs[i]->info.datatype;
		key.outputs[i].dataof = outputs[i]->dataof;
		const int nd = key.outputs[i].nd = ccv_nnc_tensor_nd(outputs[i]->info.dim);
		for (j = 0; j < nd; j++)
			key.outputs[i].dim[j] = outputs[i]->info.dim[j];
		if (CCV_IS_TENSOR_VIEW(outputs[i]))
			for (j = 0; j < nd; j++)
				key.outputs[i].stride[j] = ((ccv_nnc_tensor_view_t*)outputs[i])->stride[j];
	}
	return key;
}

// autotune cache.
static inline uint32_t twang_32from64(uint64_t key)
{
	key = (~key) + (key << 18);
	key = key ^ (key >> 31);
	key = key * 21;
	key = key ^ (key >> 11);
	key = key + (key << 6);
	key = key ^ (key >> 22);
	return (uint32_t)(key);
}

static inline khint32_t _kh_autotune_key_executable_hash_func(const ccv_nnc_cmd_autotune_key_t key)
{
	uint32_t h = key.cmd;
	int i, j;
	uint32_t* data = (uint32_t*)&key.params;
	for (i = 0; i < sizeof(key.params) / sizeof(uint32_t); i++)
		h = twang_32from64(((uint64_t)h << 32) | data[i]);
	data = (uint32_t*)&key.hint;
	for (i = 0; i < sizeof(key.hint) / sizeof(uint32_t); i++)
		h = twang_32from64(((uint64_t)h << 32) | data[i]);
	h = twang_32from64(((uint64_t)h << 32) | key.workspace_size);
	h = twang_32from64(((uint64_t)h << 32) | key.input_size);
	h = twang_32from64(((uint64_t)h << 32) | key.output_size);
	for (i = 0; i < key.input_size; i++)
	{
		h = twang_32from64(((uint64_t)h << 32) | key.inputs[i].format);
		h = twang_32from64(((uint64_t)h << 32) | key.inputs[i].datatype);
		h = twang_32from64(((uint64_t)h << 32) | key.inputs[i].dataof);
		h = twang_32from64(((uint64_t)h << 32) | key.inputs[i].nd);
		for (j = 0; j < key.inputs[i].nd; j++)
		{
			h = twang_32from64(((uint64_t)h << 32) | key.inputs[i].dim[j]);
			h = twang_32from64(((uint64_t)h << 32) | key.inputs[i].stride[j]);
		}
	}
	for (i = 0; i < key.output_size; i++)
	{
		h = twang_32from64(((uint64_t)h << 32) | key.outputs[i].format);
		h = twang_32from64(((uint64_t)h << 32) | key.outputs[i].datatype);
		h = twang_32from64(((uint64_t)h << 32) | key.outputs[i].dataof);
		h = twang_32from64(((uint64_t)h << 32) | key.outputs[i].nd);
		for (j = 0; j < key.outputs[i].nd; j++)
		{
			h = twang_32from64(((uint64_t)h << 32) | key.outputs[i].dim[j]);
			h = twang_32from64(((uint64_t)h << 32) | key.outputs[i].stride[j]);
		}
	}
	return (khint32_t)h;
}

static inline int _kh_autotune_key_executable_hash_equal(const ccv_nnc_cmd_autotune_key_t a, const ccv_nnc_cmd_autotune_key_t b)
{
	if (a.cmd != b.cmd || a.flags != b.flags || a.workspace_size != b.workspace_size || a.input_size != b.input_size || a.output_size != b.output_size)
		return 0;
	if (memcmp(&a.params, &b.params, sizeof(a.params)) != 0)
		return 0;
	if (memcmp(&a.hint, &b.hint, sizeof(a.hint)) != 0)
		return 0;
	int i, j;
	for (i = 0; i < a.input_size; i++)
	{
		if (a.inputs[i].format != b.inputs[i].format || a.inputs[i].datatype != b.inputs[i].datatype || a.inputs[i].nd != b.inputs[i].nd || a.inputs[i].dataof != b.inputs[i].dataof)
			return 0;
		for (j = 0; j < a.inputs[i].nd; j++)
			if (a.inputs[i].dim[j] != b.inputs[i].dim[j] || a.inputs[i].stride[j] != b.inputs[i].stride[j])
				return 0;
	}
	for (i = 0; i < a.output_size; i++)
	{
		if (a.outputs[i].format != b.outputs[i].format || a.outputs[i].datatype != b.outputs[i].datatype || a.outputs[i].nd != b.outputs[i].nd || a.outputs[i].dataof != b.outputs[i].dataof)
			return 0;
		for (j = 0; j < a.outputs[i].nd; j++)
			if (a.outputs[i].dim[j] != b.outputs[i].dim[j] || a.outputs[i].stride[j] != b.outputs[i].stride[j])
				return 0;
	}
	return 1;
}

typedef struct {
	int backend;
	int algorithm;
} ccv_nnc_cmd_autotune_val_t;

KHASH_INIT(autotune_executable_cache, ccv_nnc_cmd_autotune_key_t, ccv_nnc_cmd_autotune_val_t, 1, _kh_autotune_key_executable_hash_func, _kh_autotune_key_executable_hash_equal)

static khash_t(autotune_executable_cache)* g_autotune_executable_cache = 0;

static inline void ccv_nnc_cmd_autotune_key_free(ccv_nnc_cmd_autotune_key_t key)
{
	if (key.inputs)
		ccfree(key.inputs);
}

void ccv_nnc_drain_autotune_cache(void)
{
	if (!g_autotune_executable_cache)
		return;
	khiter_t k;
	for (k = kh_begin(g_autotune_executable_cache); k < kh_end(g_autotune_executable_cache); k++)
	{
		if (!kh_exist(g_autotune_executable_cache, k))
			continue;
		ccv_nnc_cmd_autotune_key_free(kh_key(g_autotune_executable_cache, k));
		kh_del(autotune_executable_cache, g_autotune_executable_cache, k);
	}
}

ccv_nnc_cmd_t ccv_nnc_cmd_autotune(const ccv_nnc_cmd_t cmd, const size_t max_workspace_size, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	// This is a custom cmd kernel, no need to autotune.
	if (cmd.cmd == CCV_NNC_NOOP ||
		cmd.cmd == CCV_NNC_GRAPH_FORWARD || cmd.cmd == CCV_NNC_GRAPH_BACKWARD ||
		cmd.cmd == CCV_NNC_CUSTOM_FORWARD || cmd.cmd == CCV_NNC_CUSTOM_BACKWARD)
		return cmd;
	int i, j, k;
	// Go through all the backends that supports the same type of memory input / output tensors support.
	int tensor_memory = 0, tensor_formats = 0, tensor_datatypes = 0;
	for (i = 0; i < input_size; i++)
		if (inputs[i])
			tensor_memory |= CCV_TENSOR_GET_MEMORY(inputs[i]->info.type), tensor_formats |= inputs[i]->info.format, tensor_datatypes |= CCV_GET_DATA_TYPE(inputs[i]->info.datatype);
	for (i = 0; i < output_size; i++)
		if (outputs[i])
			tensor_memory |= CCV_TENSOR_GET_MEMORY(outputs[i]->info.type), tensor_formats |= outputs[i]->info.format, tensor_datatypes |= CCV_GET_DATA_TYPE(outputs[i]->info.datatype);
	// In this case, we cannot determine the type of the tensor, skip auto-tune.
	if (!tensor_memory)
		return cmd;
	// Otherwise, we are good to go.
	ccv_nnc_cmd_t tuned_cmd = cmd;
	if (!g_autotune_executable_cache)
		g_autotune_executable_cache = kh_init(autotune_executable_cache);
	int ret = 0;
	ccv_nnc_cmd_autotune_key_t key = ccv_nnc_cmd_autotune_key_new(cmd, max_workspace_size, hint, flags, inputs, input_size, outputs, output_size);
	khiter_t kiter = kh_put(autotune_executable_cache, g_autotune_executable_cache, key, &ret);
	if (ret == 0)
	{
		ccv_nnc_cmd_autotune_key_free(key);
		const ccv_nnc_cmd_autotune_val_t val = kh_val(g_autotune_executable_cache, kiter);
		tuned_cmd.backend = val.backend;
		tuned_cmd.algorithm = val.algorithm;
		return tuned_cmd;
	}
	int64_t best_measured = -1;
	const int cmd_idx = _ccv_nnc_cmd_ph(cmd.cmd);
	assert(cmd_idx >= 0 && cmd_idx < sizeof(init_map) / sizeof(init_map[0]));
	int flag = 0, autotune_available_1 = 0; // This is only applicable if we have only one backend.
	for (i = 0; i < CCV_NNC_BACKEND_COUNT; i++)
	{
		const ccv_nnc_cmd_backend_registry_t api_registry = init_map[cmd_idx].backends[i];
		// We have the exec kernel, and support all the tensor memory types.
		if (api_registry.exec &&
			(api_registry.tensor_memory & tensor_memory) == tensor_memory &&
			(api_registry.tensor_formats & tensor_formats) == tensor_formats &&
			(api_registry.tensor_datatypes & tensor_datatypes) == tensor_datatypes)
		{
			if (api_registry.autotune)
				autotune_available_1 = 1;
			if ((++flag) >= 2) // If we have more than 2 suitable backend, we can do this now.
				break;
		}
	}
	if (flag == 0)
		return cmd;
	_ccv_nnc_cmd_set_device_id(inputs, input_size, outputs, output_size, stream_context);
	// Allocate inputs / outputs and fill them in.
	ccv_nnc_tensor_t** copy_inputs;
	ccv_nnc_tensor_t** copy_outputs;
	ccv_nnc_tensor_t** allocated_inputs;
	ccv_nnc_tensor_t** allocated_outputs;
	ccv_nnc_tensor_view_t** allocated_input_views;
	ccv_nnc_tensor_view_t** allocated_output_views;
	if (flag > 1 || autotune_available_1)
	{
		copy_inputs = (ccv_nnc_tensor_t**)cccalloc((input_size + output_size) * 3, sizeof(ccv_nnc_tensor_t*));
		copy_outputs = copy_inputs + input_size;
		allocated_inputs = copy_outputs + output_size;
		allocated_outputs = allocated_inputs + input_size;
		allocated_input_views = (ccv_nnc_tensor_view_t**)(allocated_outputs + output_size);
		allocated_output_views = allocated_input_views + input_size;
		int stride[CCV_NNC_MAX_DIM_ALLOC];
		for (i = 0; i < output_size; i++)
			if (outputs[i])
			{
				for (j = 0; j < input_size; j++)
					if (inputs[j])
					{
						if (outputs[i] == inputs[j])
						{
							if (!copy_inputs[j])
							{
								allocated_inputs[j] = ccv_nnc_tensor_new(0, inputs[j]->info, 0);
								if (CCV_IS_TENSOR_VIEW(inputs[j]))
								{
									ccv_nnc_tensor_get_stride(inputs[j]->info.dim, stride);
									copy_inputs[j] = (ccv_nnc_tensor_t*)(allocated_input_views[j] = ccv_nnc_tensor_view_new(allocated_inputs[j], inputs[j]->info, DIM_ALLOC(), stride));
								} else
									copy_inputs[j] = allocated_inputs[j];
							}
							copy_outputs[i] = copy_inputs[j];
							break;
						} else if (outputs[i]->data.u8 == inputs[j]->data.u8 &&
							ccv_nnc_tensor_count(outputs[i]->info) == ccv_nnc_tensor_count(inputs[j]->info)) {
							if (!copy_inputs[j])
							{
								allocated_inputs[j] = ccv_nnc_tensor_new(0, inputs[j]->info, 0);
								if (CCV_IS_TENSOR_VIEW(inputs[j]))
								{
									ccv_nnc_tensor_get_stride(inputs[j]->info.dim, stride);
									copy_inputs[j] = (ccv_nnc_tensor_t*)(allocated_input_views[j] = ccv_nnc_tensor_view_new(allocated_inputs[j], inputs[j]->info, DIM_ALLOC(), stride));
								} else
									copy_inputs[j] = allocated_inputs[j];
							}
							allocated_outputs[i] = ccv_nnc_tensor_new(copy_inputs[j]->data.u8, outputs[i]->info, 0);
							if (CCV_IS_TENSOR_VIEW(outputs[i]))
							{
									ccv_nnc_tensor_get_stride(outputs[i]->info.dim, stride);
								copy_outputs[i] = (ccv_nnc_tensor_t*)(allocated_output_views[i] = ccv_nnc_tensor_view_new(allocated_outputs[i], outputs[i]->info, DIM_ALLOC(), stride));
							} else
								copy_outputs[i] = allocated_outputs[i];
							break;
						}
					}
				if (!copy_outputs[i])
				{
					allocated_outputs[i] = ccv_nnc_tensor_new(0, outputs[i]->info, 0);
					if (CCV_IS_TENSOR_VIEW(outputs[i]))
					{
						ccv_nnc_tensor_get_stride(outputs[i]->info.dim, stride);
						copy_outputs[i] = (ccv_nnc_tensor_t*)(allocated_output_views[i] = ccv_nnc_tensor_view_new(allocated_outputs[i], outputs[i]->info, DIM_ALLOC(), stride));
					} else
						copy_outputs[i] = allocated_outputs[i];
				}
			}
		for (i = 0; i < input_size; i++)
			if (inputs[i] && !copy_inputs[i])
				copy_inputs[i] = inputs[i];
	}
	if (flag == 1)
	{
		for (i = 0; i < CCV_NNC_BACKEND_COUNT; i++)
		{
			const ccv_nnc_cmd_backend_registry_t api_registry = init_map[cmd_idx].backends[i];
			// We have the exec kernel, and support all the tensor memory types.
			if (api_registry.exec &&
				(api_registry.tensor_memory & tensor_memory) == tensor_memory &&
				(api_registry.tensor_formats & tensor_formats) == tensor_formats &&
				(api_registry.tensor_datatypes & tensor_datatypes) == tensor_datatypes)
			{
				tuned_cmd.backend = backend_init_map[i].backend;
				// If a given API exist an autotune function, use that to pick the top algorithm.
				if (api_registry.autotune)
				{
					ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, inputs, input_size, copy_inputs, input_size, stream_context);
					_ccv_nnc_cmd_set_device_id(copy_inputs, input_size, copy_outputs, output_size, stream_context);
					tuned_cmd.algorithm = api_registry.autotune(tuned_cmd, max_workspace_size, hint, flags, copy_inputs, input_size, copy_outputs, output_size, stream_context);
					// Drain the context, autotune can use excessive amount of memory. Need to drain it now.
					ccv_nnc_stream_context_drain(stream_context);
				}
				break;
			}
		}
		if (autotune_available_1)
		{
			for (i = 0; i < input_size; i++)
			{
				if (allocated_inputs[i])
					ccv_nnc_tensor_free(allocated_inputs[i]);
				if (allocated_input_views[i])
					ccv_nnc_tensor_view_free(allocated_input_views[i]);
			}
			for (i = 0; i < output_size; i++)
			{
				if (allocated_outputs[i])
					ccv_nnc_tensor_free(allocated_outputs[i]);
				if (allocated_output_views[i])
					ccv_nnc_tensor_view_free(allocated_output_views[i]);
			}
			ccfree(copy_inputs);
		}
		const ccv_nnc_cmd_autotune_val_t val = {
			.backend = tuned_cmd.backend,
			.algorithm = tuned_cmd.algorithm
		};
		kh_val(g_autotune_executable_cache, kiter) = val;
		return tuned_cmd;
	}
	// We need to have trial loop through all the data.
	for (k = 0; k < AUTO_TUNE_TRIAL_SIZE; k++)
	{
		for (i = 0; i < CCV_NNC_BACKEND_COUNT; i++)
		{
			const ccv_nnc_cmd_backend_registry_t api_registry = init_map[cmd_idx].backends[i];
			// We have the exec kernel, and support all the tensor memory types.
			if (api_registry.exec &&
				(api_registry.tensor_memory & tensor_memory) == tensor_memory &&
				(api_registry.tensor_formats & tensor_formats) == tensor_formats &&
				(api_registry.tensor_datatypes & tensor_datatypes) == tensor_datatypes)
			{
				ccv_nnc_cmd_t candid_cmd = cmd;
				candid_cmd.backend = backend_init_map[i].backend;
				// If a given API exist an autotune function, use that to pick the top algorithm.
				if (api_registry.autotune)
				{
					// Assuming k == 0 is sufficient, and we can skip.
					if (k > 0)
						continue;
					ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, inputs, input_size, copy_inputs, input_size, stream_context);
					_ccv_nnc_cmd_set_device_id(copy_inputs, input_size, copy_outputs, output_size, stream_context);
					candid_cmd.algorithm = api_registry.autotune(candid_cmd, max_workspace_size, hint, flags, copy_inputs, input_size, copy_outputs, output_size, stream_context);
					// Drain the context, autotune can use excessive amount of memory. Need to drain it now.
					ccv_nnc_stream_context_drain(stream_context);
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
					for (j = 0; j < api_registry.algorithms; j++)
					{
						candid_cmd.algorithm = j;
						ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, inputs, input_size, copy_inputs, input_size, stream_context);
						_ccv_nnc_cmd_set_device_id(copy_inputs, input_size, copy_outputs, output_size, stream_context);
						uint64_t elapsed = ccv_nnc_cmd_mono_time();
						// Ready to run.
						int status = ccv_nnc_cmd_exec(candid_cmd, hint, flags, copy_inputs, input_size, copy_outputs, output_size, stream_context);
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
	for (i = 0; i < input_size; i++)
	{
		if (allocated_inputs[i])
			ccv_nnc_tensor_free(allocated_inputs[i]);
		if (allocated_input_views[i])
			ccv_nnc_tensor_view_free(allocated_input_views[i]);
	}
	for (i = 0; i < output_size; i++)
	{
		if (allocated_outputs[i])
			ccv_nnc_tensor_free(allocated_outputs[i]);
		if (allocated_output_views[i])
			ccv_nnc_tensor_view_free(allocated_output_views[i]);
	}
	ccfree(copy_inputs);
	const ccv_nnc_cmd_autotune_val_t val = {
		.backend = tuned_cmd.backend,
		.algorithm = tuned_cmd.algorithm
	};
	kh_val(g_autotune_executable_cache, kiter) = val;
	return tuned_cmd;
}

int ccv_nnc_cmd_bitmask(const ccv_nnc_cmd_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// If it is no-op, return true, it can deal with any number of parameters.
	if (cmd.cmd == CCV_NNC_NOOP)
		return 1;
	// If it is a custom command, I cannot check it at all, return false.
	if (cmd.cmd == CCV_NNC_CUSTOM_FORWARD || cmd.cmd == CCV_NNC_CUSTOM_BACKWARD)
		return 0;
	const int cmd_idx = _ccv_nnc_cmd_ph(cmd.cmd);
	const ccv_nnc_cmd_registry_t cmd_registry = init_map[cmd_idx].registry;
	if (cmd_registry.bitmask)
		return cmd_registry.bitmask(cmd.info, input_size, output_size, input_bitmasks, input_bitmask_size, output_bitmasks, output_bitmask_size);
	// If there is not checking, none can pass.
	return 0;
}

int ccv_nnc_device_ids_for_io(ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const int tensor_type, int* const device_ids, const int max_device_id_size)
{
	int i, j;
	int device_id_size = 0;
	if (max_device_id_size <= device_id_size)
		return device_id_size;
	// The device id of the exec is determined by its outputs.
	for (i = 0; i < output_size; i++)
		if (outputs[i] &&
			CCV_TENSOR_GET_MEMORY(outputs[i]->info.type) == tensor_type &&
			CCV_TENSOR_GET_DEVICE(outputs[i]->info.type) != CCV_COMPUTE_DEVICE_ANY)
		{
			const int device_id = CCV_TENSOR_GET_DEVICE_ID(outputs[i]->info.type);
			int flag = 0;
			for (j = 0; !flag && j < device_id_size; j++)
				flag = (device_ids[j] == device_id);
			if (flag)
				continue;
			device_ids[device_id_size++] = device_id;
			if (device_id_size >= max_device_id_size)
				return device_id_size;
		}
	if (device_id_size == 0)
	{
		int device_id = -1;
		for (i = 0; i < input_size; i++)
			if (inputs[i] &&
				CCV_TENSOR_GET_MEMORY(inputs[i]->info.type) == tensor_type &&
				CCV_TENSOR_GET_DEVICE(inputs[i]->info.type) != CCV_COMPUTE_DEVICE_ANY &&
				(device_id < 0 || CCV_TENSOR_GET_DEVICE_ID(inputs[i]->info.type) < device_id))
				device_id = CCV_TENSOR_GET_DEVICE_ID(inputs[i]->info.type);
		if (device_id >= 0)
		{
			device_ids[0] = device_id;
			return 1;
		}
	}
	return device_id_size;
}

void* ccv_nnc_cmd_aux(const ccv_nnc_cmd_t cmd)
{
	if (cmd.cmd == CCV_NNC_NOOP ||
		cmd.cmd == CCV_NNC_CUSTOM_FORWARD || cmd.cmd == CCV_NNC_CUSTOM_BACKWARD ||
		cmd.cmd == CCV_NNC_GRAPH_FORWARD || cmd.cmd == CCV_NNC_GRAPH_BACKWARD)
		return 0;
	assert(cmd.backend != CCV_NNC_NO_BACKEND);
	const int cmd_idx = _ccv_nnc_cmd_ph(cmd.cmd);
	assert(cmd_idx >= 0 && cmd_idx < sizeof(init_map) / sizeof(init_map[0]));
	const int backend_idx = _ccv_nnc_cmd_backend_ph(cmd.backend);
	assert(backend_idx >= 0 && backend_idx < CCV_NNC_BACKEND_COUNT);
	const ccv_nnc_cmd_backend_registry_t api_registry = init_map[cmd_idx].backends[backend_idx];
	return api_registry.aux;
}

int ccv_nnc_cmd_exec(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	// If it is no-op, return as if succeed already.
	if (cmd.cmd == CCV_NNC_NOOP)
		return 0;
	_ccv_nnc_cmd_set_device_id(inputs, input_size, outputs, output_size, stream_context);
	// If it is a custom command, just apply it directly.
	if (cmd.cmd == CCV_NNC_CUSTOM_FORWARD || cmd.cmd == CCV_NNC_CUSTOM_BACKWARD)
	{
		int ret = cmd.isa->exec(cmd, hint, flags, inputs, input_size, outputs, output_size, stream_context);
		if (!stream_context)
			ccv_nnc_stream_context_drain(stream_context);
		return ret;
	}
	assert(cmd.cmd != CCV_NNC_GRAPH_FORWARD && cmd.cmd != CCV_NNC_GRAPH_BACKWARD);
	const int cmd_idx = _ccv_nnc_cmd_ph(cmd.cmd);
	assert(cmd_idx >= 0 && cmd_idx < sizeof(init_map) / sizeof(init_map[0]));
	int i;
	uint32_t backend = cmd.backend;
	if (backend == CCV_NNC_NO_BACKEND)
	{
		// Find a suitable backend.
		int tensor_memory = 0, tensor_formats = 0, tensor_datatypes = 0;
		for (i = 0; i < input_size; i++)
			if (inputs[i])
				tensor_memory |= CCV_TENSOR_GET_MEMORY(inputs[i]->info.type), tensor_formats |= inputs[i]->info.format, tensor_datatypes |= CCV_GET_DATA_TYPE(inputs[i]->info.datatype);
		for (i = 0; i < output_size; i++)
			if (outputs[i])
				tensor_memory |= CCV_TENSOR_GET_MEMORY(outputs[i]->info.type), tensor_formats |= outputs[i]->info.format, tensor_datatypes |= CCV_GET_DATA_TYPE(outputs[i]->info.datatype);
		backend = ccv_nnc_cmd_find_backend(cmd, tensor_memory, tensor_formats, tensor_datatypes);
	}
	assert(backend != CCV_NNC_NO_BACKEND);
	const int backend_idx = _ccv_nnc_cmd_backend_ph(backend);
	assert(backend_idx >= 0 && backend_idx < CCV_NNC_BACKEND_COUNT);
	const ccv_nnc_cmd_backend_registry_t api_registry = init_map[cmd_idx].backends[backend_idx];
	if (!api_registry.exec)
		return CCV_NNC_EXEC_NO_KERNEL;
	// Everything is out, call the underlying implementation.
	int ret = api_registry.exec(cmd, hint, flags, inputs, input_size, outputs, output_size, stream_context);
	if (!stream_context)
		ccv_nnc_stream_context_drain(stream_context);
	return ret;
}

int ccv_nnc_cmd_attr(const ccv_nnc_cmd_t cmd, const int flags)
{
	// No additional attr for noop.
	if (cmd.cmd == CCV_NNC_NOOP ||
		// If it is a custom command, just apply it directly.
		cmd.cmd == CCV_NNC_CUSTOM_FORWARD || cmd.cmd == CCV_NNC_CUSTOM_BACKWARD ||
		// If it is sub-graph, there is no additional attr as well.
		cmd.cmd == CCV_NNC_GRAPH_FORWARD || cmd.cmd == CCV_NNC_GRAPH_BACKWARD)
		return 0;
	const int cmd_idx = _ccv_nnc_cmd_ph(cmd.cmd);
	assert(cmd_idx >= 0 && cmd_idx <sizeof(init_map) / sizeof(init_map[0]));
	const ccv_nnc_cmd_registry_t cmd_registry = init_map[cmd_idx].registry;
	return !!(cmd_registry.flags & flags);
}

void ccv_nnc_set_profiler(int state)
{
#ifdef HAVE_CUDA
	cusetprofiler(state);
#endif
}

int ccv_nnc_queue_watermark(void)
{
#ifdef HAVE_MPS
	return ccv_nnc_mps_queue_watermark();
#else
	return 0;
#endif
}

void ccv_nnc_set_queue_watermark(int watermark)
{
#ifdef HAVE_MPS
	// If we need to be memory efficient, we need to bound how many in-flight command buffers there are.
	ccv_nnc_mps_set_queue_watermark(watermark);
#endif
}

void ccv_nnc_set_whole_file_mapping_size_limit(const size_t size_limit)
{
#ifdef HAVE_MPS
	ccv_nnc_mps_set_whole_file_mapping_size_limit(size_limit);
#endif
}

void ccv_nnc_set_device_permutation(const int type, const int* const device_map, const int size)
{
	if (type != CCV_STREAM_CONTEXT_GPU)
		return;
#ifdef HAVE_CUDA
	cusetdevicemap(device_map, size);
#endif
}

void ccv_nnc_set_binary_artifacts(const char** const paths_to_read, const int paths_to_read_size, const char* const path_to_write)
{
#ifdef HAVE_MPS
	ccv_nnc_mps_set_binary_artifacts(paths_to_read, paths_to_read_size, path_to_write);
#endif
}
