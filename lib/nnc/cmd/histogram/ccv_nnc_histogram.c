#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"

static int _ccv_nnc_histogram_forw_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// input, [bins]
	// output, [stats] (stats contains scalar in order of: min, max, sum, sum of squares).
	if (((input_bitmasks[0] & 1u) == 1u || (input_bitmasks[0] & 3u) == 3u) && (output_bitmasks[0] == 1u || output_bitmasks[0] == 3u))
		return 1;
	return 0;
}

static int _ccv_nnc_histogram_back_bitmask(const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	return 0;
}

static void _ccv_nnc_histogram_tensor_auto_forw(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	outputs[0] = inputs[0];
	outputs[0].datatype = CCV_32S;
	memset(outputs[0].dim, 0, sizeof(outputs[0].dim));
	switch (cmd.histogram.type)
	{
		case CCV_NNC_HISTOGRAM_BINS:
		{
			assert(input_size >= 2);
			outputs[0].dim[0] = ccv_nnc_tensor_count(inputs[1]) + 2;
			break;
		}
		case CCV_NNC_HISTOGRAM_EVEN:
		{
			outputs[0].dim[0] = cmd.histogram.bins + 3;
			break;
		}
		case CCV_NNC_HISTOGRAM_LOGARITHMIC:
		{
			const float log_base = 1.0 / logf(cmd.histogram.rate);
			const int upper_range = ceilf(logf(cmd.histogram.max / cmd.histogram.min) * log_base);
			outputs[0].dim[0] = upper_range * 2 + 2;
			break;
		}
	}
	if (output_size >= 2)
	{
		// These for stats, min, max, sum, sum of squares).
		outputs[1] = inputs[0];
		memset(outputs[1].dim, 0, sizeof(outputs[1].dim));
		outputs[1].dim[0] = 4;
	}
}

static void _ccv_nnc_histogram_tensor_auto_back(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	// Doesn't support.
}

REGISTER_COMMAND(CCV_NNC_HISTOGRAM_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_histogram_cpu_ref.c)
{
	registry->bitmask = _ccv_nnc_histogram_forw_bitmask;
	registry->tensor_auto = _ccv_nnc_histogram_tensor_auto_forw;
}

REGISTER_COMMAND(CCV_NNC_HISTOGRAM_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_histogram_cpu_ref.c)
{
	registry->bitmask = _ccv_nnc_histogram_back_bitmask;
	registry->tensor_auto = _ccv_nnc_histogram_tensor_auto_back;
}

//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_HISTOGRAM_FORWARD)
#define CMD_HISTOGRAM_EVEN(_bins, _min, _max) ccv_nnc_cmd(CCV_NNC_HISTOGRAM_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.histogram={.type=CCV_NNC_HISTOGRAM_EVEN,.bins=_bins,.min=_min,.max=_max}}), 0)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_HISTOGRAM_FORWARD)
#define CMD_HISTOGRAM_LOG_X_0() ccv_nnc_cmd(CCV_NNC_HISTOGRAM_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.histogram={.type=CCV_NNC_HISTOGRAM_LOGARITHMIC,.min=1e-12,.max=1e20,.rate=1.1}}), 0)
#define CMD_HISTOGRAM_LOG_X_F(...) ("This should not be used, you should have either 0 parameter or 3 parameters for CMD_HISTOGRAM_LOG")
#define CMD_HISTOGRAM_LOG_X_2(_min, _max, _rate) ccv_nnc_cmd(CCV_NNC_HISTOGRAM_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.histogram={.type=CCV_NNC_HISTOGRAM_LOGARITHMIC,.min=_min,.max=_max,.rate=_rate}}), 0)
#define CMD_HISTOGRAM_LOG_X_SEL(_0, _1, _2, _3, _FX, ...) _FX
#define CMD_HISTOGRAM_LOG(...) CMD_HISTOGRAM_LOG_X_SEL(CMD_HISTOGRAM_LOG_X_F, ##__VA_ARGS__, CMD_HISTOGRAM_LOG_X_3, CMD_HISTOGRAM_LOG_X_F, CMD_HISTOGRAM_LOG_X_F, CMD_HISTOGRAM_LOG_X_0)(__VA_ARGS__)
//@REGISTER_EASY_COMMAND_MACRO(CCV_NNC_HISTOGRAM_FORWARD)
#define CMD_HISTOGRAM_BINS() ccv_nnc_cmd(CCV_NNC_HISTOGRAM_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.histogram={.type=CCV_NNC_HISTOGRAM_BINS}}), 0)
