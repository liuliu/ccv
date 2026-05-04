#pragma once

#include "ln.h"

namespace layer_norm {

enum DataType {
    DATA_TYPE_FP16 = 0,
    DATA_TYPE_BF16 = 1,
    DATA_TYPE_FP32 = 2,
};

uint32_t round_hidden_size(uint32_t hidden_size);
bool run_layer_norm_fwd(LaunchParams<FwdParams>& launch_params, DataType wtype, DataType itype, DataType rtype, DataType otype, uint32_t hidden_size, bool configure_params);

} // namespace layer_norm
