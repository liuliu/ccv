#include "ln_api.h"

namespace layer_norm {

FwdRegistry FWD_FUNCS, PARALLEL_FWD_FUNCS;
BwdRegistry BWD_FUNCS, PARALLEL_BWD_FUNCS;

static uint64_t get_key(DataType wtype, DataType itype, DataType rtype, DataType otype, DataType ctype, uint32_t hidden_size)
{
    const uint64_t type_key = (uint64_t)wtype | ((uint64_t)itype << 2) | ((uint64_t)rtype << 4) | ((uint64_t)otype << 6) | ((uint64_t)ctype << 8);
    return (type_key << 32) | hidden_size;
}

uint32_t round_hidden_size(uint32_t hidden_size)
{
    const uint32_t multiple = hidden_size <= 1536 ? 256 : (hidden_size <= 3072 ? 512 : 1024);
    return (hidden_size + multiple - 1) / multiple * multiple;
}

bool run_layer_norm_fwd(LaunchParams<FwdParams>& launch_params, DataType wtype, DataType itype, DataType rtype, DataType otype, uint32_t hidden_size, bool configure_params)
{
    const uint32_t rounded_hidden_size = round_hidden_size(hidden_size);
    const uint64_t key = get_key(wtype, itype, rtype, otype, DATA_TYPE_FP32, rounded_hidden_size);
    const auto iter = FWD_FUNCS.find(key);
    if (iter == FWD_FUNCS.end())
        return false;
    iter->second(launch_params, configure_params);
    return true;
}

} // namespace layer_norm
