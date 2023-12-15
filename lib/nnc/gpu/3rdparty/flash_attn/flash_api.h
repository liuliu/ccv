#pragma once

#include "src/flash.h"

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel);
int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits);
void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream, const bool configure);
