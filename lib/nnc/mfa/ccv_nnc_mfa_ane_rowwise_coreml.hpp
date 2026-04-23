#ifndef GUARD_ccv_nnc_mfa_ane_rowwise_coreml_hpp
#define GUARD_ccv_nnc_mfa_ane_rowwise_coreml_hpp

#include <stddef.h>
#include <stdint.h>

#include "ccv_nnc_mfa_defines.hpp"

typedef struct ccv_nnc_stream_context_s ccv_nnc_stream_context_t;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ccv_nnc_mfa_ane_rowwise_coreml_cache_s ccv_nnc_mfa_ane_rowwise_coreml_cache_t;
typedef struct ccv_nnc_mfa_ane_rowwise_coreml_program_s ccv_nnc_mfa_ane_rowwise_coreml_program_t;

ccv_nnc_mfa_ane_rowwise_coreml_cache_t* ccv_nnc_mfa_ane_rowwise_coreml_cache_create(mtl_device_t* device);
void ccv_nnc_mfa_ane_rowwise_coreml_cache_destroy(ccv_nnc_mfa_ane_rowwise_coreml_cache_t* cache);

int ccv_nnc_mfa_ane_rowwise_coreml_cache_ensure_scratch(
    ccv_nnc_mfa_ane_rowwise_coreml_cache_t* cache,
    uint32_t padded_M,
    uint32_t N,
    uint32_t K,
    char* error_out,
    size_t error_out_size);

mtl_buffer_t* ccv_nnc_mfa_ane_rowwise_coreml_cache_activation_surface_buffer(
    ccv_nnc_mfa_ane_rowwise_coreml_cache_t* cache);
mtl_buffer_t* ccv_nnc_mfa_ane_rowwise_coreml_cache_output_surface_buffer(
    ccv_nnc_mfa_ane_rowwise_coreml_cache_t* cache);
mtl_buffer_t* ccv_nnc_mfa_ane_rowwise_coreml_cache_activation_scales_buffer(
    ccv_nnc_mfa_ane_rowwise_coreml_cache_t* cache);

ccv_nnc_mfa_ane_rowwise_coreml_program_t* ccv_nnc_mfa_ane_rowwise_coreml_find_or_create_program(
    uint32_t padded_M,
    uint32_t N,
    uint32_t K,
    char* error_out,
    size_t error_out_size);
void ccv_nnc_mfa_ane_rowwise_coreml_program_release(
    ccv_nnc_mfa_ane_rowwise_coreml_program_t* program);

uint32_t ccv_nnc_mfa_ane_rowwise_coreml_program_M(
    const ccv_nnc_mfa_ane_rowwise_coreml_program_t* program);
uint32_t ccv_nnc_mfa_ane_rowwise_coreml_program_N(
    const ccv_nnc_mfa_ane_rowwise_coreml_program_t* program);
uint32_t ccv_nnc_mfa_ane_rowwise_coreml_program_K(
    const ccv_nnc_mfa_ane_rowwise_coreml_program_t* program);

int ccv_nnc_mfa_ane_rowwise_coreml_evaluate(
    ccv_nnc_mfa_ane_rowwise_coreml_cache_t* cache,
    const ccv_nnc_mfa_ane_rowwise_coreml_program_t* program,
    char* error_out,
    size_t error_out_size);

int ccv_nnc_mfa_ane_rowwise_coreml_append_weight_upload(
    ccv_nnc_mfa_ane_rowwise_coreml_cache_t* cache,
    mtl_command_batch_t* command_batch,
    mtl_buffer_t* weight,
    size_t weight_offset,
    size_t weight_bytes,
    char* error_out,
    size_t error_out_size);

int ccv_nnc_mfa_ane_rowwise_fast_fence_prepare(
    ccv_nnc_mfa_ane_rowwise_coreml_cache_t* cache,
    char* error_out,
    size_t error_out_size);
uint32_t ccv_nnc_mfa_ane_rowwise_fast_fence_next_value(
    ccv_nnc_mfa_ane_rowwise_coreml_cache_t* cache,
    uint32_t fence_index);
int ccv_nnc_mfa_ane_rowwise_fast_fence_append_update(
    ccv_nnc_mfa_ane_rowwise_coreml_cache_t* cache,
    mtl_command_batch_t* command_batch,
    uint32_t fence_index,
    uint32_t value,
    char* error_out,
    size_t error_out_size);
int ccv_nnc_mfa_ane_rowwise_fast_fence_encode_wait(
    ccv_nnc_mfa_ane_rowwise_coreml_cache_t* cache,
    mtl_compute_command_encoder_t* encoder,
    uint32_t fence_index,
    uint32_t value,
    char* error_out,
    size_t error_out_size);
int ccv_nnc_mfa_ane_rowwise_fast_fence_cpu_wait(
    ccv_nnc_mfa_ane_rowwise_coreml_cache_t* cache,
    uint32_t fence_index,
    uint32_t value,
    char* error_out,
    size_t error_out_size);
void ccv_nnc_mfa_ane_rowwise_fast_fence_cpu_update(
    ccv_nnc_mfa_ane_rowwise_coreml_cache_t* cache,
    uint32_t fence_index,
    uint32_t value);

int ccv_nnc_mfa_ane_rowwise_finish_command_batch_and_wait(
    ccv_nnc_stream_context_t* stream_context,
    mtl_command_batch_t* command_batch,
    int use_mps_wrapper,
    char* error_out,
    size_t error_out_size);
int ccv_nnc_mfa_ane_rowwise_finish_command_batch_async(
    ccv_nnc_stream_context_t* stream_context,
    mtl_command_batch_t* command_batch,
    char* error_out,
    size_t error_out_size);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
