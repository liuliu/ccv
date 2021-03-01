#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_nnc_micro.h"

// MARK - Level-1 API

ccv_nnc_micro_io_t ccv_nnc_micro_input(void)
{
	return (ccv_nnc_micro_io_t){
		.type = CCV_NNC_MICRO_INPUT
	};
}

ccv_nnc_micro_io_t ccv_nnc_micro_reindex(const char* const* const shape, const char* const* const reindex, const int reindex_count, const ccv_nnc_micro_io_t x)
{
	return (ccv_nnc_micro_io_t){
		.type = CCV_NNC_MICRO_REINDEX
	};
}

ccv_nnc_micro_io_t ccv_nnc_micro_binary(const uint32_t op, const ccv_nnc_micro_io_t x, const ccv_nnc_micro_io_t y)
{
	return (ccv_nnc_micro_io_t){
		.type = CCV_NNC_MICRO_BINARY
	};
}

ccv_nnc_micro_io_t ccv_nnc_micro_reduce(const uint32_t op, const int* const axis, const int axis_count, const ccv_nnc_micro_io_t x)
{
	return (ccv_nnc_micro_io_t){
		.type = CCV_NNC_MICRO_REDUCE
	};
}

ccv_nnc_micro_io_t ccv_nnc_micro_select(const int* const axis, const int axis_count, const ccv_nnc_micro_io_t x, const ccv_nnc_micro_io_t index)
{
	return (ccv_nnc_micro_io_t){
		.type = CCV_NNC_MICRO_SELECT
	};
}

CCV_WARN_UNUSED(ccv_nnc_micro_combine_t*) ccv_nnc_micro_combine_new(const ccv_nnc_micro_io_t* const inputs, const int input_size, const char* const* const parameters, const int parameter_size, const ccv_nnc_micro_io_t* const outputs, const int output_size)
{
	return 0;
}

void ccv_nnc_micro_combine_free(ccv_nnc_micro_combine_t* const combine)
{
}

char* ccv_nnc_micro_combine_c(ccv_nnc_micro_combine_t* const combine)
{
	return 0;
}
