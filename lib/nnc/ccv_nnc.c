#include "ccv_nnc.h"

#define CCV_NNC_INIT_DECL(init_func) extern void (init_func)(ccv_nnc_api_t api[])

#define CCV_NNC_INIT_EXEC(name, init_func) do { \
		(init_func)(api_decls[name]); \
	} while (0)

enum {
	CCV_NNC_PROVIDE_CPU_REF,
	CCV_NNC_PROVIDE_GPU_REF,
	CCV_NNC_PROVIDE_GPU_CUDNN,
	CCV_NNC_PROVIDE_COUNT,
};

CCV_NNC_INIT_DECL(ccv_nnc_cpu_ref_init);
CCV_NNC_INIT_DECL(ccv_nnc_gpu_ref_init);
CCV_NNC_INIT_DECL(ccv_nnc_gpu_cudnn_init);

static ccv_nnc_api_t api_decls[CCV_NNC_PROVIDE_COUNT][CCV_NNC_TYPE_COUNT];

void ccv_nnc_init(void)
{
	// Init dynamic dispatch table.
	CCV_NNC_INIT_EXEC(CCV_NNC_PROVIDE_CPU_REF, ccv_nnc_cpu_ref_init);
	CCV_NNC_INIT_EXEC(CCV_NNC_PROVIDE_GPU_REF, ccv_nnc_gpu_ref_init);
	CCV_NNC_INIT_EXEC(CCV_NNC_PROVIDE_GPU_CUDNN, ccv_nnc_gpu_cudnn_init);
}

#define CCV_NNC_TENSOR_SIZE(params) (params.rows * params.cols * params.channels * 4)

ccv_nnc_tensor_t* ccv_nnc_tensor_new(const void* ptr, ccv_nnc_tensor_param_t params, int flags)
{
	ccv_nnc_tensor_t* tensor;
	if (ptr)
	{
		tensor = (ccv_nnc_tensor_t*)ccmalloc(sizeof(ccv_nnc_tensor_t));
		tensor->type = CCV_NO_DATA_ALLOC;
		tensor->params = params;
		tensor->data.u8 = (uint8_t*)ptr;
		return tensor;
	}
	assert((flags & CCV_TENSOR_CPU_MEMORY) || (flags == 0));
	tensor = (ccv_nnc_tensor_t*)ccmalloc(sizeof(ccv_nnc_tensor_t) + CCV_NNC_TENSOR_SIZE(params));
	tensor->type = CCV_UNMANAGED;
	tensor->params = params;
	return tensor;
}

void ccv_nnc_tensor_free(ccv_nnc_tensor_t* tensor)
{
	ccfree(tensor);
}

ccv_nnc_net_t* ccv_nnc_net_new(const void* ptr, ccv_nnc_net_param_t params, int flags)
{
	return 0;
}

void ccv_nnc_net_free(ccv_nnc_net_t* net)
{
	ccfree(net);
}

void ccv_nnc_net_inference(ccv_nnc_net_t* net, ccv_nnc_tensor_t* a, ccv_nnc_tensor_t* b)
{
	assert(net->provider < CCV_NNC_PROVIDE_COUNT);
	assert(net->type < CCV_NNC_TYPE_COUNT);
	ccv_nnc_api_t api_decl = api_decls[net->provider][net->type];
	api_decl.inference(net, a, b);
}

void ccv_nnc_net_backprop(ccv_nnc_net_t* net, ccv_nnc_tensor_t* a, ccv_nnc_tensor_t* b, ccv_nnc_tensor_t* c, ccv_nnc_tensor_t* d, ccv_nnc_tensor_t* w, ccv_nnc_tensor_t* bias)
{
	assert(net->provider < CCV_NNC_PROVIDE_COUNT);
	assert(net->type < CCV_NNC_TYPE_COUNT);
	ccv_nnc_api_t api_decl = api_decls[net->provider][net->type];
	api_decl.backprop(net, a, b, c, d, w, bias);
}
