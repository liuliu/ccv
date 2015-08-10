#include "ccv_nnc.h"

typedef void(*ccv_nnc_init_f)(ccv_nnc_api_t api[]);

typedef struct {
	int provide;
	ccv_nnc_init_f init;
} ccv_nnc_init_t;

#define CCV_NNC_INIT_DECL(init_func) extern void (init_func)(ccv_nnc_api_t api[])
#define CCV_NNC_INIT_MAP_BEGIN() static ccv_nnc_init_t init_map[] = {
#define CCV_NNC_INIT_MAP(name, init_func) { .provide = name, .init = init_func, },
#define CCV_NNC_INIT_MAP_END() };

#define CCV_NNC_INIT_EXEC(name, init_func) do { \
		(init_func)(api_decls[name]); \
	} while (0)

void ccv_nnc_cpu_ref_init(ccv_nnc_api_t api[])
{
}

void ccv_nnc_gpu_ref_init(ccv_nnc_api_t api[])
{
}

void ccv_nnc_gpu_cudnn_init(ccv_nnc_api_t api[])
{
}

// I should be able to automatically extract code below from source code.
enum {
	CCV_NNC_PROVIDE_CPU_REF,
	CCV_NNC_PROVIDE_GPU_REF,
	CCV_NNC_PROVIDE_GPU_CUDNN,
	CCV_NNC_PROVIDE_COUNT,
};

CCV_NNC_INIT_DECL(ccv_nnc_cpu_ref_init);
CCV_NNC_INIT_DECL(ccv_nnc_gpu_ref_init);
CCV_NNC_INIT_DECL(ccv_nnc_gpu_cudnn_init);

CCV_NNC_INIT_MAP_BEGIN()
CCV_NNC_INIT_MAP(CCV_NNC_PROVIDE_CPU_REF, ccv_nnc_cpu_ref_init)
CCV_NNC_INIT_MAP(CCV_NNC_PROVIDE_GPU_REF, ccv_nnc_gpu_ref_init)
CCV_NNC_INIT_MAP(CCV_NNC_PROVIDE_GPU_CUDNN, ccv_nnc_gpu_cudnn_init)
CCV_NNC_INIT_MAP_END()
// Above should be automatic generated.

static ccv_nnc_api_t api_decls[CCV_NNC_PROVIDE_COUNT][CCV_NNC_TYPE_COUNT];

void ccv_nnc_init(void)
{
	int i;
	int count = sizeof(init_map) / sizeof(ccv_nnc_init_t);
	// Init dynamic dispatch table.
	for (i = 0; i < count; i++)
		init_map[i].init(api_decls[init_map[i].provide]);
}

#define CCV_NNC_TENSOR_SIZE(params) (params.dim[0] * params.dim[1] * params.channels * 4)

ccv_nnc_tensor_t* ccv_nnc_tensor_new(const void* ptr, const ccv_nnc_tensor_param_t params, const int flags)
{
	ccv_nnc_tensor_t* tensor;
	if (ptr)
	{
		tensor = (ccv_nnc_tensor_t*)ccmalloc(sizeof(ccv_nnc_tensor_t));
		tensor->type = CCV_NO_DATA_ALLOC;
		tensor->meta = params;
		tensor->data.u8 = (uint8_t*)ptr;
		return tensor;
	}
	assert((flags & CCV_TENSOR_CPU_MEMORY) || (flags == 0));
	tensor = (ccv_nnc_tensor_t*)ccmalloc(sizeof(ccv_nnc_tensor_t) + CCV_NNC_TENSOR_SIZE(params));
	tensor->type = CCV_UNMANAGED;
	tensor->meta = params;
	return tensor;
}

void ccv_nnc_tensor_free(ccv_nnc_tensor_t* tensor)
{
	ccfree(tensor);
}

ccv_nnc_net_t* ccv_nnc_net_new(const void* ptr, const int type, const ccv_nnc_net_param_t params, const int flags)
{
	ccv_nnc_net_t* net;
	if (ptr)
	{
		net = (ccv_nnc_net_t*)ptr;
	} else {
		net = (ccv_nnc_net_t*)ccmalloc(sizeof(ccv_nnc_net_t));
	}
	net->meta = params;
	// TODO: auto-find a workable implementation.
	net->provide = CCV_NNC_PROVIDE_CPU_REF;
	net->type = type;
	return net;
}

void ccv_nnc_net_free(ccv_nnc_net_t* net)
{
	ccfree(net);
}

int ccv_nnc_net_hint_verify(const ccv_nnc_net_t* net, const ccv_nnc_net_hint_t hint, const ccv_nnc_tensor_param_t a, const ccv_nnc_tensor_param_t b)
{
	int i;
	for (i = 0; i < CCV_NNC_MAX_DIM; i++)
	{
		if ((hint.border.front[i] + hint.border.back[i] + a.dim[i] - net->meta.size.dim[i]) % hint.stride.dim[i] != 0)
			return -1;
		int expected = (hint.border.front[i] + hint.border.back[i] + a.dim[i] - net->meta.size.dim[i]) / hint.stride.dim[i] + 1;
		if (expected != b.dim[i])
			return -1;
	}
	return 0;
}

ccv_nnc_net_hint_t ccv_nnc_net_hint_guess(const ccv_nnc_net_t* net, const ccv_nnc_tensor_param_t a, const ccv_nnc_tensor_param_t b)
{
	ccv_nnc_net_hint_t guess;
	int i;
	for (i = 0; i < CCV_NNC_MAX_DIM; i++)
	{
		// This is guessed by having a stride that will approximately match the scale.
		int stride = (a.dim[i] + b.dim[i] / 2) / b.dim[i];
		guess.stride.dim[i] = stride;
		int border = (b.dim[i] - 1) * stride - a.dim[i] + net->meta.size.dim[i];
		guess.border.front[i] = border / 2;
		guess.border.back[i] = border - guess.border.front[i];
	}
	return guess;
}

void ccv_nnc_net_inference(const ccv_nnc_net_t* net, const ccv_nnc_net_hint_t hint, const ccv_nnc_tensor_t* a, ccv_nnc_tensor_t* b, const ccv_nnc_tensor_t* w, const ccv_nnc_tensor_t* bias)
{
	assert(net->provide < CCV_NNC_PROVIDE_COUNT);
	assert(net->type < CCV_NNC_TYPE_COUNT);
	ccv_nnc_api_t api_decl = api_decls[net->provide][net->type];
	assert(api_decl.tensor_formats & a->meta.format);
	assert(api_decl.tensor_formats & b->meta.format);
	api_decl.inference(net, hint, a, b, w, bias);
}

void ccv_nnc_net_backprop(const ccv_nnc_net_t* net, const ccv_nnc_net_hint_t hint, const ccv_nnc_tensor_t* a, ccv_nnc_tensor_t* b, const ccv_nnc_tensor_t* c, const ccv_nnc_tensor_t* d, ccv_nnc_tensor_t* w, ccv_nnc_tensor_t* bias)
{
	assert(net->provide < CCV_NNC_PROVIDE_COUNT);
	assert(net->type < CCV_NNC_TYPE_COUNT);
	ccv_nnc_api_t api_decl = api_decls[net->provide][net->type];
	assert(api_decl.tensor_formats & a->meta.format);
	assert(api_decl.tensor_formats & b->meta.format);
	api_decl.backprop(net, hint, a, b, c, d, w, bias);
}
