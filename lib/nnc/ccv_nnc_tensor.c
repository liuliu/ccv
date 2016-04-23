#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#ifdef HAVE_CUDA
#include "gpu/ccv_nnc_cuda.h"
#endif

ccv_nnc_tensor_t* ccv_nnc_tensor_new(const void* ptr, const ccv_nnc_tensor_param_t params, const int flags)
{
	ccv_nnc_tensor_t* tensor;
	// this specific form can be toll-free bridging to ccv_dense_matrix_t (On CPU, and 3 dims (channels, rows, cols), and channels is smaller than max channels of ccv_dense_matrix_t).
	int tfb = (params.type == CCV_TENSOR_CPU_MEMORY && params.dim[0] > 0 && params.dim[0] <= CCV_MAX_CHANNEL && params.dim[1] > 0 && params.dim[2] > 0 && params.dim[3] == 0);
	if (ptr)
	{
		tensor = (ccv_nnc_tensor_t*)ccmalloc(sizeof(ccv_nnc_tensor_t));
		tensor->sig = 0;
		tensor->refcount = 1;
		tensor->info = params;
		if (tfb)
		{
			tensor->type = CCV_NO_DATA_ALLOC | CCV_MATRIX_DENSE | CCV_32F | params.dim[0];
			// This corresponding to mat->step
			tensor->info.dim[4] = CCV_GET_STEP(params.dim[1], (CCV_32F | params.dim[0]));
		} else // This won't be recognized by ccv_dense_matrix_t
			tensor->type = CCV_NO_DATA_ALLOC | CCV_MATRIX_DENSE | CCV_32F;
		tensor->data.u8 = (uint8_t*)ptr;
		return tensor;
	}
	if (flags & CCV_TENSOR_CPU_MEMORY)
	{
		assert(CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_CPU_MEMORY);
	} else if (flags & CCV_TENSOR_GPU_MEMORY) {
		assert(CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_GPU_MEMORY);
	}
	size_t size = CCV_GET_DATA_TYPE_SIZE(CCV_32F); // Assuming 32-bit float point layout
	int i;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC; i++)
	{
		if (!params.dim[i])
			break;
		size *= params.dim[i];
	}
#ifdef HAVE_CUDA
	if (CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_GPU_MEMORY)
	{
		tensor = (ccv_nnc_tensor_t*)ccmalloc(sizeof(ccv_nnc_tensor_t));
		tensor->data.u8 = (uint8_t*)gcmalloc(size);
	} else {
		assert(CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_CPU_MEMORY);
		ccmemalign((void **)&tensor, 16, sizeof(ccv_nnc_tensor_t) + size);
		tensor->data.u8 = (uint8_t*)(tensor + 1);
	}
#else
	assert(CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_CPU_MEMORY);
	ccmemalign((void **)&tensor, 16, sizeof(ccv_nnc_tensor_t) + size);
	tensor->data.u8 = (uint8_t*)(tensor + 1);
#endif
	tensor->sig = 0;
	tensor->refcount = 1;
	tensor->info = params;
	if (tfb)
	{
		tensor->type = CCV_UNMANAGED | CCV_MATRIX_DENSE | CCV_32F | params.dim[0];
		// This corresponding to mat->step
		tensor->info.dim[4] = CCV_GET_STEP(params.dim[1], (CCV_32F | params.dim[0]));
	} else
		tensor->type = CCV_UNMANAGED | CCV_MATRIX_DENSE | CCV_32F;
	return tensor;
}

ccv_nnc_tensor_t ccv_nnc_tensor(const void* ptr, const ccv_nnc_tensor_param_t params, const int flags)
{
	// this specific form can be toll-free bridging to ccv_dense_matrix_t
	int tfb = (CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_CPU_MEMORY && params.dim[0] > 0 && params.dim[0] <= CCV_MAX_CHANNEL && params.dim[1] > 0 && params.dim[2] > 0 && params.dim[3] == 0);
	assert(ptr);
	ccv_nnc_tensor_t tensor;
	tensor.sig = 0;
	tensor.refcount = 1;
	tensor.info = params;
	if (flags & CCV_TENSOR_CPU_MEMORY)
	{
		assert(CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_CPU_MEMORY);
	} else if (flags & CCV_TENSOR_GPU_MEMORY) {
		assert(CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_GPU_MEMORY);
	}
	if (tfb)
	{
		tensor.type = CCV_NO_DATA_ALLOC | CCV_UNMANAGED | CCV_MATRIX_DENSE | CCV_32F | params.dim[0];
		// This corresponding to mat->step
		tensor.info.dim[4] = CCV_GET_STEP(params.dim[1], (CCV_32F | params.dim[0]));
	} else // This won't be recognized by ccv_dense_matrix_t
		tensor.type = CCV_NO_DATA_ALLOC | CCV_UNMANAGED | CCV_MATRIX_DENSE | CCV_32F;
	tensor.data.u8 = (uint8_t*)ptr;
	return tensor;
}

void ccv_nnc_tensor_free(ccv_nnc_tensor_t* tensor)
{
#ifdef HAVE_CUDA
	if (CCV_TENSOR_GET_MEMORY(tensor->info.type) == CCV_TENSOR_GPU_MEMORY)
		gcfree(tensor->data.u8);
#endif
	ccfree(tensor);
}

static inline void _ccv_nnc_tensor_view_set(ccv_nnc_tensor_view_t* tv, const ccv_nnc_tensor_t* tensor, const int ofs[CCV_NNC_MAX_DIM_ALLOC], const int dim[CCV_NNC_MAX_DIM_ALLOC])
{
	memcpy(tv->inc, tensor->info.dim, sizeof(float) * CCV_NNC_MAX_DIM_ALLOC);
	memcpy(tv->info.dim, dim, sizeof(float) * CCV_NNC_MAX_DIM_ALLOC);
	int i, inc = 1;
	float* p = tensor->data.f32;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && tv->info.dim[i] > 0; i++)
	{
		p += ofs[i] * inc;
		inc *= tv->inc[i];
	}
	tv->data.f32 = p;
}

ccv_nnc_tensor_view_t* ccv_nnc_tensor_view_new(const ccv_nnc_tensor_t* tensor, const int ofs[CCV_NNC_MAX_DIM_ALLOC], const int dim[CCV_NNC_MAX_DIM_ALLOC])
{
	ccv_nnc_tensor_view_t* tv = (ccv_nnc_tensor_view_t*)ccmalloc(sizeof(ccv_nnc_tensor_view_t));
	tv->type = (tensor->type & ~0xfff) | CCV_TENSOR_VIEW;
	tv->refcount = 1;
	tv->sig = 0;
	tv->info = tensor->info;
	_ccv_nnc_tensor_view_set(tv, tensor, ofs, dim);
	return tv;
}

ccv_nnc_tensor_view_t ccv_nnc_tensor_view(const ccv_nnc_tensor_t* tensor, const int ofs[CCV_NNC_MAX_DIM_ALLOC], const int dim[CCV_NNC_MAX_DIM_ALLOC])
{
	assert(!CCV_IS_TENSOR_VIEW(tensor));
	ccv_nnc_tensor_view_t tv = {
		.type = (tensor->type & ~0xfff) | CCV_TENSOR_VIEW, // clean up the channel bits, and then add CCV_TENSOR_VIEW identifier
		.refcount = 1,
		.sig = 0,
		.info = tensor->info,
	};
	_ccv_nnc_tensor_view_set(&tv, tensor, ofs, dim);
	return tv;
}

void ccv_nnc_tensor_view_free(ccv_nnc_tensor_view_t* tensor_view)
{
	ccfree(tensor_view);
}

void ccv_nnc_tensor_zero(void* tensor)
{
	ccv_nnc_tensor_view_t* tv = (ccv_nnc_tensor_view_t*)tensor;
	const int* tvinc = CCV_IS_TENSOR_VIEW(tv) ? tv->inc : tv->info.dim;
	// reset it to 0.
	int c, i[3];
	int count = 1;
	int mod[CCV_NNC_MAX_DIM_ALLOC - 3];
	int mod_inc[CCV_NNC_MAX_DIM_ALLOC - 2];
	mod_inc[0] = tvinc[0] * tvinc[1] * tvinc[2];
	int dim_count = 0;
	for (c = 3; c < CCV_NNC_MAX_DIM_ALLOC && tv->info.dim[c] > 0; c++)
	{
		// Compute the mod.
		mod[c - 3] = c == 3 ? tv->info.dim[c] : mod[c - 4] * tv->info.dim[c];
		mod_inc[c - 2] = mod_inc[c - 3] * tvinc[c];
		count *= tv->info.dim[c];
		dim_count = c - 2; // Keep track of the top of the dim.
	}
	for (c = dim_count - 1; c > 0; c--)
		mod_inc[c] = mod_inc[c - 1] * (tvinc[c + 3] - tv->info.dim[c + 3]);
	float* tvdf32 = tv->data.f32;
	for (c = 0; c < count; c++)
	{
		for (i[2] = 0; i[2] < ccv_max(1, tv->info.dim[2]); i[2]++)
		{
			float* tvp = tvdf32 + i[2] * tvinc[1] * tvinc[0];
			for (i[1] = 0; i[1] < ccv_max(1, tv->info.dim[1]); i[1]++)
			{
				memset(tvp, 0, sizeof(float) * tv->info.dim[0]);
				tvp += tvinc[0];
			}
		}
		int j;
		tvdf32 += mod_inc[0];
		for (j = 0; j < dim_count - 1; j++)
			if ((c + 1) % mod[j] != 0)
				break; // cannot be mod, break out.
			else
				tvdf32 += mod_inc[j + 1];
	}
}

int ccv_nnc_tensor_eq(const ccv_nnc_tensor_t* a, const ccv_nnc_tensor_t* b)
{
	assert(!CCV_IS_TENSOR_VIEW(a));
	assert(!CCV_IS_TENSOR_VIEW(b));
	// If a is a dense matrix, just use ccv_matrix_eq
	if (CCV_TENSOR_IS_DENSE_MATRIX(a->type))
		return ccv_matrix_eq((ccv_matrix_t*)a, (ccv_matrix_t*)b);
	// Otherwise, do our own thing.
	if (CCV_GET_DATA_TYPE(a->type) != CCV_GET_DATA_TYPE(b->type))
		return -1;
	// Only support 32F at this point.
	assert(CCV_GET_DATA_TYPE(a->type) == CCV_32F);
	int i, c = 1;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC; i++)
	{
		if (!a->info.dim[i] && !b->info.dim[i])
			break;
		if (a->info.dim[i] != b->info.dim[i])
			return -1;
		c *= a->info.dim[i];
	}
	// Read: http://www.cygnus-software.com/papers/comparingfloats/comparingfloats.htm
	// http://floating-point-gui.de/errors/comparison/
	static const float epsi = FLT_EPSILON;
	static const int32_t ulps = 128; // so that for 1 and 1.000015 will be treated as the same.
	for (i = 0; i < c; i++)
	{
		// Although this is float point, I use integer as a way to compare.
		int32_t i32a = a->data.i32[i];
		if (i32a < 0)
			i32a = 0x80000000 - i32a;
		int32_t i32b = b->data.i32[i];
		if (i32b < 0)
			i32b = 0x80000000 - i32b;
		if (abs(i32a - i32b) > ulps && fabsf(a->data.f32[i] - b->data.f32[i]) > epsi)
			return -1;
	}
	return 0;
}
