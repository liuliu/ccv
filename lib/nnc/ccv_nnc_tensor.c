#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#ifdef HAVE_CUDA
#include "gpu/ccv_nnc_compat.h"
#endif

#pragma mark - Level-1 API

const int ccv_nnc_no_ofs[CCV_NNC_MAX_DIM_ALLOC] = {0};

ccv_nnc_tensor_t* ccv_nnc_tensor_new(const void* const ptr, const ccv_nnc_tensor_param_t params, const int flags)
{
	ccv_nnc_tensor_t* tensor;
	// this specific form can be toll-free bridging to ccv_dense_matrix_t (On CPU, and 3 dims (channels, rows, cols), and channels is smaller than max channels of ccv_dense_matrix_t).
	const int tfb = (CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_CPU_MEMORY && params.format == CCV_TENSOR_FORMAT_NHWC && params.dim[2] > 0 && params.dim[2] <= CCV_MAX_CHANNEL && params.dim[0] > 0 && params.dim[1] > 0 && params.dim[3] == 0);
	if (ptr)
	{
		tensor = (ccv_nnc_tensor_t*)ccmalloc(sizeof(ccv_nnc_tensor_t));
		tensor->alias_ref = 0;
		tensor->sig = 0;
		tensor->refcount = 1;
		tensor->info = params;
		if (tfb)
		{
			tensor->type = CCV_NO_DATA_ALLOC | CCV_MATRIX_DENSE | params.datatype | params.dim[2];
			// This corresponding to mat->step
			tensor->info.dim[4] = CCV_GET_STEP(params.dim[1], (params.datatype | params.dim[2]));
		} else // This won't be recognized by ccv_dense_matrix_t
			tensor->type = CCV_NO_DATA_ALLOC | CCV_MATRIX_DENSE | params.datatype;
		tensor->data.u8 = (uint8_t*)ptr;
		return tensor;
	}
	if (flags & CCV_TENSOR_CPU_MEMORY)
	{
		assert(CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_CPU_MEMORY);
	} else if (flags & CCV_TENSOR_GPU_MEMORY) {
		assert(CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_GPU_MEMORY);
	}
	const size_t tensor_hdr_size = (sizeof(ccv_nnc_tensor_t) + 15) & -16;
	const size_t size = ccv_nnc_tensor_data_size(params);
#ifdef HAVE_CUDA
	if (CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_GPU_MEMORY)
	{
		tensor = (ccv_nnc_tensor_t*)ccmalloc(sizeof(ccv_nnc_tensor_t));
		assert(CCV_TENSOR_GET_DEVICE(params.type) != CCV_COMPUTE_DEVICE_ANY);
		tensor->data.u8 = (uint8_t*)cumalloc(CCV_TENSOR_GET_DEVICE_ID(params.type), size);
	} else {
		assert(CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_CPU_MEMORY);
		ccmemalign((void **)&tensor, 16, tensor_hdr_size + size);
		tensor->data.u8 = (uint8_t*)tensor + tensor_hdr_size;
	}
#else
	assert(CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_CPU_MEMORY);
	ccmemalign((void **)&tensor, 16, tensor_hdr_size + size);
	tensor->data.u8 = (uint8_t*)tensor + tensor_hdr_size;
#endif
	tensor->alias_ref = 0;
	tensor->data_size = size;
	tensor->sig = 0;
	tensor->refcount = 1;
	tensor->info = params;
	if (tfb)
	{
		tensor->type = CCV_UNMANAGED | CCV_MATRIX_DENSE | params.datatype | params.dim[2];
		// This corresponding to mat->step
		tensor->info.dim[4] = CCV_GET_STEP(params.dim[1], (params.datatype | params.dim[2]));
	} else
		tensor->type = CCV_UNMANAGED | CCV_MATRIX_DENSE | params.datatype;
	return tensor;
}

ccv_nnc_tensor_t* ccv_nnc_tensor_resize(ccv_nnc_tensor_t* const tensor, const ccv_nnc_tensor_param_t params)
{
	assert(!CCV_IS_TENSOR_VIEW(tensor));
	assert(tensor->type & CCV_UNMANAGED);
	assert(tensor->data_size > 0);
	assert(CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_GET_MEMORY(tensor->info.type));
	assert(CCV_TENSOR_GET_DEVICE(params.type) == CCV_TENSOR_GET_DEVICE(tensor->info.type));
	const size_t size = ccv_nnc_tensor_data_size(params);
	const int tfb = (CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_CPU_MEMORY && params.format == CCV_TENSOR_FORMAT_NHWC && params.dim[2] > 0 && params.dim[2] <= CCV_MAX_CHANNEL && params.dim[0] > 0 && params.dim[1] > 0 && params.dim[3] == 0);
	tensor->info = params;
#ifdef HAVE_CUDA
	const int pinned_mem = (tensor->type & CCV_PINNED_MEM);
#endif
	if (tfb)
	{
		tensor->type = CCV_UNMANAGED | CCV_MATRIX_DENSE | params.datatype | params.dim[2];
		// This corresponding to mat->step
		tensor->info.dim[4] = CCV_GET_STEP(params.dim[1], (params.datatype | params.dim[2]));
	} else
		tensor->type = CCV_UNMANAGED | CCV_MATRIX_DENSE | params.datatype;
	if (size <= tensor->data_size) // Nothing.
	{
#ifdef HAVE_CUDA
		if (pinned_mem)
			tensor->type |= CCV_PINNED_MEM;
#endif
		return tensor;
	}
	ccv_nnc_tensor_t* new_tensor = tensor;
	const size_t tensor_hdr_size = (sizeof(ccv_nnc_tensor_t) + 15) & -16;
#ifdef HAVE_CUDA
	if (CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_GPU_MEMORY)
	{
		assert(CCV_TENSOR_GET_DEVICE(params.type) != CCV_COMPUTE_DEVICE_ANY);
		const int device_id = CCV_TENSOR_GET_DEVICE_ID(params.type);
		assert(device_id == CCV_TENSOR_GET_DEVICE_ID(tensor->info.type));
		cufree(device_id, tensor->data.u8);
		new_tensor->data.u8 = (uint8_t*)cumalloc(device_id, size);
	} else {
		assert(CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_CPU_MEMORY);
		assert(CCV_TENSOR_GET_MEMORY(tensor->info.type) == CCV_TENSOR_CPU_MEMORY);
		// pin memory again.
		if (pinned_mem)
			cuunregister(new_tensor->data.u8);
		new_tensor = ccrealloc(new_tensor, tensor_hdr_size + size);
		new_tensor->data.u8 = (uint8_t*)new_tensor + tensor_hdr_size;
	}
#else
	assert(CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_CPU_MEMORY);
	new_tensor = ccrealloc(new_tensor, tensor_hdr_size + size);
	new_tensor->data.u8 = (uint8_t*)new_tensor + tensor_hdr_size;
#endif
	new_tensor->data_size = size;
#ifdef HAVE_CUDA
	if (pinned_mem)
		ccv_nnc_tensor_pin_memory(new_tensor);
#endif
	return new_tensor;
}

ccv_nnc_tensor_t ccv_nnc_tensor(const void* const ptr, const ccv_nnc_tensor_param_t params, const int flags)
{
	// this specific form can be toll-free bridging to ccv_dense_matrix_t
	const int tfb = (CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_CPU_MEMORY && params.format == CCV_TENSOR_FORMAT_NHWC && params.dim[2] > 0 && params.dim[2] <= CCV_MAX_CHANNEL && params.dim[0] > 0 && params.dim[1] > 0 && params.dim[3] == 0);
	ccv_nnc_tensor_t tensor;
	tensor.alias_ref = 0;
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
		tensor.type = CCV_NO_DATA_ALLOC | CCV_UNMANAGED | CCV_MATRIX_DENSE | params.datatype | params.dim[2];
		// This corresponding to mat->step
		tensor.info.dim[4] = CCV_GET_STEP(params.dim[1], (params.datatype | params.dim[2]));
	} else // This won't be recognized by ccv_dense_matrix_t
		tensor.type = CCV_NO_DATA_ALLOC | CCV_UNMANAGED | CCV_MATRIX_DENSE | params.datatype;
	tensor.data.u8 = (uint8_t*)ptr;
	tensor.data_size = 0;
	return tensor;
}

int ccv_nnc_tensor_pin_memory(ccv_nnc_tensor_t* const tensor)
{
#ifdef HAVE_CUDA
	assert(CCV_TENSOR_GET_MEMORY(tensor->info.type) == CCV_TENSOR_CPU_MEMORY);
	if (!(tensor->type & CCV_PINNED_MEM) && tensor->data_size)
	{
		const int success = curegister(tensor->data.u8, tensor->data_size);
		if (success)
			tensor->type |= CCV_PINNED_MEM;
		return success ? 0 : -1;
	}
#endif
	return 0;
}

void ccv_nnc_tensor_free(ccv_nnc_tensor_t* const tensor)
{
#ifdef HAVE_CUDA
	if (CCV_TENSOR_GET_MEMORY(tensor->info.type) == CCV_TENSOR_GPU_MEMORY &&
		!(tensor->type & CCV_NO_DATA_ALLOC)) // If this is GPU memory and it is allocated, free.
		cufree(CCV_TENSOR_GET_DEVICE_ID(tensor->info.type), tensor->data.u8);
	if (tensor->type & CCV_PINNED_MEM)
		cuunregister(tensor->data.u8);
#endif
	ccfree(tensor);
}

static inline void _ccv_nnc_tensor_view_set(ccv_nnc_tensor_view_t* const tv, const ccv_nnc_tensor_t* const tensor, const int dim[CCV_NNC_MAX_DIM_ALLOC], const int ofs[CCV_NNC_MAX_DIM_ALLOC], const int inc[CCV_NNC_MAX_DIM_ALLOC])
{
	memcpy(tv->inc, inc, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC);
	memcpy(tv->info.dim, dim, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC);
	uint8_t* const p = tensor->data.u8;
	const off_t off = tv->off = ccv_nnc_tensor_view_offset(tv->info.datatype, tv->inc, ofs);
	tv->data.u8 = p + off;
}

ccv_nnc_tensor_view_t* ccv_nnc_tensor_view_new(const ccv_nnc_tensor_t* const tensor, const ccv_nnc_tensor_param_t params, const int ofs[CCV_NNC_MAX_DIM_ALLOC], const int inc[CCV_NNC_MAX_DIM_ALLOC])
{
	ccv_nnc_tensor_view_t* tv = (ccv_nnc_tensor_view_t*)ccmalloc(sizeof(ccv_nnc_tensor_view_t));
	tv->type = (tensor->type & ~0xfff) | CCV_TENSOR_VIEW;
	tv->alias_ref = (uintptr_t)tensor;
	tv->refcount = 1;
	tv->sig = 0;
	tv->data_size = 0;
	assert(params.type == tensor->info.type);
	assert(params.datatype == tensor->info.datatype);
	tv->info = params;
	_ccv_nnc_tensor_view_set(tv, tensor, params.dim, ofs, inc);
	return tv;
}

ccv_nnc_tensor_view_t ccv_nnc_tensor_view(const ccv_nnc_tensor_t* const tensor, const ccv_nnc_tensor_param_t params, const int ofs[CCV_NNC_MAX_DIM_ALLOC], const int inc[CCV_NNC_MAX_DIM_ALLOC])
{
	assert(!CCV_IS_TENSOR_VIEW(tensor));
	assert(params.type == tensor->info.type);
	assert(params.datatype == tensor->info.datatype);
	ccv_nnc_tensor_view_t tv = {
		.alias_ref = (uintptr_t)tensor,
		.type = (tensor->type & ~0xfff) | CCV_TENSOR_VIEW, // clean up the channel bits, and then add CCV_TENSOR_VIEW identifier
		.refcount = 1,
		.sig = 0,
		.info = params,
		.data_size = 0,
	};
	_ccv_nnc_tensor_view_set(&tv, tensor, params.dim, ofs, inc);
	return tv;
}

void ccv_nnc_tensor_view_free(ccv_nnc_tensor_view_t* const tensor_view)
{
	ccfree(tensor_view);
}

void ccv_nnc_tensor_zero(void* const tensor)
{
	ccv_nnc_tensor_view_t* tv = (ccv_nnc_tensor_view_t*)tensor;
	const size_t data_size = CCV_GET_DATA_TYPE_SIZE(tv->info.datatype);
	if (!CCV_IS_TENSOR_VIEW(tv))
	{
		memset(tv->data.u8, 0, data_size * ccv_nnc_tensor_count(tv->info));
		return;
	}
	const int nd = ccv_nnc_tensor_nd(tv->info.dim);
	assert(nd >= 1);
	const int* const tvinc = tv->inc;
	// reset it to 0.
	int c, x, y;
	int count = 1;
	int mod[CCV_NNC_MAX_DIM_ALLOC - 3];
	size_t mod_inc[CCV_NNC_MAX_DIM_ALLOC - 2];
	const size_t top_mod_inc = nd > 2 ? data_size * tvinc[nd - 3] * tvinc[nd - 2] * tvinc[nd - 1] : data_size;
	if (nd > 2)
		mod_inc[nd - 3] = top_mod_inc;
	for (c = nd - 4; c >= 0; c--)
	{
		// Compute the mod.
		mod[c] = c == nd - 4 ? tv->info.dim[c] : mod[c + 1] * tv->info.dim[c];
		mod_inc[c] = mod_inc[c + 1] * tvinc[c];
		count *= tv->info.dim[c];
	}
	for (c = 0; c < nd - 3; c++)
		mod_inc[c] = mod_inc[c + 1] * (tvinc[c] - tv->info.dim[c]);
	uint8_t* tvd = tv->data.u8;
	const size_t tvinc_1 = data_size * tvinc[nd - 1];
	const size_t tvinc_21 = tvinc_1 * (nd >= 2 ? tvinc[nd - 2] : 1);
	const size_t tvdim_1 = data_size * tv->info.dim[nd - 1];
	const int max_y = ccv_max(1, nd >= 3 ? tv->info.dim[nd - 3] : 1);
	const int max_x = ccv_max(1, nd >= 2 ? tv->info.dim[nd - 2] : 1);
	for (c = 0; c < count; c++)
	{
		for (y = 0; y < max_y; y++)
		{
			uint8_t* tvp = tvd + y * tvinc_21;
			for (x = 0; x < max_x; x++)
			{
				memset(tvp, 0, tvdim_1);
				tvp += tvinc_1;
			}
		}
		tvd += top_mod_inc;
		for (y = nd - 4; y >= 0; y--)
			if ((c + 1) % mod[y] != 0)
				break; // cannot be mod, break out.
			else
				tvd += mod_inc[y];
	}
}

int ccv_nnc_tensor_eq(const ccv_nnc_tensor_t* const a, const ccv_nnc_tensor_t* const b)
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
