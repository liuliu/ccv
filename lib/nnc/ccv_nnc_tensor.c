#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#ifdef HAVE_CUDA
#include "gpu/ccv_nnc_compat.h"
#elif defined(HAVE_MPS)
#include "mps/ccv_nnc_mps.h"
#endif
#include <fcntl.h>
#include <sys/mman.h>

// MARK - Level-1 API

const int ccv_nnc_no_ofs[CCV_NNC_MAX_DIM_ALLOC] = {0};

ccv_nnc_tensor_t* ccv_nnc_tensor_new(const void* const ptr, const ccv_nnc_tensor_param_t params, const int flags)
{
	ccv_nnc_tensor_t* tensor;
	// this specific form can be toll-free bridging to ccv_dense_matrix_t (On CPU, and 3 dims (channels, rows, cols), and channels is smaller than max channels of ccv_dense_matrix_t).
	const int tfb = (CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_CPU_MEMORY && params.format == CCV_TENSOR_FORMAT_NHWC && params.dim[2] > 0 && params.dim[2] <= CCV_MAX_CHANNEL && params.dim[0] > 0 && params.dim[1] > 0 && params.dim[3] == 0);
	if (ptr || (flags & CCV_NO_DATA_ALLOC))
	{
		tensor = (ccv_nnc_tensor_t*)ccmalloc(sizeof(ccv_nnc_tensor_t));
		tensor->dataof = 0;
		tensor->alias_ref = 0;
		tensor->sig = 0;
		tensor->refcount = 1;
		tensor->info = params;
		if (tfb)
		{
			tensor->type = CCV_NO_DATA_ALLOC | CCV_MATRIX_DENSE | CCV_GET_DATA_TYPE(params.datatype) | params.dim[2];
			// This corresponding to mat->step
			tensor->info.dim[4] = CCV_GET_STEP(params.dim[1], (CCV_GET_DATA_TYPE(params.datatype) | params.dim[2]));
		} else // This won't be recognized by ccv_dense_matrix_t
			tensor->type = CCV_NO_DATA_ALLOC | CCV_MATRIX_DENSE | CCV_GET_DATA_TYPE(params.datatype);
		tensor->data.u8 = (uint8_t*)ptr;
		return tensor;
	}
	if (flags & CCV_TENSOR_CPU_MEMORY)
	{
		assert(CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_CPU_MEMORY);
	} else if (flags & CCV_TENSOR_GPU_MEMORY) {
		assert(CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_GPU_MEMORY);
	}
	const size_t tensor_hdr_size = (sizeof(ccv_nnc_tensor_t) + 63) & -64;
	const size_t size = ccv_nnc_tensor_data_size(params);
#ifdef HAVE_CUDA
	if (CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_GPU_MEMORY)
	{
		tensor = (ccv_nnc_tensor_t*)ccmalloc(sizeof(ccv_nnc_tensor_t));
		assert(CCV_TENSOR_GET_DEVICE(params.type) != CCV_COMPUTE_DEVICE_ANY);
		if (size > 0)
			tensor->data.u8 = (uint8_t*)cumalloc(CCV_TENSOR_GET_DEVICE_ID(params.type), size);
		else
			tensor->data.u8 = 0;
	} else {
		assert(CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_CPU_MEMORY);
		ccmemalign((void **)&tensor, 64, tensor_hdr_size + size);
		if (size > 0)
			tensor->data.u8 = (uint8_t*)tensor + tensor_hdr_size;
		else
			tensor->data.u8 = 0;
	}
#elif defined(HAVE_MPS)
	if (CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_GPU_MEMORY)
	{
		tensor = (ccv_nnc_tensor_t*)ccmalloc(sizeof(ccv_nnc_tensor_t));
		assert(CCV_TENSOR_GET_DEVICE(params.type) != CCV_COMPUTE_DEVICE_ANY);
		if (size > 0)
			tensor->data.u8 = (uint8_t*)mpobjmalloc(CCV_TENSOR_GET_DEVICE_ID(params.type), size);
		else
			tensor->data.u8 = 0;
	} else {
		assert(CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_CPU_MEMORY);
		ccmemalign((void **)&tensor, 64, tensor_hdr_size + size);
		if (size > 0)
			tensor->data.u8 = (uint8_t*)tensor + tensor_hdr_size;
		else
			tensor->data.u8 = 0;
	}
#else
	assert(CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_CPU_MEMORY);
	ccmemalign((void **)&tensor, 64, tensor_hdr_size + size);
	if (size > 0)
		tensor->data.u8 = (uint8_t*)tensor + tensor_hdr_size;
	else
		tensor->data.u8 = 0;
#endif
	tensor->dataof = 0;
	tensor->alias_ref = 0;
	tensor->data_size = size;
	tensor->sig = 0;
	tensor->refcount = 1;
	tensor->info = params;
	if (tfb)
	{
		tensor->type = CCV_UNMANAGED | CCV_MATRIX_DENSE | CCV_GET_DATA_TYPE(params.datatype) | params.dim[2];
		// This corresponding to mat->step
		tensor->info.dim[4] = CCV_GET_STEP(params.dim[1], (CCV_GET_DATA_TYPE(params.datatype) | params.dim[2]));
	} else
		tensor->type = CCV_UNMANAGED | CCV_MATRIX_DENSE | CCV_GET_DATA_TYPE(params.datatype);
	return tensor;
}

ccv_nnc_tensor_t* ccv_nnc_tensor_new_from_file(const ccv_nnc_tensor_param_t params, const char* const filename, const off_t offset, const int flags)
{
	ccv_nnc_tensor_t* tensor;
	// this specific form can be toll-free bridging to ccv_dense_matrix_t (On CPU, and 3 dims (channels, rows, cols), and channels is smaller than max channels of ccv_dense_matrix_t).
	const int tfb = (CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_CPU_MEMORY && params.format == CCV_TENSOR_FORMAT_NHWC && params.dim[2] > 0 && params.dim[2] <= CCV_MAX_CHANNEL && params.dim[0] > 0 && params.dim[1] > 0 && params.dim[3] == 0);
	tensor = (ccv_nnc_tensor_t*)ccmalloc(sizeof(ccv_nnc_tensor_t));
	tensor->dataof = 0;
	tensor->alias_ref = 0;
	tensor->sig = 0;
	tensor->refcount = 1;
	tensor->info = params;
	if (tfb)
	{
		tensor->type = CCV_NO_DATA_ALLOC | CCV_MATRIX_DENSE | CCV_GET_DATA_TYPE(params.datatype) | params.dim[2];
		// This corresponding to mat->step
		tensor->info.dim[4] = CCV_GET_STEP(params.dim[1], (CCV_GET_DATA_TYPE(params.datatype) | params.dim[2]));
	} else // This won't be recognized by ccv_dense_matrix_t
		tensor->type = CCV_NO_DATA_ALLOC | CCV_MATRIX_DENSE | CCV_GET_DATA_TYPE(params.datatype);
	const size_t size = ccv_nnc_tensor_data_size(params);
#ifdef HAVE_CUDA
	if (CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_GPU_MEMORY)
	{
		assert(CCV_TENSOR_GET_DEVICE(params.type) != CCV_COMPUTE_DEVICE_ANY);
		if (size > 0)
		{
			// This is not supported yet on CUDA.
			tensor->data.u8 = (uint8_t*)cumalloc(CCV_TENSOR_GET_DEVICE_ID(params.type), size);
			int fd = open(filename, O_RDONLY, 0);
			void* bufptr = mmap(0, size, PROT_READ, MAP_PRIVATE, fd, offset);
			close(fd);
			madvise(bufptr, size, MADV_SEQUENTIAL | MADV_WILLNEED);
			cumemcpy(tensor->data.u8, CCV_TENSOR_GPU_MEMORY, bufptr, CCV_TENSOR_CPU_MEMORY, size);
			munmap(bufptr, size);
		} else
			tensor->data.u8 = 0;
	} else {
		assert(CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_CPU_MEMORY);
		if (size > 0)
		{
			int fd = open(filename, O_RDONLY, 0);
			void* bufptr = mmap(0, size, PROT_READ, MAP_PRIVATE, fd, offset);
			close(fd);
			madvise(bufptr, size, MADV_SEQUENTIAL | MADV_WILLNEED);
			tensor->data.u8 = bufptr;
			tensor->type |= CCV_MAPPED_MEM;
		} else
			tensor->data.u8 = 0;
	}
#elif defined(HAVE_MPS)
	if (CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_GPU_MEMORY)
	{
		assert(CCV_TENSOR_GET_DEVICE(params.type) != CCV_COMPUTE_DEVICE_ANY);
		if (size > 0)
			tensor->data.u8 = (uint8_t*)mpmemmap(filename, size, offset, flags);
		else
			tensor->data.u8 = 0;
	} else {
		assert(CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_CPU_MEMORY);
		if (size > 0)
		{
			int fd = open(filename, O_RDONLY, 0);
			void* bufptr = mmap(0, size, PROT_READ, MAP_PRIVATE, fd, offset);
			close(fd);
			madvise(bufptr, size, MADV_SEQUENTIAL | MADV_WILLNEED);
			tensor->data.u8 = bufptr;
			tensor->type |= CCV_MAPPED_MEM;
		} else
			tensor->data.u8 = 0;
	}
#else
	assert(CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_CPU_MEMORY);
	if (size > 0)
	{
		int fd = open(filename, O_RDONLY, 0);
		void* bufptr = mmap(0, size, PROT_READ, MAP_PRIVATE, fd, offset);
		close(fd);
		madvise(bufptr, size, MADV_SEQUENTIAL | MADV_WILLNEED);
		tensor->data.u8 = bufptr;
		tensor->type |= CCV_MAPPED_MEM;
	} else
		tensor->data.u8 = 0;
#endif
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
		tensor->type = CCV_UNMANAGED | CCV_MATRIX_DENSE | CCV_GET_DATA_TYPE(params.datatype) | params.dim[2];
		// This corresponding to mat->step
		tensor->info.dim[4] = CCV_GET_STEP(params.dim[1], (CCV_GET_DATA_TYPE(params.datatype) | params.dim[2]));
	} else
		tensor->type = CCV_UNMANAGED | CCV_MATRIX_DENSE | CCV_GET_DATA_TYPE(params.datatype);
	if (size <= tensor->data_size) // Nothing.
	{
#ifdef HAVE_CUDA
		if (pinned_mem)
			tensor->type |= CCV_PINNED_MEM;
#endif
		return tensor;
	}
	ccv_nnc_tensor_t* new_tensor = tensor;
	const size_t tensor_hdr_size = (sizeof(ccv_nnc_tensor_t) + 63) & -64;
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
#elif defined(HAVE_MPS)
	if (CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_GPU_MEMORY)
	{
		assert(CCV_TENSOR_GET_DEVICE(params.type) != CCV_COMPUTE_DEVICE_ANY);
		const int device_id = CCV_TENSOR_GET_DEVICE_ID(params.type);
		assert(device_id == CCV_TENSOR_GET_DEVICE_ID(tensor->info.type));
		mpobjfree(device_id, tensor->data.u8);
		new_tensor->data.u8 = (uint8_t*)mpobjmalloc(device_id, size);
	} else {
		assert(CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_CPU_MEMORY);
		assert(CCV_TENSOR_GET_MEMORY(tensor->info.type) == CCV_TENSOR_CPU_MEMORY);
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
	tensor.dataof = 0;
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
		tensor.type = CCV_NO_DATA_ALLOC | CCV_UNMANAGED | CCV_MATRIX_DENSE | CCV_GET_DATA_TYPE(params.datatype) | params.dim[2];
		// This corresponding to mat->step
		tensor.info.dim[4] = CCV_GET_STEP(params.dim[1], (CCV_GET_DATA_TYPE(params.datatype) | params.dim[2]));
	} else // This won't be recognized by ccv_dense_matrix_t
		tensor.type = CCV_NO_DATA_ALLOC | CCV_UNMANAGED | CCV_MATRIX_DENSE | CCV_GET_DATA_TYPE(params.datatype);
	if (params.dim[0] > 0)
		tensor.data.u8 = (uint8_t*)ptr;
	else
		tensor.data.u8 = 0;
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
	if (CCV_TENSOR_GET_MEMORY(tensor->info.type) == CCV_TENSOR_CPU_MEMORY && tensor->type & CCV_MAPPED_MEM)
	{
		// The size might be different than the ones when we allocated (for example, the tensor might rewrite its size to be smaller).
		// This might cause issues in the future.
		const size_t size = ccv_nnc_tensor_data_size(tensor->info);
		munmap(tensor->data.u8, size);
	}
#ifdef HAVE_CUDA
	if (tensor->type & CCV_PINNED_MEM)
		cuunregister(tensor->data.u8);
	if (CCV_TENSOR_GET_MEMORY(tensor->info.type) == CCV_TENSOR_GPU_MEMORY &&
		!(tensor->type & CCV_NO_DATA_ALLOC)) // If this is GPU memory and it is allocated, free.
		cufree(CCV_TENSOR_GET_DEVICE_ID(tensor->info.type), tensor->data.u8);
#elif defined(HAVE_MPS)
	if (CCV_TENSOR_GET_MEMORY(tensor->info.type) == CCV_TENSOR_GPU_MEMORY &&
		!(tensor->type & CCV_NO_DATA_ALLOC)) // If this is GPU memory and it is allocated, free.
		mpobjfree(CCV_TENSOR_GET_DEVICE_ID(tensor->info.type), tensor->data.u8);
#endif
	ccfree(tensor);
}

static inline void _ccv_nnc_tensor_view_set(ccv_nnc_tensor_view_t* const tv, const ccv_nnc_tensor_t* const tensor, const int dim[CCV_NNC_MAX_DIM_ALLOC], const int ofs[CCV_NNC_MAX_DIM_ALLOC], const int stride[CCV_NNC_MAX_DIM_ALLOC])
{
	memcpy(tv->stride, stride, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC);
	memcpy(tv->info.dim, dim, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC);
	uint8_t* const p = tensor->data.u8;
	const off_t off = tv->off = ccv_nnc_tensor_view_offset(tv->info.datatype, stride, ofs);
	tv->contiguous = ccv_nnc_tensor_view_is_contiguous(dim, stride);
	assert(off + CCV_GET_DATA_TYPE_SIZE(tv->info.datatype) * ccv_nnc_dimension_upper_bound(tv->info.dim, tv->stride) <= CCV_GET_DATA_TYPE_SIZE(tensor->info.datatype) * ccv_nnc_tensor_count(tensor->info));
	ccv_nnc_tensor_data(tv->info, p, off + tensor->dataof, &tv->data, &tv->dataof);
}

ccv_nnc_tensor_view_t* ccv_nnc_tensor_view_new(const ccv_nnc_tensor_t* const tensor, const ccv_nnc_tensor_param_t params, const int ofs[CCV_NNC_MAX_DIM_ALLOC], const int stride[CCV_NNC_MAX_DIM_ALLOC])
{
	ccv_nnc_tensor_view_t* tv = (ccv_nnc_tensor_view_t*)ccmalloc(sizeof(ccv_nnc_tensor_view_t));
	tv->type = (tensor->type & ~0xfff) | CCV_TENSOR_VIEW;
	tv->dataof = 0;
	tv->alias_ref = (uintptr_t)tensor;
	tv->refcount = 1;
	tv->sig = 0;
	tv->data_size = 0;
	assert(params.type == tensor->info.type);
	assert(params.datatype == tensor->info.datatype);
	tv->info = params;
	_ccv_nnc_tensor_view_set(tv, tensor, params.dim, ofs, stride);
	return tv;
}

ccv_nnc_tensor_view_t ccv_nnc_tensor_view(const ccv_nnc_tensor_t* const tensor, const ccv_nnc_tensor_param_t params, const int ofs[CCV_NNC_MAX_DIM_ALLOC], const int stride[CCV_NNC_MAX_DIM_ALLOC])
{
	assert(!CCV_IS_TENSOR_VIEW(tensor));
	assert(params.type == tensor->info.type);
	assert(params.datatype == tensor->info.datatype);
	ccv_nnc_tensor_view_t tv = {
		.dataof = 0,
		.alias_ref = (uintptr_t)tensor,
		.type = (tensor->type & ~0xfff) | CCV_TENSOR_VIEW, // clean up the channel bits, and then add CCV_TENSOR_VIEW identifier
		.refcount = 1,
		.sig = 0,
		.info = params,
		.data_size = 0,
	};
	_ccv_nnc_tensor_view_set(&tv, tensor, params.dim, ofs, stride);
	return tv;
}

void ccv_nnc_tensor_view_free(ccv_nnc_tensor_view_t* const tensor_view)
{
	ccfree(tensor_view);
}

void _ccv_nnc_tensor_set_zero(unsigned char* u8, const int nd, const int* const dim, const int* const stride, const size_t data_size)
{
	if (nd == 1)
	{
		if (stride[0] == 1)
		{
			memset(u8, 0, data_size * dim[0]);
			return;
		}
		int i;
		for (i = 0; i < dim[0]; i++)
			memset(u8 + i * stride[0] * data_size, 0, data_size);
	} else if (nd == 2) {
		if (stride[1] == 1 && stride[0] == dim[1])
		{
			memset(u8, 0, data_size * dim[1] * dim[0]);
			return;
		}
		int x, y;
		for (y = 0; y < dim[0]; y++)
		{
			unsigned char* const u8y = u8 + y * stride[0] * data_size;
			for (x = 0; x < dim[1]; x++)
				memset(u8y + x * stride[1] * data_size, 0, data_size);
		}
	} else if (nd == 3) {
		if (stride[2] == 1 && stride[1] == dim[2] && stride[0] == dim[1] * dim[2])
		{
			memset(u8, 0, data_size * dim[2] * dim[1] * dim[0]);
			return;
		}
		int x, y, z;
		for (z = 0; z < dim[0]; z++)
		{
			unsigned char* const u8z = u8 + z * stride[0] * data_size;
			for (y = 0; y < dim[1]; y++)
			{
				unsigned char* const u8y = u8z + y * stride[1] * data_size;
				for (x = 0; x < dim[2]; x++)
					memset(u8y + x * stride[2] * data_size, 0, data_size);
			}
		}
	} else if (nd == 4) {
		if (stride[3] == 1 && stride[2] == dim[3] && stride[1] == dim[2] * dim[3] && stride[0] == dim[1] * dim[2] * dim[3])
		{
			memset(u8, 0, data_size * dim[3] * dim[2] * dim[1] * dim[0]);
			return;
		}
		int x, y, z, s;
		for (s = 0; s < dim[0]; s++)
		{
			unsigned char* const u8s = u8 + s * stride[0] * data_size;
			for (z = 0; z < dim[1]; z++)
			{
				unsigned char* const u8z = u8s + z * stride[1] * data_size;
				for (y = 0; y < dim[2]; y++)
				{
					unsigned char* const u8y = u8z + y * stride[2] * data_size;
					for (x = 0; x < dim[3]; x++)
						memset(u8y + x * stride[3] * data_size, 0, data_size);
				}
			}
		}
	} else {
		int i;
		for (i = 0; i < dim[0]; i++)
			_ccv_nnc_tensor_set_zero(u8 + i * stride[0] * data_size, nd - 1, dim + 1, stride + 1, data_size);
	}
}

void ccv_nnc_tensor_zero(void* const tensor)
{
	ccv_nnc_tensor_view_t* tv = (ccv_nnc_tensor_view_t*)tensor;
	const size_t data_size = CCV_GET_DATA_TYPE_SIZE(tv->info.datatype);
	if (CCV_IS_TENSOR_CONTIGUOUS(tv))
	{
		memset(tv->data.u8, 0, data_size * ccv_nnc_tensor_count(tv->info));
		return;
	}
	const int nd = ccv_nnc_tensor_nd(tv->info.dim);
	assert(nd >= 1);
	const int* const tvstride = tv->stride;
	// Go through this recursively.
	_ccv_nnc_tensor_set_zero(tv->data.u8, nd, tv->info.dim, tvstride, data_size);
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
	int i, c = 1;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC; i++)
	{
		if (!a->info.dim[i] && !b->info.dim[i])
			break;
		if (a->info.dim[i] != b->info.dim[i])
			return -1;
		c *= a->info.dim[i];
	}
	if (CCV_GET_DATA_TYPE(a->type) == CCV_32S)
		return memcmp(a->data.i32, b->data.i32, sizeof(int) * c) == 0 ? 0 : -1;
	// Only support 32F at this point.
	assert(CCV_GET_DATA_TYPE(a->type) == CCV_32F || CCV_GET_DATA_TYPE(a->type) == CCV_64F);
	// Read: http://www.cygnus-software.com/papers/comparingfloats/comparingfloats.htm
	// http://floating-point-gui.de/errors/comparison/
	if (CCV_GET_DATA_TYPE(a->type) == CCV_32F)
	{
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
	} else if (CCV_GET_DATA_TYPE(a->type) == CCV_64F) {
		typedef union {
			double f64;
			int64_t i64;
		} Float64;
		static const double epsi = DBL_EPSILON;
		static const int64_t ulps = 128; // so that for 1 and 1.000015 will be treated as the same.
		for (i = 0; i < c; i++)
		{
			// Although this is float point, I use integer as a way to compare.
			Float64 f64a, f64b;
			f64a.f64 = a->data.f64[i];
			f64b.f64 = b->data.f64[i];
			if (f64a.i64 < 0)
				f64a.i64 = 0x8000000000000000 - f64a.i64;
			if (f64b.i64 < 0)
				f64b.i64 = 0x8000000000000000 - f64b.i64;
			if (llabs(f64a.i64 - f64b.i64) > ulps && fabs(a->data.f64[i] - b->data.f64[i]) > epsi)
				return -1;
		}
	}
	return 0;
}

static void _strcat(char** str, int* written, size_t* len, char* from, int from_size)
{
	if (*len - *written < from_size)
	{
		*len += from_size * 2;
		*str = (char*)ccrealloc(*str, *len);
	}
	memcpy(*str + *written, from, from_size);
	*written += from_size;
}

#define _STRPRINTF(str, written, len, format, ...) \
do { \
	const int newly_written = snprintf((str) + (written), (len) - (written), format, ## __VA_ARGS__); \
	if ((len) - (written) < newly_written) \
	{ \
		(len) += newly_written * 2; \
		(str) = (char*)ccrealloc((str), (len)); \
		(written) += snprintf((str) + (written), (len) - (written), format, ## __VA_ARGS__); \
	} else \
		(written) += newly_written; \
} while (0)

static void _strv(char** str, int* written, size_t* len, const ccv_nnc_tensor_t* const a, int i)
{
	if (a->info.datatype == CCV_32F)
		_STRPRINTF(*str, *written, *len, "%10.5g", a->data.f32[i]);
	else if (a->info.datatype == CCV_64F)
		_STRPRINTF(*str, *written, *len, "%10.5g", a->data.f64[i]);
	else if (a->info.datatype == CCV_16F) {
		float v;
		ccv_half_precision_to_float((uint16_t*)(a->data.f16 + i), &v, 1);
		_STRPRINTF(*str, *written, *len, "%10.5g", v);
	} else if (a->info.datatype == CCV_32S)
		_STRPRINTF(*str, *written, *len, "%10d", a->data.i32[i]);
	else if (a->info.datatype == CCV_64S)
		_STRPRINTF(*str, *written, *len, "%12lld", (long long int)a->data.i64[i]);
	else if (a->info.datatype == CCV_8U)
		_STRPRINTF(*str, *written, *len, "%3d", (int)a->data.u8[i]);
}

static void _strt(char** str, int* written, size_t* len, const ccv_nnc_tensor_t* const a, int nd, int spacer, const int* const dim, const int* const stride, int idx)
{
	assert(nd != 1);
	if (nd == 2)
	{
		// Print columns and the rows.
		int i, j, k;
		if (dim[0] <= 8)
		{
			for (i = 0; i < dim[0]; i++)
			{
				if (i != 0)
				{
					_strcat(str, written, len, "  ", 2);
					for (k = 0; k < spacer; k++)
						_strcat(str, written, len, " ", 1);
				}
				_strcat(str, written, len, "[", 1);
				if (dim[1] <= 8)
				{
					for (j = 0; j < dim[1]; j++)
					{
						_strv(str, written, len, a, idx + i * stride[0] + j * stride[1]);
						if (j < dim[1] - 1)
							_strcat(str, written, len, ", ", 2);
					}
					if (i < dim[0] - 1)
						_strcat(str, written, len, "],\n", 3);
				} else {
					for (j = 0; j < 3; j++)
					{
						_strv(str, written, len, a, idx + i * stride[0] + j * stride[1]);
						_strcat(str, written, len, ", ", 2);
					}
					_strcat(str, written, len, " ..., ", 6);
					for (j = dim[1] - 3; j < dim[1]; j++)
					{
						_strv(str, written, len, a, idx + i * stride[0] + j * stride[1]);
						if (j < dim[1] - 1)
							_strcat(str, written, len, ", ", 2);
					}
					if (i < dim[0] - 1)
						_strcat(str, written, len, "],\n", 3);
				}
			}
			_strcat(str, written, len, "]", 1);
		} else {
			for (i = 0; i < 3; i++)
			{
				if (i != 0)
				{
					_strcat(str, written, len, "  ", 2);
					for (k = 0; k < spacer; k++)
						_strcat(str, written, len, " ", 1);
				}
				_strcat(str, written, len, "[", 1);
				if (dim[1] <= 8)
				{
					for (j = 0; j < dim[1]; j++)
					{
						_strv(str, written, len, a, idx + i * stride[0] + j * stride[1]);
						if (j < dim[1] - 1)
							_strcat(str, written, len, ", ", 2);
					}
					_strcat(str, written, len, "],\n", 3);
				} else {
					for (j = 0; j < 3; j++)
					{
						_strv(str, written, len, a, idx + i * stride[0] + j * stride[1]);
						_strcat(str, written, len, ", ", 2);
					}
					_strcat(str, written, len, " ..., ", 6);
					for (j = dim[1] - 3; j < dim[1]; j++)
					{
						_strv(str, written, len, a, idx + i * stride[0] + j * stride[1]);
						if (j < dim[1] - 1)
							_strcat(str, written, len, ", ", 2);
					}
					_strcat(str, written, len, "],\n", 3);
				}
			}
			_strcat(str, written, len, "  ", 2);
			for (k = 0; k < spacer; k++)
				_strcat(str, written, len, " ", 1);
			_strcat(str, written, len, "...,\n", 5);
			for (i = dim[0] - 3; i < dim[0]; i++)
			{
				_strcat(str, written, len, "  ", 2);
				for (k = 0; k < spacer; k++)
					_strcat(str, written, len, " ", 1);
				_strcat(str, written, len, "[", 1);
				if (dim[1] < 8)
				{
					for (j = 0; j < dim[1]; j++)
					{
						_strv(str, written, len, a, idx + i * stride[0] + j * stride[1]);
						if (j < dim[1] - 1)
							_strcat(str, written, len, ", ", 2);
					}
					if (i < dim[0] - 1)
						_strcat(str, written, len, "],\n", 3);
				} else {
					for (j = 0; j < 3; j++)
					{
						_strv(str, written, len, a, idx + i * stride[0] + j * stride[1]);
						_strcat(str, written, len, ", ", 2);
					}
					_strcat(str, written, len, " ..., ", 6);
					for (j = dim[1] - 3; j < dim[1]; j++)
					{
						_strv(str, written, len, a, idx + i * stride[0] + j * stride[1]);
						if (j < dim[1] - 1)
							_strcat(str, written, len, ", ", 2);
					}
					if (i < dim[0] - 1)
						_strcat(str, written, len, "],\n", 3);
				}
			}
			_strcat(str, written, len, "]", 1);
		}
		return;
	}
	int i, j;
	if (dim[0] > 4)
	{
		for (i = 0; i < 2; i++)
		{
			_strcat(str, written, len, "[", 1);
			_strt(str, written, len, a, nd - 1, spacer + 1, dim + 1, stride + 1, idx + stride[0] * i);
			_strcat(str, written, len, "],\n  ", 5);
			for (j = 0; j < spacer; j++)
				_strcat(str, written, len, " ", 1);
		}
		_strcat(str, written, len, "...,\n", 5);
		_strcat(str, written, len, "  ", 2);
		for (j = 0; j < spacer; j++)
			_strcat(str, written, len, " ", 1);
		for (i = dim[0] - 2; i < dim[0]; i++)
		{
			_strcat(str, written, len, "[", 1);
			_strt(str, written, len, a, nd - 1, spacer + 1, dim + 1, stride + 1, idx + stride[0] * i);
			if (i < dim[0] - 1)
			{
				_strcat(str, written, len, "],\n  ", 5);
				for (j = 0; j < spacer; j++)
					_strcat(str, written, len, " ", 1);
			}
		}
		_strcat(str, written, len, "]", 1);
	} else {
		for (i = 0; i < dim[0]; i++)
		{
			_strcat(str, written, len, "[", 1);
			_strt(str, written, len, a, nd - 1, spacer + 1, dim + 1, stride + 1, idx + stride[0] * i);
			if (i < dim[0] - 1)
			{
				_strcat(str, written, len, "],\n", 3);
				_strcat(str, written, len, "  ", 2);
				for (j = 0; j < spacer; j++)
					_strcat(str, written, len, " ", 1);
			}
		}
		_strcat(str, written, len, "]", 1);
	}
}

char* ccv_nnc_tensor_format_new(const ccv_nnc_tensor_t* const a)
{
	const int nd = ccv_nnc_tensor_nd(a->info.dim);
	int i;
	int rows = 8; // 8 rows for the first one, and then just first and last.
	for (i = 2; i < nd; i++)
		rows *= 5; // Maximum 3 rows beyond the first two.
	int columns = nd * 2 + 16 * 8;
	size_t len = sizeof(char) * columns * rows;
	// Allocate return string buffer.
	char* str = (char*)ccmalloc(len);
	int written = 0;
	int stride[CCV_NNC_MAX_DIM_ALLOC];
	if (CCV_IS_TENSOR_VIEW(a))
		memcpy(stride, ((ccv_nnc_tensor_view_t*)a)->stride, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC);
	else
		ccv_nnc_tensor_get_stride(a->info.dim, stride);
	_strcat(&str, &written, &len, "[\n  ", 4);
	if (nd == 1)
	{
		// Special casing for vector.
		if (a->info.dim[0] <= 64)
			for (i = 0; i < a->info.dim[0]; i++)
			{
				_strv(&str, &written, &len, a, i * stride[0]);
				if (i < a->info.dim[0] - 1)
				{
					if ((i + 1) % 8 == 0)
						_strcat(&str, &written, &len, ",\n  ", 4);
					else
						_strcat(&str, &written, &len, ", ", 2);
				}
			}
		else {
			// First 3 rows.
			for (i = 0; i < 24; i++)
			{
				_strv(&str, &written, &len, a, i * stride[0]);
				if ((i + 1) % 8 == 0)
					_strcat(&str, &written, &len, ",\n  ", 4);
				else
					_strcat(&str, &written, &len, ", ", 2);
			}
			_strcat(&str, &written, &len, "...,\n  ", 7);
			// Last 3 rows (aligned to 8 items per row).
			int start = ((a->info.dim[0] + 7) / 8 - 3) * 8;
			for (i = start; i < a->info.dim[0]; i++)
			{
				_strv(&str, &written, &len, a, i * stride[0]);
				if (i < a->info.dim[0] - 1)
				{
					if ((i + 1) % 8 == 0)
						_strcat(&str, &written, &len, ",\n  ", 4);
					else
						_strcat(&str, &written, &len, ", ", 2);
				}
			}
		}
	} else {
		_strt(&str, &written, &len, a, nd, 0, a->info.dim, stride, 0);
	}
	_strcat(&str, &written, &len, "\n]", 3); // Including the terminal \0.
	str = (char*)ccrealloc(str, written); // Don't need the extra spaces.
	return str;
}
