extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

#ifdef HAVE_CUDA

template<typename NUM>
__global__ void _ccv_nnc_upsample_bilinear_forw_nchw(const size_t tensor_count, const float rwidth, const float rheight, const int adim2, const int ainc2, const int adim3, const int ainc3, const NUM* const a, const int bdim2, const int binc2, const int bdim3, const int binc3, NUM* const b)
{
	CUDA_1D_KERNEL_LOOP(i, tensor_count) {
		const int xd = i % bdim3;
		const int idxyd = i / bdim3;
		const int yd = idxyd % bdim2;
		const int idx = idxyd / bdim2;
		const NUM* const ap = a + idx * ainc2;
		NUM* const bp = b + idx * binc2;
		const float xs = (xd + 0.5) * rwidth - 0.5;
		const int xsi0 = (int)xs;
		const int xsi1 = ccv_min(xsi0 + 1, adim3 - 1);
		const float xsc1 = xs - xsi0;
		const float xsc0 = 1.0 - xsc1;
		const float ys = (yd + 0.5) * rheight - 0.5;
		const int ysi0 = (int)ys;
		const int ysi1 = ccv_min(ysi0 + 1, adim2 - 1);
		const float ysc1 = ys - ysi0;
		const float ysc0 = 1.0 - ysc1;
		bp[xd + yd * binc3] = (NUM)((float)ap[xsi0 + ysi0 * ainc3] * xsc0 * ysc0 + (float)ap[xsi1 + ysi0 * ainc3] * xsc1 * ysc0 + (float)ap[xsi0 + ysi1 * ainc3] * xsc0 * ysc1 + (float)ap[xsi1 + ysi1 * ainc3] * xsc1 * ysc1);
	}
}

template<typename NUM>
__global__ void _ccv_nnc_upsample_bilinear_forw_nhwc(const size_t tensor_count, const float rwidth, const float rheight, const int ch, const int adim1, const int ainc1, const int adim2, const int ainc2, const int ainc3, const NUM* const a, const int bdim1, const int binc1, const int bdim2, const int binc2, const int binc3, NUM* const b)
{
	CUDA_1D_KERNEL_LOOP(i, tensor_count) {
		const int xd = i % bdim2;
		const int idxyd = i / bdim2;
		const int yd = idxyd % bdim1;
		const int idx = idxyd / bdim1;
		const NUM* const ap = a + idx * ainc1;
		NUM* const bp = b + idx * binc1;
		const float xs = (xd + 0.5) * rwidth - 0.5;
		const int xsi0 = (int)xs;
		const int xsi1 = ccv_min(xsi0 + 1, adim2 - 1);
		const float xsc1 = xs - xsi0;
		const float xsc0 = 1.0 - xsc1;
		const float ys = (yd + 0.5) * rheight - 0.5;
		const int ysi0 = (int)ys;
		const int ysi1 = ccv_min(ysi0 + 1, adim1 - 1);
		const float ysc1 = ys - ysi0;
		const float ysc0 = 1.0 - ysc1;
		for (int c = 0; c < ch; c++)
			bp[c + xd * binc3 + yd * binc2] = (NUM)((float)ap[c + xsi0 * ainc3 + ysi0 * ainc2] * xsc0 * ysc0 + (float)ap[c + xsi1 * ainc3 + ysi0 * ainc2] * xsc1 * ysc0 + (float)ap[c + xsi0 * ainc3 + ysi1 * ainc2] * xsc0 * ysc1 + (float)ap[c + xsi1 * ainc3 + ysi1 * ainc2] * xsc1 * ysc1);
	}
}

template<typename NUM>
__global__ void _ccv_nnc_upsample_bilinear_back_nchw(const size_t tensor_count, const float rwidth, const float rheight, const int adim2, const int ainc2, const int adim3, const int ainc3, NUM* const a, const int bdim2, const int binc2, const int bdim3, const int binc3, const NUM* const b)
{
	CUDA_1D_KERNEL_LOOP(i, tensor_count) {
		const int xd = i % bdim3;
		const int idxyd = i / bdim3;
		const int yd = idxyd % bdim2;
		const int idx = idxyd / bdim2;
		NUM* const ap = a + idx * ainc2;
		const NUM* const bp = b + idx * binc2;
		const float xs = (xd + 0.5) * rwidth - 0.5;
		const int xsi0 = (int)xs;
		const int xsi1 = ccv_min(xsi0 + 1, adim3 - 1);
		const float xsc1 = xs - xsi0;
		const float xsc0 = 1.0 - xsc1;
		const float ys = (yd + 0.5) * rheight - 0.5;
		const int ysi0 = (int)ys;
		const int ysi1 = ccv_min(ysi0 + 1, adim2 - 1);
		const float ysc1 = ys - ysi0;
		const float ysc0 = 1.0 - ysc1;
		const float bpi = (float)__ldg(bp + xd + yd * binc3);
		atomicAdd(&ap[xsi0 + ysi0 * ainc3], (NUM)(bpi * xsc0 * ysc0));
		atomicAdd(&ap[xsi1 + ysi0 * ainc3], (NUM)(bpi * xsc1 * ysc0));
		atomicAdd(&ap[xsi0 + ysi1 * ainc3], (NUM)(bpi * xsc0 * ysc1));
		atomicAdd(&ap[xsi1 + ysi1 * ainc3], (NUM)(bpi * xsc1 * ysc1));
	}
}

template<typename NUM>
__global__ void _ccv_nnc_upsample_bilinear_back_nhwc(const size_t tensor_count, const float rwidth, const float rheight, const int ch, const int adim1, const int ainc1, const int adim2, const int ainc2, const int ainc3, NUM* const a, const int bdim1, const int binc1, const int bdim2, const int binc2, const int binc3, const NUM* const b)
{
	CUDA_1D_KERNEL_LOOP(i, tensor_count) {
		const int xd = i % bdim2;
		const int idxyd = i / bdim2;
		const int yd = idxyd % bdim1;
		const int idx = idxyd / bdim1;
		NUM* const ap = a + idx * ainc1;
		const NUM* const bp = b + idx * binc1;
		const float xs = (xd + 0.5) * rwidth - 0.5;
		const int xsi0 = (int)xs;
		const int xsi1 = ccv_min(xsi0 + 1, adim2 - 1);
		const float xsc1 = xs - xsi0;
		const float xsc0 = 1.0 - xsc1;
		const float ys = (yd + 0.5) * rheight - 0.5;
		const int ysi0 = (int)ys;
		const int ysi1 = ccv_min(ysi0 + 1, adim1 - 1);
		const float ysc1 = ys - ysi0;
		const float ysc0 = 1.0 - ysc1;
		for (int c = 0; c < ch; c++)
		{
			const float bpi = (float)__ldg(bp + c + xd * binc3 + yd * binc2);
			atomicAdd(&ap[c + xsi0 * ainc3 + ysi0 * ainc2], (NUM)(bpi * xsc0 * ysc0));
			atomicAdd(&ap[c + xsi1 * ainc3 + ysi0 * ainc2], (NUM)(bpi * xsc1 * ysc0));
			atomicAdd(&ap[c + xsi0 * ainc3 + ysi1 * ainc2], (NUM)(bpi * xsc0 * ysc1));
			atomicAdd(&ap[c + xsi1 * ainc3 + ysi1 * ainc2], (NUM)(bpi * xsc1 * ysc1));
		}
	}
}

static int _ccv_nnc_upsample_bilinear_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 1);
	assert(output_size >= 1);
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(ccv_nnc_tensor_nd(a->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(b->info.dim) <= CCV_NNC_MAX_DIM + 2);
	// Assuming this is float 32.
	int adim[CCV_NNC_MAX_DIM_ALLOC];
	int bdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(a, adim);
	ccv_nnc_tensor_view_get_dim(b, bdim);
	int ainc[CCV_NNC_MAX_DIM_ALLOC];
	int binc[CCV_NNC_MAX_DIM_ALLOC];
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	ccv_nnc_tensor_view_get_inc(a, ainc);
	ccv_nnc_tensor_view_get_inc(b, binc);
	assert(a->info.format == b->info.format);
	assert(a->info.datatype == b->info.datatype);
	if (a->info.format == CCV_TENSOR_FORMAT_NCHW)
	{
		const size_t tensor_count = ccv_nnc_tensor_count(b->info);
		const float rheight = (float)adim[2] / bdim[2];
		const float rwidth = (float)adim[3] / bdim[3];
		assert(rheight <= 1);
		assert(rwidth <= 1);
		if (a->info.datatype == CCV_16F)
		{
			_ccv_nnc_upsample_bilinear_forw_nchw<<<CUDA_GET_BLOCKS(tensor_count), CUDA_NUM_THREADS, 0, stream>>>(tensor_count, rwidth, rheight, adim[2], ainc[2] * ainc[3], adim[3], ainc[3], (__half*)a->data.f16, bdim[2], binc[2] * binc[3], bdim[3], binc[3], (__half*)b->data.f16);
		} else if (a->info.datatype == CCV_32F) {
			_ccv_nnc_upsample_bilinear_forw_nchw<<<CUDA_GET_BLOCKS(tensor_count), CUDA_NUM_THREADS, 0, stream>>>(tensor_count, rwidth, rheight, adim[2], ainc[2] * ainc[3], adim[3], ainc[3], a->data.f32, bdim[2], binc[2] * binc[3], bdim[3], binc[3], b->data.f32);
		}
	} else {
		assert(a->info.format == CCV_TENSOR_FORMAT_NHWC || a->info.format == CCV_TENSOR_FORMAT_CHWN);
		const float rheight = (float)adim[1] / bdim[1];
		const float rwidth = (float)adim[2] / bdim[2];
		assert(rheight <= 1);
		assert(rwidth <= 1);
		const size_t tensor_count = ccv_nnc_tensor_count(b->info) / adim[3];
		if (a->info.datatype == CCV_16F)
		{
			_ccv_nnc_upsample_bilinear_forw_nhwc<<<CUDA_GET_BLOCKS(tensor_count), CUDA_NUM_THREADS, 0, stream>>>(tensor_count, rwidth, rheight, adim[3], adim[1], ainc[1] * ainc[2] * ainc[3], adim[2], ainc[2] * ainc[3], ainc[3], (__half*)a->data.f16, bdim[1], binc[1] * binc[2] * binc[3], bdim[2], binc[2] * binc[3], binc[3], (__half*)b->data.f16);
		} else if (a->info.datatype == CCV_32F) {
			_ccv_nnc_upsample_bilinear_forw_nhwc<<<CUDA_GET_BLOCKS(tensor_count), CUDA_NUM_THREADS, 0, stream>>>(tensor_count, rwidth, rheight, adim[3], adim[1], ainc[1] * ainc[2] * ainc[3], adim[2], ainc[2] * ainc[3], ainc[3], a->data.f32, bdim[1], binc[1] * binc[2] * binc[3], bdim[2], binc[2] * binc[3], binc[3], b->data.f32);
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_upsample_bilinear_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 1);
	assert(output_size >= 1);
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)inputs[0];
	assert(ccv_nnc_tensor_nd(a->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(b->info.dim) <= CCV_NNC_MAX_DIM + 2);
	// Assuming this is float 32.
	int adim[CCV_NNC_MAX_DIM_ALLOC];
	int bdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(a, adim);
	ccv_nnc_tensor_view_get_dim(b, bdim);
	int ainc[CCV_NNC_MAX_DIM_ALLOC];
	int binc[CCV_NNC_MAX_DIM_ALLOC];
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	ccv_nnc_tensor_view_get_inc(a, ainc);
	ccv_nnc_tensor_view_get_inc(b, binc);
	assert(a->info.format == b->info.format);
	assert(a->info.datatype == b->info.datatype);
	if (a->info.format == CCV_TENSOR_FORMAT_NCHW)
	{
		const size_t tensor_count = ccv_nnc_tensor_count(b->info);
		const float rheight = (float)adim[2] / bdim[2];
		const float rwidth = (float)adim[3] / bdim[3];
		assert(rheight <= 1);
		assert(rwidth <= 1);
		if (a->info.datatype == CCV_16F)
		{
			_ccv_nnc_upsample_bilinear_back_nchw<<<CUDA_GET_BLOCKS(tensor_count), CUDA_NUM_THREADS, 0, stream>>>(tensor_count, rwidth, rheight, adim[2], ainc[2] * ainc[3], adim[3], ainc[3], (__half*)a->data.f16, bdim[2], binc[2] * binc[3], bdim[3], binc[3], (__half*)b->data.f16);
		} else if (a->info.datatype == CCV_32F) {
			_ccv_nnc_upsample_bilinear_back_nchw<<<CUDA_GET_BLOCKS(tensor_count), CUDA_NUM_THREADS, 0, stream>>>(tensor_count, rwidth, rheight, adim[2], ainc[2] * ainc[3], adim[3], ainc[3], a->data.f32, bdim[2], binc[2] * binc[3], bdim[3], binc[3], b->data.f32);
		}
	} else {
		assert(a->info.format == CCV_TENSOR_FORMAT_NHWC || a->info.format == CCV_TENSOR_FORMAT_CHWN);
		const float rheight = (float)adim[1] / bdim[1];
		const float rwidth = (float)adim[2] / bdim[2];
		assert(rheight <= 1);
		assert(rwidth <= 1);
		const size_t tensor_count = ccv_nnc_tensor_count(b->info) / adim[3];
		if (a->info.datatype == CCV_16F)
		{
			_ccv_nnc_upsample_bilinear_back_nhwc<<<CUDA_GET_BLOCKS(tensor_count), CUDA_NUM_THREADS, 0, stream>>>(tensor_count, rwidth, rheight, adim[3], adim[1], ainc[1] * ainc[2] * ainc[3], adim[2], ainc[2] * ainc[3], ainc[3], (__half*)a->data.f16, bdim[1], binc[1] * binc[2] * binc[3], bdim[2], binc[2] * binc[3], binc[3], (__half*)b->data.f16);
		} else if (a->info.datatype == CCV_32F) {
			_ccv_nnc_upsample_bilinear_back_nhwc<<<CUDA_GET_BLOCKS(tensor_count), CUDA_NUM_THREADS, 0, stream>>>(tensor_count, rwidth, rheight, adim[3], adim[1], ainc[1] * ainc[2] * ainc[3], adim[2], ainc[2] * ainc[3], ainc[3], a->data.f32, bdim[1], binc[1] * binc[2] * binc[3], bdim[2], binc[2] * binc[3], binc[3], b->data.f32);
		}
	}
	return CCV_NNC_EXEC_INVALID;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_UPSAMPLE_BILINEAR_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDA
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_upsample_bilinear_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_UPSAMPLE_BILINEAR_BACKWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDA
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_upsample_bilinear_back;
#endif
}
