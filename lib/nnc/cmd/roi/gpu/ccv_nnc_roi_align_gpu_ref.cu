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
__global__ void _ccv_nnc_roi_align_forw_nchw(const int chw, const int w, const int h, const int adim1, const int adim2, const NUM* const ap, const NUM* const bp, const int pool_w, const int pool_h, const int cdim1, const int cdim2, NUM* const cp)
{
	const float roi_x = bp[0] * w; // These assumed it is real-coordinate, with range between 0 to w - 1.
	const float roi_y = bp[1] * h;
	const float roi_w = bp[2] * w;
	const float roi_h = bp[3] * h;
	const int bin_h = (int)ceilf(roi_h / pool_h); // How many bins in each point of the pool. We slightly sampling at higher resolution (due to ceiling) with bilinear interpolation.
	const int bin_w = (int)ceilf(roi_w / pool_w);
	const int bin_pool_h = bin_h * pool_h; // Before averaging, what's the size of the region in integral term.
	const int bin_pool_w = bin_w * pool_w;
	const float scale_y = roi_h / bin_pool_h; // The scale to multiply back to get original coordinate.
	const float scale_x = roi_w / bin_pool_w;
	const int pool_hw = pool_h * pool_w;
	CUDA_1D_KERNEL_LOOP(i, chw) {
		const int k = i / pool_hw;
		const int xy = i % pool_hw;
		const int y = xy / pool_w;
		const int x = xy % pool_w;
		const int py = y * bin_h;
		const int px = x * bin_w;
		float v = 0;
		int count = 0;
		const float* const apz = ap + k * adim1;
		for (int by = 0; by < bin_h; by++)
		{
			const float ay = roi_y + (by + py + 0.5) * scale_y - 0.5;
			const int iy = (int)ay;
			if (iy + 1 < 0 || iy > h - 1)
				continue;
			const float ry = ay - iy;
			const int iy0 = ccv_clamp(iy, 0, h - 1);
			const int iy1 = ccv_clamp(iy + 1, 0, h - 1);
			for (int bx = 0; bx < bin_w; bx++)
			{
				const float ax = roi_x + (bx + px + 0.5) * scale_x - 0.5;
				const int ix = (int)ax;
				if (ix + 1 < 0 || ix > w - 1)
					continue;
				const float rx = ax - ix;
				const int ix0 = ccv_clamp(ix, 0, w - 1);
				const int ix1 = ccv_clamp(ix + 1, 0, w - 1);
				const float c00 = (1 - ry) * (1 - rx);
				const float c01 = (1 - ry) * rx;
				const float c10 = ry * (1 - rx);
				const float c11 = ry * rx;
				const float ap00 = apz[iy0 * adim2 + ix0];
				const float ap01 = apz[iy0 * adim2 + ix1];
				const float ap10 = apz[iy1 * adim2 + ix0];
				const float ap11 = apz[iy1 * adim2 + ix1];
				v += ap00 * c00 + ap01 * c01 + ap10 * c10 + ap11 * c11;
				++count;
			}
		}
		cp[k * cdim1 + y * cdim2 + x] = count > 0 ? v / count : 0;
	}
}

template<typename NUM>
__global__ void _ccv_nnc_roi_align_forw_nhwc(const int hw, const int w, const int h, const int ch, const int adim1, const int adim2, const NUM* const ap, const NUM* const bp, const int pool_w, const int pool_h, const int cdim1, const int cdim2, NUM* const cp)
{
	const float roi_x = bp[0] * w; // These assumed it is real-coordinate, with range between 0 to w - 1.
	const float roi_y = bp[1] * h;
	const float roi_w = bp[2] * w;
	const float roi_h = bp[3] * h;
	const int bin_h = (int)ceilf(roi_h / pool_h); // How many bins in each point of the pool. We slightly sampling at higher resolution (due to ceiling) with bilinear interpolation.
	const int bin_w = (int)ceilf(roi_w / pool_w);
	const int bin_pool_h = bin_h * pool_h; // Before averaging, what's the size of the region in integral term.
	const int bin_pool_w = bin_w * pool_w;
	const float scale_y = roi_h / bin_pool_h; // The scale to multiply back to get original coordinate.
	const float scale_x = roi_w / bin_pool_w;
	const int pool_hw = pool_h * pool_w;
	CUDA_1D_KERNEL_LOOP(i, hw) {
		const int y = i / pool_w;
		const int x = i % pool_w;
		const int py = y * bin_h;
		const int px = x * bin_w;
		int count = 0;
		float* const cpz = cp + y * cdim1 + x * cdim2;
		for (int k = 0; k < ch; k++)
			cpz[k] = 0;
		for (int by = 0; by < bin_h; by++)
		{
			const float ay = roi_y + (by + py + 0.5) * scale_y - 0.5;
			const int iy = (int)ay;
			if (iy + 1 < 0 || iy > h - 1)
				continue;
			const float ry = ay - iy;
			const int iy0 = ccv_clamp(iy, 0, h - 1);
			const int iy1 = ccv_clamp(iy + 1, 0, h - 1);
			for (int bx = 0; bx < bin_w; bx++)
			{
				const float ax = roi_x + (bx + px + 0.5) * scale_x - 0.5;
				const int ix = (int)ax;
				if (ix + 1 < 0 || ix > w - 1)
					continue;
				const float rx = ax - ix;
				const int ix0 = ccv_clamp(ix, 0, w - 1);
				const int ix1 = ccv_clamp(ix + 1, 0, w - 1);
				const float c00 = (1 - ry) * (1 - rx);
				const float c01 = (1 - ry) * rx;
				const float c10 = ry * (1 - rx);
				const float c11 = ry * rx;
				const float* const ap00 = ap + iy0 * adim1 + ix0 * adim2;
				const float* const ap01 = ap + iy0 * adim1 + ix1 * adim2;
				const float* const ap10 = ap + iy1 * adim1 + ix0 * adim2;
				const float* const ap11 = ap + iy1 * adim1 + ix1 * adim2;
				for (int k = 0; k < ch; k++)
					cpz[k] += ap00[k] * c00 + ap01[k] * c01 + ap10[k] * c10 + ap11[k] * c11;
				++count;
			}
		}
		if (count > 0)
			for (int k = 0; k < ch; k++)
				cpz[k] = cpz[k] / count;
	}
}

static int _ccv_nnc_roi_align_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 2);
	const ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[0];
	assert(output_size == 1);
	const ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)inputs[1];
	ccv_nnc_tensor_view_t* c = (ccv_nnc_tensor_view_t*)outputs[0];
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	assert(a_nd == CCV_NNC_MAX_DIM + 1 || a_nd == CCV_NNC_MAX_DIM + 2);
	const int* adim = (a_nd == CCV_NNC_MAX_DIM + 1) ? a->info.dim : a->info.dim + 1;
	const int c_nd = ccv_nnc_tensor_nd(c->info.dim);
	assert(c_nd == CCV_NNC_MAX_DIM + 1 || c_nd == CCV_NNC_MAX_DIM + 2);
	const int* cdim = (c_nd == CCV_NNC_MAX_DIM + 1) ? c->info.dim : c->info.dim + 1;
	const int* ainc = CCV_IS_TENSOR_VIEW(a) ? ((a_nd == CCV_NNC_MAX_DIM + 1) ?  a->inc : a->inc + 1) : adim;
	const int* cinc = CCV_IS_TENSOR_VIEW(c) ? ((c_nd == CCV_NNC_MAX_DIM + 1) ?  c->inc : c->inc + 1) : cdim;
	assert(a->info.format == c->info.format);
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	const int n = ccv_nnc_tensor_get_n(a->info);
	assert(n == 1);
	if (a->info.format == CCV_TENSOR_FORMAT_NCHW)
	{
		const int h = adim[1];
		const int w = adim[2];
		const int pool_h = cdim[1];
		const int pool_w = cdim[2];
		assert(cdim[0] == adim[0]);
		const int ch = cdim[0];
		const int chw = pool_h * pool_w * ch;
		_ccv_nnc_roi_align_forw_nchw<<<CUDA_GET_BLOCKS(chw), CUDA_NUM_THREADS, 0, stream>>>(chw, w, h, ainc[1] * ainc[2], ainc[2], a->data.f32, b->data.f32, pool_w, pool_h, cinc[1] * cinc[2], cinc[2], c->data.f32);
	} else {
		assert(a->info.format == CCV_TENSOR_FORMAT_NHWC);
		const int h = adim[0];
		const int w = adim[1];
		const int pool_h = cdim[0];
		const int pool_w = cdim[1];
		assert(cdim[2] == adim[2]);
		const int ch = cdim[2];
		const int hw = pool_h * pool_w;
		_ccv_nnc_roi_align_forw_nhwc<<<CUDA_GET_BLOCKS(hw), CUDA_NUM_THREADS, 0, stream>>>(hw, w, h, ch, ainc[1] * ainc[2], ainc[2], a->data.f32, b->data.f32, pool_w, pool_h, cinc[1] * cinc[2], cinc[2], c->data.f32);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_roi_align_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_SUCCESS;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_ROI_ALIGN_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDA
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_roi_align_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_ROI_ALIGN_BACKWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDA
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F; // Currently only support CCV_32F because atomicAdd only supports __half at sm_70. I will revisit this by either get rid of atomicAdd or deprecate support for Jetson Nano / TX2.
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_roi_align_back;
#endif
}
