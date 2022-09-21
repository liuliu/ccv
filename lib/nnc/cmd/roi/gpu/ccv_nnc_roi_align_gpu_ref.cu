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
__global__ void _ccv_nnc_roi_align_forw_nchw(const int nchw, const int ch, const int w, const int h, const int a_n, const int adim0, const int adim1, const int adim2, const NUM* const ap, const int b_n, const int bdim0, const NUM* const bp, const int pool_w, const int pool_h, const int cdim0, const int cdim1, const int cdim2, NUM* const cp)
{
	const int pool_chw = ch * pool_h * pool_w;
	const int pool_hw = pool_h * pool_w;
	CUDA_1D_KERNEL_LOOP(i, nchw) {
		const int n = i / pool_chw;
		const int cxy = i % pool_chw;
		const int k = cxy / pool_hw;
		const int xy = cxy % pool_hw;
		const int y = xy / pool_w;
		const int x = xy % pool_w;
		const float roi_x = bp[(n % b_n) * bdim0] * w; // These assumed it is real-coordinate, with range between 0 to w - 1.
		const float roi_y = bp[(n % b_n) * bdim0 + 1] * h;
		const float roi_w = bp[(n % b_n) * bdim0 + 2] * w;
		const float roi_h = bp[(n % b_n) * bdim0 + 3] * h;
		const int bin_h = (int)ceilf(roi_h / pool_h); // How many bins in each point of the pool. We slightly sampling at higher resolution (due to ceiling) with bilinear interpolation.
		const int bin_w = (int)ceilf(roi_w / pool_w);
		const int bin_pool_h = bin_h * pool_h; // Before averaging, what's the size of the region in integral term.
		const int bin_pool_w = bin_w * pool_w;
		const float scale_y = roi_h / bin_pool_h; // The scale to multiply back to get original coordinate.
		const float scale_x = roi_w / bin_pool_w;
		const int py = y * bin_h;
		const int px = x * bin_w;
		float v = 0;
		int count = 0;
		const float* const apz = ap + (n % a_n) * adim0 + k * adim1;
		for (int by = 0; by < bin_h; by++)
		{
			const float ay = roi_y + (by + py + 0.5) * scale_y - 0.5;
			const int iy = (int)floorf(ay);
			if (iy + 1 < 0 || iy > h - 1)
				continue;
			const float ry = ay - iy;
			const int iy0 = ccv_clamp(iy, 0, h - 1);
			const int iy1 = ccv_clamp(iy + 1, 0, h - 1);
			for (int bx = 0; bx < bin_w; bx++)
			{
				const float ax = roi_x + (bx + px + 0.5) * scale_x - 0.5;
				const int ix = (int)floorf(ax);
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
		cp[n * cdim0 + k * cdim1 + y * cdim2 + x] = count > 0 ? v / count : 0;
	}
}

template<typename NUM>
__global__ void _ccv_nnc_roi_align_forw_nhwc(const int nchw, const int ch, const int w, const int h, const int a_n, const int adim0, const int adim1, const int adim2, const NUM* const ap, const int b_n, const int bdim0, const NUM* const bp, const int pool_w, const int pool_h, const int cdim0, const int cdim1, const int cdim2, NUM* const cp)
{
	const int pool_chw = ch * pool_h * pool_w;
	const int pool_hw = pool_h * pool_w;
	CUDA_1D_KERNEL_LOOP(i, nchw) {
		const int n = i / pool_chw;
		const int cxy = i % pool_chw;
		const int k = cxy / pool_hw;
		const int xy = cxy % pool_hw;
		const int y = xy / pool_w;
		const int x = xy % pool_w;
		const float roi_x = bp[(n % b_n) * bdim0] * w; // These assumed it is real-coordinate, with range between 0 to w - 1.
		const float roi_y = bp[(n % b_n) * bdim0 + 1] * h;
		const float roi_w = bp[(n % b_n) * bdim0 + 2] * w;
		const float roi_h = bp[(n % b_n) * bdim0 + 3] * h;
		const int bin_h = (int)ceilf(roi_h / pool_h); // How many bins in each point of the pool. We slightly sampling at higher resolution (due to ceiling) with bilinear interpolation.
		const int bin_w = (int)ceilf(roi_w / pool_w);
		const int bin_pool_h = bin_h * pool_h; // Before averaging, what's the size of the region in integral term.
		const int bin_pool_w = bin_w * pool_w;
		const float scale_y = roi_h / bin_pool_h; // The scale to multiply back to get original coordinate.
		const float scale_x = roi_w / bin_pool_w;
		const int py = y * bin_h;
		const int px = x * bin_w;
		float v = 0;
		int count = 0;
		const float* const apz = ap + (n % a_n) * adim0 + k;
		for (int by = 0; by < bin_h; by++)
		{
			const float ay = roi_y + (by + py + 0.5) * scale_y - 0.5;
			const int iy = (int)floorf(ay);
			if (iy + 1 < 0 || iy > h - 1)
				continue;
			const float ry = ay - iy;
			const int iy0 = ccv_clamp(iy, 0, h - 1);
			const int iy1 = ccv_clamp(iy + 1, 0, h - 1);
			for (int bx = 0; bx < bin_w; bx++)
			{
				const float ax = roi_x + (bx + px + 0.5) * scale_x - 0.5;
				const int ix = (int)floorf(ax);
				if (ix + 1 < 0 || ix > w - 1)
					continue;
				const float rx = ax - ix;
				const int ix0 = ccv_clamp(ix, 0, w - 1);
				const int ix1 = ccv_clamp(ix + 1, 0, w - 1);
				const float c00 = (1 - ry) * (1 - rx);
				const float c01 = (1 - ry) * rx;
				const float c10 = ry * (1 - rx);
				const float c11 = ry * rx;
				const float ap00 = apz[iy0 * adim1 + ix0 * adim2];
				const float ap01 = apz[iy0 * adim1 + ix1 * adim2];
				const float ap10 = apz[iy1 * adim1 + ix0 * adim2];
				const float ap11 = apz[iy1 * adim1 + ix1 * adim2];
				v += ap00 * c00 + ap01 * c01 + ap10 * c10 + ap11 * c11;
				++count;
			}
		}
		cp[n * cdim0 + y * cdim1 + x * cdim2 + k] = count > 0 ? v / count : 0;
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
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_stride(a, astride);
	int cstride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_stride(c, cstride);
	const int a_n = ccv_nnc_tensor_get_n(a->info);
	const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
	assert(b_nd == 1 || b_nd == 2);
	const int b_n = b_nd == 1 ? 1 : b->info.dim[0];
	const int c_n = ccv_nnc_tensor_get_n(c->info);
	assert(c_n == ccv_max(a_n, b_n));
	const int aninc = a_nd == CCV_NNC_MAX_DIM + 1 ? 0 : astride[0];
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_stride(b, bstride);
	const int bninc = b_nd == 1 ? 0 : bstride[CCV_NNC_MAX_DIM + 2 - b_nd];
	const int cninc = c_nd == CCV_NNC_MAX_DIM + 1 ? 0 : cstride[0];
	assert(a->info.format == c->info.format);
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	if (a->info.format == CCV_TENSOR_FORMAT_NCHW)
	{
		const int h = adim[1];
		const int w = adim[2];
		const int pool_h = cdim[1];
		const int pool_w = cdim[2];
		assert(cdim[0] == adim[0]);
		const int ch = cdim[0];
		const int nchw = c_n * pool_h * pool_w * ch;
		_ccv_nnc_roi_align_forw_nchw<<<CUDA_GET_BLOCKS(nchw), CUDA_NUM_THREADS, 0, stream>>>(nchw, ch, w, h, a_n, aninc, astride[1], astride[2], a->data.f32, b_n, bninc, b->data.f32, pool_w, pool_h, cninc, cstride[1], cstride[2], c->data.f32);
	} else {
		assert(a->info.format == CCV_TENSOR_FORMAT_NHWC);
		const int h = adim[0];
		const int w = adim[1];
		const int pool_h = cdim[0];
		const int pool_w = cdim[1];
		assert(cdim[2] == adim[2]);
		const int ch = cdim[2];
		const int nchw = c_n * pool_h * pool_w * ch;
		_ccv_nnc_roi_align_forw_nhwc<<<CUDA_GET_BLOCKS(nchw), CUDA_NUM_THREADS, 0, stream>>>(nchw, ch, w, h, a_n, aninc, astride[1], astride[2], a->data.f32, b_n, bninc, b->data.f32, pool_w, pool_h, cninc, cstride[1], cstride[2], c->data.f32);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

template<typename NUM>
__global__ void _ccv_nnc_zero_back(const size_t tensor_count, NUM* const a)
{
	CUDA_1D_KERNEL_LOOP(i, tensor_count) {
		a[i] = 0;
	}
}

template<typename NUM>
__global__ void _ccv_nnc_roi_align_back_nchw(const int nchw, const int ch, const int w, const int h, const int o_n, const int odim0, const int odim1, const int odim2, NUM* const op, const int b_n, const int bdim0, const NUM* const bp, const int pool_w, const int pool_h, const int gdim0, const int gdim1, const int gdim2, const NUM* const gp)
{
	const int pool_chw = ch * pool_h * pool_w;
	const int pool_hw = pool_h * pool_w;
	CUDA_1D_KERNEL_LOOP(i, nchw) {
		const int n = i / pool_chw;
		const int cxy = i % pool_chw;
		const int k = cxy / pool_hw;
		const int xy = cxy % pool_hw;
		const int y = xy / pool_w;
		const int x = xy % pool_w;
		const float roi_x = bp[(n % b_n) * bdim0] * w; // These assumed it is real-coordinate, with range between 0 to w - 1.
		const float roi_y = bp[(n % b_n) * bdim0 + 1] * h;
		const float roi_w = bp[(n % b_n) * bdim0 + 2] * w;
		const float roi_h = bp[(n % b_n) * bdim0 + 3] * h;
		const int bin_h = (int)ceilf(roi_h / pool_h); // How many bins in each point of the pool. We slightly sampling at higher resolution (due to ceiling) with bilinear interpolation.
		const int bin_w = (int)ceilf(roi_w / pool_w);
		const int bin_pool_h = bin_h * pool_h; // Before averaging, what's the size of the region in integral term.
		const int bin_pool_w = bin_w * pool_w;
		const float scale_y = roi_h / bin_pool_h; // The scale to multiply back to get original coordinate.
		const float scale_x = roi_w / bin_pool_w;
		const int py = y * bin_h;
		const int px = x * bin_w;
		float* const opz = op + (n % o_n) * odim0 + k * odim1;
		// Need to be careful about float-point accuracy. For both min / max, we started with edge case:
		// what if I am at h already for iy? In that case, if we exceed even a little, we tip over, hence, floor.
		// What if I am at -1 already for iy? In that case, if we exceed even a little, we tip over, hence, ceil.
		// Notice that for upper limit, I use h? If I use h - 1, it is OK to tip over, because we floor(ay), hence,
		// the extra will be trimmed. Only if I am so close to h, tip over is not acceptable.
		const int bin_h_at_y = ccv_min(bin_h - 1, (int)floorf((h + 0.5 - roi_y) / scale_y - 0.5 - py)) - ccv_max(0, (int)ceilf((-0.5 - roi_y) / scale_y - 0.5 - py)) + 1;
		const int bin_w_at_x = ccv_min(bin_w - 1, (int)floorf((w + 0.5 - roi_x) / scale_x - 0.5 - px)) - ccv_max(0, (int)ceilf((-0.5 - roi_x) / scale_x - 0.5 - px)) + 1;
		const int count = ccv_max(0, bin_h_at_y) * ccv_max(0, bin_w_at_x);
		const float v = count > 0 ? gp[n * gdim0 + k * gdim1 + y * gdim2 + x] / count : 0;
		for (int by = 0; by < bin_h; by++)
		{
			const float ay = roi_y + (by + py + 0.5) * scale_y - 0.5;
			const int iy = (int)floorf(ay);
			if (iy + 1 < 0 || iy > h - 1)
				continue;
			const float ry = ay - iy;
			const int iy0 = ccv_clamp(iy, 0, h - 1);
			const int iy1 = ccv_clamp(iy + 1, 0, h - 1);
			for (int bx = 0; bx < bin_w; bx++)
			{
				const float ax = roi_x + (bx + px + 0.5) * scale_x - 0.5;
				const int ix = (int)floorf(ax);
				if (ix + 1 < 0 || ix > w - 1)
					continue;
				const float rx = ax - ix;
				const int ix0 = ccv_clamp(ix, 0, w - 1);
				const int ix1 = ccv_clamp(ix + 1, 0, w - 1);
				const float c00 = (1 - ry) * (1 - rx);
				const float c01 = (1 - ry) * rx;
				const float c10 = ry * (1 - rx);
				const float c11 = ry * rx;
				atomicAdd(&opz[iy0 * odim2 + ix0], (NUM)(v * c00));
				atomicAdd(&opz[iy0 * odim2 + ix1], (NUM)(v * c01));
				atomicAdd(&opz[iy1 * odim2 + ix0], (NUM)(v * c10));
				atomicAdd(&opz[iy1 * odim2 + ix1], (NUM)(v * c11));
			}
		}
	}
}

template<typename NUM>
__global__ void _ccv_nnc_roi_align_back_nhwc(const int nchw, const int ch, const int w, const int h, const int o_n, const int odim0, const int odim1, const int odim2, NUM* const op, const int b_n, const int bdim0, const NUM* const bp, const int pool_w, const int pool_h, const int gdim0, const int gdim1, const int gdim2, const NUM* const gp)
{
	const int pool_chw = ch * pool_h * pool_w;
	const int pool_hw = pool_h * pool_w;
	CUDA_1D_KERNEL_LOOP(i, nchw) {
		const int n = i / pool_chw;
		const int cxy = i % pool_chw;
		const int k = cxy / pool_hw;
		const int xy = cxy % pool_hw;
		const int y = xy / pool_w;
		const int x = xy % pool_w;
		const float roi_x = bp[(n % b_n) * bdim0] * w; // These assumed it is real-coordinate, with range between 0 to w - 1.
		const float roi_y = bp[(n % b_n) * bdim0 + 1] * h;
		const float roi_w = bp[(n % b_n) * bdim0 + 2] * w;
		const float roi_h = bp[(n % b_n) * bdim0 + 3] * h;
		const int bin_h = (int)ceilf(roi_h / pool_h); // How many bins in each point of the pool. We slightly sampling at higher resolution (due to ceiling) with bilinear interpolation.
		const int bin_w = (int)ceilf(roi_w / pool_w);
		const int bin_pool_h = bin_h * pool_h; // Before averaging, what's the size of the region in integral term.
		const int bin_pool_w = bin_w * pool_w;
		const float scale_y = roi_h / bin_pool_h; // The scale to multiply back to get original coordinate.
		const float scale_x = roi_w / bin_pool_w;
		const int py = y * bin_h;
		const int px = x * bin_w;
		float* const opz = op + (n % o_n) * odim0 + k;
		const int bin_h_at_y = ccv_min(bin_h - 1, (int)floorf((h + 0.5 - roi_y) / scale_y - 0.5 - py)) - ccv_max(0, (int)ceilf((-0.5 - roi_y) / scale_y - 0.5 - py)) + 1;
		const int bin_w_at_x = ccv_min(bin_w - 1, (int)floorf((w + 0.5 - roi_x) / scale_x - 0.5 - px)) - ccv_max(0, (int)ceilf((-0.5 - roi_x) / scale_x - 0.5 - px)) + 1;
		const int count = ccv_max(0, bin_h_at_y) * ccv_max(0, bin_w_at_x);
		const float v = count > 0 ? gp[n * gdim0 + y * gdim1 + x * gdim2 + k] / count : 0;
		for (int by = 0; by < bin_h; by++)
		{
			const float ay = roi_y + (by + py + 0.5) * scale_y - 0.5;
			const int iy = (int)floorf(ay);
			if (iy + 1 < 0 || iy > h - 1)
				continue;
			const float ry = ay - iy;
			const int iy0 = ccv_clamp(iy, 0, h - 1);
			const int iy1 = ccv_clamp(iy + 1, 0, h - 1);
			for (int bx = 0; bx < bin_w; bx++)
			{
				const float ax = roi_x + (bx + px + 0.5) * scale_x - 0.5;
				const int ix = (int)floorf(ax);
				if (ix + 1 < 0 || ix > w - 1)
					continue;
				const float rx = ax - ix;
				const int ix0 = ccv_clamp(ix, 0, w - 1);
				const int ix1 = ccv_clamp(ix + 1, 0, w - 1);
				const float c00 = (1 - ry) * (1 - rx);
				const float c01 = (1 - ry) * rx;
				const float c10 = ry * (1 - rx);
				const float c11 = ry * rx;
				atomicAdd(&opz[iy0 * odim1 + ix0 * odim2], (NUM)(v * c00));
				atomicAdd(&opz[iy0 * odim1 + ix1 * odim2], (NUM)(v * c01));
				atomicAdd(&opz[iy1 * odim1 + ix0 * odim2], (NUM)(v * c10));
				atomicAdd(&opz[iy1 * odim1 + ix1 * odim2], (NUM)(v * c11));
			}
		}
	}
}

static int _ccv_nnc_roi_align_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 3);
	const ccv_nnc_tensor_view_t* g = (ccv_nnc_tensor_view_t*)inputs[0];
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* o = (ccv_nnc_tensor_view_t*)outputs[0];
	const int g_nd = ccv_nnc_tensor_nd(g->info.dim);
	assert(g_nd == CCV_NNC_MAX_DIM + 1 || g_nd == CCV_NNC_MAX_DIM + 2);
	const int* gdim = (g_nd == CCV_NNC_MAX_DIM + 1) ? g->info.dim : g->info.dim + 1;
	const int o_nd = ccv_nnc_tensor_nd(o->info.dim);
	assert(o_nd == CCV_NNC_MAX_DIM + 1 || o_nd == CCV_NNC_MAX_DIM + 2);
	const int* odim = (o_nd == CCV_NNC_MAX_DIM + 1) ? o->info.dim : o->info.dim + 1;
	int gstride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_stride(g, gstride);
	int ostride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_stride(o, ostride);
	const ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)inputs[2];
	const int o_n = ccv_nnc_tensor_get_n(o->info);
	const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
	assert(b_nd == 1 || b_nd == 2);
	const int b_n = b_nd == 1 ? 1 : b->info.dim[0];
	const int g_n = ccv_nnc_tensor_get_n(g->info);
	assert(g_n == ccv_max(o_n, b_n));
	const int oninc = o_nd == CCV_NNC_MAX_DIM + 1 ? 0 : ostride[0];
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_stride(b, bstride);
	const int bninc = b_nd == 1 ? 0 : bstride[CCV_NNC_MAX_DIM + 2 - b_nd];
	const int gninc = g_nd == CCV_NNC_MAX_DIM + 1 ? 0 : gstride[0];
	assert(g->info.format == o->info.format);
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	const size_t o_tensor_count = ccv_nnc_tensor_count(o->info);
	_ccv_nnc_zero_back<<<CUDA_GET_BLOCKS(o_tensor_count), CUDA_NUM_THREADS, 0, stream>>>(o_tensor_count, o->data.f32);
	if (o->info.format == CCV_TENSOR_FORMAT_NCHW)
	{
		const int h = odim[1];
		const int w = odim[2];
		const int pool_h = gdim[1];
		const int pool_w = gdim[2];
		assert(gdim[0] == odim[0]);
		const int ch = gdim[0];
		const int nchw = g_n * pool_h * pool_w * ch;
		_ccv_nnc_roi_align_back_nchw<<<CUDA_GET_BLOCKS(nchw), CUDA_NUM_THREADS, 0, stream>>>(nchw, ch, w, h, o_n, oninc, ostride[1], ostride[2], o->data.f32, b_n, bninc, b->data.f32, pool_w, pool_h, gninc, gstride[1], gstride[2], g->data.f32);
	} else {
		assert(o->info.format == CCV_TENSOR_FORMAT_NHWC);
		const int h = odim[0];
		const int w = odim[1];
		const int pool_h = gdim[0];
		const int pool_w = gdim[1];
		assert(gdim[2] == odim[2]);
		const int ch = gdim[2];
		const int nchw = g_n * pool_h * pool_w * ch;
		_ccv_nnc_roi_align_back_nhwc<<<CUDA_GET_BLOCKS(nchw), CUDA_NUM_THREADS, 0, stream>>>(nchw, ch, w, h, o_n, oninc, ostride[1], ostride[2], o->data.f32, b_n, bninc, b->data.f32, pool_w, pool_h, gninc, gstride[1], gstride[2], g->data.f32);
	}
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
