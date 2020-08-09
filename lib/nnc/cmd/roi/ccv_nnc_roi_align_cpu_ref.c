#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#ifdef USE_OPENMP
#include <omp.h>
#endif
#ifdef USE_DISPATCH
#include <dispatch/dispatch.h>
#endif

typedef struct {
	int i0, i1, mute;
	float r;
} roi_align_coeffs_t;

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
	const int h = adim[0];
	const int w = adim[1];
	const int c_nd = ccv_nnc_tensor_nd(c->info.dim);
	assert(c_nd == CCV_NNC_MAX_DIM + 1 || c_nd == CCV_NNC_MAX_DIM + 2);
	const int* cdim = (c_nd == CCV_NNC_MAX_DIM + 1) ? c->info.dim : c->info.dim + 1;
	const int pool_h = cdim[0];
	const int pool_w = cdim[1];
	assert(cdim[2] == adim[2]);
	const int ch = cdim[2];
	float* ap = a->data.f32;
	const int* ainc = CCV_IS_TENSOR_VIEW(a) ? ((a_nd == CCV_NNC_MAX_DIM + 1) ?  a->inc : a->inc + 1) : adim;
	const float* const bp = b->data.f32;
	const float roi_x = bp[0] * w; // These assumed it is real-coordinate, and 0.5 is the center of a pixel.
	const float roi_y = bp[1] * h;
	const float roi_w = bp[2] * w;
	const float roi_h = bp[3] * h;
	float* cp = c->data.f32;
	const int* cinc = CCV_IS_TENSOR_VIEW(c) ? ((c_nd == CCV_NNC_MAX_DIM + 1) ?  c->inc : c->inc + 1) : cdim;
	const int bin_h = (int)ceilf(roi_h / pool_h); // How many bins in each point of the pool. We slightly sampling at higher resolution (due to ceiling) with bilinear interpolation.
	const int bin_w = (int)ceilf(roi_w / pool_w);
	const int bin_pool_h = bin_h * pool_h; // Before averaging, what's the size of the region in integral term.
	const int bin_pool_w = bin_w * pool_w;
	const float scale_y = roi_h / bin_pool_h; // The scale to multiply back to get original coordinate.
	const float scale_x = roi_w / bin_pool_w;
	ccv_nnc_tensor_zero(c);
	int x, y, i, j, k;
	roi_align_coeffs_t* const y_coeffs = (roi_align_coeffs_t*)ccv_nnc_stream_context_get_workspace(stream_context, sizeof(roi_align_coeffs_t) * (bin_pool_h + bin_pool_w) + sizeof(int) * (pool_h + pool_w), CCV_TENSOR_CPU_MEMORY);
	roi_align_coeffs_t* const x_coeffs = y_coeffs + bin_pool_h;
	int* const bin_h_at_y = (int*)(x_coeffs + bin_pool_w);
	int* const bin_w_at_x = bin_h_at_y + pool_h;
	for (i = 0; i < pool_h; i++)
	{
		const int pi = i * bin_h;
		int count = 0;
		for (y = 0; y < bin_h; y++)
		{
			const float ay = roi_y + ((y + pi + 0.5) * scale_y - 0.5) - 0.5; // Since from the beginning, we assumed the coordinate center is 0.5 (per pixel), - 0.5 would tell us which pixel to bind.
			const int iy = (int)ay;
			const float ry = ay - iy;
			const int iy0 = ccv_clamp(iy, 0, h - 1);
			const int iy1 = ccv_clamp(iy + 1, 0, h - 1);
			y_coeffs[pi + y].i0 = iy0;
			y_coeffs[pi + y].i1 = iy1;
			y_coeffs[pi + y].r = ry;
			const int mute = (iy + 1 < 0 || iy > h - 1);
			y_coeffs[pi + y].mute = mute;
			if (!mute)
				++count;
		}
		bin_h_at_y[i] = count;
	}
	int start_h = pool_h;
	for (i = 0; start_h == pool_h && i < pool_h; i++)
		if (bin_h_at_y[i] > 0)
			start_h = i;
	int end_h = 0;
	for (i = pool_h - 1; end_h == 0 && i >= 0; i--)
		if (bin_h_at_y[i] > 0)
			end_h = i + 1;
	for (j = 0; j < pool_w; j++)
	{
		const int pj = j * bin_w;
		int count = 0;
		for (x = 0; x < bin_w; x++)
		{
			const float ax = roi_x + ((x + pj + 0.5) * scale_x - 0.5) - 0.5;
			const int ix = (int)ax;
			const float rx = ax - ix;
			const int ix0 = ccv_clamp(ix, 0, w - 1);
			const int ix1 = ccv_clamp(ix + 1, 0, w - 1);
			x_coeffs[pj + x].i0 = ix0;
			x_coeffs[pj + x].i1 = ix1;
			x_coeffs[pj + x].r = rx;
			const int mute = (ix + 1 < 0 || ix > w - 1);
			x_coeffs[pj + x].mute = mute;
			if (!mute)
				++count;
		}
		bin_w_at_x[j] = count;
	}
	int start_w = pool_w;
	for (j = 0; start_w == pool_w && j < pool_w; j++)
		if (bin_w_at_x[j] > 0)
			start_w = j;
	int end_w = 0;
	for (j = pool_w - 1; end_w == 0 && j >= 0; j--)
		if (bin_w_at_x[j] > 0)
			end_w = j + 1;
	for (i = start_h; i < end_h; i++)
	{
		const int pi = i * bin_h;
		const int bin_hz = bin_h_at_y[i];
		for (j = start_w; j < end_w; j++)
		{
			const int pj = j * bin_w;
			const int bin_wz = bin_w_at_x[j];
			const float inv = 1.0 / (bin_hz * bin_wz);
			float* const cpz = cp + j * cinc[CCV_NNC_MAX_DIM];
			for (y = 0; y < bin_h; y++)
			{
				if (y_coeffs[pi + y].mute)
					continue;
				const float ry = y_coeffs[pi + y].r;
				const int iy0 = y_coeffs[pi + y].i0;
				const int iy1 = y_coeffs[pi + y].i1;
				for (x = 0; x < bin_w; x++)
				{
					if (x_coeffs[pj + x].mute)
						continue;
					const float rx = x_coeffs[pj + x].r;
					const int ix0 = x_coeffs[pj + x].i0;
					const int ix1 = x_coeffs[pj + x].i1;
					const float c00 = (1 - ry) * (1 - rx);
					const float c01 = (1 - ry) * rx;
					const float c10 = ry * (1 - rx);
					const float c11 = ry * rx;
					const float* const ap00 = ap + (iy0 * ainc[CCV_NNC_MAX_DIM - 1] + ix0) * ainc[CCV_NNC_MAX_DIM];
					const float* const ap01 = ap + (iy0 * ainc[CCV_NNC_MAX_DIM - 1] + ix1) * ainc[CCV_NNC_MAX_DIM];
					const float* const ap10 = ap + (iy1 * ainc[CCV_NNC_MAX_DIM - 1] + ix0) * ainc[CCV_NNC_MAX_DIM];
					const float* const ap11 = ap + (iy1 * ainc[CCV_NNC_MAX_DIM - 1] + ix1) * ainc[CCV_NNC_MAX_DIM];
					for (k = 0; k < ch; k++)
						cpz[k] += ap00[k] * c00 + ap01[k] * c01 + ap10[k] * c10 + ap11[k] * c11;
				}
			}
			for (k = 0; k < ch; k++)
				cpz[k] *= inv;
		}
		cp += cinc[CCV_NNC_MAX_DIM - 1] * cinc[CCV_NNC_MAX_DIM];
	}
	return CCV_NNC_EXEC_SUCCESS;
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
	const int pool_h = gdim[0];
	const int pool_w = gdim[1];
	const int o_nd = ccv_nnc_tensor_nd(o->info.dim);
	assert(o_nd == CCV_NNC_MAX_DIM + 1 || o_nd == CCV_NNC_MAX_DIM + 2);
	const int* odim = (o_nd == CCV_NNC_MAX_DIM + 1) ? o->info.dim : o->info.dim + 1;
	const int h = odim[0];
	const int w = odim[1];
	assert(gdim[2] == odim[2]);
	const int ch = gdim[2];
	float* gp = g->data.f32;
	const int* ginc = CCV_IS_TENSOR_VIEW(g) ? ((g_nd == CCV_NNC_MAX_DIM + 1) ? g->inc : g->inc + 1) : gdim;
	float* op = o->data.f32;
	const int* oinc = CCV_IS_TENSOR_VIEW(o) ? ((o_nd == CCV_NNC_MAX_DIM + 1) ? o->inc : o->inc + 1) : odim;
	const ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)inputs[2];
	const float* const bp = b->data.f32;
	const float roi_x = bp[0] * w; // These assumed it is real-coordinate, and 0.5 is the center of a pixel.
	const float roi_y = bp[1] * h;
	const float roi_w = bp[2] * w;
	const float roi_h = bp[3] * h;
	const int bin_h = (int)ceilf(roi_h / pool_h); // How many bins in each point of the pool. We slightly sampling at higher resolution (due to ceiling) with bilinear interpolation.
	const int bin_w = (int)ceilf(roi_w / pool_w);
	const int bin_pool_h = bin_h * pool_h; // Before averaging, what's the size of the region in integral term.
	const int bin_pool_w = bin_w * pool_w;
	const float scale_y = roi_h / bin_pool_h; // The scale to multiply back to get original coordinate.
	const float scale_x = roi_w / bin_pool_w;
	ccv_nnc_tensor_zero(o);
	int x, y, i, j, k;
	roi_align_coeffs_t* const y_coeffs = (roi_align_coeffs_t*)ccv_nnc_stream_context_get_workspace(stream_context, sizeof(roi_align_coeffs_t) * (bin_pool_h + bin_pool_w) + sizeof(int) * (pool_h + pool_w), CCV_TENSOR_CPU_MEMORY);
	roi_align_coeffs_t* const x_coeffs = y_coeffs + bin_pool_h;
	int* const bin_h_at_y = (int*)(x_coeffs + bin_pool_w);
	int* const bin_w_at_x = bin_h_at_y + pool_h;
	for (i = 0; i < pool_h; i++)
	{
		const int pi = i * bin_h;
		int count = 0;
		for (y = 0; y < bin_h; y++)
		{
			const float ay = roi_y + ((y + pi + 0.5) * scale_y - 0.5) - 0.5; // Since from the beginning, we assumed the coordinate center is 0.5 (per pixel), - 0.5 would tell us which pixel to bind.
			const int iy = (int)ay;
			const float ry = ay - iy;
			const int iy0 = ccv_clamp(iy, 0, h - 1);
			const int iy1 = ccv_clamp(iy + 1, 0, h - 1);
			y_coeffs[pi + y].i0 = iy0;
			y_coeffs[pi + y].i1 = iy1;
			y_coeffs[pi + y].r = ry;
			const int mute = (iy + 1 < 0 || iy > h - 1);
			y_coeffs[pi + y].mute = mute;
			if (!mute)
				++count;
		}
		bin_h_at_y[i] = count;
	}
	int start_h = pool_h;
	for (i = 0; start_h == pool_h && i < pool_h; i++)
		if (bin_h_at_y[i] > 0)
			start_h = i;
	int end_h = 0;
	for (i = pool_h - 1; end_h == 0 && i >= 0; i--)
		if (bin_h_at_y[i] > 0)
			end_h = i + 1;
	for (j = 0; j < pool_w; j++)
	{
		const int pj = j * bin_w;
		int count = 0;
		for (x = 0; x < bin_w; x++)
		{
			const float ax = roi_x + ((x + pj + 0.5) * scale_x - 0.5) - 0.5;
			const int ix = (int)ax;
			const float rx = ax - ix;
			const int ix0 = ccv_clamp(ix, 0, w - 1);
			const int ix1 = ccv_clamp(ix + 1, 0, w - 1);
			x_coeffs[pj + x].i0 = ix0;
			x_coeffs[pj + x].i1 = ix1;
			x_coeffs[pj + x].r = rx;
			const int mute = (ix + 1 < 0 || ix > w - 1);
			x_coeffs[pj + x].mute = mute;
			if (!mute)
				++count;
		}
		bin_w_at_x[j] = count;
	}
	int start_w = pool_w;
	for (j = 0; start_w == pool_w && j < pool_w; j++)
		if (bin_w_at_x[j] > 0)
			start_w = j;
	int end_w = 0;
	for (j = pool_w - 1; end_w == 0 && j >= 0; j--)
		if (bin_w_at_x[j] > 0)
			end_w = j + 1;
	for (i = 0; i < pool_h; i++)
	{
		const int pi = i * bin_h;
		const int bin_hz = bin_h_at_y[i];
		for (j = 0; j < pool_w; j++)
		{
			const int pj = j * bin_w;
			const int bin_wz = bin_w_at_x[j];
			const float inv = 1.0 / (bin_hz * bin_wz);
			const float* const gpz = gp + j * ginc[CCV_NNC_MAX_DIM];
			for (y = 0; y < bin_h; y++)
			{
				if (y_coeffs[pi + y].mute)
					continue;
				const float ry = y_coeffs[pi + y].r;
				const int iy0 = y_coeffs[pi + y].i0;
				const int iy1 = y_coeffs[pi + y].i1;
				for (x = 0; x < bin_w; x++)
				{
					if (x_coeffs[pj + x].mute)
						continue;
					const float rx = x_coeffs[pj + x].r;
					const int ix0 = x_coeffs[pj + x].i0;
					const int ix1 = x_coeffs[pj + x].i1;
					const float c00 = (1 - ry) * (1 - rx);
					const float c01 = (1 - ry) * rx;
					const float c10 = ry * (1 - rx);
					const float c11 = ry * rx;
					float* const op00 = op + (iy0 * oinc[CCV_NNC_MAX_DIM - 1] + ix0) * oinc[CCV_NNC_MAX_DIM];
					float* const op01 = op + (iy0 * oinc[CCV_NNC_MAX_DIM - 1] + ix1) * oinc[CCV_NNC_MAX_DIM];
					float* const op10 = op + (iy1 * oinc[CCV_NNC_MAX_DIM - 1] + ix0) * oinc[CCV_NNC_MAX_DIM];
					float* const op11 = op + (iy1 * oinc[CCV_NNC_MAX_DIM - 1] + ix1) * oinc[CCV_NNC_MAX_DIM];
					for (k = 0; k < ch; k++)
					{
						op00[k] += gpz[k] * c00 * inv;
						op01[k] += gpz[k] * c01 * inv;
						op10[k] += gpz[k] * c10 * inv;
						op11[k] += gpz[k] * c11 * inv;
					}
				}
			}
		}
		gp += ginc[CCV_NNC_MAX_DIM - 1] * ginc[CCV_NNC_MAX_DIM];
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_ROI_ALIGN_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_roi_align_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_ROI_ALIGN_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_roi_align_back;
}
