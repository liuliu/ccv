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

static void _ccv_nnc_bilinear_coeffs(ccv_nnc_stream_context_t* const stream_context, const int h, const int w, const float roi_y, const float roi_x, const float roi_h, const float roi_w, const int pool_h, const int pool_w, int* const bin_h_ref, int* const bin_w_ref, roi_align_coeffs_t** const y_coeffs_ref, roi_align_coeffs_t** const x_coeffs_ref, int** const bin_h_at_y_ref, int** const bin_w_at_x_ref, int* const start_h_ref, int* const start_w_ref, int* const end_h_ref, int* const end_w_ref)
{
	const int bin_h = (int)ceilf(roi_h / pool_h); // How many bins in each point of the pool. We slightly sampling at higher resolution (due to ceiling) with bilinear interpolation.
	const int bin_w = (int)ceilf(roi_w / pool_w);
	const int bin_pool_h = bin_h * pool_h; // Before averaging, what's the size of the region in integral term.
	const int bin_pool_w = bin_w * pool_w;
	const float scale_y = roi_h / bin_pool_h; // The scale to multiply back to get original coordinate.
	const float scale_x = roi_w / bin_pool_w;
	int x, y, i, j;
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
			const float ay = roi_y + (y + pi + 0.5) * scale_y - 0.5;
			const int iy = (int)floorf(ay);
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
			const float ax = roi_x + (x + pj + 0.5) * scale_x - 0.5;
			const int ix = (int)floorf(ax);
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
	*bin_h_ref = bin_h;
	*bin_w_ref = bin_w;
	*y_coeffs_ref = y_coeffs;
	*x_coeffs_ref = x_coeffs;
	*bin_h_at_y_ref = bin_h_at_y;
	*bin_w_at_x_ref = bin_w_at_x;
	*start_h_ref = start_h;
	*start_w_ref = start_w;
	*end_h_ref = end_h;
	*end_w_ref = end_w;
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
	const int h = adim[0];
	const int w = adim[1];
	const int c_nd = ccv_nnc_tensor_nd(c->info.dim);
	assert(c_nd == CCV_NNC_MAX_DIM + 1 || c_nd == CCV_NNC_MAX_DIM + 2);
	const int* cdim = (c_nd == CCV_NNC_MAX_DIM + 1) ? c->info.dim : c->info.dim + 1;
	const int pool_h = cdim[0];
	const int pool_w = cdim[1];
	assert(cdim[2] == adim[2]);
	const int ch = cdim[2];
	const float* const ap = a->data.f32;
	const int* ainc = CCV_IS_TENSOR_VIEW(a) ? ((a_nd == CCV_NNC_MAX_DIM + 1) ?  a->inc : a->inc + 1) : adim;
	const float* const bp = b->data.f32;
	float* cp = c->data.f32;
	const int* cinc = CCV_IS_TENSOR_VIEW(c) ? ((c_nd == CCV_NNC_MAX_DIM + 1) ?  c->inc : c->inc + 1) : cdim;
	const int a_n = ccv_nnc_tensor_get_n(a->info);
	const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
	assert(b_nd == 1 || b_nd == 2);
	const int b_n = b_nd == 1 ? 1 : b->info.dim[0];
	const int c_n = ccv_nnc_tensor_get_n(c->info);
	assert(c_n == ccv_max(a_n, b_n));
	const int aninc = a_nd == CCV_NNC_MAX_DIM + 1 ? 0 : ainc[0] * ainc[1] * ainc[2];
	const int* binc = CCV_IS_TENSOR_VIEW(b) ? b->inc : b->info.dim;
	const int bninc = b_nd == 1 ? 0 : binc[1];
	const int cninc = c_nd == CCV_NNC_MAX_DIM + 1 ? 0 : cinc[0] * cinc[1] * cinc[2];
	ccv_nnc_tensor_zero(c);
	int bin_h, bin_w;
	roi_align_coeffs_t* y_coeffs;
	roi_align_coeffs_t* x_coeffs;
	int* bin_h_at_y;
	int* bin_w_at_x;
	int start_h, start_w, end_h, end_w;
	int n;
	for (n = 0; n < c_n; n++)
	{
		const float* const apn = ap + (n % a_n) * aninc;
		float* cpn = cp + n * cninc;
		const float roi_x = bp[(n % b_n) * bninc] * w; // These assumed it is real-coordinate, with range between 0 to w - 1.
		const float roi_y = bp[(n % b_n) * bninc + 1] * h;
		const float roi_w = bp[(n % b_n) * bninc + 2] * w;
		const float roi_h = bp[(n % b_n) * bninc + 3] * h;
		// Re-compute the offsets if b changes or it is the first time.
		if ((b_n == 1 && n == 0) || b_n > 1)
			_ccv_nnc_bilinear_coeffs(stream_context, h, w, roi_y, roi_x, roi_h, roi_w, pool_h, pool_w, &bin_h, &bin_w, &y_coeffs, &x_coeffs, &bin_h_at_y, &bin_w_at_x, &start_h, &start_w, &end_h, &end_w);
		int i, j, x, y, k;
		for (i = start_h; i < end_h; i++)
		{
			const int pi = i * bin_h;
			const int bin_hz = bin_h_at_y[i];
			for (j = start_w; j < end_w; j++)
			{
				const int pj = j * bin_w;
				const int bin_wz = bin_w_at_x[j];
				const float inv = 1.0 / (bin_hz * bin_wz);
				float* const cpz = cpn + j * cinc[CCV_NNC_MAX_DIM];
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
						const float* const ap00 = apn + (iy0 * ainc[CCV_NNC_MAX_DIM - 1] + ix0) * ainc[CCV_NNC_MAX_DIM];
						const float* const ap01 = apn + (iy0 * ainc[CCV_NNC_MAX_DIM - 1] + ix1) * ainc[CCV_NNC_MAX_DIM];
						const float* const ap10 = apn + (iy1 * ainc[CCV_NNC_MAX_DIM - 1] + ix0) * ainc[CCV_NNC_MAX_DIM];
						const float* const ap11 = apn + (iy1 * ainc[CCV_NNC_MAX_DIM - 1] + ix1) * ainc[CCV_NNC_MAX_DIM];
						for (k = 0; k < ch; k++)
							cpz[k] += ap00[k] * c00 + ap01[k] * c01 + ap10[k] * c10 + ap11[k] * c11;
					}
				}
				for (k = 0; k < ch; k++)
					cpz[k] *= inv;
			}
			cpn += cinc[CCV_NNC_MAX_DIM - 1] * cinc[CCV_NNC_MAX_DIM];
		}
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
	const int o_n = ccv_nnc_tensor_get_n(o->info);
	const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
	assert(b_nd == 1 || b_nd == 2);
	const int b_n = b_nd == 1 ? 1 : b->info.dim[0];
	const int g_n = ccv_nnc_tensor_get_n(g->info);
	assert(g_n == ccv_max(o_n, b_n));
	const int oninc = o_nd == CCV_NNC_MAX_DIM + 1 ? 0 : oinc[0] * oinc[1] * oinc[2];
	const int* binc = CCV_IS_TENSOR_VIEW(b) ? b->inc : b->info.dim;
	const int bninc = b_nd == 1 ? 0 : binc[1];
	const int gninc = g_nd == CCV_NNC_MAX_DIM + 1 ? 0 : ginc[0] * ginc[1] * ginc[2];
	int bin_h, bin_w;
	roi_align_coeffs_t* y_coeffs;
	roi_align_coeffs_t* x_coeffs;
	int* bin_h_at_y;
	int* bin_w_at_x;
	int start_h, start_w, end_h, end_w;
	int n;
	ccv_nnc_tensor_zero(o);
	for (n = 0; n < g_n; n++)
	{
		const float roi_x = bp[(n % b_n) * bninc] * w; // These assumed it is real-coordinate, with range between 0 to w - 1.
		const float roi_y = bp[(n % b_n) * bninc + 1] * h;
		const float roi_w = bp[(n % b_n) * bninc + 2] * w;
		const float roi_h = bp[(n % b_n) * bninc + 3] * h;
		// Re-compute the offsets if b changes or it is the first time.
		if ((b_n == 1 && n == 0) || b_n > 1)
			_ccv_nnc_bilinear_coeffs(stream_context, h, w, roi_y, roi_x, roi_h, roi_w, pool_h, pool_w, &bin_h, &bin_w, &y_coeffs, &x_coeffs, &bin_h_at_y, &bin_w_at_x, &start_h, &start_w, &end_h, &end_w);
		const float* gpn = gp + n * gninc;
		float* const opn = op + (n % o_n) * oninc;
		int x, y, i, j, k;
		for (i = 0; i < pool_h; i++)
		{
			const int pi = i * bin_h;
			const int bin_hz = bin_h_at_y[i];
			for (j = 0; j < pool_w; j++)
			{
				const int pj = j * bin_w;
				const int bin_wz = bin_w_at_x[j];
				const float inv = 1.0 / (bin_hz * bin_wz);
				const float* const gpz = gpn + j * ginc[CCV_NNC_MAX_DIM];
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
						float* const op00 = opn + (iy0 * oinc[CCV_NNC_MAX_DIM - 1] + ix0) * oinc[CCV_NNC_MAX_DIM];
						float* const op01 = opn + (iy0 * oinc[CCV_NNC_MAX_DIM - 1] + ix1) * oinc[CCV_NNC_MAX_DIM];
						float* const op10 = opn + (iy1 * oinc[CCV_NNC_MAX_DIM - 1] + ix0) * oinc[CCV_NNC_MAX_DIM];
						float* const op11 = opn + (iy1 * oinc[CCV_NNC_MAX_DIM - 1] + ix1) * oinc[CCV_NNC_MAX_DIM];
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
			gpn += ginc[CCV_NNC_MAX_DIM - 1] * ginc[CCV_NNC_MAX_DIM];
		}
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
