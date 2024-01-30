#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#ifdef USE_OPENMP
#include <omp.h>
#endif
#ifdef USE_DISPATCH
#include <dispatch/dispatch.h>
#endif

// Shared methods.
#include "../_ccv_nnc_cpu_ref.h"

static int _ccv_nnc_upsample_nearest_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 1);
	assert(output_size >= 1);
	ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(ccv_nnc_tensor_nd(a->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(b->info.dim) <= CCV_NNC_MAX_DIM + 2);
	// Assuming this is float 32.
	int adim[CCV_NNC_MAX_DIM_ALLOC];
	int bdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(a, adim);
	ccv_nnc_tensor_view_get_dim(b, bdim);
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	ccv_nnc_tensor_view_get_stride(a, astride);
	ccv_nnc_tensor_view_get_stride(b, bstride);
	int i[CCV_NNC_MAX_DIM + 2];
	int xd, yd, cd;
	const float* ap = a->data.f32;
	float* const bp = b->data.f32;
	assert(a->info.format == b->info.format);
	const int align_corners = cmd.info.upsample.align_corners;
	if (a->info.format == CCV_TENSOR_FORMAT_NCHW)
	{
		const float rheight = align_corners ? (float)(adim[2] - 1) / ccv_max(1, bdim[2] - 1) : (float)adim[2] / bdim[2];
		const float rwidth = align_corners ? (float)(adim[3] - 1) / ccv_max(1, bdim[3] - 1) : (float)adim[3] / bdim[3];
		assert(rheight <= 1);
		assert(rwidth <= 1);
		int* const xcoeff = (int*)ccv_nnc_stream_context_get_workspace(stream_context, sizeof(int) * (bdim[3]), CCV_TENSOR_CPU_MEMORY);
		for (xd = 0; xd < bdim[3]; xd++)
			xcoeff[xd] = ccv_min(align_corners ? (int)(xd * rwidth + 0.5) : (int)((xd + 0.5) * rwidth), adim[3] - 1);
		assert(adim[0] == bdim[0]);
		assert(adim[1] == bdim[1]);
		for (i[0] = 0; i[0] < adim[0]; i[0]++)
		{
			const float* ap0 = ap + i[0] * astride[0];
			float* const bp0 = bp + i[0] * bstride[0];
			for (i[1] = 0; i[1] < adim[1]; i[1]++)
			{
				int pysi0 = 0;
				const float* ap1 = ap0;
				float* bp1 = bp0 + i[1] * bstride[1];
				for (yd = 0; yd < bdim[2]; yd++)
				{
					const int ysi0 = ccv_min(align_corners ? (int)(yd * rheight + 0.5) : (int)((yd + 0.5) * rheight), adim[2] - 1);
					if (pysi0 < ysi0) // Move to ay1 line.
					{
						ap1 += (ysi0 - pysi0) * astride[2];
						pysi0 = ysi0;
					}
					for (xd = 0; xd < bdim[3]; xd++)
						bp1[xd] = ap1[xcoeff[xd]];
					bp1 += bstride[2];
				}
				ap0 += astride[1];
			}
		}
	} else {
		// Any case, this is either NHWC or CHWN
		assert(a->info.format == CCV_TENSOR_FORMAT_NHWC || a->info.format == CCV_TENSOR_FORMAT_CHWN);
		const float rheight = align_corners ? (float)(adim[1] - 1) / ccv_max(1, bdim[1] - 1) : (float)adim[1] / bdim[1];
		const float rwidth = align_corners ? (float)(adim[2] - 1) / ccv_max(1, bdim[2] - 1) : (float)adim[2] / bdim[2];
		assert(rheight <= 1);
		assert(rwidth <= 1);
		int* const xcoeff = (int*)ccv_nnc_stream_context_get_workspace(stream_context, sizeof(int) * (bdim[2]), CCV_TENSOR_CPU_MEMORY);
		for (xd = 0; xd < bdim[2]; xd++)
			xcoeff[xd] = ccv_min(align_corners ? (int)(xd * rwidth + 0.5) : (int)((xd + 0.5) * rwidth), adim[2] - 1);
		assert(adim[0] == bdim[0]);
		assert(adim[3] == bdim[3]);
		for (i[0] = 0; i[0] < adim[0]; i[0]++)
		{
			int pysi0 = 0;
			const float* ap0 = ap;
			float* const bp0 = bp + i[0] * bstride[0];
			for (yd = 0; yd < bdim[1]; yd++)
			{
				const int ysi0 = ccv_min(align_corners ? (int)(yd * rheight + 0.5) : (int)((yd + 0.5) * rheight), adim[1] - 1);
				if (pysi0 < ysi0) // Move to ay1 line.
				{
					ap0 += (ysi0 - pysi0) * astride[1];
					pysi0 = ysi0;
				}
				float* bp1 = bp0 + yd * bstride[1];
				for (xd = 0; xd < bdim[2]; xd++)
				{
					const float* const ap00 = ap0 + xcoeff[xd] * astride[2];
					for (cd = 0; cd < bdim[3]; cd++)
						bp1[cd] = ap00[cd];
					bp1 += bstride[2];
				}
			}
			ap += astride[0];
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_upsample_nearest_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 1);
	assert(output_size >= 1);
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(ccv_nnc_tensor_nd(a->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(b->info.dim) <= CCV_NNC_MAX_DIM + 2);
	// Assuming this is float 32.
	int adim[CCV_NNC_MAX_DIM_ALLOC];
	int bdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(a, adim);
	ccv_nnc_tensor_view_get_dim(b, bdim);
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	ccv_nnc_tensor_view_get_stride(a, astride);
	ccv_nnc_tensor_view_get_stride(b, bstride);
	int i[CCV_NNC_MAX_DIM + 2];
	int xd, yd, cd;
	_ccv_nnc_tensor_set_cpu_ref_f32(a, 0);
	float* ap = a->data.f32;
	const float* bp = b->data.f32;
	const int align_corners = cmd.info.upsample.align_corners;
	assert(a->info.format == b->info.format);
	if (a->info.format == CCV_TENSOR_FORMAT_NCHW)
	{
		const float rheight = align_corners ? (float)(adim[2] - 1) / ccv_max(1, bdim[2] - 1) : (float)adim[2] / bdim[2];
		const float rwidth = align_corners ? (float)(adim[3] - 1) / ccv_max(1, bdim[3] - 1) : (float)adim[3] / bdim[3];
		assert(rheight <= 1);
		assert(rwidth <= 1);
		int* const xcoeff = (int*)ccv_nnc_stream_context_get_workspace(stream_context, sizeof(int) * (bdim[3]), CCV_TENSOR_CPU_MEMORY);
		for (xd = 0; xd < bdim[3]; xd++)
			xcoeff[xd] = ccv_min(align_corners ? (int)(xd * rwidth + 0.5) : (int)((xd + 0.5) * rwidth), adim[3] - 1);
		assert(adim[0] == bdim[0]);
		assert(adim[1] == bdim[1]);
		for (i[0] = 0; i[0] < adim[0]; i[0]++)
		{
			float* ap0 = ap + i[0] * astride[0];
			const float* bp0 = bp + i[0] * bstride[0];
			for (i[1] = 0; i[1] < adim[1]; i[1]++)
			{
				int pysi0 = 0;
				float* ap1 = ap0;
				const float* bp1 = bp0 + i[1] * bstride[1];
				for (yd = 0; yd < bdim[2]; yd++)
				{
					const int ysi0 = ccv_min(align_corners ? (int)(yd * rheight + 0.5) : (int)((yd + 0.5) * rheight), adim[2] - 1);
					if (pysi0 < ysi0) // Move to ay1 line.
					{
						ap1 += (ysi0 - pysi0) * astride[2];
						pysi0 = ysi0;
					}
					for (xd = 0; xd < bdim[3]; xd++)
						ap1[xcoeff[xd]] += bp1[xd];
					bp1 += bstride[2];
				}
				ap0 += astride[1];
			}
		}
	} else {
		// Any case, this is either NHWC or CHWN
		assert(a->info.format == CCV_TENSOR_FORMAT_NHWC || a->info.format == CCV_TENSOR_FORMAT_CHWN);
		const float rheight = align_corners ? (float)(adim[1] - 1) / ccv_max(1, bdim[1] - 1) : (float)adim[1] / bdim[1];
		const float rwidth = align_corners ? (float)(adim[2] - 1) / ccv_max(1, bdim[2] - 1) : (float)adim[2] / bdim[2];
		assert(rheight <= 1);
		assert(rwidth <= 1);
		int* const xcoeff = (int*)ccv_nnc_stream_context_get_workspace(stream_context, sizeof(int) * (bdim[2]), CCV_TENSOR_CPU_MEMORY);
		for (xd = 0; xd < bdim[2]; xd++)
			xcoeff[xd] = ccv_min(align_corners ? (int)(xd * rwidth + 0.5) : (int)((xd + 0.5) * rwidth), adim[2] - 1);
		assert(adim[0] == bdim[0]);
		assert(adim[3] == bdim[3]);
		for (i[0] = 0; i[0] < adim[0]; i[0]++)
		{
			int pysi0 = 0;
			float* ap0 = ap;
			const float* const bp0 = bp + i[0] * bstride[0];
			for (yd = 0; yd < bdim[1]; yd++)
			{
				const int ysi0 = ccv_min(align_corners ? (int)(yd * rheight + 0.5) : (int)((yd + 0.5) * rheight), adim[1] - 1);
				if (pysi0 < ysi0) // Move to ay1 line.
				{
					ap0 += (ysi0 - pysi0) * astride[1];
					pysi0 = ysi0;
				}
				const float* bp1 = bp0 + yd * bstride[1];
				for (xd = 0; xd < bdim[2]; xd++)
				{
					float* const ap00 = ap0 + xcoeff[xd] * astride[2];
					for (cd = 0; cd < bdim[3]; cd++)
						ap00[cd] += bp1[cd];
					bp1 += bstride[2];
				}
			}
			ap += astride[0];
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

typedef struct {
	int si[2];
	float sc[2];
} ccv_nnc_bi_coeffs_t;

static void _ccv_nnc_init_bi_coeffs(const int ss, const int sz, const float s, ccv_nnc_bi_coeffs_t* const coeff, const int align_corners)
{
	int i;
	if (align_corners)
	{
		for (i = 0; i < sz; i++)
		{
			const float xs = i * s;
			coeff[i].si[0] = (int)xs;
			coeff[i].si[1] = ccv_min((int)(xs + 1), ss - 1);
			coeff[i].sc[1] = xs - coeff[i].si[0];
			coeff[i].sc[0] = 1.0 - coeff[i].sc[1];
		}
	} else {
		for (i = 0; i < sz; i++)
		{
			const float xs = (i + 0.5) * s - 0.5;
			coeff[i].si[0] = (int)xs;
			coeff[i].si[1] = ccv_min((int)(xs + 1), ss - 1);
			coeff[i].sc[1] = xs - coeff[i].si[0];
			coeff[i].sc[0] = 1.0 - coeff[i].sc[1];
		}
	}
}

static int _ccv_nnc_upsample_bilinear_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 1);
	assert(output_size >= 1);
	ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(ccv_nnc_tensor_nd(a->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(b->info.dim) <= CCV_NNC_MAX_DIM + 2);
	// Assuming this is float 32.
	int adim[CCV_NNC_MAX_DIM_ALLOC];
	int bdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(a, adim);
	ccv_nnc_tensor_view_get_dim(b, bdim);
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	ccv_nnc_tensor_view_get_stride(a, astride);
	ccv_nnc_tensor_view_get_stride(b, bstride);
	int i[CCV_NNC_MAX_DIM + 2];
	int xd, yd, cd;
	const float* ap = a->data.f32;
	float* bp = b->data.f32;
	assert(a->info.format == b->info.format);
	const int align_corners = cmd.info.upsample.align_corners;
	if (a->info.format == CCV_TENSOR_FORMAT_NCHW)
	{
		const float rheight = align_corners ? (float)(adim[2] - 1) / ccv_max(1, bdim[2] - 1) : (float)adim[2] / bdim[2];
		const float rwidth = align_corners ? (float)(adim[3] - 1) / ccv_max(1, bdim[3] - 1) : (float)adim[3] / bdim[3];
		assert(rheight <= 1);
		assert(rwidth <= 1);
		ccv_nnc_bi_coeffs_t* const ycoeff = (ccv_nnc_bi_coeffs_t*)ccv_nnc_stream_context_get_workspace(stream_context, sizeof(ccv_nnc_bi_coeffs_t) * (bdim[2] + bdim[3]), CCV_TENSOR_CPU_MEMORY);
		ccv_nnc_bi_coeffs_t* const xcoeff = ycoeff + bdim[2];
		_ccv_nnc_init_bi_coeffs(adim[2], bdim[2], rheight, ycoeff, align_corners);
		_ccv_nnc_init_bi_coeffs(adim[3], bdim[3], rwidth, xcoeff, align_corners);
		assert(adim[0] == bdim[0]);
		assert(adim[1] == bdim[1]);
		for (i[0] = 0; i[0] < adim[0]; i[0]++)
		{
			const float* ap0 = ap + i[0] * astride[0];
			float* const bp0 = bp + i[0] * bstride[0];
			for (i[1] = 0; i[1] < adim[1]; i[1]++)
			{
				int pysi0 = 0;
				const float* ap1 = ap0;
				float* bp1 = bp0 + i[1] * bstride[1];
				for (yd = 0; yd < bdim[2]; yd++)
				{
					const int ysi0 = ycoeff[yd].si[0];
					const int ysi1 = ycoeff[yd].si[1] - ysi0;
					const float ysc0 = ycoeff[yd].sc[0];
					const float ysc1 = ycoeff[yd].sc[1];
					if (pysi0 < ysi0) // Move to ay1 line.
					{
						ap1 += (ysi0 - pysi0) * astride[2];
						pysi0 = ysi0;
					}
					for (xd = 0; xd < bdim[3]; xd++)
					{
						const ccv_nnc_bi_coeffs_t cof = xcoeff[xd];
						bp1[xd] = ap1[cof.si[0]] * cof.sc[0] * ysc0 + ap1[cof.si[1]] * cof.sc[1] * ysc0 +
							ap1[cof.si[0] + astride[2] * ysi1] * cof.sc[0] * ysc1 + ap1[cof.si[1] + astride[2] * ysi1] * cof.sc[1] * ysc1;
					}
					bp1 += bstride[2];
				}
				ap0 += astride[1];
			}
		}
	} else {
		// Any case, this is either NHWC or CHWN
		assert(a->info.format == CCV_TENSOR_FORMAT_NHWC || a->info.format == CCV_TENSOR_FORMAT_CHWN);
		const float rheight = align_corners ? (float)(adim[1] - 1) / ccv_max(1, bdim[1] - 1) : (float)adim[1] / bdim[1];
		const float rwidth = align_corners ? (float)(adim[2] - 1) / ccv_max(1, bdim[2] - 1) : (float)adim[2] / bdim[2];
		assert(rheight <= 1);
		assert(rwidth <= 1);
		ccv_nnc_bi_coeffs_t* const ycoeff = (ccv_nnc_bi_coeffs_t*)ccv_nnc_stream_context_get_workspace(stream_context, sizeof(ccv_nnc_bi_coeffs_t) * (bdim[1] + bdim[2]), CCV_TENSOR_CPU_MEMORY);
		ccv_nnc_bi_coeffs_t* const xcoeff = ycoeff + bdim[1];
		_ccv_nnc_init_bi_coeffs(adim[1], bdim[1], rheight, ycoeff, align_corners);
		_ccv_nnc_init_bi_coeffs(adim[2], bdim[2], rwidth, xcoeff, align_corners);
		assert(adim[0] == bdim[0]);
		assert(adim[3] == bdim[3]);
		for (i[0] = 0; i[0] < adim[0]; i[0]++)
		{
			int pysi0 = 0;
			const float* ap0 = ap;
			float* const bp0 = bp + i[0] * bstride[0];
			for (yd = 0; yd < bdim[1]; yd++)
			{
				const int ysi0 = ycoeff[yd].si[0];
				const int ysi1 = ycoeff[yd].si[1] - ysi0;
				const float ysc0 = ycoeff[yd].sc[0];
				const float ysc1 = ycoeff[yd].sc[1];
				if (pysi0 < ysi0) // Move to ay1 line.
				{
					ap0 += (ysi0 - pysi0) * astride[1];
					pysi0 = ysi0;
				}
				float* bp1 = bp0 + yd * bstride[1];
				for (xd = 0; xd < bdim[2]; xd++)
				{
					const ccv_nnc_bi_coeffs_t cof = xcoeff[xd];
					const float c00 = cof.sc[0] * ysc0;
					const float c01 = cof.sc[1] * ysc0;
					const float c10 = cof.sc[0] * ysc1;
					const float c11 = cof.sc[1] * ysc1;
					const float* const ap00 = ap0 + cof.si[0] * astride[2];
					const float* const ap01 = ap0 + cof.si[1] * astride[2];
					const float* const ap10 = ap00 + ysi1 * astride[1];
					const float* const ap11 = ap01 + ysi1 * astride[1];
					for (cd = 0; cd < bdim[3]; cd++)
						bp1[cd] = ap00[cd] * c00 + ap01[cd] * c01 +
							ap10[cd] * c10 + ap11[cd] * c11;
					bp1 += bstride[2];
				}
			}
			ap += astride[0];
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_upsample_bilinear_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 1);
	assert(output_size >= 1);
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(ccv_nnc_tensor_nd(a->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(b->info.dim) <= CCV_NNC_MAX_DIM + 2);
	// Assuming this is float 32.
	int adim[CCV_NNC_MAX_DIM_ALLOC];
	int bdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(a, adim);
	ccv_nnc_tensor_view_get_dim(b, bdim);
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	ccv_nnc_tensor_view_get_stride(a, astride);
	ccv_nnc_tensor_view_get_stride(b, bstride);
	int i[CCV_NNC_MAX_DIM + 2];
	int xd, yd, cd;
	_ccv_nnc_tensor_set_cpu_ref_f32(a, 0);
	float* ap = a->data.f32;
	const float* bp = b->data.f32;
	assert(a->info.format == b->info.format);
	const int align_corners = cmd.info.upsample.align_corners;
	if (a->info.format == CCV_TENSOR_FORMAT_NCHW)
	{
		const float rheight = align_corners ? (float)(adim[2] - 1) / ccv_max(1, bdim[2] - 1) : (float)adim[2] / bdim[2];
		const float rwidth = align_corners ? (float)(adim[3] - 1) / ccv_max(1, bdim[3] - 1) : (float)adim[3] / bdim[3];
		assert(rheight <= 1);
		assert(rwidth <= 1);
		ccv_nnc_bi_coeffs_t* const ycoeff = (ccv_nnc_bi_coeffs_t*)ccv_nnc_stream_context_get_workspace(stream_context, sizeof(ccv_nnc_bi_coeffs_t) * (bdim[2] + bdim[3]), CCV_TENSOR_CPU_MEMORY);
		ccv_nnc_bi_coeffs_t* const xcoeff = ycoeff + bdim[2];
		_ccv_nnc_init_bi_coeffs(adim[2], bdim[2], rheight, ycoeff, align_corners);
		_ccv_nnc_init_bi_coeffs(adim[3], bdim[3], rwidth, xcoeff, align_corners);
		assert(adim[0] == bdim[0]);
		assert(adim[1] == bdim[1]);
		for (i[0] = 0; i[0] < adim[0]; i[0]++)
		{
			float* ap0 = ap + i[0] * astride[0];
			const float* const bp0 = bp + i[0] * bstride[0];
			for (i[1] = 0; i[1] < adim[1]; i[1]++)
			{
				int pysi0 = 0;
				float* ap1 = ap0;
				const float* bp1 = bp0 + i[1] * bstride[1];
				for (yd = 0; yd < bdim[2]; yd++)
				{
					const int ysi0 = ycoeff[yd].si[0];
					const int ysi1 = ycoeff[yd].si[1] - ysi0;
					const float ysc0 = ycoeff[yd].sc[0];
					const float ysc1 = ycoeff[yd].sc[1];
					if (pysi0 < ysi0) // Move to ay1 line.
					{
						ap1 += (ysi0 - pysi0) * astride[2];
						pysi0 = ysi0;
					}
					for (xd = 0; xd < bdim[3]; xd++)
					{
						const ccv_nnc_bi_coeffs_t cof = xcoeff[xd];
						ap1[cof.si[0]] += bp1[xd] * ysc0 * cof.sc[0];
						ap1[cof.si[1]] += bp1[xd] * ysc0 * cof.sc[1];
						ap1[cof.si[0] + astride[2] * ysi1] += bp1[xd] * ysc1 * cof.sc[0];
						ap1[cof.si[1] + astride[2] * ysi1] += bp1[xd] * ysc1 * cof.sc[1];
					}
					bp1 += bstride[2];
				}
				ap0 += astride[1];
			}
		}
	} else {
		// Any case, this is either NHWC or CHWN
		assert(a->info.format == CCV_TENSOR_FORMAT_NHWC || a->info.format == CCV_TENSOR_FORMAT_CHWN);
		const float rheight = align_corners ? (float)(adim[1] - 1) / ccv_max(1, bdim[1] - 1) : (float)adim[1] / bdim[1];
		const float rwidth = align_corners ? (float)(adim[2] - 1) / ccv_max(1, bdim[2] - 1) : (float)adim[2] / bdim[2];
		assert(rheight <= 1);
		assert(rwidth <= 1);
		ccv_nnc_bi_coeffs_t* const ycoeff = (ccv_nnc_bi_coeffs_t*)ccv_nnc_stream_context_get_workspace(stream_context, sizeof(ccv_nnc_bi_coeffs_t) * (bdim[1] + bdim[2]), CCV_TENSOR_CPU_MEMORY);
		ccv_nnc_bi_coeffs_t* const xcoeff = ycoeff + bdim[1];
		_ccv_nnc_init_bi_coeffs(adim[1], bdim[1], rheight, ycoeff, align_corners);
		_ccv_nnc_init_bi_coeffs(adim[2], bdim[2], rwidth, xcoeff, align_corners);
		assert(adim[0] == bdim[0]);
		assert(adim[3] == bdim[3]);
		for (i[0] = 0; i[0] < adim[0]; i[0]++)
		{
			int pysi0 = 0;
			float* ap0 = ap;
			const float* const bp0 = bp + i[0] * bstride[0];
			for (yd = 0; yd < bdim[1]; yd++)
			{
				const int ysi0 = ycoeff[yd].si[0];
				const int ysi1 = ycoeff[yd].si[1] - ysi0;
				const float ysc0 = ycoeff[yd].sc[0];
				const float ysc1 = ycoeff[yd].sc[1];
				if (pysi0 < ysi0) // Move to ay1 line.
				{
					ap0 += (ysi0 - pysi0) * astride[1];
					pysi0 = ysi0;
				}
				const float* bp1 = bp0 + yd * bstride[1];
				for (xd = 0; xd < bdim[2]; xd++)
				{
					const ccv_nnc_bi_coeffs_t cof = xcoeff[xd];
					const float c00 = cof.sc[0] * ysc0;
					const float c01 = cof.sc[1] * ysc0;
					const float c10 = cof.sc[0] * ysc1;
					const float c11 = cof.sc[1] * ysc1;
					float* const ap00 = ap0 + cof.si[0] * astride[2];
					float* const ap01 = ap0 + cof.si[1] * astride[2];
					float* const ap10 = ap00 + ysi1 * astride[1];
					float* const ap11 = ap01 + ysi1 * astride[1];
					for (cd = 0; cd < bdim[3]; cd++)
					{
						ap00[cd] += bp1[cd] * c00;
						ap01[cd] += bp1[cd] * c01;
						ap10[cd] += bp1[cd] * c10;
						ap11[cd] += bp1[cd] * c11;
					}
					bp1 += bstride[2];
				}
			}
			ap += astride[0];
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_upsample_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	if (cmd.info.upsample.type == CCV_NNC_UPSAMPLE_NEAREST)
		return _ccv_nnc_upsample_nearest_forw(cmd, hint, flags, inputs, input_size, outputs, output_size, stream_context);
	else if (cmd.info.upsample.type == CCV_NNC_UPSAMPLE_BILINEAR)
		return _ccv_nnc_upsample_bilinear_forw(cmd, hint, flags, inputs, input_size, outputs, output_size, stream_context);
	return CCV_NNC_EXEC_INVALID;
}

static int _ccv_nnc_upsample_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	if (cmd.info.upsample.type == CCV_NNC_UPSAMPLE_NEAREST)
		return _ccv_nnc_upsample_nearest_back(cmd, hint, flags, inputs, input_size, outputs, output_size, stream_context);
	else if (cmd.info.upsample.type == CCV_NNC_UPSAMPLE_BILINEAR)
		return _ccv_nnc_upsample_bilinear_back(cmd, hint, flags, inputs, input_size, outputs, output_size, stream_context);
	return CCV_NNC_EXEC_INVALID;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_UPSAMPLE_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_upsample_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_UPSAMPLE_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_upsample_back;
}
