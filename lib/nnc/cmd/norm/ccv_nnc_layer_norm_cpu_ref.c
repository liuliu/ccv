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

static int _ccv_nnc_layer_norm_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 3);
	ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const scale = (ccv_nnc_tensor_view_t*)inputs[1];
	ccv_nnc_tensor_view_t* const bias = (ccv_nnc_tensor_view_t*)inputs[2];
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* const saved_mean = (ccv_nnc_tensor_view_t*)outputs[1];
	ccv_nnc_tensor_view_t* const saved_inv_std = (ccv_nnc_tensor_view_t*)outputs[2];
	assert(ccv_nnc_tensor_nd(a->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(b->info.dim) <= CCV_NNC_MAX_DIM + 2);
	// Assuming this is float 32.
	int adim[CCV_NNC_MAX_DIM_ALLOC];
	int rdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(a, adim);
	ccv_nnc_tensor_view_get_dim(saved_mean, rdim);
	assert(ccv_nnc_tensor_view_check_dim(saved_inv_std, rdim));
	assert(ccv_nnc_tensor_view_check_dim(b, adim));
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
	int scale_stride[CCV_NNC_MAX_DIM_ALLOC];
	int bias_stride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_stride(a, astride);
	ccv_nnc_tensor_view_get_stride(scale, scale_stride);
	ccv_nnc_tensor_view_get_stride(bias, bias_stride);
	ccv_nnc_tensor_view_get_stride(b, bstride);
	// The epsilon is used a little bit differently from batch norm, it is outside of the sqrt in this case.
	const float epsilon = cmd.info.lnorm.epsilon;
	int saved_mean_stride[CCV_NNC_MAX_DIM_ALLOC];
	int saved_inv_std_stride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_stride(saved_mean, saved_mean_stride);
	ccv_nnc_tensor_view_get_stride(saved_inv_std, saved_inv_std_stride);
	int x;
	int n = 1;
	for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
		n *= adim[x];
	for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
		n /= rdim[x];
	const float inv_n = 1. / n;
	_ccv_nnc_reduce_sum_forw_cpu_ref(a, saved_mean);
	_ccv_nnc_mul_forw_cpu_ref(inv_n, saved_mean, 0, saved_mean);
	ccv_nnc_tensor_zero(saved_inv_std);
	float* const ap = a->data.f32;
	float* const meanp = saved_mean->data.f32;
	float* const varp = saved_inv_std->data.f32;
	int i[CCV_NNC_MAX_DIM + 2];
	for (i[0] = 0; i[0] < adim[0]; i[0]++)
	{
		float* const ap0 = ap + i[0] * astride[0];
		float* const meanp0 = rdim[0] == 1 ? meanp : meanp + i[0] * saved_mean_stride[0];
		float* const varp0 = rdim[0] == 1 ? varp : varp + i[0] * saved_inv_std_stride[0];
		for (i[1] = 0; i[1] < adim[1]; i[1]++)
		{
			float* ap1 = ap0 + i[1] * astride[1];
			float* const meanp1 = rdim[1] == 1 ? meanp0 : meanp0 + i[1] * saved_mean_stride[1];
			float* const varp1 = rdim[1] == 1 ? varp0 : varp0 + i[1] * saved_inv_std_stride[1];
			for (i[2] = 0; i[2] < adim[2]; i[2]++)
			{
				float* const meanp2 = rdim[2] == 1 ? meanp1 : meanp1 + i[2] * saved_mean_stride[2];
				float* const varp2 = rdim[2] == 1 ? varp1 : varp1 + i[2] * saved_inv_std_stride[2];
				if (rdim[3] == 1)
					for (x = 0; x < adim[3]; x++)
					{
						float w = ap1[x * astride[3]] - meanp2[0];
						varp2[0] += w * w;
					}
				else
					for (x = 0; x < adim[3]; x++)
					{
						float w = ap1[x * astride[3]] - meanp2[x];
						varp2[x] += w * w;
					}
				ap1 += astride[2];
			}
		}
	}
	for (i[0] = 0; i[0] < rdim[0]; i[0]++)
	{
		float* const varp0 = varp + i[0] * saved_inv_std_stride[0];
		for (i[1] = 0; i[1] < rdim[1]; i[1]++)
		{
			float* const varp1 = varp0 + i[1] * saved_inv_std_stride[1];
			for (i[2] = 0; i[2] < rdim[2]; i[2]++)
			{
				float* const varp2 = varp1 + i[2] * saved_inv_std_stride[2];
				for (x = 0; x < rdim[3]; x++)
					varp2[x] = 1. / sqrtf(varp2[x] * inv_n + epsilon);
			}
		}
	}
	float* const scalep = scale->data.f32;
	float* const biasp = bias->data.f32;
	int sdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(scale, sdim);
	int bias_dim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(bias, bias_dim);
	// Do the straight-forward one, y = (x - mean) * inv_std * scale + bias, we cannot allocate extra memory to help.
	// There is no need for precompute since scale / bias is per element.
	float* const bp = b->data.f32;
	for (i[0] = 0; i[0] < adim[0]; i[0]++)
	{
		float* const ap0 = ap + i[0] * astride[0];
		float* const bp0 = bp + i[0] * bstride[0];
		float* const meanp0 = rdim[0] == 1 ? meanp : meanp + i[0] * saved_mean_stride[0];
		float* const varp0 = rdim[0] == 1 ? varp : varp + i[0] * saved_inv_std_stride[0];
		float* const scalep0 = sdim[0] == 1 ? scalep : scalep + i[0] * scale_stride[0];
		float* const biasp0 = bias_dim[0] == 1 ? biasp : biasp + i[0] * bias_stride[0];
		for (i[1] = 0; i[1] < adim[1]; i[1]++)
		{
			float* ap1 = ap0 + i[1] * astride[1];
			float* bp1 = bp0 + i[1] * bstride[1];
			float* const meanp1 = rdim[1] == 1 ? meanp0 : meanp0 + i[1] * saved_mean_stride[1];
			float* const varp1 = rdim[1] == 1 ? varp0 : varp0 + i[1] * saved_inv_std_stride[1];
			float* const scalep1 = sdim[1] == 1 ? scalep0 : scalep0 + i[1] * scale_stride[1];
			float* const biasp1 = bias_dim[1] == 1 ? biasp0 : biasp0 + i[1] * bias_stride[1];
			for (i[2] = 0; i[2] < adim[2]; i[2]++)
			{
				float* const meanp2 = rdim[2] == 1 ? meanp1 : meanp1 + i[2] * saved_mean_stride[2];
				float* const varp2 = rdim[2] == 1 ? varp1 : varp1 + i[2] * saved_inv_std_stride[2];
				float* const scalep2 = sdim[2] == 1 ? scalep1 : scalep1 + i[2] * scale_stride[2];
				float* const biasp2 = bias_dim[2] == 1 ? biasp1 : biasp1 + i[2] * bias_stride[2];
				if (rdim[3] == 1)
					for (x = 0; x < adim[3]; x++)
						bp1[x] = (ap1[x * astride[3]] - meanp2[0]) * varp2[0] * scalep2[sdim[3] == 1 ? 0 : x] + biasp2[bias_dim[3] == 1 ? 0 : x];
				else
					for (x = 0; x < adim[3]; x++)
						bp1[x] = (ap1[x * astride[3]] - meanp2[x]) * varp2[x] * scalep2[sdim[3] == 1 ? 0 : x] + biasp2[bias_dim[3] == 1 ? 0 : x];
				ap1 += astride[2];
				bp1 += bstride[2];
			}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_layer_norm_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 9);
	assert(output_size >= 1);
	ccv_nnc_tensor_view_t* const g = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[3];
	ccv_nnc_tensor_view_t* const scale = (ccv_nnc_tensor_view_t*)inputs[4];
	ccv_nnc_tensor_view_t* const saved_mean = (ccv_nnc_tensor_view_t*)inputs[7];
	ccv_nnc_tensor_view_t* const saved_inv_std = (ccv_nnc_tensor_view_t*)inputs[8];
	ccv_nnc_tensor_view_t* const h = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* const dscale = output_size > 1 ? (ccv_nnc_tensor_view_t*)outputs[1] : 0;
	ccv_nnc_tensor_view_t* const dbias = output_size > 2 ? (ccv_nnc_tensor_view_t*)outputs[2] : 0;
	assert(ccv_nnc_tensor_nd(g->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(a->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(h->info.dim) <= CCV_NNC_MAX_DIM + 2);
	// Assuming this is float 32.
	int gdim[CCV_NNC_MAX_DIM_ALLOC];
	int rdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(g, gdim);
	ccv_nnc_tensor_view_get_dim(saved_mean, rdim);
	assert(ccv_nnc_tensor_view_check_dim(saved_inv_std, rdim));
	int sdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(scale, sdim);
	if (dscale)
		{ assert(ccv_nnc_tensor_view_check_dim(dscale, sdim)); }
	assert(ccv_nnc_tensor_view_check_dim(a, gdim));
	assert(ccv_nnc_tensor_view_check_dim(h, gdim));
	if (dbias)
		_ccv_nnc_reduce_sum_forw_cpu_ref(g, dbias);
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	int gstride[CCV_NNC_MAX_DIM_ALLOC];
	int hstride[CCV_NNC_MAX_DIM_ALLOC];
	int scale_stride[CCV_NNC_MAX_DIM_ALLOC];
	int mean_stride[CCV_NNC_MAX_DIM_ALLOC];
	int inv_std_stride[CCV_NNC_MAX_DIM_ALLOC];
	int dscale_stride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_stride(a, astride);
	ccv_nnc_tensor_view_get_stride(g, gstride);
	ccv_nnc_tensor_view_get_stride(h, hstride);
	ccv_nnc_tensor_view_get_stride(scale, scale_stride);
	ccv_nnc_tensor_view_get_stride(saved_mean, mean_stride);
	ccv_nnc_tensor_view_get_stride(saved_inv_std, inv_std_stride);
	if (dscale)
		ccv_nnc_tensor_view_get_stride(dscale, dscale_stride);
	// Need to allocate two additional memory:
	// 1. normalized a;
	// 2. scale * inv_std / n;
	assert(!(flags & CCV_NNC_ZERO_MEMORY_ALLOC));
	int x;
	int n = 1;
	for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
		n *= gdim[x];
	for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
		n /= rdim[x];
	int gcount = 1, rcount = 1;
	for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
		gcount *= gdim[x], rcount *= rdim[x];
	float* const ah = (float*)ccv_nnc_stream_context_get_workspace(stream_context, sizeof(float) * gcount * 2 + sizeof(float) * rcount * 2, CCV_TENSOR_CPU_MEMORY);
	float* const gss = ah + gcount; // g * scale * inv_std
	float* const gssr = gss + gcount; // gss reduced to inv_std dimension
	float* const ahgssr = gssr + rcount; // ah * gss then reduced to inv_std dimension.
	int i[CCV_NNC_MAX_DIM + 2];
	float* ahp = ah;
	const float* const meanp = saved_mean->data.f32;
	const float* const inv_stdp = saved_inv_std->data.f32;
	const float* const ap = a->data.f32;
	for (i[0] = 0; i[0] < gdim[0]; i[0]++)
	{
		const float* const ap0 = ap + i[0] * astride[0];
		const float* const meanp0 = rdim[0] == 1 ? meanp : meanp + i[0] * mean_stride[0];
		const float* const inv_stdp0 = rdim[0] == 1 ? inv_stdp : inv_stdp + i[0] * inv_std_stride[0];
		for (i[1] = 0; i[1] < gdim[1]; i[1]++)
		{
			const float* ap1 = ap0 + i[1] * astride[1];
			const float* const meanp1 = rdim[1] == 1 ? meanp0 : meanp0 + i[1] * mean_stride[1];
			const float* const inv_stdp1 = rdim[1] == 1 ? inv_stdp0 : inv_stdp0 + i[1] * inv_std_stride[1];
			for (i[2] = 0; i[2] < gdim[2]; i[2]++)
			{
				const float* const meanp2 = rdim[2] == 1 ? meanp1 : meanp1 + i[2] * mean_stride[2];
				const float* const inv_stdp2 = rdim[2] == 1 ? inv_stdp1 : inv_stdp1 + i[2] * inv_std_stride[2];
				if (rdim[3] == 1)
					for (x = 0; x < gdim[3]; x++)
						ahp[x] = (ap1[x] - meanp2[0]) * inv_stdp2[0];
				else
					for (x = 0; x < gdim[3]; x++)
						ahp[x] = (ap1[x] - meanp2[x]) * inv_stdp2[x];
				ap1 += astride[2];
				ahp += gdim[3];
			}
		}
	}
	if (dscale)
	{
		ccv_nnc_tensor_zero(dscale);
		ahp = ah;
		float* gssp = gss;
		const float* const gp = g->data.f32;
		const float* const scalep = scale->data.f32;
		float* const dscalep = dscale->data.f32;
		for (i[0] = 0; i[0] < gdim[0]; i[0]++)
		{
			const float* const gp0 = gp + i[0] * gstride[0];
			const float* const inv_stdp0 = rdim[0] == 1 ? inv_stdp : inv_stdp + i[0] * inv_std_stride[0];
			const float* const scalep0 = sdim[0] == 1 ? scalep : scalep + i[0] * scale_stride[0];
			float* const dscalep0 = sdim[0] == 1 ? dscalep : dscalep + i[0] * dscale_stride[0];
			for (i[1] = 0; i[1] < gdim[1]; i[1]++)
			{
				const float* gp1 = gp0 + i[1] * gstride[1];
				const float* const inv_stdp1 = rdim[1] == 1 ? inv_stdp0 : inv_stdp0 + i[1] * inv_std_stride[1];
				const float* const scalep1 = sdim[1] == 1 ? scalep0 : scalep0 + i[1] * scale_stride[1];
				float* const dscalep1 = sdim[1] == 1 ? dscalep0 : dscalep0 + i[1] * dscale_stride[1];
				for (i[2] = 0; i[2] < gdim[2]; i[2]++)
				{
					const float* const inv_stdp2 = rdim[2] == 1 ? inv_stdp1 : inv_stdp1 + i[2] * inv_std_stride[2];
					const float* const scalep2 = sdim[2] == 1 ? scalep1 : scalep1 + i[2] * scale_stride[2];
					float* const dscalep2 = sdim[2] == 1 ? dscalep1 : dscalep1 + i[2] * dscale_stride[2];
					if (sdim[3] == 1)
						for (x = 0; x < gdim[3]; x++)
						{
							gssp[x] = gp1[x] * scalep2[0] * inv_stdp2[rdim[3] == 1 ? 0 : x];
							dscalep2[0] += ahp[x] * gp1[x];
						}
					else
						for (x = 0; x < gdim[3]; x++)
						{
							gssp[x] = gp1[x] * scalep2[x] * inv_stdp2[rdim[3] == 1 ? 0 : x];
							dscalep2[x] += ahp[x] * gp1[x];
						}
					gp1 += gstride[2];
					ahp += gdim[3];
					gssp += gdim[3];
				}
			}
		}
	} else {
		ahp = ah;
		float* gssp = gss;
		const float* const gp = g->data.f32;
		const float* const scalep = scale->data.f32;
		for (i[0] = 0; i[0] < gdim[0]; i[0]++)
		{
			const float* const gp0 = gp + i[0] * gstride[0];
			const float* const inv_stdp0 = rdim[0] == 1 ? inv_stdp : inv_stdp + i[0] * inv_std_stride[0];
			const float* const scalep0 = sdim[0] == 1 ? scalep : scalep + i[0] * scale_stride[0];
			for (i[1] = 0; i[1] < gdim[1]; i[1]++)
			{
				const float* gp1 = gp0 + i[1] * gstride[1];
				const float* const inv_stdp1 = rdim[1] == 1 ? inv_stdp0 : inv_stdp0 + i[1] * inv_std_stride[1];
				const float* const scalep1 = sdim[1] == 1 ? scalep0 : scalep0 + i[1] * scale_stride[1];
				for (i[2] = 0; i[2] < gdim[2]; i[2]++)
				{
					const float* const inv_stdp2 = rdim[2] == 1 ? inv_stdp1 : inv_stdp1 + i[2] * inv_std_stride[2];
					const float* const scalep2 = sdim[2] == 1 ? scalep1 : scalep1 + i[2] * scale_stride[2];
					if (sdim[3] == 1)
						for (x = 0; x < gdim[3]; x++)
							gssp[x] = gp1[x] * scalep2[0] * inv_stdp2[rdim[3] == 1 ? 0 : x];
					else
						for (x = 0; x < gdim[3]; x++)
							gssp[x] = gp1[x] * scalep2[x] * inv_stdp2[rdim[3] == 1 ? 0 : x];
					gp1 += gstride[2];
					ahp += gdim[3];
					gssp += gdim[3];
				}
			}
		}
	}
	ccv_nnc_tensor_t gsst = ccv_nnc_tensor(gss, g->info, 0);
	ccv_nnc_tensor_t gssrt = ccv_nnc_tensor(gssr, saved_mean->info, 0);
	_ccv_nnc_reduce_sum_forw_cpu_ref((ccv_nnc_tensor_view_t*)&gsst, (ccv_nnc_tensor_view_t*)&gssrt);
	ahp = ah;
	float* gssp = gss;
	ccv_nnc_tensor_t ahgssrt = ccv_nnc_tensor(ahgssr, saved_mean->info, 0);
	ccv_nnc_tensor_zero(&ahgssrt);
	float* const ahgssrp = ahgssr;
	for (i[0] = 0; i[0] < gdim[0]; i[0]++)
	{
		float* const ahgssrp0 = rdim[0] == 1 ? ahgssrp : ahgssrp + i[0] * rdim[1] * rdim[2] * rdim[3];
		for (i[1] = 0; i[1] < gdim[1]; i[1]++)
		{
			float* const ahgssrp1 = rdim[1] == 1 ? ahgssrp0 : ahgssrp0 + i[1] * rdim[2] * rdim[3];
			for (i[2] = 0; i[2] < gdim[2]; i[2]++)
			{
				float* const ahgssrp2 = rdim[2] == 1 ? ahgssrp1 : ahgssrp1 + i[2] * rdim[3];
				if (rdim[3] == 1)
					for (x = 0; x < gdim[3]; x++)
						ahgssrp2[0] += ahp[x] * gssp[x];
				else
					for (x = 0; x < gdim[3]; x++)
						ahgssrp2[x] += ahp[x] * gssp[x];
				ahp += gdim[3];
				gssp += gdim[3];
			}
		}
	}
	// Now the part to compute dx (h).
	float* const hp = h->data.f32;
	ahp = ah;
	const float inv_n = 1. / n;
	gssp = gss;
	const float* const gssrp = gssr;
	for (i[0] = 0; i[0] < gdim[0]; i[0]++)
	{
		float* const hp0 = hp + i[0] * hstride[0];
		const float* const gssrp0 = rdim[0] == 1 ? gssrp : gssrp + i[0] * rdim[1] * rdim[2] * rdim[3];
		const float* const ahgssrp0 = rdim[0] == 1 ? ahgssrp : ahgssrp + i[0] * rdim[1] * rdim[2] * rdim[3];
		for (i[1] = 0; i[1] < gdim[1]; i[1]++)
		{
			float* hp1 = hp0 + i[1] * hstride[1];
			const float* const gssrp1 = rdim[1] == 1 ? gssrp0 : gssrp0 + i[1] * rdim[2] * rdim[3];
			const float* const ahgssrp1 = rdim[1] == 1 ? ahgssrp0 : ahgssrp0 + i[1] * rdim[2] * rdim[3];
			for (i[2] = 0; i[2] < gdim[2]; i[2]++)
			{
				const float* const gssrp2 = rdim[2] == 1 ? gssrp1 : gssrp1 + i[2] * rdim[3];
				const float* const ahgssrp2 = rdim[2] == 1 ? ahgssrp1 : ahgssrp1 + i[2] * rdim[3];
				if (rdim[3] == 1)
					for (x = 0; x < gdim[3]; x++)
						hp1[x] = gssp[x] - inv_n * (gssrp2[0] + ahp[x] * ahgssrp2[0]);
				else
					for (x = 0; x < gdim[3]; x++)
						hp1[x] = gssp[x] - inv_n * (gssrp2[x] + ahp[x] * ahgssrp2[x]);
				hp1 += hstride[2];
				ahp += gdim[3];
				gssp += gdim[3];
			}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_LAYER_NORM_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_layer_norm_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_LAYER_NORM_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_layer_norm_back;
}
