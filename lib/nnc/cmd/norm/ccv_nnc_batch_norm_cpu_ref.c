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

static int _ccv_nnc_batch_norm_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 5);
	ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const scale = (ccv_nnc_tensor_view_t*)inputs[1];
	ccv_nnc_tensor_view_t* const bias = (ccv_nnc_tensor_view_t*)inputs[2];
	ccv_nnc_tensor_view_t* const mean = (ccv_nnc_tensor_view_t*)inputs[3];
	ccv_nnc_tensor_view_t* const var = (ccv_nnc_tensor_view_t*)inputs[4];
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(ccv_nnc_tensor_nd(a->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(b->info.dim) <= CCV_NNC_MAX_DIM + 2);
	// Assuming this is float 32.
	int adim[CCV_NNC_MAX_DIM_ALLOC];
	int rdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(a, adim);
	ccv_nnc_tensor_view_get_dim(scale, rdim);
	assert(ccv_nnc_tensor_view_check_dim(bias, rdim));
	assert(ccv_nnc_tensor_view_check_dim(mean, rdim));
	assert(ccv_nnc_tensor_view_check_dim(var, rdim));
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
	const float epsilon = cmd.info.bnorm.epsilon;
	if (!cmd.info.bnorm.is_test)
	{
		assert(output_size == 5);
		// Both are inplace.
		assert(inputs[3]->data.f32 == outputs[1]->data.f32);
		assert(inputs[4]->data.f32 == outputs[2]->data.f32);
		ccv_nnc_tensor_view_t* const saved_mean = (ccv_nnc_tensor_view_t*)outputs[3];
		ccv_nnc_tensor_view_t* const saved_inv_std = (ccv_nnc_tensor_view_t*)outputs[4];
		assert(ccv_nnc_tensor_view_check_dim(saved_mean, rdim));
		assert(ccv_nnc_tensor_view_check_dim(saved_inv_std, rdim));
		int saved_mean_stride[CCV_NNC_MAX_DIM_ALLOC];
		int saved_inv_std_stride[CCV_NNC_MAX_DIM_ALLOC];
		ccv_nnc_tensor_view_get_stride(saved_mean, saved_mean_stride);
		ccv_nnc_tensor_view_get_stride(saved_inv_std, saved_inv_std_stride);
		int i[CCV_NNC_MAX_DIM + 2];
		int x;
		int batch_size = 1;
		for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
			batch_size *= adim[x];
		for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
			batch_size /= rdim[x];
		const float inv_batch_size = 1. / batch_size;
		_ccv_nnc_reduce_sum_forw_cpu_ref(a, saved_mean);
		_ccv_nnc_mul_forw_cpu_ref(inv_batch_size, saved_mean, 0, saved_mean);
		// Copy this into running mean / var.
		_ccv_nnc_add_forw_cpu_ref(cmd.info.bnorm.momentum, 1. - cmd.info.bnorm.momentum, mean, saved_mean, mean);
		ccv_nnc_tensor_zero(saved_inv_std);
		float* const ap = a->data.f32;
		float* const meanp = saved_mean->data.f32;
		float* const varp = saved_inv_std->data.f32;
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
							float w = ap1[x] - meanp2[0];
							varp2[0] += w * w;
						}
					else
						for (x = 0; x < adim[3]; x++)
						{
							float w = ap1[x] - meanp2[x];
							varp2[x] += w * w;
						}
					ap1 += astride[2];
				}
			}
		}
		_ccv_nnc_mul_forw_cpu_ref(inv_batch_size, saved_inv_std, 0, saved_inv_std);
		_ccv_nnc_add_forw_cpu_ref(cmd.info.bnorm.momentum, 1. - cmd.info.bnorm.momentum, var, saved_inv_std, var);
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
						varp2[x] = 1. / sqrtf(varp2[x] + epsilon);
				}
			}
		}
		float* const scalep = scale->data.f32;
		float* const biasp = bias->data.f32;
		// Now, after mean and inv_std computed, go and stretch a.
		if (flags & CCV_NNC_ZERO_MEMORY_ALLOC)
		{
			// Do the straight-forward one, y = (x - mean) * inv_std * scale + bias, we cannot allocate extra memory to help.
			float* const bp = b->data.f32;
			for (i[0] = 0; i[0] < adim[0]; i[0]++)
			{
				float* const ap0 = ap + i[0] * astride[0];
				float* const bp0 = bp + i[0] * bstride[0];
				float* const meanp0 = rdim[0] == 1 ? meanp : meanp + i[0] * saved_mean_stride[0];
				float* const varp0 = rdim[0] == 1 ? varp : varp + i[0] * saved_inv_std_stride[0];
				float* const scalep0 = rdim[0] == 1 ? scalep : scalep + i[0] * scale_stride[0];
				float* const biasp0 = rdim[0] == 1 ? biasp : biasp + i[0] * bias_stride[0];
				for (i[1] = 0; i[1] < adim[1]; i[1]++)
				{
					float* ap1 = ap0 + i[1] * astride[1];
					float* bp1 = bp0 + i[1] * bstride[1];
					float* const meanp1 = rdim[1] == 1 ? meanp0 : meanp0 + i[1] * saved_mean_stride[1];
					float* const varp1 = rdim[1] == 1 ? varp0 : varp0 + i[1] * saved_inv_std_stride[1];
					float* const scalep1 = rdim[1] == 1 ? scalep0 : scalep0 + i[1] * scale_stride[1];
					float* const biasp1 = rdim[1] == 1 ? biasp0 : biasp0 + i[1] * bias_stride[1];
					for (i[2] = 0; i[2] < adim[2]; i[2]++)
					{
						float* const meanp2 = rdim[2] == 1 ? meanp1 : meanp1 + i[2] * saved_mean_stride[2];
						float* const varp2 = rdim[2] == 1 ? varp1 : varp1 + i[2] * saved_inv_std_stride[2];
						float* const scalep2 = rdim[2] == 1 ? scalep1 : scalep1 + i[2] * scale_stride[2];
						float* const biasp2 = rdim[2] == 1 ? biasp1 : biasp1 + i[2] * bias_stride[2];
						if (rdim[3] == 1)
							for (x = 0; x < adim[3]; x++)
								bp1[x] = (ap1[x] - meanp2[0]) * varp2[0] * scalep2[0] + biasp2[0];
						else
							for (x = 0; x < adim[3]; x++)
								bp1[x] = (ap1[x] - meanp2[x]) * varp2[x] * scalep2[x] + biasp2[x];
						ap1 += astride[2];
						bp1 += bstride[2];
					}
				}
			}
		} else {
			// If we allocate extra memory, we can convert y = (x - mean) * inv_std * scale + bias
			// to y = x * inv_std * scale + (bias - mean * inv_std * scale)
			// we can pre-compute nscale = inv_std * scale, nbias = bias - mean * inv_std * scale
			int count = 1;
			for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
				count *= rdim[x];
			float* const nscalep = (float*)ccv_nnc_stream_context_get_workspace(stream_context, sizeof(float) * count * 2, CCV_TENSOR_CPU_MEMORY);
			float* const nbiasp = nscalep + count;
			for (i[0] = 0; i[0] < rdim[0]; i[0]++)
			{
				float* const meanp0 = meanp + i[0] * saved_mean_stride[0];
				float* const varp0 = varp + i[0] * saved_inv_std_stride[0];
				float* const scalep0 = scalep + i[0] * scale_stride[0];
				float* const biasp0 = biasp + i[0] * bias_stride[0];
				float* const nscalep0 = nscalep + i[0] * rdim[1] * rdim[2] * rdim[3];
				float* const nbiasp0 = nbiasp + i[0] * rdim[1] * rdim[2] * rdim[3];
				for (i[1] = 0; i[1] < rdim[1]; i[1]++)
				{
					float* const meanp1 = meanp0 + i[1] * saved_mean_stride[1];
					float* const varp1 = varp0 + i[1] * saved_inv_std_stride[1];
					float* const scalep1 = scalep0 + i[1] * scale_stride[1];
					float* const biasp1 = biasp0 + i[1] * bias_stride[1];
					float* const nscalep1 = nscalep0 + i[1] * rdim[2] * rdim[3];
					float* const nbiasp1 = nbiasp0 + i[1] * rdim[2] * rdim[3];
					for (i[2] = 0; i[2] < rdim[2]; i[2]++)
					{
						float* const meanp2 = meanp1 + i[2] * saved_mean_stride[2];
						float* const varp2 = varp1 + i[2] * saved_inv_std_stride[2];
						float* const scalep2 = scalep1 + i[2] * scale_stride[2];
						float* const biasp2 = biasp1 + i[2] * bias_stride[2];
						float* const nscalep2 = nscalep1 + i[2] * rdim[3];
						float* const nbiasp2 = nbiasp1 + i[2] * rdim[3];
						for (x = 0; x < rdim[3]; x++)
						{
							const float w = varp2[x] * scalep2[x];
							nscalep2[x] = w;
							nbiasp2[x] = biasp2[x] - meanp2[x] * w;
						}
					}
				}
			}
			float* const bp = b->data.f32;
			for (i[0] = 0; i[0] < adim[0]; i[0]++)
			{
				float* const ap0 = ap + i[0] * astride[0];
				float* const bp0 = bp + i[0] * bstride[0];
				float* const nscalep0 = rdim[0] == 1 ? nscalep : nscalep + i[0] * rdim[1] * rdim[2] * rdim[3];
				float* const nbiasp0 = rdim[0] == 1 ? nbiasp : nbiasp + i[0] * rdim[1] * rdim[2] * rdim[3];
				for (i[1] = 0; i[1] < adim[1]; i[1]++)
				{
					float* ap1 = ap0 + i[1] * astride[1];
					float* bp1 = bp0 + i[1] * bstride[1];
					float* const nscalep1 = rdim[1] == 1 ? nscalep0 : nscalep0 + i[1] * rdim[2] * rdim[3];
					float* const nbiasp1 = rdim[1] == 1 ? nbiasp0 : nbiasp0 + i[1] * rdim[2] * rdim[3];
					for (i[2] = 0; i[2] < adim[2]; i[2]++)
					{
						float* const nscalep2 = rdim[2] == 1 ? nscalep1 : nscalep1 + i[2] * rdim[3];
						float* const nbiasp2 = rdim[2] == 1 ? nbiasp1 : nbiasp1 + i[2] * rdim[3];
						if (rdim[3] == 1)
							for (x = 0; x < adim[3]; x++)
								bp1[x] = ap1[x] * nscalep2[0] + nbiasp2[0];
						else
							for (x = 0; x < adim[3]; x++)
								bp1[x] = ap1[x] * nscalep2[x] + nbiasp2[x];
						ap1 += astride[2];
						bp1 += bstride[2];
					}
				}
			}
		}
	} else {
		assert(output_size >= 1);
		int mean_stride[CCV_NNC_MAX_DIM_ALLOC];
		int var_stride[CCV_NNC_MAX_DIM_ALLOC];
		ccv_nnc_tensor_view_get_stride(mean, mean_stride);
		ccv_nnc_tensor_view_get_stride(var, var_stride);
		int i[CCV_NNC_MAX_DIM + 2];
		int x;
		assert(!(flags & CCV_NNC_ZERO_MEMORY_ALLOC));
		int count = 1;
		for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
			count *= rdim[x];
		float* const meanp = mean->data.f32;
		float* const varp = var->data.f32;
		float* const scalep = scale->data.f32;
		float* const biasp = bias->data.f32;
		float* const nscalep = (float*)ccv_nnc_stream_context_get_workspace(stream_context, sizeof(float) * count * 2, CCV_TENSOR_CPU_MEMORY);
		float* const nbiasp = nscalep + count;
		for (i[0] = 0; i[0] < rdim[0]; i[0]++)
		{
			float* const meanp0 = meanp + i[0] * mean_stride[0];
			float* const varp0 = varp + i[0] * var_stride[0];
			float* const scalep0 = scalep + i[0] * scale_stride[0];
			float* const biasp0 = biasp + i[0] * bias_stride[0];
			float* const nscalep0 = nscalep + i[0] * rdim[1] * rdim[2] * rdim[3];
			float* const nbiasp0 = nbiasp + i[0] * rdim[1] * rdim[2] * rdim[3];
			for (i[1] = 0; i[1] < rdim[1]; i[1]++)
			{
				float* const meanp1 = meanp0 + i[1] * mean_stride[1];
				float* const varp1 = varp0 + i[1] * var_stride[1];
				float* const scalep1 = scalep0 + i[1] * scale_stride[1];
				float* const biasp1 = biasp0 + i[1] * bias_stride[1];
				float* const nscalep1 = nscalep0 + i[1] * rdim[2] * rdim[3];
				float* const nbiasp1 = nbiasp0 + i[1] * rdim[2] * rdim[3];
				for (i[2] = 0; i[2] < rdim[2]; i[2]++)
				{
					float* const meanp2 = meanp1 + i[2] * mean_stride[2];
					float* const varp2 = varp1 + i[2] * var_stride[2];
					float* const scalep2 = scalep1 + i[2] * scale_stride[2];
					float* const biasp2 = biasp1 + i[2] * bias_stride[2];
					float* const nscalep2 = nscalep1 + i[2] * rdim[3];
					float* const nbiasp2 = nbiasp1 + i[2] * rdim[3];
					for (x = 0; x < rdim[3]; x++)
					{
						const float w = scalep2[x] / (sqrtf(varp2[x]) + epsilon);
						nscalep2[x] = w;
						nbiasp2[x] = biasp2[x] - meanp2[x] * w;
					}
				}
			}
		}
		float* const ap = a->data.f32;
		float* const bp = b->data.f32;
		for (i[0] = 0; i[0] < adim[0]; i[0]++)
		{
			float* const ap0 = ap + i[0] * astride[0];
			float* const bp0 = bp + i[0] * bstride[0];
			float* const nscalep0 = rdim[0] == 1 ? nscalep : nscalep + i[0] * rdim[1] * rdim[2] * rdim[3];
			float* const nbiasp0 = rdim[0] == 1 ? nbiasp : nbiasp + i[0] * rdim[1] * rdim[2] * rdim[3];
			for (i[1] = 0; i[1] < adim[1]; i[1]++)
			{
				float* ap1 = ap0 + i[1] * astride[1];
				float* bp1 = bp0 + i[1] * bstride[1];
				float* const nscalep1 = rdim[1] == 1 ? nscalep0 : nscalep0 + i[1] * rdim[2] * rdim[3];
				float* const nbiasp1 = rdim[1] == 1 ? nbiasp0 : nbiasp0 + i[1] * rdim[2] * rdim[3];
				for (i[2] = 0; i[2] < adim[2]; i[2]++)
				{
					float* const nscalep2 = rdim[2] == 1 ? nscalep1 : nscalep1 + i[2] * rdim[3];
					float* const nbiasp2 = rdim[2] == 1 ? nbiasp1 : nbiasp1 + i[2] * rdim[3];
					if (rdim[3] == 1)
						for (x = 0; x < adim[3]; x++)
							bp1[x] = ap1[x] * nscalep2[0] + nbiasp2[0];
					else
						for (x = 0; x < adim[3]; x++)
							bp1[x] = ap1[x] * nscalep2[x] + nbiasp2[x];
					ap1 += astride[2];
					bp1 += bstride[2];
				}
			}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_batch_norm_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 15);
	assert(output_size >= 3);
	ccv_nnc_tensor_view_t* const g = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[5];
	ccv_nnc_tensor_view_t* const scale = (ccv_nnc_tensor_view_t*)inputs[6];
	ccv_nnc_tensor_view_t* const saved_mean = (ccv_nnc_tensor_view_t*)inputs[13];
	ccv_nnc_tensor_view_t* const saved_inv_std = (ccv_nnc_tensor_view_t*)inputs[14];
	ccv_nnc_tensor_view_t* const h = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* const dscale = (ccv_nnc_tensor_view_t*)outputs[1];
	ccv_nnc_tensor_view_t* const dbias = (ccv_nnc_tensor_view_t*)outputs[2];
	assert(ccv_nnc_tensor_nd(g->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(a->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(h->info.dim) <= CCV_NNC_MAX_DIM + 2);
	// Assuming this is float 32.
	int gdim[CCV_NNC_MAX_DIM_ALLOC];
	int rdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(g, gdim);
	ccv_nnc_tensor_view_get_dim(scale, rdim);
	assert(ccv_nnc_tensor_view_check_dim(saved_mean, rdim));
	assert(ccv_nnc_tensor_view_check_dim(saved_inv_std, rdim));
	assert(ccv_nnc_tensor_view_check_dim(dscale, rdim));
	assert(ccv_nnc_tensor_view_check_dim(dbias, rdim));
	assert(ccv_nnc_tensor_view_check_dim(a, gdim));
	assert(ccv_nnc_tensor_view_check_dim(h, gdim));
	_ccv_nnc_reduce_sum_forw_cpu_ref(g, dbias);
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	int gstride[CCV_NNC_MAX_DIM_ALLOC];
	int hstride[CCV_NNC_MAX_DIM_ALLOC];
	int mean_stride[CCV_NNC_MAX_DIM_ALLOC];
	int inv_std_stride[CCV_NNC_MAX_DIM_ALLOC];
	int dscale_stride[CCV_NNC_MAX_DIM_ALLOC];
	int dbias_stride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_stride(a, astride);
	ccv_nnc_tensor_view_get_stride(g, gstride);
	ccv_nnc_tensor_view_get_stride(h, hstride);
	ccv_nnc_tensor_view_get_stride(saved_mean, mean_stride);
	ccv_nnc_tensor_view_get_stride(saved_inv_std, inv_std_stride);
	ccv_nnc_tensor_view_get_stride(dscale, dscale_stride);
	ccv_nnc_tensor_view_get_stride(dbias, dbias_stride);
	// Need to allocate two additional memory:
	// 1. normalized a;
	// 2. scale * inv_std / batch_size;
	assert(!(flags & CCV_NNC_ZERO_MEMORY_ALLOC));
	int x;
	int batch_size = 1;
	for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
		batch_size *= gdim[x];
	for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
		batch_size /= rdim[x];
	int gcount = 1, rcount = 1;
	for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
		gcount *= gdim[x], rcount *= rdim[x];
	float* const ah = (float*)ccv_nnc_stream_context_get_workspace(stream_context, sizeof(float) * gcount + sizeof(float) * rcount, CCV_TENSOR_CPU_MEMORY);
	float* const sisb = ah + gcount;
	ccv_nnc_tensor_t sisbt = ccv_nnc_tensor(sisb, scale->info, 0);
	_ccv_nnc_mul_forw_cpu_ref(1. / batch_size, scale, saved_inv_std, (ccv_nnc_tensor_view_t*)&sisbt);
	int i[CCV_NNC_MAX_DIM + 2];
	float* const ap = a->data.f32;
	float* ahp = ah;
	float* const meanp = saved_mean->data.f32;
	float* const inv_stdp = saved_inv_std->data.f32;
	for (i[0] = 0; i[0] < gdim[0]; i[0]++)
	{
		float* const ap0 = ap + i[0] * astride[0];
		float* const meanp0 = rdim[0] == 1 ? meanp : meanp + i[0] * mean_stride[0];
		float* const inv_stdp0 = rdim[0] == 1 ? inv_stdp : inv_stdp + i[0] * inv_std_stride[0];
		for (i[1] = 0; i[1] < gdim[1]; i[1]++)
		{
			float* ap1 = ap0 + i[1] * astride[1];
			float* const meanp1 = rdim[1] == 1 ? meanp0 : meanp0 + i[1] * mean_stride[1];
			float* const inv_stdp1 = rdim[1] == 1 ? inv_stdp0 : inv_stdp0 + i[1] * inv_std_stride[1];
			for (i[2] = 0; i[2] < gdim[2]; i[2]++)
			{
				float* const meanp2 = rdim[2] == 1 ? meanp1 : meanp1 + i[2] * mean_stride[2];
				float* const inv_stdp2 = rdim[2] == 1 ? inv_stdp1 : inv_stdp1 + i[2] * inv_std_stride[2];
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
	ccv_nnc_tensor_zero(dscale);
	ahp = ah;
	float* const gp = g->data.f32;
	float* const dscalep = dscale->data.f32;
	for (i[0] = 0; i[0] < gdim[0]; i[0]++)
	{
		float* const gp0 = gp + i[0] * gstride[0];
		float* const dscalep0 = rdim[0] == 1 ? dscalep : dscalep + i[0] * dscale_stride[0];
		for (i[1] = 0; i[1] < gdim[1]; i[1]++)
		{
			float* gp1 = gp0 + i[1] * gstride[1];
			float* const dscalep1 = rdim[1] == 1 ? dscalep0 : dscalep0 + i[1] * dscale_stride[1];
			for (i[2] = 0; i[2] < gdim[2]; i[2]++)
			{
				float* const dscalep2 = rdim[2] == 1 ? dscalep1 : dscalep1 + i[2] * dscale_stride[2];
				if (rdim[3] == 1)
					for (x = 0; x < gdim[3]; x++)
						dscalep2[0] += ahp[x] * gp1[x];
				else
					for (x = 0; x < gdim[3]; x++)
						dscalep2[x] += ahp[x] * gp1[x];
				gp1 += gstride[2];
				ahp += gdim[3];
			}
		}
	}
	// Now the part to compute dx (h).
	float* const hp = h->data.f32;
	ahp = ah;
	float* const sisbp = sisb;
	float* const dbiasp = dbias->data.f32;
	for (i[0] = 0; i[0] < gdim[0]; i[0]++)
	{
		float* const gp0 = gp + i[0] * gstride[0];
		float* const hp0 = hp + i[0] * hstride[0];
		float* const sisbp0 = rdim[0] == 1 ? sisbp : sisbp + i[0] * rdim[1] * rdim[2] * rdim[3];
		float* const dscalep0 = rdim[0] == 1 ? dscalep : dscalep + i[0] * dscale_stride[0];
		float* const dbiasp0 = rdim[0] == 1 ? dbiasp : dbiasp + i[0] * dbias_stride[0];
		for (i[1] = 0; i[1] < gdim[1]; i[1]++)
		{
			float* gp1 = gp0 + i[1] * gstride[1];
			float* hp1 = hp0 + i[1] * hstride[1];
			float* const sisbp1 = rdim[1] == 1 ? sisbp0 : sisbp0 + i[1] * rdim[2] * rdim[3];
			float* const dscalep1 = rdim[1] == 1 ? dscalep0 : dscalep0 + i[1] * dscale_stride[1];
			float* const dbiasp1 = rdim[1] == 1 ? dbiasp0 : dbiasp0 + i[1] * dbias_stride[1];
			for (i[2] = 0; i[2] < gdim[2]; i[2]++)
			{
				float* const sisbp2 = rdim[2] == 1 ? sisbp1 : sisbp1 + i[2] * rdim[3];
				float* const dscalep2 = rdim[2] == 1 ? dscalep1 : dscalep1 + i[2] * dscale_stride[2];
				float* const dbiasp2 = rdim[2] == 1 ? dbiasp1 : dbiasp1 + i[2] * dbias_stride[2];
				if (rdim[3] == 1)
					for (x = 0; x < gdim[3]; x++)
						hp1[x] = sisbp2[0] * (batch_size * gp1[x] - dbiasp2[0] - ahp[x] * dscalep2[0]);
				else
					for (x = 0; x < gdim[3]; x++)
						hp1[x] = sisbp2[x] * (batch_size * gp1[x] - dbiasp2[x] - ahp[x] * dscalep2[x]);
				gp1 += gstride[2];
				hp1 += hstride[2];
				ahp += gdim[3];
			}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_BATCH_NORM_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_batch_norm_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_BATCH_NORM_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_batch_norm_back;
}
