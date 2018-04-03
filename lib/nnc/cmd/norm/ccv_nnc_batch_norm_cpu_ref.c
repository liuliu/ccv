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

// Shared methods.
#include "../_ccv_nnc_cpu_ref.h"

static int _ccv_nnc_batch_norm_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 5);
	ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const scale = (ccv_nnc_tensor_view_t*)inputs[1];
	ccv_nnc_tensor_view_t* const bias = (ccv_nnc_tensor_view_t*)inputs[2];
	ccv_nnc_tensor_view_t* const mean = (ccv_nnc_tensor_view_t*)inputs[3];
	ccv_nnc_tensor_view_t* const var = (ccv_nnc_tensor_view_t*)inputs[4];
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(a->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	// Assuming this is float 32.
	int adim[CCV_NNC_MAX_DIM + 2];
	int rdim[CCV_NNC_MAX_DIM + 2];
	ccv_nnc_tensor_view_get_dim(a, adim);
	ccv_nnc_tensor_view_get_dim(scale, rdim);
	assert(ccv_nnc_tensor_view_check_dim(bias, rdim));
	assert(ccv_nnc_tensor_view_check_dim(mean, rdim));
	assert(ccv_nnc_tensor_view_check_dim(var, rdim));
	assert(ccv_nnc_tensor_view_check_dim(b, adim));
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	int ainc[CCV_NNC_MAX_DIM + 2];
	int binc[CCV_NNC_MAX_DIM + 2];
	int scale_inc[CCV_NNC_MAX_DIM + 2];
	int bias_inc[CCV_NNC_MAX_DIM + 2];
	ccv_nnc_tensor_view_get_inc(a, ainc);
	ccv_nnc_tensor_view_get_inc(scale, scale_inc);
	ccv_nnc_tensor_view_get_inc(bias, bias_inc);
	ccv_nnc_tensor_view_get_inc(b, binc);
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
		int saved_mean_inc[CCV_NNC_MAX_DIM + 2];
		int saved_inv_std_inc[CCV_NNC_MAX_DIM + 2];
		ccv_nnc_tensor_view_get_inc(saved_mean, saved_mean_inc);
		ccv_nnc_tensor_view_get_inc(saved_inv_std, saved_inv_std_inc);
		int i[CCV_NNC_MAX_DIM + 2];
		int x;
		int batch_size = 1;
		for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
			batch_size *= adim[x];
		for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
			batch_size /= rdim[x];
		float inv_batch_size = 1. / batch_size;
		_ccv_nnc_reduce_sum_forw_cpu_ref(a, saved_mean);
		_ccv_nnc_mul_forw_cpu_ref(inv_batch_size, saved_mean, 0, saved_mean);
		// Copy this into running mean / var.
		_ccv_nnc_add_forw_cpu_ref(cmd.info.bnorm.momentum, 1. - cmd.info.bnorm.momentum, mean, saved_mean, mean);
		ccv_nnc_tensor_zero(saved_inv_std);
		float* ap = a->data.f32;
		float* const meanp = saved_mean->data.f32;
		float* const varp = saved_inv_std->data.f32;
		for (i[0] = 0; i[0] < adim[0]; i[0]++)
		{
			float* const meanp0 = rdim[0] == 1 ? meanp : meanp + i[0] * saved_mean_inc[1] * saved_mean_inc[2] * saved_mean_inc[3];
			float* const varp0 = rdim[0] == 1 ? varp : varp + i[0] * saved_inv_std_inc[1] * saved_inv_std_inc[2] * saved_inv_std_inc[3];
			for (i[1] = 0; i[1] < adim[1]; i[1]++)
			{
				float* const meanp1 = rdim[1] == 1 ? meanp0 : meanp0 + i[1] * saved_mean_inc[2] * saved_mean_inc[3];
				float* const varp1 = rdim[1] == 1 ? varp0 : varp0 + i[1] * saved_inv_std_inc[2] * saved_inv_std_inc[3];
				for (i[2] = 0; i[2] < adim[2]; i[2]++)
				{
					float* const meanp2 = rdim[2] == 1 ? meanp1 : meanp1 + i[2] * saved_mean_inc[3];
					float* const varp2 = rdim[2] == 1 ? varp1 : varp1 + i[2] * saved_inv_std_inc[3];
					if (rdim[3] == 1)
						for (x = 0; x < adim[3]; x++)
						{
							float w = ap[x] - meanp2[0];
							varp2[0] += w * w;
						}
					else
						for (x = 0; x < adim[3]; x++)
						{
							float w = ap[x] - meanp2[x];
							varp2[x] += w * w;
						}
					ap += ainc[3];
				}
				ap += (ainc[2] - adim[2]) * ainc[3];
			}
			ap += (ainc[1] - adim[1]) * ainc[2] * ainc[3];
		}
		for (i[0] = 0; i[0] < rdim[0]; i[0]++)
		{
			float* const varp0 = varp + i[0] * saved_inv_std_inc[1] * saved_inv_std_inc[2] * saved_inv_std_inc[3];
			for (i[1] = 0; i[1] < rdim[1]; i[1]++)
			{
				float* const varp1 = varp0 + i[1] * saved_inv_std_inc[2] * saved_inv_std_inc[3];
				for (i[2] = 0; i[2] < rdim[2]; i[2]++)
				{
					float* const varp2 = varp1 + i[2] * saved_inv_std_inc[3];
					for (x = 0; x < rdim[3]; x++)
						varp2[x] = varp2[x] * inv_batch_size;
				}
			}
		}
		_ccv_nnc_add_forw_cpu_ref(cmd.info.bnorm.momentum, 1. - cmd.info.bnorm.momentum, var, saved_inv_std, var);
		for (i[0] = 0; i[0] < rdim[0]; i[0]++)
		{
			float* const varp0 = varp + i[0] * saved_inv_std_inc[1] * saved_inv_std_inc[2] * saved_inv_std_inc[3];
			for (i[1] = 0; i[1] < rdim[1]; i[1]++)
			{
				float* const varp1 = varp0 + i[1] * saved_inv_std_inc[2] * saved_inv_std_inc[3];
				for (i[2] = 0; i[2] < rdim[2]; i[2]++)
				{
					float* const varp2 = varp1 + i[2] * saved_inv_std_inc[3];
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
			ap = a->data.f32;
			float* bp = b->data.f32;
			for (i[0] = 0; i[0] < adim[0]; i[0]++)
			{
				float* const meanp0 = rdim[0] == 1 ? meanp : meanp + i[0] * saved_mean_inc[1] * saved_mean_inc[2] * saved_mean_inc[3];
				float* const varp0 = rdim[0] == 1 ? varp : varp + i[0] * saved_inv_std_inc[1] * saved_inv_std_inc[2] * saved_inv_std_inc[3];
				float* const scalep0 = rdim[0] == 1 ? scalep : scalep + i[0] * scale_inc[1] * scale_inc[2] * scale_inc[3];
				float* const biasp0 = rdim[0] == 1 ? biasp : biasp + i[0] * bias_inc[1] * bias_inc[2] * bias_inc[3];
				for (i[1] = 0; i[1] < adim[1]; i[1]++)
				{
					float* const meanp1 = rdim[1] == 1 ? meanp0 : meanp0 + i[1] * saved_mean_inc[2] * saved_mean_inc[3];
					float* const varp1 = rdim[1] == 1 ? varp0 : varp0 + i[1] * saved_inv_std_inc[2] * saved_inv_std_inc[3];
					float* const scalep1 = rdim[1] == 1 ? scalep0 : scalep0 + i[1] * scale_inc[2] * scale_inc[3];
					float* const biasp1 = rdim[1] == 1 ? biasp0 : biasp0 + i[1] * bias_inc[2] * bias_inc[3];
					for (i[2] = 0; i[2] < adim[2]; i[2]++)
					{
						float* const meanp2 = rdim[2] == 1 ? meanp1 : meanp1 + i[2] * saved_mean_inc[3];
						float* const varp2 = rdim[2] == 1 ? varp1 : varp1 + i[2] * saved_inv_std_inc[3];
						float* const scalep2 = rdim[2] == 1 ? scalep1 : scalep1 + i[2] * scale_inc[3];
						float* const biasp2 = rdim[2] == 1 ? biasp1 : biasp1 + i[2] * bias_inc[3];
						if (rdim[3] == 1)
							for (x = 0; x < adim[3]; x++)
								bp[x] = (ap[x] - meanp2[0]) * varp2[0] * scalep2[0] + biasp2[0];
						else
							for (x = 0; x < adim[3]; x++)
								bp[x] = (ap[x] - meanp2[x]) * varp2[x] * scalep2[x] + biasp2[x];
						ap += ainc[3];
						bp += binc[3];
					}
					ap += (ainc[2] - adim[2]) * ainc[3];
					bp += (binc[2] - adim[2]) * binc[3];
				}
				ap += (ainc[1] - adim[1]) * ainc[2] * ainc[3];
				bp += (binc[1] - adim[1]) * binc[2] * binc[3];
			}
		} else {
			// If we allocate extra memory, we can convert y = (x - mean) * inv_std * scale + bias
			// to y = x * inv_std * scale + (bias - mean * inv_std * scale)
			// we can pre-compute nscale = inv_std * scale, nbias = bias - mean * inv_std * scale
			int count = 1;
			for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
				count *= rdim[x];
			float* const nscalep = ccmalloc(sizeof(float) * count * 2);
			float* const nbiasp = nscalep + count;
			for (i[0] = 0; i[0] < rdim[0]; i[0]++)
			{
				float* const meanp0 = meanp + i[0] * saved_mean_inc[1] * saved_mean_inc[2] * saved_mean_inc[3];
				float* const varp0 = varp + i[0] * saved_inv_std_inc[1] * saved_inv_std_inc[2] * saved_inv_std_inc[3];
				float* const scalep0 = scalep + i[0] * scale_inc[1] * scale_inc[2] * scale_inc[3];
				float* const biasp0 = biasp + i[0] * bias_inc[1] * bias_inc[2] * bias_inc[3];
				float* const nscalep0 = nscalep + i[0] * rdim[1] * rdim[2] * rdim[3];
				float* const nbiasp0 = nbiasp + i[0] * rdim[1] * rdim[2] * rdim[3];
				for (i[1] = 0; i[1] < rdim[1]; i[1]++)
				{
					float* const meanp1 = meanp0 + i[1] * saved_mean_inc[2] * saved_mean_inc[3];
					float* const varp1 = varp0 + i[1] * saved_inv_std_inc[2] * saved_inv_std_inc[3];
					float* const scalep1 = scalep0 + i[1] * scale_inc[2] * scale_inc[3];
					float* const biasp1 = biasp0 + i[1] * bias_inc[2] * bias_inc[3];
					float* const nscalep1 = nscalep0 + i[1] * rdim[2] * rdim[3];
					float* const nbiasp1 = nbiasp0 + i[1] * rdim[2] * rdim[3];
					for (i[2] = 0; i[2] < rdim[2]; i[2]++)
					{
						float* const meanp2 = meanp1 + i[2] * saved_mean_inc[3];
						float* const varp2 = varp1 + i[2] * saved_inv_std_inc[3];
						float* const scalep2 = scalep1 + i[2] * scale_inc[3];
						float* const biasp2 = biasp1 + i[2] * bias_inc[3];
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
			ap = a->data.f32;
			float* bp = b->data.f32;
			for (i[0] = 0; i[0] < adim[0]; i[0]++)
			{
				float* const nscalep0 = rdim[0] == 1 ? nscalep : scalep + i[0] * rdim[1] * rdim[2] * rdim[3];
				float* const nbiasp0 = rdim[0] == 1 ? nbiasp : biasp + i[0] * rdim[1] * rdim[2] * rdim[3];
				for (i[1] = 0; i[1] < adim[1]; i[1]++)
				{
					float* const nscalep1 = rdim[1] == 1 ? nscalep0 : nscalep0 + i[1] * rdim[2] * rdim[3];
					float* const nbiasp1 = rdim[1] == 1 ? nbiasp0 : nbiasp0 + i[1] * rdim[2] * rdim[3];
					for (i[2] = 0; i[2] < adim[2]; i[2]++)
					{
						float* const nscalep2 = rdim[2] == 1 ? nscalep1 : nscalep1 + i[2] * rdim[3];
						float* const nbiasp2 = rdim[2] == 1 ? nbiasp1 : nbiasp1 + i[2] * rdim[3];
						if (rdim[3] == 1)
							for (x = 0; x < adim[3]; x++)
								bp[x] = ap[x] * nscalep2[0] + nbiasp2[0];
						else
							for (x = 0; x < adim[3]; x++)
								bp[x] = ap[x] * nscalep2[x] + nbiasp2[x];
						ap += ainc[3];
						bp += binc[3];
					}
					ap += (ainc[2] - adim[2]) * ainc[3];
					bp += (binc[2] - adim[2]) * binc[3];
				}
				ap += (ainc[1] - adim[1]) * ainc[2] * ainc[3];
				bp += (binc[1] - adim[1]) * binc[2] * binc[3];
			}
			ccfree(nscalep);
		}
	} else {
		assert(output_size == 1);
		int mean_inc[CCV_NNC_MAX_DIM + 2];
		int var_inc[CCV_NNC_MAX_DIM + 2];
		ccv_nnc_tensor_view_get_inc(mean, mean_inc);
		ccv_nnc_tensor_view_get_inc(var, var_inc);
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
		float* const nscalep = ccmalloc(sizeof(float) * count * 2);
		float* const nbiasp = nscalep + count;
		for (i[0] = 0; i[0] < rdim[0]; i[0]++)
		{
			float* const meanp0 = meanp + i[0] * mean_inc[1] * mean_inc[2] * mean_inc[3];
			float* const varp0 = varp + i[0] * var_inc[1] * var_inc[2] * var_inc[3];
			float* const scalep0 = scalep + i[0] * scale_inc[1] * scale_inc[2] * scale_inc[3];
			float* const biasp0 = biasp + i[0] * bias_inc[1] * bias_inc[2] * bias_inc[3];
			float* const nscalep0 = nscalep + i[0] * rdim[1] * rdim[2] * rdim[3];
			float* const nbiasp0 = nbiasp + i[0] * rdim[1] * rdim[2] * rdim[3];
			for (i[1] = 0; i[1] < rdim[1]; i[1]++)
			{
				float* const meanp1 = meanp0 + i[1] * mean_inc[2] * mean_inc[3];
				float* const varp1 = varp0 + i[1] * var_inc[2] * var_inc[3];
				float* const scalep1 = scalep0 + i[1] * scale_inc[2] * scale_inc[3];
				float* const biasp1 = biasp0 + i[1] * bias_inc[2] * bias_inc[3];
				float* const nscalep1 = nscalep0 + i[1] * rdim[2] * rdim[3];
				float* const nbiasp1 = nbiasp0 + i[1] * rdim[2] * rdim[3];
				for (i[2] = 0; i[2] < rdim[2]; i[2]++)
				{
					float* const meanp2 = meanp1 + i[2] * mean_inc[3];
					float* const varp2 = varp1 + i[2] * var_inc[3];
					float* const scalep2 = scalep1 + i[2] * scale_inc[3];
					float* const biasp2 = biasp1 + i[2] * bias_inc[3];
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
		float* ap = a->data.f32;
		float* bp = b->data.f32;
		for (i[0] = 0; i[0] < adim[0]; i[0]++)
		{
			float* const nscalep0 = rdim[0] == 1 ? nscalep : scalep + i[0] * rdim[1] * rdim[2] * rdim[3];
			float* const nbiasp0 = rdim[0] == 1 ? nbiasp : biasp + i[0] * rdim[1] * rdim[2] * rdim[3];
			for (i[1] = 0; i[1] < adim[1]; i[1]++)
			{
				float* const nscalep1 = rdim[1] == 1 ? nscalep0 : nscalep0 + i[1] * rdim[2] * rdim[3];
				float* const nbiasp1 = rdim[1] == 1 ? nbiasp0 : nbiasp0 + i[1] * rdim[2] * rdim[3];
				for (i[2] = 0; i[2] < adim[2]; i[2]++)
				{
					float* const nscalep2 = rdim[2] == 1 ? nscalep1 : nscalep1 + i[2] * rdim[3];
					float* const nbiasp2 = rdim[2] == 1 ? nbiasp1 : nbiasp1 + i[2] * rdim[3];
					if (rdim[3] == 1)
						for (x = 0; x < adim[3]; x++)
							bp[x] = ap[x] * nscalep2[0] + nbiasp2[0];
					else
						for (x = 0; x < adim[3]; x++)
							bp[x] = ap[x] * nscalep2[x] + nbiasp2[x];
					ap += ainc[3];
					bp += binc[3];
				}
				ap += (ainc[2] - adim[2]) * ainc[3];
				bp += (binc[2] - adim[2]) * binc[3];
			}
			ap += (ainc[1] - adim[1]) * ainc[2] * ainc[3];
			bp += (binc[1] - adim[1]) * binc[2] * binc[3];
		}
		ccfree(nscalep);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_batch_norm_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 15);
	assert(output_size == 5);
	ccv_nnc_tensor_view_t* const g = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[5];
	ccv_nnc_tensor_view_t* const scale = (ccv_nnc_tensor_view_t*)inputs[6];
	ccv_nnc_tensor_view_t* const saved_mean = (ccv_nnc_tensor_view_t*)inputs[13];
	ccv_nnc_tensor_view_t* const saved_inv_std = (ccv_nnc_tensor_view_t*)inputs[14];
	ccv_nnc_tensor_view_t* const h = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* const dscale = (ccv_nnc_tensor_view_t*)outputs[1];
	ccv_nnc_tensor_view_t* const dbias = (ccv_nnc_tensor_view_t*)outputs[2];
	assert(g->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	assert(a->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	assert(h->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	// Assuming this is float 32.
	int gdim[CCV_NNC_MAX_DIM + 2];
	int rdim[CCV_NNC_MAX_DIM + 2];
	ccv_nnc_tensor_view_get_dim(g, gdim);
	ccv_nnc_tensor_view_get_dim(scale, rdim);
	assert(ccv_nnc_tensor_view_check_dim(saved_mean, rdim));
	assert(ccv_nnc_tensor_view_check_dim(saved_inv_std, rdim));
	assert(ccv_nnc_tensor_view_check_dim(dscale, rdim));
	assert(ccv_nnc_tensor_view_check_dim(dbias, rdim));
	assert(ccv_nnc_tensor_view_check_dim(a, gdim));
	assert(ccv_nnc_tensor_view_check_dim(h, gdim));
	_ccv_nnc_reduce_sum_forw_cpu_ref(g, dbias);
	int ainc[CCV_NNC_MAX_DIM + 2];
	int ginc[CCV_NNC_MAX_DIM + 2];
	int hinc[CCV_NNC_MAX_DIM + 2];
	int mean_inc[CCV_NNC_MAX_DIM + 2];
	int inv_std_inc[CCV_NNC_MAX_DIM + 2];
	int dscale_inc[CCV_NNC_MAX_DIM + 2];
	int dbias_inc[CCV_NNC_MAX_DIM + 2];
	ccv_nnc_tensor_view_get_inc(a, ainc);
	ccv_nnc_tensor_view_get_inc(g, ginc);
	ccv_nnc_tensor_view_get_inc(h, hinc);
	ccv_nnc_tensor_view_get_inc(saved_mean, mean_inc);
	ccv_nnc_tensor_view_get_inc(saved_inv_std, inv_std_inc);
	ccv_nnc_tensor_view_get_inc(dscale, dscale_inc);
	ccv_nnc_tensor_view_get_inc(dbias, dbias_inc);
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
	float* const ah = ccmalloc(sizeof(float) * gcount + sizeof(float) * rcount);
	float* const sisb = ah + gcount;
	ccv_nnc_tensor_t sisbt = ccv_nnc_tensor(sisb, scale->info, 0);
	_ccv_nnc_mul_forw_cpu_ref(1. / batch_size, scale, saved_inv_std, (ccv_nnc_tensor_view_t*)&sisbt);
	int i[CCV_NNC_MAX_DIM + 2];
	float* ap = a->data.f32;
	float* ahp = ah;
	float* const meanp = saved_mean->data.f32;
	float* const inv_stdp = saved_inv_std->data.f32;
	for (i[0] = 0; i[0] < gdim[0]; i[0]++)
	{
		float* const meanp0 = rdim[0] == 1 ? meanp : meanp + i[0] * mean_inc[1] * mean_inc[2] * mean_inc[3];
		float* const inv_stdp0 = rdim[0] == 1 ? inv_stdp : inv_stdp + i[0] * inv_std_inc[1] * inv_std_inc[2] * inv_std_inc[3];
		for (i[1] = 0; i[1] < gdim[1]; i[1]++)
		{
			float* const meanp1 = rdim[1] == 1 ? meanp0 : meanp0 + i[1] * mean_inc[2] * mean_inc[3];
			float* const inv_stdp1 = rdim[1] == 1 ? inv_stdp0 : inv_stdp0 + i[1] * inv_std_inc[2] * inv_std_inc[3];
			for (i[2] = 0; i[2] < gdim[2]; i[2]++)
			{
				float* const meanp2 = rdim[2] == 1 ? meanp1 : meanp1 + i[2] * mean_inc[3];
				float* const inv_stdp2 = rdim[2] == 1 ? inv_stdp1 : inv_stdp1 + i[2] * inv_std_inc[3];
				if (rdim[3] == 1)
					for (x = 0; x < gdim[3]; x++)
						ahp[x] = (ap[x] - meanp2[0]) * inv_stdp2[0];
				else
					for (x = 0; x < gdim[3]; x++)
						ahp[x] = (ap[x] - meanp2[x]) * inv_stdp2[x];
				ap += ainc[3];
				ahp += gdim[3];
			}
			ap += (ainc[2] - gdim[2]) * ainc[3];
		}
		ap += (ainc[1] - gdim[1]) * ainc[2] * ainc[3];
	}
	ccv_nnc_tensor_zero(dscale);
	ahp = ah;
	float* gp = g->data.f32;
	float* const dscalep = dscale->data.f32;
	for (i[0] = 0; i[0] < gdim[0]; i[0]++)
	{
		float* const dscalep0 = rdim[0] == 1 ? dscalep : dscalep + i[0] * dscale_inc[1] * dscale_inc[2] * dscale_inc[3];
		for (i[1] = 0; i[1] < gdim[1]; i[1]++)
		{
			float* const dscalep1 = rdim[1] == 1 ? dscalep0 : dscalep0 + i[1] * dscale_inc[2] * dscale_inc[3];
			for (i[2] = 0; i[2] < gdim[2]; i[2]++)
			{
				float* const dscalep2 = rdim[2] == 1 ? dscalep1 : dscalep1 + i[2] * dscale_inc[3];
				if (rdim[3] == 1)
					for (x = 0; x < gdim[3]; x++)
						dscalep2[0] += ahp[x] * gp[x];
				else
					for (x = 0; x < gdim[3]; x++)
						dscalep2[x] += ahp[x] * gp[x];
				gp += ginc[3];
				ahp += gdim[3];
			}
			gp += (ginc[2] - gdim[2]) * ginc[3];
		}
		gp += (ginc[1] - gdim[1]) * ginc[2] * ginc[3];
	}
	// Now the part to compute dx (h).
	ap = a->data.f32;
	gp = g->data.f32;
	float* hp = h->data.f32;
	ahp = ah;
	float* const sisbp = sisb;
	float* const dbiasp = dbias->data.f32;
	for (i[0] = 0; i[0] < gdim[0]; i[0]++)
	{
		float* const sisbp0 = rdim[0] == 1 ? sisbp : sisbp + i[0] * rdim[1] * rdim[2] * rdim[3];
		float* const dscalep0 = rdim[0] == 1 ? dscalep : dscalep + i[0] * dscale_inc[1] * dscale_inc[2] * dscale_inc[3];
		float* const dbiasp0 = rdim[0] == 1 ? dbiasp : dbiasp + i[0] * dbias_inc[1] * dbias_inc[2] * dbias_inc[3];
		for (i[1] = 0; i[1] < gdim[1]; i[1]++)
		{
			float* const sisbp1 = rdim[1] == 1 ? sisbp0 : sisbp0 + i[1] * rdim[2] * rdim[3];
			float* const dscalep1 = rdim[1] == 1 ? dscalep0 : dscalep0 + i[1] * dscale_inc[2] * dscale_inc[3];
			float* const dbiasp1 = rdim[1] == 1 ? dbiasp0 : dbiasp0 + i[1] * dbias_inc[2] * dbias_inc[3];
			for (i[2] = 0; i[2] < gdim[2]; i[2]++)
			{
				float* const sisbp2 = rdim[2] == 1 ? sisbp1 : sisbp1 + i[2] * rdim[3];
				float* const dscalep2 = rdim[2] == 1 ? dscalep1 : dscalep1 + i[2] * dscale_inc[3];
				float* const dbiasp2 = rdim[2] == 1 ? dbiasp1 : dbiasp1 + i[2] * dbias_inc[3];
				if (rdim[3] == 1)
					for (x = 0; x < gdim[3]; x++)
						hp[x] = sisbp2[0] * (batch_size * gp[x] - dbiasp2[0] - ahp[x] * dscalep2[0]);
				else
					for (x = 0; x < gdim[3]; x++)
						hp[x] = sisbp2[x] * (batch_size * gp[x] - dbiasp2[x] - ahp[x] * dscalep2[x]);
				ap += ainc[3];
				gp += ginc[3];
				hp += hinc[3];
				ahp += gdim[3];
			}
			ap += (ainc[2] - gdim[2]) * ainc[3];
			gp += (ginc[2] - gdim[2]) * ginc[3];
			hp += (hinc[2] - gdim[2]) * hinc[3];
		}
		ap += (ainc[1] - gdim[1]) * ainc[2] * ainc[3];
		gp += (ginc[1] - gdim[1]) * ginc[2] * ginc[3];
		hp += (hinc[1] - gdim[1]) * hinc[2] * hinc[3];
	}
	ccfree(ah);
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
