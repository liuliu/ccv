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
	if (!cmd.info.bnorm.is_test)
	{
		assert(output_size == 5);
		// Both are inplace.
		assert(inputs[3]->data.f32 == outputs[1]->data.f32);
		assert(inputs[4]->data.f32 == outputs[2]->data.f32);
		ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[0];
		ccv_nnc_tensor_view_t* const scale = (ccv_nnc_tensor_view_t*)inputs[1];
		ccv_nnc_tensor_view_t* const bias = (ccv_nnc_tensor_view_t*)inputs[2];
		ccv_nnc_tensor_view_t* const mean = (ccv_nnc_tensor_view_t*)inputs[3];
		ccv_nnc_tensor_view_t* const var = (ccv_nnc_tensor_view_t*)inputs[4];
		ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
		ccv_nnc_tensor_view_t* const saved_mean = (ccv_nnc_tensor_view_t*)outputs[3];
		ccv_nnc_tensor_view_t* const saved_inv_std = (ccv_nnc_tensor_view_t*)outputs[4];
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
		assert(ccv_nnc_tensor_view_check_dim(saved_mean, rdim));
		assert(ccv_nnc_tensor_view_check_dim(saved_inv_std, rdim));
		int ainc[CCV_NNC_MAX_DIM + 2];
		int binc[CCV_NNC_MAX_DIM + 2];
		int scale_inc[CCV_NNC_MAX_DIM + 2];
		int bias_inc[CCV_NNC_MAX_DIM + 2];
		int saved_mean_inc[CCV_NNC_MAX_DIM + 2];
		int saved_inv_std_inc[CCV_NNC_MAX_DIM + 2];
		assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
		ccv_nnc_tensor_view_get_inc(a, ainc);
		ccv_nnc_tensor_view_get_inc(scale, scale_inc);
		ccv_nnc_tensor_view_get_inc(bias, bias_inc);
		ccv_nnc_tensor_view_get_inc(b, binc);
		ccv_nnc_tensor_view_get_inc(saved_mean, saved_mean_inc);
		ccv_nnc_tensor_view_get_inc(saved_inv_std, saved_inv_std_inc);
		int i[CCV_NNC_MAX_DIM + 2];
		int x;
		int batch_size = 1;
		for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
			batch_size *= adim[x];
		for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
			batch_size /= rdim[x];
		ccv_nnc_tensor_zero(saved_mean);
		float* ap = a->data.f32;
		float* const meanp = saved_mean->data.f32;
		for (i[0] = 0; i[0] < adim[0]; i[0]++)
		{
			float* const meanp0 = rdim[0] == 1 ? meanp : meanp + i[0] * saved_mean_inc[1] * saved_mean_inc[2] * saved_mean_inc[3];
			for (i[1] = 0; i[1] < adim[1]; i[1]++)
			{
				float* const meanp1 = rdim[1] == 1 ? meanp0 : meanp0 + i[1] * saved_mean_inc[2] * saved_mean_inc[3];
				for (i[2] = 0; i[2] < adim[2]; i[2]++)
				{
					float* const meanp2 = rdim[2] == 1 ? meanp1 : meanp1 + i[2] * saved_mean_inc[3];
					if (rdim[3] == 1)
						for (x = 0; x < adim[3]; x++)
							meanp2[0] += ap[x];
					else
						for (x = 0; x < adim[3]; x++)
							meanp2[x] += ap[x];
					ap += ainc[3];
				}
				ap += (ainc[2] - adim[2]) * ainc[3];
			}
			ap += (ainc[1] - adim[1]) * ainc[2] * ainc[3];
		}
		float inv_batch_size = 1.0 / batch_size;
		for (i[0] = 0; i[0] < rdim[0]; i[0]++)
		{
			float* const meanp0 = meanp + i[0] * saved_mean_inc[1] * saved_mean_inc[2] * saved_mean_inc[3];
			for (i[1] = 0; i[1] < rdim[1]; i[1]++)
			{
				float* const meanp1 = meanp0 + i[1] * saved_mean_inc[2] * saved_mean_inc[3];
				for (i[2] = 0; i[2] < rdim[2]; i[2]++)
				{
					float* const meanp2 = meanp1 + i[2] * saved_mean_inc[3];
					for (x = 0; x < rdim[3]; x++)
						meanp2[x] = meanp2[x] * inv_batch_size;
				}
			}
		}
		// Copy this into running mean / var.
		ccv_nnc_cmd_t mul_cmd = {
			.cmd = CCV_NNC_MUL_FORWARD,
			.backend = cmd.backend,
			.info = {
				.blas = {
					.a = {
						cmd.info.bnorm.momentum, 1. - cmd.info.bnorm.momentum
					}
				}
			}
		};
		_ccv_nnc_mul_forw_cpu_ref(mul_cmd, ccv_nnc_no_hint, flags, (ccv_nnc_tensor_t*[]){ (ccv_nnc_tensor_t*)mean, (ccv_nnc_tensor_t*)saved_mean }, 2, (ccv_nnc_tensor_t**)&mean, 1, stream_context);
		ccv_nnc_tensor_zero(saved_inv_std);
		ap = a->data.f32;
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
		_ccv_nnc_mul_forw_cpu_ref(mul_cmd, ccv_nnc_no_hint, flags, (ccv_nnc_tensor_t*[]){ (ccv_nnc_tensor_t*)var, (ccv_nnc_tensor_t*)saved_inv_std }, 2, (ccv_nnc_tensor_t**)&var, 1, stream_context);
		const float epsilon = cmd.info.bnorm.epsilon;
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
						varp2[x] = 1.0 / (sqrtf(varp2[x]) + epsilon);
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
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_batch_norm_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
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
