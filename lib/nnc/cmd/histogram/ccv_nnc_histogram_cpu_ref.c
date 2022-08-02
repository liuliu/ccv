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

static int _upper_bound(const float v, const int size, const float* const bounds)
{
	int upper_bound = size;
	int lower_bound = -1;
	while (lower_bound + 1 < upper_bound)
	{
		const int middle = ((upper_bound - lower_bound) >> 1) + lower_bound;
		if (v < bounds[middle])
			upper_bound = middle;
		else
			lower_bound = middle;
	}
	return upper_bound;
}

static int _ccv_nnc_histogram_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 1);
	const ccv_nnc_tensor_t* a = inputs[0];
	assert(a->info.datatype == CCV_32F);
	const ccv_nnc_tensor_t* h = input_size > 1 ? inputs[1] : 0;
	if (h)
		{ assert(CCV_IS_TENSOR_CONTIGUOUS(h)); }
	assert(output_size >= 1);
	ccv_nnc_tensor_t* b = outputs[0];
	ccv_nnc_tensor_t* s = output_size > 1 ? outputs[1] : 0;
	assert(CCV_IS_TENSOR_CONTIGUOUS(b));
	ccv_nnc_tensor_zero(b);
	assert(b->info.datatype == CCV_32S);
	int* bp = b->data.i32;
	float a_min = a->data.f32[0];
	float a_max = a_min;
	double a_sum = 0;
	double a_sum_of_squares = 0;
	if (CCV_IS_TENSOR_CONTIGUOUS(a))
	{
		float* ap = a->data.f32;
		int i, count = ccv_nnc_tensor_count(a->info);
		switch (cmd.info.histogram.type)
		{
			case CCV_NNC_HISTOGRAM_EVEN:
			{
				const int bins = cmd.info.histogram.bins;
				assert(ccv_nnc_tensor_count(b->info) == bins + 3);
				const float min = cmd.info.histogram.min;
				const float max = cmd.info.histogram.max;
				assert(cmd.info.histogram.max > cmd.info.histogram.min);
				const float range = bins / (max - min);
				for (i = 0; i < count; i++)
				{
					a_min = ccv_min(a_min, ap[i]);
					a_max = ccv_min(a_max, ap[i]);
					a_sum += ap[i];
					a_sum_of_squares += ap[i] * ap[i];
					if (isnanf(ap[i]))
						++bp[bins + 2];
					else if (ap[i] < min)
						++bp[0];
					else if (ap[i] >= max)
						++bp[bins + 1];
					else {
						int idx = (int)((ap[i] - min) * range) + 1;
						idx = ccv_min(ccv_max(idx, 1), bins);
						++bp[idx];
					}
				}
				break;
			}
			case CCV_NNC_HISTOGRAM_LOGARITHMIC:
			{
				const float log_base = 1.0 / logf(cmd.info.histogram.rate);
				assert(cmd.info.histogram.max > 0);
				assert(cmd.info.histogram.min > 0);
				assert(cmd.info.histogram.max > cmd.info.histogram.min);
				const float min = cmd.info.histogram.min;
				const float max = cmd.info.histogram.max;
				const int upper_range = ceilf(logf(cmd.info.histogram.max / cmd.info.histogram.min) * log_base);
				const float min_inv = 1.0 / cmd.info.histogram.min;
				for (i = 0; i < count; i++)
				{
					a_min = ccv_min(a_min, ap[i]);
					a_max = ccv_min(a_max, ap[i]);
					a_sum += ap[i];
					a_sum_of_squares += ap[i] * ap[i];
					// Range from 1e-12 to 1e20, with 1.1 ratio. We reserve 0, count - 2 for -inf and inf, count - 1 for nan.
					if (isnanf(ap[i]))
						++bp[upper_range * 2 + 1];
					else if (ap[i] >= max)
						++bp[upper_range * 2];
					else if (ap[i] <= -max)
						++bp[0];
					else if (ap[i] < min && ap[i] > -min)
						++bp[upper_range];
					else {
						int idx = ceilf(logf(fabsf(ap[i]) * min_inv) * log_base);
						idx = ap[i] > 0 ? idx + upper_range : upper_range - idx;
						idx = ccv_min(ccv_max(idx, 0), upper_range * 2);
						++bp[idx];
					}
				}
				break;
			}
			case CCV_NNC_HISTOGRAM_BINS:
			{
				assert(h);
				const int upper_range = ccv_nnc_tensor_count(h->info);
				assert(ccv_nnc_tensor_count(b->info) == upper_range + 2);
				for (i = 0; i < count; i++)
				{
					a_min = ccv_min(a_min, ap[i]);
					a_max = ccv_min(a_max, ap[i]);
					a_sum += ap[i];
					a_sum_of_squares += ap[i] * ap[i];
					if (isnanf(ap[i]))
						++bp[upper_range + 1];
					else {
						const int idx = _upper_bound(ap[i], upper_range, h->data.f32);
						++bp[idx];
					}
				}
				break;
			}
		}
		if (s)
		{
			assert(ccv_nnc_tensor_count(s->info) >= 4);
			assert(s->info.datatype == CCV_32F);
			s->data.f32[0] = a_min;
			s->data.f32[1] = a_max;
			s->data.f32[2] = a_sum;
			s->data.f32[3] = a_sum_of_squares;
		}
		return CCV_NNC_EXEC_SUCCESS;
	}
	ccv_nnc_tensor_view_t* tv = (ccv_nnc_tensor_view_t*)a;
	assert(CCV_IS_TENSOR_VIEW(tv));
	const int nd = ccv_nnc_tensor_nd(tv->info.dim);
	assert(nd >= 1);
	const int* const tvinc = tv->inc;
	// reset it to 0.
	int c, x, y, i;
	int count = 1;
	int mod[CCV_NNC_MAX_DIM_ALLOC - 3];
	int mod_inc[CCV_NNC_MAX_DIM_ALLOC - 2];
	const int top_mod_inc = nd > 2 ? tvinc[nd - 3] * tvinc[nd - 2] * tvinc[nd - 1] : 1;
	if (nd > 2)
		mod_inc[nd - 3] = top_mod_inc;
	for (c = nd - 4; c >= 0; c--)
	{
		// Compute the mod.
		mod[c] = c == nd - 4 ? tv->info.dim[c] : mod[c + 1] * tv->info.dim[c];
		mod_inc[c] = mod_inc[c + 1] * tvinc[c];
		count *= tv->info.dim[c];
	}
	for (c = 0; c < nd - 3; c++)
		mod_inc[c] = mod_inc[c + 1] * (tvinc[c] - tv->info.dim[c]);
	float* tvd = tv->data.f32;
	const int tvinc_1 = tvinc[nd - 1];
	const int tvinc_21 = tvinc_1 * (nd >= 2 ? tvinc[nd - 2] : 1);
	const int tvdim_1 = tv->info.dim[nd - 1];
	const int max_y = ccv_max(1, nd >= 3 ? tv->info.dim[nd - 3] : 1);
	const int max_x = ccv_max(1, nd >= 2 ? tv->info.dim[nd - 2] : 1);
	switch (cmd.info.histogram.type)
	{
		case CCV_NNC_HISTOGRAM_EVEN:
		{
			const int bins = cmd.info.histogram.bins;
			assert(ccv_nnc_tensor_count(b->info) == bins + 3);
			const float min = cmd.info.histogram.min;
			const float max = cmd.info.histogram.max;
			assert(cmd.info.histogram.max > cmd.info.histogram.min);
			const float range = bins / (max - min);
			for (c = 0; c < count; c++)
			{
				for (y = 0; y < max_y; y++)
				{
					float* tvp = tvd + y * tvinc_21;
					for (x = 0; x < max_x; x++)
					{
						for (i = 0; i < tvdim_1; i++)
						{
							a_min = ccv_min(a_min, tvp[i]);
							a_max = ccv_min(a_max, tvp[i]);
							a_sum += tvp[i];
							a_sum_of_squares += tvp[i] * tvp[i];
							if (isnanf(tvp[i]))
								++bp[bins + 2];
							else if (tvp[i] < min)
								++bp[0];
							else if (tvp[i] >= max)
								++bp[bins + 1];
							else {
								int idx = (int)((tvp[i] - min) * range) + 1;
								idx = ccv_min(ccv_max(idx, 1), bins);
								++bp[idx];
							}
						}
						tvp += tvinc_1;
					}
				}
				tvd += top_mod_inc;
				for (y = nd - 4; y >= 0; y--)
					if ((c + 1) % mod[y] != 0)
						break; // cannot be mod, break out.
					else
						tvd += mod_inc[y];
			}
			break;
		}
		case CCV_NNC_HISTOGRAM_LOGARITHMIC:
		{
			const float log_base = 1.0 / logf(cmd.info.histogram.rate);
			assert(cmd.info.histogram.max > 0);
			assert(cmd.info.histogram.min > 0);
			assert(cmd.info.histogram.max > cmd.info.histogram.min);
			const float min = cmd.info.histogram.min;
			const float max = cmd.info.histogram.max;
			const int upper_range = ceilf(logf(cmd.info.histogram.max / cmd.info.histogram.min) * log_base);
			const float min_inv = 1.0 / cmd.info.histogram.min;
			for (c = 0; c < count; c++)
			{
				for (y = 0; y < max_y; y++)
				{
					float* tvp = tvd + y * tvinc_21;
					for (x = 0; x < max_x; x++)
					{
						for (i = 0; i < tvdim_1; i++)
						{
							a_min = ccv_min(a_min, tvp[i]);
							a_max = ccv_min(a_max, tvp[i]);
							a_sum += tvp[i];
							a_sum_of_squares += tvp[i] * tvp[i];
							// Range from 1e-12 to 1e20, with 1.1 ratio. We reserve 0, count - 2 for -inf and inf, count - 1 for nan.
							if (isnanf(tvp[i]))
								++bp[upper_range * 2 + 1];
							else if (tvp[i] >= max)
								++bp[upper_range * 2];
							else if (tvp[i] <= -max)
								++bp[0];
							else if (tvp[i] <= -max)
								++bp[0];
							else if (tvp[i] < min && tvp[i] > -min)
								++bp[upper_range];
							else {
								int idx = ceilf(logf(fabsf(tvp[i]) * min_inv) * log_base);
								idx = tvp[i] > 0 ? idx + upper_range : upper_range - idx;
								idx = ccv_min(ccv_max(idx, 0), upper_range * 2);
								++bp[idx];
							}
						}
						tvp += tvinc_1;
					}
				}
				tvd += top_mod_inc;
				for (y = nd - 4; y >= 0; y--)
					if ((c + 1) % mod[y] != 0)
						break; // cannot be mod, break out.
					else
						tvd += mod_inc[y];
			}
			break;
		}
		case CCV_NNC_HISTOGRAM_BINS:
		{
			assert(h);
			const int upper_range = ccv_nnc_tensor_count(h->info);
			assert(ccv_nnc_tensor_count(b->info) == upper_range + 2);
			for (c = 0; c < count; c++)
			{
				for (y = 0; y < max_y; y++)
				{
					float* tvp = tvd + y * tvinc_21;
					for (x = 0; x < max_x; x++)
					{
						for (i = 0; i < tvdim_1; i++)
						{
							a_min = ccv_min(a_min, tvp[i]);
							a_max = ccv_min(a_max, tvp[i]);
							a_sum += tvp[i];
							a_sum_of_squares += tvp[i] * tvp[i];
							if (isnanf(tvp[i]))
								++bp[upper_range + 1];
							else {
								const int idx = _upper_bound(tvp[i], upper_range, h->data.f32);
								++bp[idx];
							}
						}
						tvp += tvinc_1;
					}
				}
				tvd += top_mod_inc;
				for (y = nd - 4; y >= 0; y--)
					if ((c + 1) % mod[y] != 0)
						break; // cannot be mod, break out.
					else
						tvd += mod_inc[y];
			}
			break;
		}
	}
	if (s)
	{
		assert(ccv_nnc_tensor_count(s->info) >= 4);
		assert(s->info.datatype == CCV_32F);
		s->data.f32[0] = a_min;
		s->data.f32[1] = a_max;
		s->data.f32[2] = a_sum;
		s->data.f32[3] = a_sum_of_squares;
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_histogram_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_INVALID;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_HISTOGRAM_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_histogram_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_HISTOGRAM_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_histogram_back;
}
