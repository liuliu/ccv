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

void _ccv_nnc_tensor_histogram_even(float* ap, int* bp, const int nd, const int* const dim, const int* const stride, const float max, const float min, const int bins, const int range, float* a_max, float* a_min, double* a_sum, double* a_sum_of_squares)
{
	if (nd == 1)
	{
		int i;
		for (i = 0; i < dim[0]; i++)
		{
			const float av = ap[i * stride[0]];
			*a_min = ccv_min(*a_min, av);
			*a_max = ccv_max(*a_max, av);
			*a_sum += av;
			*a_sum_of_squares += av * av;
			if (isnan(av))
				++bp[bins + 2];
			else if (av < min)
				++bp[0];
			else if (av >= max)
				++bp[bins + 1];
			else {
				int idx = (int)((av - min) * range) + 1;
				idx = ccv_min(ccv_max(idx, 1), bins);
				++bp[idx];
			}
		}
	} else if (nd == 2) {
		int x, y;
		for (y = 0; y < dim[0]; y++)
		{
			float* const apy = ap + y * stride[0];
			for (x = 0; x < dim[1]; x++)
			{
				const float av = apy[x * stride[1]];
				*a_min = ccv_min(*a_min, av);
				*a_max = ccv_max(*a_max, av);
				*a_sum += av;
				*a_sum_of_squares += av * av;
				if (isnan(av))
					++bp[bins + 2];
				else if (av < min)
					++bp[0];
				else if (av >= max)
					++bp[bins + 1];
				else {
					int idx = (int)((av - min) * range) + 1;
					idx = ccv_min(ccv_max(idx, 1), bins);
					++bp[idx];
				}
			}
		}
	} else if (nd == 3) {
		int x, y, z;
		for (z = 0; z < dim[0]; z++)
		{
			float* const apz = ap + z * stride[0];
			for (y = 0; y < dim[1]; y++)
			{
				float* const apy = apz + y * stride[1];
				for (x = 0; x < dim[2]; x++)
				{
					const float av = apy[x * stride[2]];
					*a_min = ccv_min(*a_min, av);
					*a_max = ccv_max(*a_max, av);
					*a_sum += av;
					*a_sum_of_squares += av * av;
					if (isnan(av))
						++bp[bins + 2];
					else if (av < min)
						++bp[0];
					else if (av >= max)
						++bp[bins + 1];
					else {
						int idx = (int)((av - min) * range) + 1;
						idx = ccv_min(ccv_max(idx, 1), bins);
						++bp[idx];
					}
				}
			}
		}
	} else if (nd == 4) {
		int x, y, z, s;
		for (s = 0; s < dim[0]; s++)
		{
			float* const aps = ap + s * stride[0];
			for (z = 0; z < dim[1]; z++)
			{
				float* const apz = aps + z * stride[1];
				for (y = 0; y < dim[2]; y++)
				{
					float* const apy = apz + y * stride[2];
					for (x = 0; x < dim[3]; x++)
					{
						const float av = apy[x * stride[3]];
						*a_min = ccv_min(*a_min, av);
						*a_max = ccv_max(*a_max, av);
						*a_sum += av;
						*a_sum_of_squares += av * av;
						if (isnan(av))
							++bp[bins + 2];
						else if (av < min)
							++bp[0];
						else if (av >= max)
							++bp[bins + 1];
						else {
							int idx = (int)((av - min) * range) + 1;
							idx = ccv_min(ccv_max(idx, 1), bins);
							++bp[idx];
						}
					}
				}
			}
		}
	} else {
		int i;
		for (i = 0; i < dim[0]; i++)
			_ccv_nnc_tensor_histogram_even(ap + i * stride[0], bp, nd - 1, dim + 1, stride + 1, max, min, bins, range, a_max, a_min, a_sum, a_sum_of_squares);
	}
}

void _ccv_nnc_tensor_histogram_logarithmic(float* ap, int* bp, const int nd, const int* const dim, const int* const stride, const float max, const float min, const int upper_range, const float min_inv, const float log_base, float* a_max, float* a_min, double* a_sum, double* a_sum_of_squares)
{
	if (nd == 1)
	{
		int i;
		for (i = 0; i < dim[0]; i++)
		{
			const float av = ap[i * stride[0]];
			*a_min = ccv_min(*a_min, av);
			*a_max = ccv_max(*a_max, av);
			*a_sum += av;
			*a_sum_of_squares += av * av;
			if (isnan(av))
				++bp[upper_range * 2 + 1];
			else if (av >= max)
				++bp[upper_range * 2];
			else if (av <= -max)
				++bp[0];
			else if (av <= -max)
				++bp[0];
			else if (av < min && av > -min)
				++bp[upper_range];
			else {
				int idx = ceilf(logf(fabsf(av) * min_inv) * log_base);
				idx = av > 0 ? idx + upper_range : upper_range - idx;
				idx = ccv_min(ccv_max(idx, 0), upper_range * 2);
				++bp[idx];
			}
		}
	} else if (nd == 2) {
		int x, y;
		for (y = 0; y < dim[0]; y++)
		{
			float* const apy = ap + y * stride[0];
			for (x = 0; x < dim[1]; x++)
			{
				const float av = apy[x * stride[1]];
				*a_min = ccv_min(*a_min, av);
				*a_max = ccv_max(*a_max, av);
				*a_sum += av;
				*a_sum_of_squares += av * av;
				if (isnan(av))
					++bp[upper_range * 2 + 1];
				else if (av >= max)
					++bp[upper_range * 2];
				else if (av <= -max)
					++bp[0];
				else if (av <= -max)
					++bp[0];
				else if (av < min && av > -min)
					++bp[upper_range];
				else {
					int idx = ceilf(logf(fabsf(av) * min_inv) * log_base);
					idx = av > 0 ? idx + upper_range : upper_range - idx;
					idx = ccv_min(ccv_max(idx, 0), upper_range * 2);
					++bp[idx];
				}
			}
		}
	} else if (nd == 3) {
		int x, y, z;
		for (z = 0; z < dim[0]; z++)
		{
			float* const apz = ap + z * stride[0];
			for (y = 0; y < dim[1]; y++)
			{
				float* const apy = apz + y * stride[1];
				for (x = 0; x < dim[2]; x++)
				{
					const float av = apy[x * stride[2]];
					*a_min = ccv_min(*a_min, av);
					*a_max = ccv_max(*a_max, av);
					*a_sum += av;
					*a_sum_of_squares += av * av;
					if (isnan(av))
						++bp[upper_range * 2 + 1];
					else if (av >= max)
						++bp[upper_range * 2];
					else if (av <= -max)
						++bp[0];
					else if (av <= -max)
						++bp[0];
					else if (av < min && av > -min)
						++bp[upper_range];
					else {
						int idx = ceilf(logf(fabsf(av) * min_inv) * log_base);
						idx = av > 0 ? idx + upper_range : upper_range - idx;
						idx = ccv_min(ccv_max(idx, 0), upper_range * 2);
						++bp[idx];
					}
				}
			}
		}
	} else if (nd == 4) {
		int x, y, z, s;
		for (s = 0; s < dim[0]; s++)
		{
			float* const aps = ap + s * stride[0];
			for (z = 0; z < dim[1]; z++)
			{
				float* const apz = aps + z * stride[1];
				for (y = 0; y < dim[2]; y++)
				{
					float* const apy = apz + y * stride[2];
					for (x = 0; x < dim[3]; x++)
					{
						const float av = apy[x * stride[3]];
						*a_min = ccv_min(*a_min, av);
						*a_max = ccv_max(*a_max, av);
						*a_sum += av;
						*a_sum_of_squares += av * av;
						if (isnan(av))
							++bp[upper_range * 2 + 1];
						else if (av >= max)
							++bp[upper_range * 2];
						else if (av <= -max)
							++bp[0];
						else if (av <= -max)
							++bp[0];
						else if (av < min && av > -min)
							++bp[upper_range];
						else {
							int idx = ceilf(logf(fabsf(av) * min_inv) * log_base);
							idx = av > 0 ? idx + upper_range : upper_range - idx;
							idx = ccv_min(ccv_max(idx, 0), upper_range * 2);
							++bp[idx];
						}
					}
				}
			}
		}
	} else {
		int i;
		for (i = 0; i < dim[0]; i++)
			_ccv_nnc_tensor_histogram_logarithmic(ap + i * stride[0], bp, nd - 1, dim + 1, stride + 1, max, min, upper_range, min_inv, log_base, a_max, a_min, a_sum, a_sum_of_squares);
	}
}

void _ccv_nnc_tensor_histogram_bins(float* ap, float* hp, int* bp, const int nd, const int* const dim, const int* const stride, const int upper_range, float* a_max, float* a_min, double* a_sum, double* a_sum_of_squares)
{
	if (nd == 1)
	{
		int i;
		for (i = 0; i < dim[0]; i++)
		{
			const float av = ap[i * stride[0]];
			*a_min = ccv_min(*a_min, av);
			*a_max = ccv_max(*a_max, av);
			*a_sum += av;
			*a_sum_of_squares += av * av;
			if (isnan(av))
				++bp[upper_range + 1];
			else {
				const int idx = _upper_bound(av, upper_range, hp);
				++bp[idx];
			}
		}
	} else if (nd == 2) {
		int x, y;
		for (y = 0; y < dim[0]; y++)
		{
			float* const apy = ap + y * stride[0];
			for (x = 0; x < dim[1]; x++)
			{
				const float av = apy[x * stride[1]];
				*a_min = ccv_min(*a_min, av);
				*a_max = ccv_max(*a_max, av);
				*a_sum += av;
				*a_sum_of_squares += av * av;
				if (isnan(av))
					++bp[upper_range + 1];
				else {
					const int idx = _upper_bound(av, upper_range, hp);
					++bp[idx];
				}
			}
		}
	} else if (nd == 3) {
		int x, y, z;
		for (z = 0; z < dim[0]; z++)
		{
			float* const apz = ap + z * stride[0];
			for (y = 0; y < dim[1]; y++)
			{
				float* const apy = apz + y * stride[1];
				for (x = 0; x < dim[2]; x++)
				{
					const float av = apy[x * stride[2]];
					*a_min = ccv_min(*a_min, av);
					*a_max = ccv_max(*a_max, av);
					*a_sum += av;
					*a_sum_of_squares += av * av;
					if (isnan(av))
						++bp[upper_range + 1];
					else {
						const int idx = _upper_bound(av, upper_range, hp);
						++bp[idx];
					}
				}
			}
		}
	} else if (nd == 4) {
		int x, y, z, s;
		for (s = 0; s < dim[0]; s++)
		{
			float* const aps = ap + s * stride[0];
			for (z = 0; z < dim[1]; z++)
			{
				float* const apz = aps + z * stride[1];
				for (y = 0; y < dim[2]; y++)
				{
					float* const apy = apz + y * stride[2];
					for (x = 0; x < dim[3]; x++)
					{
						const float av = apy[x * stride[3]];
						*a_min = ccv_min(*a_min, av);
						*a_max = ccv_max(*a_max, av);
						*a_sum += av;
						*a_sum_of_squares += av * av;
						if (isnan(av))
							++bp[upper_range + 1];
						else {
							const int idx = _upper_bound(av, upper_range, hp);
							++bp[idx];
						}
					}
				}
			}
		}
	} else {
		int i;
		for (i = 0; i < dim[0]; i++)
			_ccv_nnc_tensor_histogram_bins(ap + i * stride[0], hp, bp, nd - 1, dim + 1, stride + 1, upper_range, a_max, a_min, a_sum, a_sum_of_squares);
	}
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
					a_max = ccv_max(a_max, ap[i]);
					a_sum += ap[i];
					a_sum_of_squares += ap[i] * ap[i];
					if (isnan(ap[i]))
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
					a_max = ccv_max(a_max, ap[i]);
					a_sum += ap[i];
					a_sum_of_squares += ap[i] * ap[i];
					// Range from 1e-12 to 1e20, with 1.1 ratio. We reserve 0, count - 2 for -inf and inf, count - 1 for nan.
					if (isnan(ap[i]))
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
					a_max = ccv_max(a_max, ap[i]);
					a_sum += ap[i];
					a_sum_of_squares += ap[i] * ap[i];
					if (isnan(ap[i]))
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
	// reset it to 0.
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
			_ccv_nnc_tensor_histogram_even(tv->data.f32, bp, nd, tv->info.dim, tv->stride, max, min, bins, range, &a_max, &a_min, &a_sum, &a_sum_of_squares);
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
			_ccv_nnc_tensor_histogram_logarithmic(tv->data.f32, bp, nd, tv->info.dim, tv->stride, max, min, upper_range, min_inv, log_base, &a_max, &a_min, &a_sum, &a_sum_of_squares);
			break;
		}
		case CCV_NNC_HISTOGRAM_BINS:
		{
			assert(h);
			const int upper_range = ccv_nnc_tensor_count(h->info);
			assert(ccv_nnc_tensor_count(b->info) == upper_range + 2);
			_ccv_nnc_tensor_histogram_bins(tv->data.f32, h->data.f32, bp, nd, tv->info.dim, tv->stride, upper_range, &a_max, &a_min, &a_sum, &a_sum_of_squares);
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
