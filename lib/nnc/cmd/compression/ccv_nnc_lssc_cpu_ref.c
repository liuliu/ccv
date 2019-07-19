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

static int _ccv_nnc_lssc_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(output_size <= input_size);
	int n;
	ccv_float16_t a16[16];
	float a32[16];
	float bm[2];
	for (n = 0; n < output_size; n++)
	{
		const ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[n];
		ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[n];
		int i[CCV_NNC_MAX_DIM];
		int j[CCV_NNC_MAX_DIM];
		int c, k;
		const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
		assert(a_nd == CCV_NNC_MAX_DIM + 1 || a_nd == CCV_NNC_MAX_DIM + 2);
		const int* adim = (a_nd == CCV_NNC_MAX_DIM + 1) ? a->info.dim : a->info.dim + 1;
		const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
		assert(b_nd == CCV_NNC_MAX_DIM + 1 || b_nd == CCV_NNC_MAX_DIM + 2);
		const int* bdim = (b_nd == CCV_NNC_MAX_DIM + 1) ? b->info.dim : b->info.dim + 1;
		ccv_float16_t* ap = a->data.f16;
		const int* ainc = CCV_IS_TENSOR_VIEW(a) ? ((a_nd == CCV_NNC_MAX_DIM + 1) ?  a->inc : a->inc + 1) : adim;
		ccv_float16_t* bp = b->data.f16;
		const int* binc = CCV_IS_TENSOR_VIEW(b) ? ((b_nd == CCV_NNC_MAX_DIM + 1) ?  b->inc : b->inc + 1) : bdim;
		for (k = 0; k < bdim[0]; k++)
		{
			for (i[0] = 0; i[0] < bdim[1]; i[0]++)
			{
				assert(bdim[CCV_NNC_MAX_DIM] % 4 == 0);
				const int bw = bdim[CCV_NNC_MAX_DIM] / 4;
				for (i[1] = 0; i[1] < bw; i[1]++)
				{
					ccv_float16_t* apz = ap + i[0] * 4 * ainc[CCV_NNC_MAX_DIM] + i[1] * 4;
					const int h = ccv_min(i[0] * 4 + 4, adim[1]) - i[0] * 4;
					const int w = ccv_min(i[1] * 4 + 4, adim[CCV_NNC_MAX_DIM]) - i[1] * 4;
					for (c = 0; c < 16; c++)
						a16[c] = apz[0];
					for (j[0] = 0; j[0] < h; j[0]++)
						for (j[1] = 0; j[1] < w; j[1]++)
							a16[j[0] * 4 + j[1]] = apz[j[0] * ainc[CCV_NNC_MAX_DIM] + j[1]];
					ccv_half_precision_to_float((uint16_t*)a16, a32, 16);
					float amax = a32[0];
					float amin = a32[0];
					for (c = 1; c < 16; c++)
						amax = ccv_max(a32[c], amax), amin = ccv_min(a32[c], amin);
					bm[0] = amin;
					bm[1] = amax;
					ccv_float16_t* bpz = bp + i[0] * binc[CCV_NNC_MAX_DIM] + i[1] * 4;
					uint16_t* const bpz16 = (uint16_t*)bpz;
					ccv_float_to_half_precision(bm, bpz16, 2);
					const float abottom = amin * 7 / 6 - amax / 6;
					const float ascale = 3 / ccv_max(amax - amin, 1e-6);
					bpz16[2] = 0;
					for (c = 0; c < 8; c++)
						bpz16[2] |= ((ccv_clamp((int)((a32[c] - abottom) * ascale), 0, 3)) << (c << 1));
					bpz16[3] = 0;
					for (c = 0; c < 8; c++)
						bpz16[3] |= ((ccv_clamp((int)((a32[8 + c] - abottom) * ascale), 0, 3)) << (c << 1));
				}
			}
			bp += binc[CCV_NNC_MAX_DIM - 1] * binc[CCV_NNC_MAX_DIM];
			ap += ainc[CCV_NNC_MAX_DIM - 1] * ainc[CCV_NNC_MAX_DIM];
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_lssc_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(output_size <= input_size);
	int n;
	ccv_float16_t a16[16];
	float a32[16];
	float bm[4];
	for (n = 0; n < output_size; n++)
	{
		const ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)inputs[n];
		ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)outputs[n];
		int i[CCV_NNC_MAX_DIM];
		int j[CCV_NNC_MAX_DIM];
		int c, k;
		const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
		assert(a_nd == CCV_NNC_MAX_DIM + 1 || a_nd == CCV_NNC_MAX_DIM + 2);
		const int* adim = (a_nd == CCV_NNC_MAX_DIM + 1) ? a->info.dim : a->info.dim + 1;
		const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
		assert(b_nd == CCV_NNC_MAX_DIM + 1 || b_nd == CCV_NNC_MAX_DIM + 2);
		const int* bdim = (b_nd == CCV_NNC_MAX_DIM + 1) ? b->info.dim : b->info.dim + 1;
		ccv_float16_t* ap = a->data.f16;
		const int* ainc = CCV_IS_TENSOR_VIEW(a) ? ((a_nd == CCV_NNC_MAX_DIM + 1) ?  a->inc : a->inc + 1) : adim;
		ccv_float16_t* bp = b->data.f16;
		const int* binc = CCV_IS_TENSOR_VIEW(b) ? ((b_nd == CCV_NNC_MAX_DIM + 1) ?  b->inc : b->inc + 1) : bdim;
		for (k = 0; k < bdim[0]; k++)
		{
			for (i[0] = 0; i[0] < bdim[1]; i[0]++)
			{
				assert(bdim[CCV_NNC_MAX_DIM] % 4 == 0);
				const int bw = bdim[CCV_NNC_MAX_DIM] / 4;
				for (i[1] = 0; i[1] < bw; i[1]++)
				{
					ccv_float16_t* bpz = bp + i[0] * binc[CCV_NNC_MAX_DIM] + i[1] * 4;
					uint16_t* const bpz16 = (uint16_t*)bpz;
					ccv_half_precision_to_float(bpz16, bm, 2);
					bm[3] = bm[1];
					bm[1] = bm[3] / 3 + bm[0] * 2 / 3;
					bm[2] = bm[3] * 2 / 3 + bm[0] / 3;
					for (c = 0; c < 8; c++)
						a32[c] = bm[((bpz16[2] >> (c << 1)) & 3)];
					for (c = 0; c < 8; c++)
						a32[8 + c] = bm[((bpz16[3] >> (c << 1)) & 3)];
					ccv_float_to_half_precision(a32, (uint16_t*)a16, 16);
					ccv_float16_t* apz = ap + i[0] * 4 * ainc[CCV_NNC_MAX_DIM] + i[1] * 4;
					const int h = ccv_min(i[0] * 4 + 4, adim[1]) - i[0] * 4;
					const int w = ccv_min(i[1] * 4 + 4, adim[CCV_NNC_MAX_DIM]) - i[1] * 4;
					for (j[0] = 0; j[0] < h; j[0]++)
						for (j[1] = 0; j[1] < w; j[1]++)
							 apz[j[0] * ainc[CCV_NNC_MAX_DIM] + j[1]] = a16[j[0] * 4 + j[1]];
				}
			}
			bp += binc[CCV_NNC_MAX_DIM - 1] * binc[CCV_NNC_MAX_DIM];
			ap += ainc[CCV_NNC_MAX_DIM - 1] * ainc[CCV_NNC_MAX_DIM];
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_COMPRESSION_LSSC_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_16F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_lssc_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_COMPRESSION_LSSC_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_16F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_lssc_back;
}
