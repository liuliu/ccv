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

// n[x] is the start point for the filter on y axis, so that we can avoid computing the padding.
// m[x] shows how long we should loop for filter on y axis, avoid computing the padding too.
#define set_n_m_dim(x, wd, ad) \
	do { \
		n[x] = ccv_max(i[x] * hint.stride.dim[x + 1] - hint.border.begin[x + 1], 0) - (i[x] * hint.stride.dim[x + 1] - hint.border.begin[x + 1]); \
		m[x] = wd[x + 1] - n[x] - (i[x] * hint.stride.dim[x + 1] + wd[x + 1] - ccv_min(ad[x + 1] + hint.border.end[x + 1], i[x] * hint.stride.dim[x + 1] + wd[x + 1])); \
	} while (0)

inline static void _ccv_nnc_winograd_2x2_3x3_gwtg(const float* w, const int c, float* gwtg)
{
	int i;
	for (i = 0; i < c; i++)
	{
		/*
		 * a0, b1, c2
		 * d3, e4, f5
		 * g6, h7, i8
		 * {{a, 1/2 (a + b + c), 1/2 (a - b + c), c},
		 * {1/2 (a + d + g), 1/4 (a + b + c + d + e + f + g + h + i), 1/4 (a - b + c + d - e + f + g - h + i), 1/2 (c + f + i)},
		 * {1/2 (a - d + g), 1/4 (a + b + c - d - e - f + g + h + i), 1/4 (a - b + c - d + e - f + g - h + i), 1/2 (c - f + i)},
		 * {g, 1/2 (g + h + i), 1/2 (g - h + i), i}}
		 */
		/* row 1 */
		gwtg[0] = w[i];
		gwtg[1] = (w[i] + w[c + i] + w[2 * c + i]) * 0.5;
		gwtg[2] = (w[i] - w[c + i] + w[2 * c + i]) * 0.5;
		gwtg[3] = w[2 * c + i];
		/* row 2 */
		gwtg[4] = (w[i] + w[3 * c + i] + w[6 * c + i]) * 0.5;
		gwtg[5] = (w[i] + w[c + i] + w[2 * c + i] + w[3 * c + i] + w[4 * c + i] + w[5 * c + i] + w[6 * c + i] + w[7 * c + i] + w[8 * c + i]) * 0.25;
		gwtg[6] = (w[i] - w[c + i] + w[2 * c + i] + w[3 * c + i] - w[4 * c + i] + w[5 * c + i] + w[6 * c + i] - w[7 * c + i] + w[8 * c + i]) * 0.25;
		gwtg[7] = (w[2 * c + i] + w[5 * c + i] + w[8 * c + i]) * 0.5;
		/* row 3 */
		gwtg[8] = (w[i] - w[3 * c + i] + w[6 * c + i]) * 0.5;
		gwtg[9] = (w[i] + w[c + i] + w[2 * c + i] - w[3 * c + i] - w[4 * c + i] - w[5 * c + i] + w[6 * c + i] + w[7 * c + i] + w[8 * c + i]) * 0.25;
		gwtg[10] = (w[i] - w[c + i] + w[2 * c + i] - w[3 * c + i] + w[4 * c + i] - w[5 * c + i] + w[6 * c + i] - w[7 * c + i] + w[8 * c + i]) * 0.25;
		gwtg[11] = (w[2 * c + i] - w[5 * c + i] + w[8 * c + i]) * 0.5;
		/* row 4 */
		gwtg[12] = w[6 * c + i];
		gwtg[13] = (w[6 * c + i] + w[7 * c + i] + w[8 * c + i]) * 0.5;
		gwtg[14] = (w[6 * c + i] - w[7 * c + i] + w[8 * c + i]) * 0.5;
		gwtg[15] = w[8 * c + i];
		gwtg += 16;
	}
}

static int _ccv_nnc_conv_forw_2x2_3x3_winograd(const ccv_nnc_tensor_view_t* a, const ccv_nnc_tensor_t* w, const ccv_nnc_tensor_t* bias, const ccv_nnc_hint_t hint, ccv_nnc_tensor_view_t* b)
{
	const int* ainc = CCV_IS_TENSOR_VIEW(a) ? a->inc : a->info.dim;
	const int* binc = CCV_IS_TENSOR_VIEW(b) ? b->inc : b->info.dim;
	assert(hint.border.begin[1] == 0);
	assert(hint.border.begin[2] == 0);
	assert(w->info.dim[1] == 3);
	assert(w->info.dim[2] == 3);
	// Convert w to a 4x4 matrix, by computing G.w.T(G) // T for transpose.
	float* const gwtg = ccmalloc(sizeof(float) * 16 * w->info.dim[0] * w->info.dim[3]);
	parallel_for(k, w->info.dim[3]) {
		_ccv_nnc_winograd_2x2_3x3_gwtg(w->data.f32 + k * w->info.dim[2] * w->info.dim[1] * w->info.dim[0], w->info.dim[0], gwtg + k * 16 * w->info.dim[0]);
	} parallel_endfor
	const int jump_dim[CCV_NNC_MAX_DIM] = {
		b->info.dim[1] / 2, b->info.dim[2] / 2
	};
	float* const ws = (float*)ccmalloc(sizeof(float) * 16 * a->info.dim[0] * jump_dim[1]);
	// kernel weight for one dim.
	const float* const biasval = bias->data.f32;
	// This block will be cause in each for-loop, therefore, you can use it to generate some temporary variables.
	parallel_for(i, jump_dim[1]) {
		int x, k, c;
		float* ap = a->data.f32 + i * 2 * ainc[1] * ainc[0];
		float* bp = b->data.f32 + i * 2 * binc[1] * binc[0];
		for (x = 0; x < b->info.dim[1] - 1; x += 2)
		{
			float* g = ws + i * 16 * a->info.dim[0];
			for (c = 0; c < a->info.dim[0]; c++)
			{
				float* apz = ap + x * ainc[0] + c;
				float t[16];
				t[0] = apz[0];
				t[1] = apz[ainc[0]];
				t[2] = apz[2 * ainc[0]];
				t[3] = apz[3 * ainc[0]];
				apz += ainc[1] * ainc[0];
				t[4] = apz[0];
				t[5] = apz[ainc[0]];
				t[6] = apz[2 * ainc[0]];
				t[7] = apz[3 * ainc[0]];
				apz += ainc[1] * ainc[0];
				t[8] = apz[0];
				t[9] = apz[ainc[0]];
				t[10] = apz[2 * ainc[0]];
				t[11] = apz[3 * ainc[0]];
				apz += ainc[1] * ainc[0];
				t[12] = apz[0];
				t[13] = apz[ainc[0]];
				t[14] = apz[2 * ainc[0]];
				t[15] = apz[3 * ainc[0]];
				/*
				 * a0, b1, c2, d3
				 * e4, f5, g6, h7
				 * i8, j9, k10 l11
				 * m12 n13 o14 p16
				 * {{a - i, b - j, c - k, d - l},
				 * {e + i, f + j, g + k, h + l},
				 * {-e + i, -f + j, -g + k, -h + l},
				 * {e - m, f - n, g - o, h - p}}
				 */
				float d[16];
				/* BT.d */
				/* row 1 */
				d[0] = t[0] - t[8];
				d[1] = t[1] - t[9];
				d[2] = t[2] - t[10];
				d[3] = t[3] - t[11];
				/* row 2 */
				d[4] = t[4] + t[8];
				d[5] = t[5] + t[9];
				d[6] = t[6] + t[10];
				d[7] = t[7] + t[11];
				/* row 3 */
				d[8] = t[8] - t[4];
				d[9] = t[9] - t[5];
				d[10] = t[10] - t[6];
				d[11] = t[11] - t[7];
				/* row 4 */
				d[12] = t[4] - t[12];
				d[13] = t[5] - t[13];
				d[14] = t[6] - t[14];
				d[15] = t[7] - t[15];
				/*
				 * a0, b1, c2, d3
				 * e4, f5, g6, h7
				 * i8, j9, k10 l11
				 * m12 n13 o14 p16
				 * {{a - c, b + c, -b + c, b - d},
				 * {e - g, f + g, -f + g, f - h},
				 * {i - k, j + k, -j + k, j - l},
				 * {m - o, n + o, -n + o, n - p}}
				 */
				/* BT.d.B */
				/* row 1 */
				g[0] = d[0] - d[2];
				g[1] = d[1] + d[2];
				g[2] = d[2] - d[1];
				g[3] = d[1] - d[3];
				/* row 2 */
				g[4] = d[4] - d[6];
				g[5] = d[5] + d[6];
				g[6] = d[6] - d[5];
				g[7] = d[5] - d[7];
				/* row 3 */
				g[8] = d[8] - d[10];
				g[9] = d[9] + d[10];
				g[10] = d[10] - d[9];
				g[11] = d[9] - d[11];
				/* row 4 */
				g[12] = d[12] - d[14];
				g[13] = d[13] + d[14];
				g[14] = d[14] - d[13];
				g[15] = d[13] - d[15];
				// move to the next channel
				g += 16;
			}
			float* wpz = gwtg;
			for (k = 0; k < w->info.dim[3]; k++)
			{
				float q[16] = {0};
				g = ws + i * 16 * a->info.dim[0];
				for (c = 0; c < a->info.dim[0]; c++)
				{
					q[0] += g[0] * wpz[0];
					q[1] += g[1] * wpz[1];
					q[2] += g[2] * wpz[2];
					q[3] += g[3] * wpz[3];
					q[4] += g[4] * wpz[4];
					q[5] += g[5] * wpz[5];
					q[6] += g[6] * wpz[6];
					q[7] += g[7] * wpz[7];
					q[8] += g[8] * wpz[8];
					q[9] += g[9] * wpz[9];
					q[10] += g[10] * wpz[10];
					q[11] += g[11] * wpz[11];
					q[12] += g[12] * wpz[12];
					q[13] += g[13] * wpz[13];
					q[14] += g[14] * wpz[14];
					q[15] += g[15] * wpz[15];
					g += 16;
					wpz += 16;
				}
				/*
				 * a0, b1, c2, d3
				 * e4, f5, g6, h7
				 * i8, j9, k10 l11
				 * m12 n13 o14 p16
				 * {{a + e + i, b + f + j, c + g + k, d + h + l},
				 * {e - i - m, f - j - n, g - k - o, h - l - p}}
				 */
				float d[8];
				/* row 1 */
				d[0] = q[0] + q[4] + q[8];
				d[1] = q[1] + q[5] + q[9];
				d[2] = q[2] + q[6] + q[10];
				d[3] = q[3] + q[7] + q[11];
				/* row 2 */
				d[4] = q[4] - q[8] - q[12];
				d[5] = q[5] - q[9] - q[13];
				d[6] = q[6] - q[10] - q[14];
				d[7] = q[7] - q[11] - q[15];
				/*
				 * {{a + b + c, b - c - d},
				 * {e + f + g, f - g - h}}
				 */
				bp[x * binc[0] + k] = d[0] + d[1] + d[2] + biasval[k];
				bp[(x + 1) * binc[0] + k] = d[1] - d[2] - d[3] + biasval[k];
				bp[(binc[1] + x) * binc[0] + k] = d[4] + d[5] + d[6] + biasval[k];
				bp[(binc[1] + x + 1) * binc[0] + k] = d[5] - d[6] - d[7] + biasval[k];
			}
		}
	} parallel_endfor
	ccfree(ws);
	ccfree(gwtg);
	return CCV_NNC_EXEC_SUCCESS;
}

/*
static int _ccv_nnc_conv_forw_2x2_3x3_winograd_neon(const ccv_nnc_tensor_view_t* a, const ccv_nnc_tensor_t* w, const ccv_nnc_tensor_t* bias, const ccv_nnc_hint_t hint, ccv_nnc_tensor_view_t* b)
{
	return CCV_NNC_EXEC_INVALID;
}

static int _ccv_nnc_conv_forw_2x2_3x3_winograd_sse2(const ccv_nnc_tensor_view_t* a, const ccv_nnc_tensor_t* w, const ccv_nnc_tensor_t* bias, const ccv_nnc_hint_t hint, ccv_nnc_tensor_view_t* b)
{
	return CCV_NNC_EXEC_INVALID;
}

static int _ccv_nnc_conv_forw_neon(const ccv_nnc_tensor_view_t* a, const ccv_nnc_tensor_t* w, const ccv_nnc_tensor_t* bias, const ccv_nnc_hint_t hint, ccv_nnc_tensor_view_t* b)
{
	return CCV_NNC_EXEC_INVALID;
}

static int _ccv_nnc_conv_forw_sse2(const ccv_nnc_tensor_view_t* a, const ccv_nnc_tensor_t* w, const ccv_nnc_tensor_t* bias, const ccv_nnc_hint_t hint, ccv_nnc_tensor_view_t* b)
{
	return CCV_NNC_EXEC_INVALID;
}
*/

static int _ccv_nnc_conv_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
{
	assert(input_size == 3);
	const ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_t* w = inputs[1];
	assert(!CCV_IS_TENSOR_VIEW(w));
	const ccv_nnc_tensor_t* bias = inputs[2];
	assert(!CCV_IS_TENSOR_VIEW(bias));
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(w->info.dim[0] == cmd.info.size.dim[0]);
	assert(w->info.dim[0] == a->info.dim[0]);
	assert(b->info.dim[0] == cmd.info.convolutional.count);
	int i;
	// Make sure the weights dimension matches the network dimension
	for (i = 1; i < CCV_NNC_MAX_DIM_ALLOC; i++)
	{
		if (w->info.dim[i] == 0 || cmd.info.size.dim[i] == 0)
			break;
		assert(w->info.dim[i] == cmd.info.size.dim[i]);
	}
	// Make sure the weights output dimension matches the network convolutional kernels
	for (i = CCV_NNC_MAX_DIM_ALLOC - 1; i > 0; i--)
		if (w->info.dim[i] == 0 && w->info.dim[i])
		{
			assert(w->info.dim[i] == cmd.info.convolutional.count);
			break;
		}
	if (w->info.dim[1] != 3 || w->info.dim[2] != 3)
		return CCV_NNC_EXEC_INVALID;
	if (hint.stride.dim[1] > 1 || hint.stride.dim[2] > 1)
		return CCV_NNC_EXEC_INVALID;
	return _ccv_nnc_conv_forw_2x2_3x3_winograd(a, w, bias, hint, b);
}

//@ccv_nnc_init CCV_NNC_BACKEND_CPU_OPT
void ccv_nnc_cpu_opt_init(ccv_nnc_cmd_api_t cmd_api[])
{
	/*TODO: I don't think any of these methods handles batch input, and I better to handle CHWN as well. */
	/* Convolutional layer */
	cmd_api[CCV_NNC_COMPUTE_CONVOLUTIONAL_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_CONVOLUTIONAL_FORWARD].exec = _ccv_nnc_conv_forw;
	/* Full connect layer */
	/* Max pool layer */
	/* Average pool layer */
	/* Softmax layer */
	/* ReLU activation */
}
