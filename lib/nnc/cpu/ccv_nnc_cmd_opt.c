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

static void _ccv_nnc_winograd_2x2_3x3_gwtg(const ccv_nnc_tensor_t* w, ccv_nnc_tensor_t* gwtg)
{
	float* gwtgp = gwtg->data.f32;
	float* wp = w->data.f32;
	const int c = w->info.dim[0];
	int i, j;
	assert(w->info.dim[1] == 3);
	assert(w->info.dim[2] == 3);
	assert(gwtg->info.dim[0] == 4);
	assert(gwtg->info.dim[1] == 4);
	for (i = 0; i < w->info.dim[3]; i++)
	{
		for (j = 0; j < c; j++)
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
			gwtgp[0] = wp[j];
			gwtgp[1] = (wp[j] + wp[c + j] + wp[2 * c + j]) * 0.5;
			gwtgp[2] = (wp[j] - wp[c + j] + wp[2 * c + j]) * 0.5;
			gwtgp[3] = wp[2 * c + j];
			/* row 2 */
			gwtgp[4] = (wp[j] + wp[3 * c + j] + wp[6 * c + j]) * 0.5;
			gwtgp[5] = (wp[j] + wp[c + j] + wp[2 * c + j] + wp[3 * c + j] + wp[4 * c + j] + wp[5 * c + j] + wp[6 * c + j] + wp[7 * c + j] + wp[8 * c + j]) * 0.25;
			gwtgp[6] = (wp[j] - wp[c + j] + wp[2 * c + j] + wp[3 * c + j] - wp[4 * c + j] + wp[5 * c + j] + wp[6 * c + j] - wp[7 * c + j] + wp[8 * c + j]) * 0.25;
			gwtgp[7] = (wp[2 * c + j] + wp[5 * c + j] + wp[8 * c + j]) * 0.5;
			/* row 3 */
			gwtgp[8] = (wp[j] - wp[3 * c + j] + wp[6 * c + j]) * 0.5;
			gwtgp[9] = (wp[j] + wp[c + j] + wp[2 * c + j] - wp[3 * c + j] - wp[4 * c + j] - wp[5 * c + j] + wp[6 * c + j] + wp[7 * c + j] + wp[8 * c + j]) * 0.25;
			gwtgp[10] = (wp[j] - wp[c + j] + wp[2 * c + j] - wp[3 * c + j] + wp[4 * c + j] - wp[5 * c + j] + wp[6 * c + j] - wp[7 * c + j] + wp[8 * c + j]) * 0.25;
			gwtgp[11] = (wp[2 * c + j] - wp[5 * c + j] + wp[8 * c + j]) * 0.5;
			/* row 4 */
			gwtgp[12] = wp[6 * c + j];
			gwtgp[13] = (wp[6 * c + j] + wp[7 * c + j] + wp[8 * c + j]) * 0.5;
			gwtgp[14] = (wp[6 * c + j] - wp[7 * c + j] + wp[8 * c + j]) * 0.5;
			gwtgp[15] = wp[8 * c + j];
			gwtgp += 16;
		}
		wp += 9 * c;
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
	ccv_nnc_tensor_param_t gwtg_params = w->info;
	gwtg_params.dim[0] = gwtg_params.dim[1] = 4;
	gwtg_params.dim[2] = w->info.dim[0];
	ccv_nnc_tensor_t* gwtg = ccv_nnc_tensor_new(0, gwtg_params, 0);
	_ccv_nnc_winograd_2x2_3x3_gwtg(w, gwtg);
	parallel_for(k, w->info.dim[3]) {
		float* ap = a->data.f32;
		float* bp = b->data.f32 + k;
		// kernel weight for one dim.
		float* gwtgp = gwtg->data.f32 + k * gwtg->info.dim[2] * gwtg->info.dim[1] * gwtg->info.dim[0];
		float biasval = bias->data.f32[k];
		// This block will be cause in each for-loop, therefore, you can use it to generate some temporary variables.
		int i[CCV_NNC_MAX_DIM];
		int j, c;
		for (i[1] = 0; i[1] < b->info.dim[2] - 1; i[1] += 2)
		{
			for (i[0] = 0; i[0] < b->info.dim[1] - 1; i[0] += 2)
			{
				float* wpz = gwtgp;
				float p[4] = {
					biasval, biasval, biasval, biasval
				};
				for (c = 0; c < a->info.dim[0]; c++)
				{
					float* apz = ap + ccv_max(i[0] - hint.border.begin[1], 0) * ainc[0] + c;
					float t[16];
					for (j = 0; j < 4; j++)
					{
						t[j * 4] = apz[0];
						t[j * 4 + 1] = apz[ainc[0]];
						t[j * 4 + 2] = apz[2 * ainc[0]];
						t[j * 4 + 3] = apz[3 * ainc[0]];
						apz += ainc[1] * ainc[0];
					}
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
					t[0] = d[0] - d[2];
					t[1] = d[1] + d[2];
					t[2] = d[2] - d[1];
					t[3] = d[1] - d[3];
					/* row 2 */
					t[4] = d[4] - d[6];
					t[5] = d[5] + d[6];
					t[6] = d[6] - d[5];
					t[7] = d[5] - d[7];
					/* row 3 */
					t[8] = d[8] - d[10];
					t[9] = d[9] + d[10];
					t[10] = d[10] - d[9];
					t[11] = d[9] - d[11];
					/* row 4 */
					t[12] = d[12] - d[14];
					t[13] = d[13] + d[14];
					t[14] = d[14] - d[13];
					t[15] = d[13] - d[15];
					// unroll'ed for loop for multiplication
					t[0] *= wpz[0];
					t[1] *= wpz[1];
					t[2] *= wpz[2];
					t[3] *= wpz[3];
					t[4] *= wpz[4];
					t[5] *= wpz[5];
					t[6] *= wpz[6];
					t[7] *= wpz[7];
					t[8] *= wpz[8];
					t[9] *= wpz[9];
					t[10] *= wpz[10];
					t[11] *= wpz[11];
					t[12] *= wpz[12];
					t[13] *= wpz[13];
					t[14] *= wpz[14];
					t[15] *= wpz[15];
					/*
					 * a0, b1, c2, d3
					 * e4, f5, g6, h7
					 * i8, j9, k10 l11
					 * m12 n13 o14 p16
					 * {{a + e + i, b + f + j, c + g + k, d + h + l},
					 * {e - i - m, f - j - n, g - k - o, h - l - p}}
					 */
					/* row 1 */
					d[0] = t[0] + t[4] + t[8];
					d[1] = t[1] + t[5] + t[9];
					d[2] = t[2] + t[6] + t[10];
					d[3] = t[3] + t[7] + t[11];
					/* row 2 */
					d[4] = t[4] - t[8] - t[12];
					d[5] = t[5] - t[9] - t[13];
					d[6] = t[6] - t[10] - t[14];
					d[7] = t[7] - t[11] - t[15];
					/*
					 * {{a + b + c, b - c - d},
					 * {e + f + g, f - g - h}}
					 */
					p[0] += d[0] + d[1] + d[2];
					p[1] += d[1] - d[2] - d[3];
					p[2] += d[4] + d[5] + d[6];
					p[3] += d[5] - d[6] - d[7];
					// move to the next channel
					wpz += 16;
				}
				bp[i[0] * binc[0]] = p[0];
				bp[(i[0] + 1) * binc[0]] = p[1];
				bp[(binc[1] + i[0]) * binc[0]] = p[2];
				bp[(binc[1] + i[0] + 1) * binc[0]] = p[3];
			}
			bp += binc[1] * binc[0] * 2;
			ap += ainc[1] * ainc[0] * (ccv_max((i[1] + 2) - hint.border.begin[2], 0) - ccv_max(i[1] - hint.border.begin[2], 0));
		}
	} parallel_endfor
	ccv_nnc_tensor_free(gwtg);
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
