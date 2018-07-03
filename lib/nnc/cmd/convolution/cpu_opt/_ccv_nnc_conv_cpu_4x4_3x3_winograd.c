#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#if defined(HAVE_SSE2)
#include <xmmintrin.h>
#elif defined(HAVE_NEON)
#include <arm_neon.h>
#endif
#ifdef USE_OPENMP
#include <omp.h>
#endif
#ifdef USE_DISPATCH
#include <dispatch/dispatch.h>
#endif
#include "../_ccv_nnc_conv_cpu_opt.h"

#define set_n_m_dim(i, x, wd, ad) \
	do { \
		n[x] = ccv_max((i) * hint.stride.dim[x] - hint.border.begin[x], 0) - ((i) * hint.stride.dim[x] - hint.border.begin[x]); \
		m[x] = wd[x + 1] - n[x] - ((i) * hint.stride.dim[x] - hint.border.begin[x] + wd[x + 1] - ccv_min(ad[x], (i) * hint.stride.dim[x] - hint.border.begin[x] + wd[x + 1])); \
	} while (0)

inline static void _ccv_nnc_winograd_4x4_3x3_gwtg_ref(const float* const w, const int c, float* gwtg)
{
	int i;
	for (i = 0; i < c; i++)
	{
		float g[18];
		/*
		 * a0, b1, c2
		 * d3, e4, f5
		 * g6, h7, i8
		 * {{a/4, b/4, c/4},
		 * {1/6 (-a - d - g), 1/6 (-b - e - h), 1/6 (-c - f - i)},
		 * {1/6 (-a + d - g), 1/6 (-b + e - h), 1/6 (-c + f - i)},
		 * {1/24 (a + 2 d + 4 g), 1/24 (b + 2 e + 4 h), 1/24 (c + 2 f + 4 i)},
		 * {1/24 (a - 2 d + 4 g), 1/24 (b - 2 e + 4 h), 1/24 (c - 2 f + 4 i)},
		 * {g, h, i}}
		 */
		/* row 1 */
		g[0] = w[i] / 4;
		g[1] = w[c + i] / 4;
		g[2] = w[2 * c + i] / 4;
		/* row 2 */
		g[3] = -(w[i] + w[3 * c + i] + w[6 * c + i]) / 6;
		g[4] = -(w[c + i] + w[4 * c + i] + w[7 * c + i]) / 6;
		g[5] = -(w[2 * c + i] + w[5 * c + i] + w[8 * c + i]) / 6;
		/* row 3 */
		g[6] = (-w[i] + w[3 * c + i] - w[6 * c + i]) / 6;
		g[7] = (-w[c + i] + w[4 * c + i] - w[7 * c + i]) / 6;
		g[8] = (-w[2 * c + i] + w[5 * c + i] - w[8 * c + i]) / 6;
		/* row 4 */
		g[9] = (w[i] + 2 * w[3 * c + i] + 4 * w[6 * c + i]) / 24;
		g[10] = (w[c + i] + 2 * w[4 * c + i] + 4 * w[7 * c + i]) / 24;
		g[11] = (w[2 * c + i] + 2 * w[5 * c + i] + 4 * w[8 * c + i]) / 24;
		/* row 5 */
		g[12] = (w[i] - 2 * w[3 * c + i] + 4 * w[6 * c + i]) / 24;
		g[13] = (w[c + i] - 2 * w[4 * c + i] + 4 * w[7 * c + i]) / 24;
		g[14] = (w[2 * c + i] - 2 * w[5 * c + i] + 4 * w[8 * c + i]) / 24;
		/* row 6 */
		g[15] = w[6 * c + i];
		g[16] = w[7 * c + i];
		g[17] = w[8 * c + i];
		/*
		 * a0, b1, c2
		 * d3, e4, f5
		 * g6, h7, i8
		 * j9, k10,l11
		 * m12,n13,o14
		 * p15,q16,r17
		 * {{a/4, 1/6 (-a - b - c), 1/6 (-a + b - c), 1/24 (a + 2 b + 4 c), 1/24 (a - 2 b + 4 c), c},
		 * {d/4, 1/6 (-d - e - f), 1/6 (-d + e - f), 1/24 (d + 2 e + 4 f), 1/24 (d - 2 e + 4 f), f},
		 * {g/4, 1/6 (-g - h - i), 1/6 (-g + h - i), 1/24 (g + 2 h + 4 i), 1/24 (g - 2 h + 4 i), i},
		 * {j/4, 1/6 (-j - k - l), 1/6 (-j + k - l), 1/24 (j + 2 k + 4 l), 1/24 (j - 2 k + 4 l), l},
		 * {m/4, 1/6 (-m - n - o), 1/6 (-m + n - o), 1/24 (m + 2 n + 4 o), 1/24 (m - 2 n + 4 o), o},
		 * {p/4, 1/6 (-p - q - r), 1/6 (-p + q - r), 1/24 (p + 2 q + 4 r), 1/24 (p - 2 q + 4 r), r}}
		 */
		/* row 1 */
		gwtg[0] = g[0] / 4;
		gwtg[c] = -(g[0] + g[1] + g[2]) / 6;
		gwtg[2 * c] = (-g[0] + g[1] - g[2]) / 6;
		gwtg[3 * c] = (g[0] + 2 * g[1] + 4 * g[2]) / 24;
		gwtg[4 * c] = (g[0] - 2 * g[1] + 4 * g[2]) / 24;
		gwtg[5 * c] = g[2];
		/* row 2 */
		gwtg[6 * c] = g[3] / 4;
		gwtg[7 * c] = -(g[3] + g[4] + g[5]) / 6;
		gwtg[8 * c] = (-g[3] + g[4] - g[5]) / 6;
		gwtg[9 * c] = (g[3] + 2 * g[4] + 4 * g[5]) / 24;
		gwtg[10 * c] = (g[3] - 2 * g[4] + 4 * g[5]) / 24;
		gwtg[11 * c] = g[5];
		/* row 3 */
		gwtg[12 * c] = g[6] / 4;
		gwtg[13 * c] = -(g[6] + g[7] + g[8]) / 6;
		gwtg[14 * c] = (-g[6] + g[7] - g[8]) / 6;
		gwtg[15 * c] = (g[6] + 2 * g[7] + 4 * g[8]) / 24;
		gwtg[16 * c] = (g[6] - 2 * g[7] + 4 * g[8]) / 24;
		gwtg[17 * c] = g[8];
		/* row 4 */
		gwtg[18 * c] = g[9] / 4;
		gwtg[19 * c] = -(g[9] + g[10] + g[11]) / 6;
		gwtg[20 * c] = (-g[9] + g[10] - g[11]) / 6;
		gwtg[21 * c] = (g[9] + 2 * g[10] + 4 * g[11]) / 24;
		gwtg[22 * c] = (g[9] - 2 * g[10] + 4 * g[11]) / 24;
		gwtg[23 * c] = g[11];
		/* row 5 */
		gwtg[24 * c] = g[12] / 4;
		gwtg[25 * c] = -(g[12] + g[13] + g[14]) / 6;
		gwtg[26 * c] = (-g[12] + g[13] - g[14]) / 6;
		gwtg[27 * c] = (g[12] + 2 * g[13] + 4 * g[14]) / 24;
		gwtg[28 * c] = (g[12] - 2 * g[13] + 4 * g[14]) / 24;
		gwtg[29 * c] = g[14];
		/* row 6 */
		gwtg[30 * c] = g[15] / 4;
		gwtg[31 * c] = -(g[15] + g[16] + g[17]) / 6;
		gwtg[32 * c] = (-g[15] + g[16] - g[17]) / 6;
		gwtg[33 * c] = (g[15] + 2 * g[16] + 4 * g[17]) / 24;
		gwtg[34 * c] = (g[15] - 2 * g[16] + 4 * g[17]) / 24;
		gwtg[35 * c] = g[17];
		++gwtg;
	}
}

static int _ccv_nnc_conv_forw_4x4_3x3_winograd_ref(const ccv_nnc_tensor_view_t* const a, const ccv_nnc_tensor_t* const w, const ccv_nnc_tensor_t* const bias, const ccv_nnc_hint_t hint, ccv_nnc_tensor_view_t* const b)
{
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	assert(a_nd == CCV_NNC_MAX_DIM + 1 || a_nd == CCV_NNC_MAX_DIM + 2);
	const int* adim = (a_nd == CCV_NNC_MAX_DIM + 1) ? a->info.dim : a->info.dim + 1;
	const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
	assert(b_nd == CCV_NNC_MAX_DIM + 1 || b_nd == CCV_NNC_MAX_DIM + 2);
	const int* bdim = (b_nd == CCV_NNC_MAX_DIM + 1) ? b->info.dim : b->info.dim + 1;
	const int* ainc = CCV_IS_TENSOR_VIEW(a) ? ((a_nd == CCV_NNC_MAX_DIM + 1) ? a->inc : a->inc + 1) : adim;
	const int* binc = CCV_IS_TENSOR_VIEW(b) ? ((b_nd == CCV_NNC_MAX_DIM + 1) ? b->inc : b->inc + 1) : bdim;
	assert(hint.border.begin[0] <= 1);
	assert(hint.border.begin[1] <= 1);
	assert(w->info.dim[1] == 3);
	assert(w->info.dim[2] == 3);
	const int jump_dim = (bdim[0] + 3) / 4;
	// allocating workspace memory for kernel reshaping and input reshaping.
#if FOR_IS_PARALLEL
	// If we do parallel for, we need to allocate input reshaping for each block.
	float* const workmem = (float*)ccmalloc(sizeof(float) * (36 * adim[2] * jump_dim + 36 * w->info.dim[0] * w->info.dim[3]));
#else
	// Otherwise, just one block.
	float* const workmem = (float*)ccmalloc(sizeof(float) * (36 * adim[2] + 36 * w->info.dim[0] * w->info.dim[3]));
#endif
	if (!workmem)
		return CCV_NNC_EXEC_OOM;
	// Convert w to a 6x6 matrix, by computing G.w.T(G) // T for transpose.
	float* const gwtg = workmem;
	float* const btdb = workmem + 36 * w->info.dim[0] * w->info.dim[3];
	parallel_for(k, w->info.dim[0]) {
		_ccv_nnc_winograd_4x4_3x3_gwtg_ref(w->data.f32 + k * w->info.dim[3] * w->info.dim[2] * w->info.dim[1], w->info.dim[3], gwtg + k * 36 * w->info.dim[3]);
	} parallel_endfor
	// kernel weight for one dim.
	// Workaround issues of dispatch_apply (cannot reference to on-stack array)
	const int tile_dim_s[CCV_NNC_MAX_DIM_ALLOC] = {
		w->info.dim[0], 6, 6, w->info.dim[3]
	};
	const int* const tile_dim = tile_dim_s;
	// This block will be cause in each for-loop, therefore, you can use it to generate some temporary variables.
	if (bias)
	{
		const float* const biasval = bias->data.f32;
		parallel_for(i, jump_dim) {
			const int y = i * 4; // i is unsigned.
			int j, x, k, c;
			int n[CCV_NNC_MAX_DIM];
			int m[CCV_NNC_MAX_DIM];
			int z[CCV_NNC_MAX_DIM];
			set_n_m_dim(y, 0, tile_dim, adim);
			z[0] = ccv_min(y + 4, bdim[0]) - y;
			const float* ap = a->data.f32 + ccv_max(y - hint.border.begin[0], 0) * ainc[1] * ainc[2];
			float* bp = b->data.f32 + y * binc[1] * binc[2];
			for (x = 0; x < bdim[1]; x += 4)
			{
				set_n_m_dim(x, 1, tile_dim, adim);
				z[1] = ccv_min(x + 4, bdim[1]) - x;
#if FOR_IS_PARALLEL
				float* g = btdb + i * 36 * adim[2];
#else
				float* g = btdb;
#endif
				// zero g such that we can have zero-padding.
				memset(g, 0, sizeof(float) * 36 * adim[2]);
				int dx, dy;
				const float* apz = ap + ccv_max(x - hint.border.begin[1], 0) * ainc[2];
				float* gz = g + (n[0] * 6 + n[1]) * adim[2];
				unroll_for(dy, m[0], 6) {
					unroll_for(dx, m[1], 6) {
						float* const gzu = gz + (dy * 6 + dx) * adim[2];
						for (c = 0; c < adim[2]; c++)
							gzu[c] = apz[dx * ainc[2] + c];
					} unroll_endfor
					apz += ainc[1] * ainc[2];
				} unroll_endfor
				for (c = 0; c < adim[2]; c++)
				{
					/*
					 * a0, a1, a2, a3, a4, a5,
					 * b6, b7, b8, b9, b10,l11,
					 * c12,c13,c14,c15,c16,c17,
					 * d18,d19,d20,d21,d22,d23,
					 * e24,e25,e26,e27,e28,e29,
					 * f30,f31,f32,f33,f34,f35
					 * {{4 a0 - 5 c12 + e24, 4 a1 - 5 c13 + e25, 4 a2 - 5 c14 + e26, 4 a3 - 5 c15 + e27, 4 a4 - 5 c16 + e28, 4 a5 - 5 c17 + e29},
					 * {-4 b6 - 4 c12 + d18 + e24, -4 b7 - 4 c13 + d19 + e25, -4 b8 - 4 c14 + d20 + e26, -4 b9 - 4 c15 + d21 + e27, -4 b10 - 4 c16 + d22 + e28, -4 b11 - 4 c17 + d23 + e29},
					 * {4 b6 - 4 c12 - d18 + e24, 4 b7 - 4 c13 - d19 + e25, 4 b8 - 4 c14 - d20 + e26, 4 b9 - 4 c15 - d21 + e27, 4 b10 - 4 c16 - d22 + e28, 4 b11 - 4 c17 - d23 + e29},
					 * {-2 b6 - c12 + 2 d18 + e24, -2 b7 - c13 + 2 d19 + e25, -2 b8 - c14 + 2 d20 + e26, -2 b9 - c15 + 2 d21 + e27, -2 b10 - c16 + 2 d22 + e28, -2 b11 - c17 + 2 d23 + e29},
					 * {2 b6 - c12 - 2 d18 + e24, 2 b7 - c13 - 2 d19 + e25, 2 b8 - c14 - 2 d20 + e26, 2 b9 - c15 - 2 d21 + e27, 2 b10 - c16 - 2 d22 + e28, 2 b11 - c17 - 2 d23 + e29},
					 * {4 b6 - 5 d18 + f30, 4 b7 - 5 d19 + f31, 4 b8 - 5 d20 + f32, 4 b9 - 5 d21 + f33, 4 b10 - 5 d22 + f34, 4 b11 - 5 d23 + f35}}
					 */
					float d[36];
					/* BT.d */
					unroll_for(j, 6) {
						float g0 = g[j * adim[2]];
						float g12 = g[(12 + j) * adim[2]];
						float g24 = g[(24 + j) * adim[2]];
						/* row 1 */
						d[j] = 4 * g0 - 5 * g12 + g24;
						float g6 = g[(6 + j) * adim[2]];
						float g18 = g[(18 + j) * adim[2]];
						/* row 2 */
						d[6 + j] = -4 * (g6 + g12) + g18 + g24;
						/* row 3 */
						d[12 + j] = 4 * (g6 - g12) - g18 + g24;
						/* row 4 */
						d[18 + j] = 2 * (g18 - g6) - g12 + g24;
						/* row 5 */
						d[24 + j] = 2 * (g6 - g18) - g12 + g24;
						float g30 = g[(30 + j) * adim[2]];
						/* row 6 */
						d[30 + j] = 4 * g6 - 5 * g18 + g30;
					} unroll_endfor
					/*
					 * a0, a1, a2, a3, a4, a5,
					 * b6, b7, b8, b9, b10,l11,
					 * c12,c13,c14,c15,c16,c17,
					 * d18,d19,d20,d21,d22,d23,
					 * e24,e25,e26,e27,e28,e29,
					 * f30,f31,f32,f33,f34,f35
					 * {{4 a0 - 5 a2 + a4, -4 a1 - 4 a2 + a3 + a4, 4 a1 - 4 a2 - a3 + a4, -2 a1 - a2 + 2 a3 + a4, 2 a1 - a2 - 2 a3 + a4, 4 a1 - 5 a3 + a5},
					 * {b10 + 4 b6 - 5 b8, b10 - 4 b7 - 4 b8 + b9, b10 + 4 b7 - 4 b8 - b9, b10 - 2 b7 - b8 + 2 b9, b10 + 2 b7 - b8 - 2 b9, b11 + 4 b7 - 5 b9},
					 * {4 c12 - 5 c14 + c16, -4 c13 - 4 c14 + c15 + c16, 4 c13 - 4 c14 - c15 + c16, -2 c13 - c14 + 2 c15 + c16, 2 c13 - c14 - 2 c15 + c16, 4 c13 - 5 c15 + c17},
					 * {4 d18 - 5 d20 + d22, -4 d19 - 4 d20 + d21 + d22, 4 d19 - 4 d20 - d21 + d22, -2 d19 - d20 + 2 d21 + d22, 2 d19 - d20 - 2 d21 + d22, 4 d19 - 5 d21 + d23},
					 * {4 e24 - 5 e26 + e28, -4 e25 - 4 e26 + e27 + e28, 4 e25 - 4 e26 - e27 + e28, -2 e25 - e26 + 2 e27 + e28, 2 e25 - e26 - 2 e27 + e28, 4 e25 - 5 e27 + e29},
					 * {4 f30 - 5 f32 + f34, -4 f31 - 4 f32 + f33 + f34, 4 f31 - 4 f32 - f33 + f34, -2 f31 - f32 + 2 f33 + f34, 2 f31 - f32 - 2 f33 + f34, 4 f31 - 5 f33 + f35}}
					 */
					/* BT.d.B */
					unroll_for(j, 6) {
						/* row 1 - 6 */
						float* const gz = g + j * 6 * adim[2];
						float* const dz = d + j * 6;
						gz[0] = 4 * dz[0] - 5 * dz[2] + dz[4];
						gz[adim[2]] = -4 * (dz[1] + dz[2]) + dz[3] + dz[4];
						gz[2 * adim[2]] = 4 * (dz[1] - dz[2]) - dz[3] + dz[4];
						gz[3 * adim[2]] = 2 * (dz[3] - dz[1]) - dz[2] + dz[4];
						gz[4 * adim[2]] = 2 * (dz[1] - dz[3]) - dz[2] + dz[4];
						gz[5 * adim[2]] = 4 * dz[1] - 5 * dz[3] + dz[5];
					} unroll_endfor
					// move to the next channel
					++g;
				}
				const float* wpz = gwtg;
				for (k = 0; k < w->info.dim[0]; k++)
				{
					float q[36];
#if FOR_IS_PARALLEL
					g = btdb + i * 36 * adim[2];
#else
					g = btdb;
#endif
					for (j = 0; j < 36; j++)
					{
						float b = 0;
						for (c = 0; c < adim[2]; c++)
							b += g[c] * wpz[c];
						q[j] = b;
						g += adim[2];
						wpz += adim[2];
					}
					/*
					 * a0, a1, a2, a3, a4, a5,
					 * b6, b7, b8, b9, b10,l11,
					 * c12,c13,c14,c15,c16,c17,
					 * d18,d19,d20,d21,d22,d23,
					 * e24,e25,e26,e27,e28,e29,
					 * f30,f31,f32,f33,f34,f35
					 * {{a0 + b6 + c12 + d18 + e24, a1 + b7 + c13 + d19 + e25, a2 + b8 + c14 + d20 + e26, a3 + b9 + c15 + d21 + e27, a4 + b10 + c16 + d22 + e28, a5 + b11 + c17 + d23 + e29},
					 * {b6 - c12 + 2 d18 - 2 e24, b7 - c13 + 2 d19 - 2 e25, b8 - c14 + 2 d20 - 2 e26, b9 - c15 + 2 d21 - 2 e27, b10 - c16 + 2 d22 - 2 e28, b11 - c17 + 2 d23 - 2 e29},
					 * {b6 + c12 + 4 (d18 + e24), b7 + c13 + 4 (d19 + e25), b8 + c14 + 4 (d20 + e26), b9 + c15 + 4 (d21 + e27), b10 + c16 + 4 (d22 + e28), b11 + c17 + 4 (d23 + e29)},
					 * {b6 - c12 + 8 d18 - 8 e24 + f30, b7 - c13 + 8 d19 - 8 e25 + f31, b8 - c14 + 8 d20 - 8 e26 + f32, b9 - c15 + 8 d21 - 8 e27 + f33, b10 - c16 + 8 d22 - 8 e28 + f34, b11 - c17 + 8 d23 - 8 e29 + f35}}
					 */
					float d[24];
					/* row 1 */
					d[0] = q[0] + q[6] + q[12] + q[18] + q[24];
					d[1] = q[1] + q[7] + q[13] + q[19] + q[25];
					d[2] = q[2] + q[8] + q[14] + q[20] + q[26];
					d[3] = q[3] + q[9] + q[15] + q[21] + q[27];
					d[4] = q[4] + q[10] + q[16] + q[22] + q[28];
					d[5] = q[5] + q[11] + q[17] + q[23] + q[29];
					/* row 2 */
					d[6] = q[6] - q[12] + 2 * (q[18] - q[24]);
					d[7] = q[7] - q[13] + 2 * (q[19] - q[25]);
					d[8] = q[8] - q[14] + 2 * (q[20] - q[26]);
					d[9] = q[9] - q[15] + 2 * (q[21] - q[27]);
					d[10] = q[10] - q[16] + 2 * (q[22] - q[28]);
					d[11] = q[11] - q[17] + 2 * (q[23] - q[29]);
					/* row 3 */
					d[12] = q[6] + q[12] + 4 * (q[18] + q[24]);
					d[13] = q[7] + q[13] + 4 * (q[19] + q[25]);
					d[14] = q[8] + q[14] + 4 * (q[20] + q[26]);
					d[15] = q[9] + q[15] + 4 * (q[21] + q[27]);
					d[16] = q[10] + q[16] + 4 * (q[22] + q[28]);
					d[17] = q[11] + q[17] + 4 * (q[23] + q[29]);
					/* row 4 */
					d[18] = q[6] - q[12] + 8 * (q[18] - q[24]) + q[30];
					d[19] = q[7] - q[13] + 8 * (q[19] - q[25]) + q[31];
					d[20] = q[8] - q[14] + 8 * (q[20] - q[26]) + q[32];
					d[21] = q[9] - q[15] + 8 * (q[21] - q[27]) + q[33];
					d[22] = q[10] - q[16] + 8 * (q[22] - q[28]) + q[34];
					d[23] = q[11] - q[17] + 8 * (q[23] - q[29]) + q[35];
					/*
					 * {{a0 + a1 + a2 + a3 + a4, a1 - a2 + 2 a3 - 2 a4, a1 + a2 + 4 (a3 + a4), a1 - a2 + 8 a3 - 8 a4 + a5},
					 * {b10 + b6 + b7 + b8 + b9, -2 b10 + b7 - b8 + 2 b9, 4 b10 + b7 + b8 + 4 b9, -8 b10 + b11 + b7 - b8 + 8 b9},
					 * {c12 + c13 + c14 + c15 + c16, c13 - c14 + 2 c15 - 2 c16, c13 + c14 + 4 (c15 + c16), c13 - c14 + 8 c15 - 8 c16 + c17},
					 * {d18 + d19 + d20 + d21 + d22, d19 - d20 + 2 d21 - 2 d22, d19 + d20 + 4 (d21 + d22), d19 - d20 + 8 d21 - 8 d22 + d23}}
					 */
					float* bpz = bp + x * binc[2] + k;
					unroll_for(dy, z[0], 4) {
						float r[] = {
							d[dy * 6 + 0] + d[dy * 6 + 1] + d[dy * 6 + 2] + d[dy * 6 + 3] + d[dy * 6 + 4] + biasval[k],
							d[dy * 6 + 1] - d[dy * 6 + 2] + 2 * (d[dy * 6 + 3] - d[dy * 6 + 4]) + biasval[k],
							d[dy * 6 + 1] + d[dy * 6 + 2] + 4 * (d[dy * 6 + 3] + d[dy * 6 + 4]) + biasval[k],
							d[dy * 6 + 1] - d[dy * 6 + 2] + 8 * (d[dy * 6 + 3] - d[dy * 6 + 4]) + d[dy * 6 + 5] + biasval[k],
						};
						unroll_for(dx, z[1], 4) {
							bpz[dx * binc[2]] = r[dx];
						} unroll_endfor
						bpz += binc[1] * binc[2];
					} unroll_endfor
				}
			}
		} parallel_endfor
	} else {
		parallel_for(i, jump_dim) {
			const int y = i * 4; // i is unsigned.
			int j, x, k, c;
			int n[CCV_NNC_MAX_DIM];
			int m[CCV_NNC_MAX_DIM];
			int z[CCV_NNC_MAX_DIM];
			set_n_m_dim(y, 0, tile_dim, adim);
			z[0] = ccv_min(y + 4, bdim[0]) - y;
			const float* ap = a->data.f32 + ccv_max(y - hint.border.begin[0], 0) * ainc[1] * ainc[2];
			float* bp = b->data.f32 + y * binc[1] * binc[2];
			for (x = 0; x < bdim[1]; x += 4)
			{
				set_n_m_dim(x, 1, tile_dim, adim);
				z[1] = ccv_min(x + 4, bdim[1]) - x;
#if FOR_IS_PARALLEL
				float* g = btdb + i * 36 * adim[2];
#else
				float* g = btdb;
#endif
				// zero g such that we can have zero-padding.
				memset(g, 0, sizeof(float) * 36 * adim[2]);
				int dx, dy;
				const float* apz = ap + ccv_max(x - hint.border.begin[1], 0) * ainc[2];
				float* gz = g + (n[0] * 6 + n[1]) * adim[2];
				unroll_for(dy, m[0], 6) {
					unroll_for(dx, m[1], 6) {
						float* const gzu = gz + (dy * 6 + dx) * adim[2];
						for (c = 0; c < adim[2]; c++)
							gzu[c] = apz[dx * ainc[2] + c];
					} unroll_endfor
					apz += ainc[1] * ainc[2];
				} unroll_endfor
				for (c = 0; c < adim[2]; c++)
				{
					/*
					 * a0, a1, a2, a3, a4, a5,
					 * b6, b7, b8, b9, b10,l11,
					 * c12,c13,c14,c15,c16,c17,
					 * d18,d19,d20,d21,d22,d23,
					 * e24,e25,e26,e27,e28,e29,
					 * f30,f31,f32,f33,f34,f35
					 * {{4 a0 - 5 c12 + e24, 4 a1 - 5 c13 + e25, 4 a2 - 5 c14 + e26, 4 a3 - 5 c15 + e27, 4 a4 - 5 c16 + e28, 4 a5 - 5 c17 + e29},
					 * {-4 b6 - 4 c12 + d18 + e24, -4 b7 - 4 c13 + d19 + e25, -4 b8 - 4 c14 + d20 + e26, -4 b9 - 4 c15 + d21 + e27, -4 b10 - 4 c16 + d22 + e28, -4 b11 - 4 c17 + d23 + e29},
					 * {4 b6 - 4 c12 - d18 + e24, 4 b7 - 4 c13 - d19 + e25, 4 b8 - 4 c14 - d20 + e26, 4 b9 - 4 c15 - d21 + e27, 4 b10 - 4 c16 - d22 + e28, 4 b11 - 4 c17 - d23 + e29},
					 * {-2 b6 - c12 + 2 d18 + e24, -2 b7 - c13 + 2 d19 + e25, -2 b8 - c14 + 2 d20 + e26, -2 b9 - c15 + 2 d21 + e27, -2 b10 - c16 + 2 d22 + e28, -2 b11 - c17 + 2 d23 + e29},
					 * {2 b6 - c12 - 2 d18 + e24, 2 b7 - c13 - 2 d19 + e25, 2 b8 - c14 - 2 d20 + e26, 2 b9 - c15 - 2 d21 + e27, 2 b10 - c16 - 2 d22 + e28, 2 b11 - c17 - 2 d23 + e29},
					 * {4 b6 - 5 d18 + f30, 4 b7 - 5 d19 + f31, 4 b8 - 5 d20 + f32, 4 b9 - 5 d21 + f33, 4 b10 - 5 d22 + f34, 4 b11 - 5 d23 + f35}}
					 */
					float d[36];
					/* BT.d */
					unroll_for(j, 6) {
						float g0 = g[j * adim[2]];
						float g12 = g[(12 + j) * adim[2]];
						float g24 = g[(24 + j) * adim[2]];
						/* row 1 */
						d[j] = 4 * g0 - 5 * g12 + g24;
						float g6 = g[(6 + j) * adim[2]];
						float g18 = g[(18 + j) * adim[2]];
						/* row 2 */
						d[6 + j] = -4 * (g6 + g12) + g18 + g24;
						/* row 3 */
						d[12 + j] = 4 * (g6 - g12) - g18 + g24;
						/* row 4 */
						d[18 + j] = 2 * (g18 - g6) - g12 + g24;
						/* row 5 */
						d[24 + j] = 2 * (g6 - g18) - g12 + g24;
						float g30 = g[(30 + j) * adim[2]];
						/* row 6 */
						d[30 + j] = 4 * g6 - 5 * g18 + g30;
					} unroll_endfor
					/*
					 * a0, a1, a2, a3, a4, a5,
					 * b6, b7, b8, b9, b10,l11,
					 * c12,c13,c14,c15,c16,c17,
					 * d18,d19,d20,d21,d22,d23,
					 * e24,e25,e26,e27,e28,e29,
					 * f30,f31,f32,f33,f34,f35
					 * {{4 a0 - 5 a2 + a4, -4 a1 - 4 a2 + a3 + a4, 4 a1 - 4 a2 - a3 + a4, -2 a1 - a2 + 2 a3 + a4, 2 a1 - a2 - 2 a3 + a4, 4 a1 - 5 a3 + a5},
					 * {b10 + 4 b6 - 5 b8, b10 - 4 b7 - 4 b8 + b9, b10 + 4 b7 - 4 b8 - b9, b10 - 2 b7 - b8 + 2 b9, b10 + 2 b7 - b8 - 2 b9, b11 + 4 b7 - 5 b9},
					 * {4 c12 - 5 c14 + c16, -4 c13 - 4 c14 + c15 + c16, 4 c13 - 4 c14 - c15 + c16, -2 c13 - c14 + 2 c15 + c16, 2 c13 - c14 - 2 c15 + c16, 4 c13 - 5 c15 + c17},
					 * {4 d18 - 5 d20 + d22, -4 d19 - 4 d20 + d21 + d22, 4 d19 - 4 d20 - d21 + d22, -2 d19 - d20 + 2 d21 + d22, 2 d19 - d20 - 2 d21 + d22, 4 d19 - 5 d21 + d23},
					 * {4 e24 - 5 e26 + e28, -4 e25 - 4 e26 + e27 + e28, 4 e25 - 4 e26 - e27 + e28, -2 e25 - e26 + 2 e27 + e28, 2 e25 - e26 - 2 e27 + e28, 4 e25 - 5 e27 + e29},
					 * {4 f30 - 5 f32 + f34, -4 f31 - 4 f32 + f33 + f34, 4 f31 - 4 f32 - f33 + f34, -2 f31 - f32 + 2 f33 + f34, 2 f31 - f32 - 2 f33 + f34, 4 f31 - 5 f33 + f35}}
					 */
					/* BT.d.B */
					unroll_for(j, 6) {
						/* row 1 - 6 */
						float* const gz = g + j * 6 * adim[2];
						float* const dz = d + j * 6;
						gz[0] = 4 * dz[0] - 5 * dz[2] + dz[4];
						gz[adim[2]] = -4 * (dz[1] + dz[2]) + dz[3] + dz[4];
						gz[2 * adim[2]] = 4 * (dz[1] - dz[2]) - dz[3] + dz[4];
						gz[3 * adim[2]] = 2 * (dz[3] - dz[1]) - dz[2] + dz[4];
						gz[4 * adim[2]] = 2 * (dz[1] - dz[3]) - dz[2] + dz[4];
						gz[5 * adim[2]] = 4 * dz[1] - 5 * dz[3] + dz[5];
					} unroll_endfor
					// move to the next channel
					++g;
				}
				const float* wpz = gwtg;
				for (k = 0; k < w->info.dim[0]; k++)
				{
					float q[36];
#if FOR_IS_PARALLEL
					g = btdb + i * 36 * adim[2];
#else
					g = btdb;
#endif
					for (j = 0; j < 36; j++)
					{
						float b = 0;
						for (c = 0; c < adim[2]; c++)
							b += g[c] * wpz[c];
						q[j] = b;
						g += adim[2];
						wpz += adim[2];
					}
					/*
					 * a0, a1, a2, a3, a4, a5,
					 * b6, b7, b8, b9, b10,l11,
					 * c12,c13,c14,c15,c16,c17,
					 * d18,d19,d20,d21,d22,d23,
					 * e24,e25,e26,e27,e28,e29,
					 * f30,f31,f32,f33,f34,f35
					 * {{a0 + b6 + c12 + d18 + e24, a1 + b7 + c13 + d19 + e25, a2 + b8 + c14 + d20 + e26, a3 + b9 + c15 + d21 + e27, a4 + b10 + c16 + d22 + e28, a5 + b11 + c17 + d23 + e29},
					 * {b6 - c12 + 2 d18 - 2 e24, b7 - c13 + 2 d19 - 2 e25, b8 - c14 + 2 d20 - 2 e26, b9 - c15 + 2 d21 - 2 e27, b10 - c16 + 2 d22 - 2 e28, b11 - c17 + 2 d23 - 2 e29},
					 * {b6 + c12 + 4 (d18 + e24), b7 + c13 + 4 (d19 + e25), b8 + c14 + 4 (d20 + e26), b9 + c15 + 4 (d21 + e27), b10 + c16 + 4 (d22 + e28), b11 + c17 + 4 (d23 + e29)},
					 * {b6 - c12 + 8 d18 - 8 e24 + f30, b7 - c13 + 8 d19 - 8 e25 + f31, b8 - c14 + 8 d20 - 8 e26 + f32, b9 - c15 + 8 d21 - 8 e27 + f33, b10 - c16 + 8 d22 - 8 e28 + f34, b11 - c17 + 8 d23 - 8 e29 + f35}}
					 */
					float d[24];
					/* row 1 */
					d[0] = q[0] + q[6] + q[12] + q[18] + q[24];
					d[1] = q[1] + q[7] + q[13] + q[19] + q[25];
					d[2] = q[2] + q[8] + q[14] + q[20] + q[26];
					d[3] = q[3] + q[9] + q[15] + q[21] + q[27];
					d[4] = q[4] + q[10] + q[16] + q[22] + q[28];
					d[5] = q[5] + q[11] + q[17] + q[23] + q[29];
					/* row 2 */
					d[6] = q[6] - q[12] + 2 * (q[18] - q[24]);
					d[7] = q[7] - q[13] + 2 * (q[19] - q[25]);
					d[8] = q[8] - q[14] + 2 * (q[20] - q[26]);
					d[9] = q[9] - q[15] + 2 * (q[21] - q[27]);
					d[10] = q[10] - q[16] + 2 * (q[22] - q[28]);
					d[11] = q[11] - q[17] + 2 * (q[23] - q[29]);
					/* row 3 */
					d[12] = q[6] + q[12] + 4 * (q[18] + q[24]);
					d[13] = q[7] + q[13] + 4 * (q[19] + q[25]);
					d[14] = q[8] + q[14] + 4 * (q[20] + q[26]);
					d[15] = q[9] + q[15] + 4 * (q[21] + q[27]);
					d[16] = q[10] + q[16] + 4 * (q[22] + q[28]);
					d[17] = q[11] + q[17] + 4 * (q[23] + q[29]);
					/* row 4 */
					d[18] = q[6] - q[12] + 8 * (q[18] - q[24]) + q[30];
					d[19] = q[7] - q[13] + 8 * (q[19] - q[25]) + q[31];
					d[20] = q[8] - q[14] + 8 * (q[20] - q[26]) + q[32];
					d[21] = q[9] - q[15] + 8 * (q[21] - q[27]) + q[33];
					d[22] = q[10] - q[16] + 8 * (q[22] - q[28]) + q[34];
					d[23] = q[11] - q[17] + 8 * (q[23] - q[29]) + q[35];
					/*
					 * {{a0 + a1 + a2 + a3 + a4, a1 - a2 + 2 a3 - 2 a4, a1 + a2 + 4 (a3 + a4), a1 - a2 + 8 a3 - 8 a4 + a5},
					 * {b10 + b6 + b7 + b8 + b9, -2 b10 + b7 - b8 + 2 b9, 4 b10 + b7 + b8 + 4 b9, -8 b10 + b11 + b7 - b8 + 8 b9},
					 * {c12 + c13 + c14 + c15 + c16, c13 - c14 + 2 c15 - 2 c16, c13 + c14 + 4 (c15 + c16), c13 - c14 + 8 c15 - 8 c16 + c17},
					 * {d18 + d19 + d20 + d21 + d22, d19 - d20 + 2 d21 - 2 d22, d19 + d20 + 4 (d21 + d22), d19 - d20 + 8 d21 - 8 d22 + d23}}
					 */
					float* bpz = bp + x * binc[2] + k;
					unroll_for(dy, z[0], 4) {
						float r[] = {
							d[dy * 6 + 0] + d[dy * 6 + 1] + d[dy * 6 + 2] + d[dy * 6 + 3] + d[dy * 6 + 4],
							d[dy * 6 + 1] - d[dy * 6 + 2] + 2 * (d[dy * 6 + 3] - d[dy * 6 + 4]),
							d[dy * 6 + 1] + d[dy * 6 + 2] + 4 * (d[dy * 6 + 3] + d[dy * 6 + 4]),
							d[dy * 6 + 1] - d[dy * 6 + 2] + 8 * (d[dy * 6 + 3] - d[dy * 6 + 4]) + d[dy * 6 + 5],
						};
						unroll_for(dx, z[1], 4) {
							bpz[dx * binc[2]] = r[dx];
						} unroll_endfor
						bpz += binc[1] * binc[2];
					} unroll_endfor
				}
			}
		} parallel_endfor
	}
	ccfree(workmem);
	return CCV_NNC_EXEC_SUCCESS;
}

#ifdef HAVE_SSE2
inline static void _ccv_nnc_winograd_4x4_3x3_gwtg_sse2(const float* const w, const int* const dim, float* const gwtg)
{
	const int jump_dim = dim[0] / 4;
	const int dimCx4 = (dim[3] + 3) & -4;
	parallel_for(k, jump_dim) {
		int i, j;
		float* gwtgz = gwtg + k * 4 * 36 * dimCx4;
		const float* wz[] = {
			w + (k * 4) * 9 * dim[3],
			w + (k * 4 + 1) * 9 * dim[3],
			w + (k * 4 + 2) * 9 * dim[3],
			w + (k * 4 + 3) * 9 * dim[3],
		};
		for (i = 0; i < dim[3]; i++)
		{
			float x9w[9 * 4] __attribute__ ((__aligned__(16)));
			unroll_for(j, 9) {
				x9w[j * 4] = wz[0][j * dim[3] + i];
				x9w[j * 4 + 1] = wz[1][j * dim[3] + i];
				x9w[j * 4 + 2] = wz[2][j * dim[3] + i];
				x9w[j * 4 + 3] = wz[3][j * dim[3] + i];
			} unroll_endfor
			float g[18 * 4] __attribute__ ((__aligned__(16)));
			__m128 x9w0 = _mm_load_ps(x9w);
			__m128 x9w1 = _mm_load_ps(x9w + 4);
			__m128 x9w2 = _mm_load_ps(x9w + 8);
			__m128 x9w3 = _mm_load_ps(x9w + 12);
			__m128 x9w4 = _mm_load_ps(x9w + 16);
			__m128 x9w5 = _mm_load_ps(x9w + 20);
			__m128 x9w6 = _mm_load_ps(x9w + 24);
			__m128 x9w7 = _mm_load_ps(x9w + 28);
			__m128 x9w8 = _mm_load_ps(x9w + 32);
			/* row 1 */
			__m128 c1_4 = _mm_set1_ps(1.0 / 4);
			_mm_store_ps(g, _mm_mul_ps(x9w0, c1_4));
			_mm_store_ps(g + 4, _mm_mul_ps(x9w1, c1_4));
			_mm_store_ps(g + 8, _mm_mul_ps(x9w2, c1_4));
			/* row 2 */
			__m128 cn1_6 = _mm_set1_ps(-1.0 / 6);
			_mm_store_ps(g + 12, _mm_mul_ps(_mm_add_ps(_mm_add_ps(x9w0, x9w6), x9w3), cn1_6));
			_mm_store_ps(g + 16, _mm_mul_ps(_mm_add_ps(_mm_add_ps(x9w1, x9w7), x9w4), cn1_6));
			_mm_store_ps(g + 20, _mm_mul_ps(_mm_add_ps(_mm_add_ps(x9w2, x9w8), x9w5), cn1_6));
			/* row 3 */
			_mm_store_ps(g + 24, _mm_mul_ps(_mm_sub_ps(_mm_add_ps(x9w0, x9w6), x9w3), cn1_6));
			_mm_store_ps(g + 28, _mm_mul_ps(_mm_sub_ps(_mm_add_ps(x9w1, x9w7), x9w4), cn1_6));
			_mm_store_ps(g + 32, _mm_mul_ps(_mm_sub_ps(_mm_add_ps(x9w2, x9w8), x9w5), cn1_6));
			/* row 6 */
			_mm_store_ps(g + 60, x9w6);
			_mm_store_ps(g + 64, x9w7);
			_mm_store_ps(g + 68, x9w8);
			/* w[x] * 2 */
			x9w3 = _mm_add_ps(x9w3, x9w3);
			x9w4 = _mm_add_ps(x9w4, x9w4);
			x9w5 = _mm_add_ps(x9w5, x9w5);
			/* w[x] * 4 */
			x9w6 = _mm_add_ps(x9w6, x9w6);
			x9w6 = _mm_add_ps(x9w6, x9w6);
			x9w7 = _mm_add_ps(x9w7, x9w7);
			x9w7 = _mm_add_ps(x9w7, x9w7);
			x9w8 = _mm_add_ps(x9w8, x9w8);
			x9w8 = _mm_add_ps(x9w8, x9w8);
			/* row 4 */
			__m128 c1_24 = _mm_set1_ps(1.0 / 24);
			_mm_store_ps(g + 36, _mm_mul_ps(_mm_add_ps(_mm_add_ps(x9w0, x9w6), x9w3), c1_24));
			_mm_store_ps(g + 40, _mm_mul_ps(_mm_add_ps(_mm_add_ps(x9w1, x9w7), x9w4), c1_24));
			_mm_store_ps(g + 44, _mm_mul_ps(_mm_add_ps(_mm_add_ps(x9w2, x9w8), x9w5), c1_24));
			/* row 5 */
			_mm_store_ps(g + 48, _mm_mul_ps(_mm_sub_ps(_mm_add_ps(x9w0, x9w6), x9w3), c1_24));
			_mm_store_ps(g + 52, _mm_mul_ps(_mm_sub_ps(_mm_add_ps(x9w1, x9w7), x9w4), c1_24));
			_mm_store_ps(g + 56, _mm_mul_ps(_mm_sub_ps(_mm_add_ps(x9w2, x9w8), x9w5), c1_24));
			unroll_for(j, 6) {
				const float* const gz = g + j * 12;
				float* const gwtgzu = gwtgz + j * 24 * dimCx4;
				__m128 g0 = _mm_load_ps(gz);
				__m128 g1 = _mm_load_ps(gz + 4);
				__m128 g2 = _mm_load_ps(gz + 8);
				_mm_store_ps(gwtgzu, _mm_mul_ps(g0, c1_4));
				_mm_store_ps(gwtgzu + 4 * dimCx4, _mm_mul_ps(_mm_add_ps(_mm_add_ps(g0, g2), g1), cn1_6));
				_mm_store_ps(gwtgzu + 8 * dimCx4, _mm_mul_ps(_mm_sub_ps(_mm_add_ps(g0, g2), g1), cn1_6));
				_mm_store_ps(gwtgzu + 20 * dimCx4, g2);
				/* g[1] * 2 */
				g1 = _mm_add_ps(g1, g1);
				/* g[2] * 4 */
				g2 = _mm_add_ps(g2, g2);
				g2 = _mm_add_ps(g2, g2);
				_mm_store_ps(gwtgzu + 12 * dimCx4, _mm_mul_ps(_mm_add_ps(_mm_add_ps(g0, g2), g1), c1_24));
				_mm_store_ps(gwtgzu + 16 * dimCx4, _mm_mul_ps(_mm_sub_ps(_mm_add_ps(g0, g2), g1), c1_24));
			} unroll_endfor
			gwtgz += 4;
		}
	} parallel_endfor
}

static int _ccv_nnc_conv_forw_4x4_3x3_winograd_sse2(const ccv_nnc_tensor_view_t* const a, const ccv_nnc_tensor_t* const w, const ccv_nnc_tensor_t* const bias, const ccv_nnc_hint_t hint, ccv_nnc_tensor_view_t* const b)
{
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	assert(a_nd == CCV_NNC_MAX_DIM + 1 || a_nd == CCV_NNC_MAX_DIM + 2);
	const int* adim = (a_nd == CCV_NNC_MAX_DIM + 1) ? a->info.dim : a->info.dim + 1;
	const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
	assert(b_nd == CCV_NNC_MAX_DIM + 1 || b_nd == CCV_NNC_MAX_DIM + 2);
	const int* bdim = (b_nd == CCV_NNC_MAX_DIM + 1) ? b->info.dim : b->info.dim + 1;
	const int* ainc = CCV_IS_TENSOR_VIEW(a) ? ((a_nd == CCV_NNC_MAX_DIM + 1) ? a->inc : a->inc + 1) : adim;
	const int* binc = CCV_IS_TENSOR_VIEW(b) ? ((b_nd == CCV_NNC_MAX_DIM + 1) ? b->inc : b->inc + 1) : bdim;
	assert(hint.border.begin[0] <= 1);
	assert(hint.border.begin[1] <= 1);
	assert(w->info.dim[0] % 4 == 0);
	assert(w->info.dim[1] == 3);
	assert(w->info.dim[2] == 3);
	const int jump_dim = (bdim[0] + 3) / 4;
	const int dimCx4 = (adim[2] + 3) & -4;
	// allocating workspace memory for kernel reshaping and input reshaping.
	float* workmem = 0;
#if FOR_IS_PARALLEL
	// If we do parallel for, we need to allocate input reshaping for each block.
	ccmemalign((void **)&workmem, 16, sizeof(float) * (36 * dimCx4 * jump_dim + 36 * dimCx4 * w->info.dim[0]));
#else
	// Otherwise, just one block.
	ccmemalign((void **)&workmem, 16, sizeof(float) * (36 * dimCx4 + 36 * dimCx4 * w->info.dim[0]));
#endif
	if (!workmem)
		return CCV_NNC_EXEC_OOM;
	// Convert w to a 6x6 matrix, by computing G.w.T(G) // T for transpose.
	float* const gwtg = workmem;
	float* const btdb = workmem + 36 * dimCx4 * w->info.dim[0];
	memset(gwtg, 0, sizeof(float) * 36 * dimCx4 * w->info.dim[0]);
	_ccv_nnc_winograd_4x4_3x3_gwtg_sse2(w->data.f32, w->info.dim, gwtg);
	// kernel weight for one dim.
	// Workaround issues of dispatch_apply (cannot reference to on-stack array)
	const int tile_dim_s[CCV_NNC_MAX_DIM_ALLOC] = {
		w->info.dim[0], 6, 6, w->info.dim[3]
	};
	const int* const tile_dim = tile_dim_s;
	if (bias)
	{
		const float* const biasval = bias->data.f32;
		// This block will be cause in each for-loop, therefore, you can use it to generate some temporary variables.
		parallel_for(i, jump_dim) {
			const int y = i * 4; // i is unsigned.
			int j, x, k, c;
			int n[CCV_NNC_MAX_DIM];
			int m[CCV_NNC_MAX_DIM];
			int z[CCV_NNC_MAX_DIM];
			set_n_m_dim(y, 0, tile_dim, adim);
			z[0] = ccv_min(y + 4, bdim[0]) - y;
			const float* ap = a->data.f32 + ccv_max(y - hint.border.begin[0], 0) * ainc[1] * ainc[2];
			float* bp = b->data.f32 + y * binc[1] * binc[2];
			for (x = 0; x < bdim[1]; x += 4)
			{
				set_n_m_dim(x, 1, tile_dim, adim);
				z[1] = ccv_min(x + 4, bdim[1]) - x;
#if FOR_IS_PARALLEL
				float* g = btdb + i * 36 * dimCx4;
#else
				float* g = btdb;
#endif
				// zero g such that we can have zero-padding.
				memset(g, 0, sizeof(float) * 36 * dimCx4);
				int dx, dy;
				const float* apz = ap + ccv_max(x - hint.border.begin[1], 0) * ainc[2];
				float* gz = g + (n[0] * 6 + n[1]) * dimCx4;
				unroll_for(dy, m[0], 6) {
					unroll_for(dx, m[1], 6) {
						float* const gzu = gz + (dy * 6 + dx) * dimCx4;
						for (c = 0; c < adim[2]; c++)
							gzu[c] = apz[dx * ainc[2] + c];
					} unroll_endfor
					apz += ainc[1] * ainc[2];
				} unroll_endfor
				for (c = 0; c < adim[2]; c += 4)
				{
					float d[36 * 4]  __attribute__ ((__aligned__(16)));
					/* BT.d */
					unroll_for(j, 6) {
						/* row 1 */
						const float* const gz = g + j * dimCx4;
						float* dz = d + j * 4;
						__m128 g0 = _mm_load_ps(gz);
						__m128 g12 = _mm_load_ps(gz + 12 * dimCx4);
						__m128 g18 = _mm_load_ps(gz + 18 * dimCx4);
						__m128 g24 = _mm_load_ps(gz + 24 * dimCx4);
						g0 = _mm_add_ps(g0, g0);
						g0 = _mm_add_ps(g0, g0);
						__m128 g12x2 = _mm_add_ps(g12, g12);
						g12x2 = _mm_add_ps(g12x2, g12x2);
						g12x2 = _mm_add_ps(g12x2, g12);
						_mm_store_ps(dz, _mm_sub_ps(_mm_add_ps(g0, g24), g12x2));
						/* row 2 */
						__m128 g6 = _mm_load_ps(gz + 6 * dimCx4);
						__m128 g6x12 = _mm_add_ps(g6, g12);
						g6x12 = _mm_add_ps(g6x12, g6x12);
						g6x12 = _mm_add_ps(g6x12, g6x12);
						_mm_store_ps(dz + 24, _mm_sub_ps(_mm_add_ps(g18, g24), g6x12));
						/* row 3 */
						g6x12 = _mm_sub_ps(g6, g12);
						g6x12 = _mm_add_ps(g6x12, g6x12);
						g6x12 = _mm_add_ps(g6x12, g6x12);
						_mm_store_ps(dz + 48, _mm_add_ps(_mm_sub_ps(g24, g18), g6x12));
						/* row 4 */
						__m128 g18x6 = _mm_sub_ps(g18, g6);
						g18x6 = _mm_add_ps(g18x6, g18x6);
						_mm_store_ps(dz + 72, _mm_add_ps(_mm_sub_ps(g24, g12), g18x6));
						/* row 5 */
						_mm_store_ps(dz + 96, _mm_sub_ps(_mm_sub_ps(g24, g12), g18x6));
						/* row 6 */
						__m128 g30 = _mm_load_ps(gz + 30 * dimCx4);
						__m128 g18x2 = _mm_add_ps(g18, g18);
						g18x2 = _mm_add_ps(g18x2, g18x2);
						g18x2 = _mm_add_ps(g18, g18x2);
						g6 = _mm_add_ps(g6, g6);
						g6 = _mm_add_ps(g6, g6);
						_mm_store_ps(dz + 120, _mm_sub_ps(_mm_add_ps(g6, g30), g18x2));
					} unroll_endfor
					/* BT.d.B */
					unroll_for(j, 6) {
						float* gz = g + j * 6 * dimCx4;
						const float* const dz = d + j * 24;
						__m128 d0 = _mm_load_ps(dz);
						__m128 d1 = _mm_load_ps(dz + 4);
						__m128 d2 = _mm_load_ps(dz + 8);
						__m128 d3 = _mm_load_ps(dz + 12);
						__m128 d4 = _mm_load_ps(dz + 16);
						__m128 d5 = _mm_load_ps(dz + 20);
						d0 = _mm_add_ps(d0, d0);
						d0 = _mm_add_ps(d0, d0);
						__m128 d2x5 = _mm_add_ps(d2, d2);
						d2x5 = _mm_add_ps(d2x5, d2x5);
						d2x5 = _mm_add_ps(d2, d2x5);
						_mm_store_ps(gz, _mm_sub_ps(_mm_add_ps(d0, d4), d2x5));
						__m128 d1x2 = _mm_add_ps(d1, d2);
						d1x2 = _mm_add_ps(d1x2, d1x2);
						d1x2 = _mm_add_ps(d1x2, d1x2);
						_mm_store_ps(gz + dimCx4, _mm_sub_ps(_mm_add_ps(d3, d4), d1x2));
						d1x2 = _mm_sub_ps(d1, d2);
						d1x2 = _mm_add_ps(d1x2, d1x2);
						d1x2 = _mm_add_ps(d1x2, d1x2);
						_mm_store_ps(gz + 2 * dimCx4, _mm_add_ps(_mm_sub_ps(d4, d3), d1x2));
						__m128 d3x1 = _mm_sub_ps(d3, d1);
						d3x1 = _mm_add_ps(d3x1, d3x1);
						_mm_store_ps(gz + 3 * dimCx4, _mm_add_ps(_mm_sub_ps(d4, d2), d3x1));
						_mm_store_ps(gz + 4 * dimCx4, _mm_sub_ps(_mm_sub_ps(d4, d2), d3x1));
						d1 = _mm_add_ps(d1, d1);
						d1 = _mm_add_ps(d1, d1);
						__m128 d3x5 = _mm_add_ps(d3, d3);
						d3x5 = _mm_add_ps(d3x5, d3x5);
						d3x5 = _mm_add_ps(d3, d3x5);
						_mm_store_ps(gz + 5 * dimCx4, _mm_sub_ps(_mm_add_ps(d1, d5), d3x5));
					} unroll_endfor
					// move to the next channel
					g += 4;
				}
				const float* wpz = gwtg;
				for (k = 0; k < w->info.dim[0]; k += 4)
				{
					float q[36 * 4] __attribute__ ((__aligned__(16)));
#if FOR_IS_PARALLEL
					g = btdb + i * 36 * dimCx4;
#else
					g = btdb;
#endif
					for (j = 0; j < 36; j++)
					{
						__m128 v40 = _mm_setzero_ps();
						__m128 v41 = _mm_setzero_ps();
						__m128 v42 = _mm_setzero_ps();
						__m128 v43 = _mm_setzero_ps();
						for (c = 0; c < adim[2]; c += 4)
						{
							__m128 g4 = _mm_load_ps(g);
							__m128 w40 = _mm_load_ps(wpz);
							__m128 w41 = _mm_load_ps(wpz + 4);
							__m128 w42 = _mm_load_ps(wpz + 8);
							__m128 w43 = _mm_load_ps(wpz + 12);
							__m128 g40 = _mm_shuffle_ps(g4, g4, 0x00);
							__m128 g41 = _mm_shuffle_ps(g4, g4, 0x55);
							__m128 g42 = _mm_shuffle_ps(g4, g4, 0xAA);
							__m128 g43 = _mm_shuffle_ps(g4, g4, 0xFF);
							v40 = _mm_add_ps(_mm_mul_ps(w40, g40), v40);
							v41 = _mm_add_ps(_mm_mul_ps(w41, g41), v41);
							v42 = _mm_add_ps(_mm_mul_ps(w42, g42), v42);
							v43 = _mm_add_ps(_mm_mul_ps(w43, g43), v43);
							g += 4;
							wpz += 16;
						}
						v40 = _mm_add_ps(v40, v41);
						v42 = _mm_add_ps(v42, v43);
						_mm_store_ps(q + j * 4, _mm_add_ps(v40, v42));
					}
					float d[24 * 4] __attribute__ ((__aligned__(16)));
					unroll_for(j, 6) {
						const float* const qz = q + j * 4;
						float* const dz = d + j * 4;
						__m128 q0 = _mm_load_ps(qz);
						__m128 q6 = _mm_load_ps(qz + 24);
						__m128 q12 = _mm_load_ps(qz + 48);
						__m128 q18 = _mm_load_ps(qz + 72);
						__m128 q24 = _mm_load_ps(qz + 96);
						__m128 qs6x12 = _mm_add_ps(q6, q12);
						__m128 qs18x24 = _mm_add_ps(q18, q24);
						__m128 qss = _mm_add_ps(qs6x12, q0);
						/* row 1 */
						_mm_store_ps(dz, _mm_add_ps(qss, qs18x24));
						__m128 qn6x12 = _mm_sub_ps(q6, q12);
						__m128 qn18x24 = _mm_sub_ps(q18, q24);
						qn18x24 = _mm_add_ps(qn18x24, qn18x24);
						/* row 2 */
						_mm_store_ps(dz + 24, _mm_add_ps(qn6x12, qn18x24));
						qs18x24 = _mm_add_ps(qs18x24, qs18x24);
						qs18x24 = _mm_add_ps(qs18x24, qs18x24);
						/* row 3 */
						_mm_store_ps(dz + 48, _mm_add_ps(qs6x12, qs18x24));
						qn18x24 = _mm_add_ps(qn18x24, qn18x24);
						qn18x24 = _mm_add_ps(qn18x24, qn18x24);
						__m128 q30 = _mm_load_ps(qz + 120);
						/* row 4 */
						_mm_store_ps(dz + 72, _mm_add_ps(_mm_add_ps(qn6x12, q30), qn18x24));
					} unroll_endfor
					float* bpz = bp + x * binc[2] + k;
					__m128 bias4 = _mm_loadu_ps(biasval + k);
					switch (z[1]) {
						case 1:
							unroll_for(dy, z[0], 4) {
								const float* const dz = d + dy * 24;
								__m128 d0 = _mm_load_ps(dz);
								__m128 d1 = _mm_load_ps(dz + 4);
								__m128 d2 = _mm_load_ps(dz + 8);
								__m128 d3 = _mm_load_ps(dz + 12);
								__m128 d4 = _mm_load_ps(dz + 16);
								__m128 ds1x2 = _mm_add_ps(d1, d2);
								__m128 ds3x4 = _mm_add_ps(d3, d4);
								ds1x2 = _mm_add_ps(ds1x2, bias4);
								_mm_stream_ps(bpz, _mm_add_ps(ds1x2, _mm_add_ps(d0, ds3x4)));
								bpz += binc[1] * binc[2];
							} unroll_endfor
							break;
						case 2:
							unroll_for(dy, z[0], 4) {
								const float* const dz = d + dy * 24;
								__m128 d0 = _mm_load_ps(dz);
								__m128 d1 = _mm_load_ps(dz + 4);
								__m128 d2 = _mm_load_ps(dz + 8);
								__m128 d3 = _mm_load_ps(dz + 12);
								__m128 d4 = _mm_load_ps(dz + 16);
								__m128 ds1x2 = _mm_add_ps(d1, d2);
								__m128 ds3x4 = _mm_add_ps(d3, d4);
								ds1x2 = _mm_add_ps(ds1x2, bias4);
								_mm_stream_ps(bpz, _mm_add_ps(ds1x2, _mm_add_ps(d0, ds3x4)));
								__m128 dn1x2 = _mm_sub_ps(d1, d2);
								__m128 dn3x4 = _mm_sub_ps(d3, d4);
								dn3x4 = _mm_add_ps(dn3x4, dn3x4);
								dn1x2 = _mm_add_ps(dn1x2, bias4);
								_mm_stream_ps(bpz + binc[2], _mm_add_ps(dn1x2, dn3x4));
								bpz += binc[1] * binc[2];
							} unroll_endfor
							break;
						case 3:
							unroll_for(dy, z[0], 4) {
								const float* const dz = d + dy * 24;
								__m128 d0 = _mm_load_ps(dz);
								__m128 d1 = _mm_load_ps(dz + 4);
								__m128 d2 = _mm_load_ps(dz + 8);
								__m128 d3 = _mm_load_ps(dz + 12);
								__m128 d4 = _mm_load_ps(dz + 16);
								__m128 ds1x2 = _mm_add_ps(d1, d2);
								__m128 ds3x4 = _mm_add_ps(d3, d4);
								ds1x2 = _mm_add_ps(ds1x2, bias4);
								_mm_stream_ps(bpz, _mm_add_ps(ds1x2, _mm_add_ps(d0, ds3x4)));
								__m128 dn1x2 = _mm_sub_ps(d1, d2);
								__m128 dn3x4 = _mm_sub_ps(d3, d4);
								dn3x4 = _mm_add_ps(dn3x4, dn3x4);
								dn1x2 = _mm_add_ps(dn1x2, bias4);
								_mm_stream_ps(bpz + binc[2], _mm_add_ps(dn1x2, dn3x4));
								ds3x4 = _mm_add_ps(ds3x4, ds3x4);
								ds3x4 = _mm_add_ps(ds3x4, ds3x4);
								_mm_stream_ps(bpz + 2 * binc[2], _mm_add_ps(ds1x2, ds3x4));
								bpz += binc[1] * binc[2];
							} unroll_endfor
							break;
						case 4:
							unroll_for(dy, z[0], 4) {
								const float* const dz = d + dy * 24;
								__m128 d0 = _mm_load_ps(dz);
								__m128 d1 = _mm_load_ps(dz + 4);
								__m128 d2 = _mm_load_ps(dz + 8);
								__m128 d3 = _mm_load_ps(dz + 12);
								__m128 d4 = _mm_load_ps(dz + 16);
								__m128 ds1x2 = _mm_add_ps(d1, d2);
								__m128 ds3x4 = _mm_add_ps(d3, d4);
								ds1x2 = _mm_add_ps(ds1x2, bias4);
								_mm_stream_ps(bpz, _mm_add_ps(ds1x2, _mm_add_ps(d0, ds3x4)));
								__m128 dn1x2 = _mm_sub_ps(d1, d2);
								__m128 dn3x4 = _mm_sub_ps(d3, d4);
								dn3x4 = _mm_add_ps(dn3x4, dn3x4);
								dn1x2 = _mm_add_ps(dn1x2, bias4);
								_mm_stream_ps(bpz + binc[2], _mm_add_ps(dn1x2, dn3x4));
								ds3x4 = _mm_add_ps(ds3x4, ds3x4);
								ds3x4 = _mm_add_ps(ds3x4, ds3x4);
								_mm_stream_ps(bpz + 2 * binc[2], _mm_add_ps(ds1x2, ds3x4));
								__m128 d5 = _mm_load_ps(dz + 20);
								dn3x4 = _mm_add_ps(dn3x4, dn3x4);
								dn3x4 = _mm_add_ps(dn3x4, dn3x4);
								_mm_stream_ps(bpz + 3 * binc[2], _mm_add_ps(_mm_add_ps(dn1x2, d5), dn3x4));
								bpz += binc[1] * binc[2];
							} unroll_endfor
							break;
					};
				}
			}
		} parallel_endfor
	} else {
		// This block will be cause in each for-loop, therefore, you can use it to generate some temporary variables.
		parallel_for(i, jump_dim) {
			const int y = i * 4; // i is unsigned.
			int j, x, k, c;
			int n[CCV_NNC_MAX_DIM];
			int m[CCV_NNC_MAX_DIM];
			int z[CCV_NNC_MAX_DIM];
			set_n_m_dim(y, 0, tile_dim, adim);
			z[0] = ccv_min(y + 4, bdim[0]) - y;
			const float* ap = a->data.f32 + ccv_max(y - hint.border.begin[0], 0) * ainc[1] * ainc[2];
			float* bp = b->data.f32 + y * binc[1] * binc[2];
			for (x = 0; x < bdim[1]; x += 4)
			{
				set_n_m_dim(x, 1, tile_dim, adim);
				z[1] = ccv_min(x + 4, bdim[1]) - x;
#if FOR_IS_PARALLEL
				float* g = btdb + i * 36 * dimCx4;
#else
				float* g = btdb;
#endif
				// zero g such that we can have zero-padding.
				memset(g, 0, sizeof(float) * 36 * dimCx4);
				int dx, dy;
				const float* apz = ap + ccv_max(x - hint.border.begin[1], 0) * ainc[2];
				float* gz = g + (n[0] * 6 + n[1]) * dimCx4;
				unroll_for(dy, m[0], 6) {
					unroll_for(dx, m[1], 6) {
						float* const gzu = gz + (dy * 6 + dx) * dimCx4;
						for (c = 0; c < adim[2]; c++)
							gzu[c] = apz[dx * ainc[2] + c];
					} unroll_endfor
					apz += ainc[1] * ainc[2];
				} unroll_endfor
				for (c = 0; c < adim[2]; c += 4)
				{
					float d[36 * 4]  __attribute__ ((__aligned__(16)));
					/* BT.d */
					unroll_for(j, 6) {
						/* row 1 */
						const float* const gz = g + j * dimCx4;
						float* dz = d + j * 4;
						__m128 g0 = _mm_load_ps(gz);
						__m128 g12 = _mm_load_ps(gz + 12 * dimCx4);
						__m128 g18 = _mm_load_ps(gz + 18 * dimCx4);
						__m128 g24 = _mm_load_ps(gz + 24 * dimCx4);
						g0 = _mm_add_ps(g0, g0);
						g0 = _mm_add_ps(g0, g0);
						__m128 g12x2 = _mm_add_ps(g12, g12);
						g12x2 = _mm_add_ps(g12x2, g12x2);
						g12x2 = _mm_add_ps(g12x2, g12);
						_mm_store_ps(dz, _mm_sub_ps(_mm_add_ps(g0, g24), g12x2));
						/* row 2 */
						__m128 g6 = _mm_load_ps(gz + 6 * dimCx4);
						__m128 g6x12 = _mm_add_ps(g6, g12);
						g6x12 = _mm_add_ps(g6x12, g6x12);
						g6x12 = _mm_add_ps(g6x12, g6x12);
						_mm_store_ps(dz + 24, _mm_sub_ps(_mm_add_ps(g18, g24), g6x12));
						/* row 3 */
						g6x12 = _mm_sub_ps(g6, g12);
						g6x12 = _mm_add_ps(g6x12, g6x12);
						g6x12 = _mm_add_ps(g6x12, g6x12);
						_mm_store_ps(dz + 48, _mm_add_ps(_mm_sub_ps(g24, g18), g6x12));
						/* row 4 */
						__m128 g18x6 = _mm_sub_ps(g18, g6);
						g18x6 = _mm_add_ps(g18x6, g18x6);
						_mm_store_ps(dz + 72, _mm_add_ps(_mm_sub_ps(g24, g12), g18x6));
						/* row 5 */
						_mm_store_ps(dz + 96, _mm_sub_ps(_mm_sub_ps(g24, g12), g18x6));
						/* row 6 */
						__m128 g30 = _mm_load_ps(gz + 30 * dimCx4);
						__m128 g18x2 = _mm_add_ps(g18, g18);
						g18x2 = _mm_add_ps(g18x2, g18x2);
						g18x2 = _mm_add_ps(g18, g18x2);
						g6 = _mm_add_ps(g6, g6);
						g6 = _mm_add_ps(g6, g6);
						_mm_store_ps(dz + 120, _mm_sub_ps(_mm_add_ps(g6, g30), g18x2));
					} unroll_endfor
					/* BT.d.B */
					unroll_for(j, 6) {
						float* gz = g + j * 6 * dimCx4;
						const float* const dz = d + j * 24;
						__m128 d0 = _mm_load_ps(dz);
						__m128 d1 = _mm_load_ps(dz + 4);
						__m128 d2 = _mm_load_ps(dz + 8);
						__m128 d3 = _mm_load_ps(dz + 12);
						__m128 d4 = _mm_load_ps(dz + 16);
						__m128 d5 = _mm_load_ps(dz + 20);
						d0 = _mm_add_ps(d0, d0);
						d0 = _mm_add_ps(d0, d0);
						__m128 d2x5 = _mm_add_ps(d2, d2);
						d2x5 = _mm_add_ps(d2x5, d2x5);
						d2x5 = _mm_add_ps(d2, d2x5);
						_mm_store_ps(gz, _mm_sub_ps(_mm_add_ps(d0, d4), d2x5));
						__m128 d1x2 = _mm_add_ps(d1, d2);
						d1x2 = _mm_add_ps(d1x2, d1x2);
						d1x2 = _mm_add_ps(d1x2, d1x2);
						_mm_store_ps(gz + dimCx4, _mm_sub_ps(_mm_add_ps(d3, d4), d1x2));
						d1x2 = _mm_sub_ps(d1, d2);
						d1x2 = _mm_add_ps(d1x2, d1x2);
						d1x2 = _mm_add_ps(d1x2, d1x2);
						_mm_store_ps(gz + 2 * dimCx4, _mm_add_ps(_mm_sub_ps(d4, d3), d1x2));
						__m128 d3x1 = _mm_sub_ps(d3, d1);
						d3x1 = _mm_add_ps(d3x1, d3x1);
						_mm_store_ps(gz + 3 * dimCx4, _mm_add_ps(_mm_sub_ps(d4, d2), d3x1));
						_mm_store_ps(gz + 4 * dimCx4, _mm_sub_ps(_mm_sub_ps(d4, d2), d3x1));
						d1 = _mm_add_ps(d1, d1);
						d1 = _mm_add_ps(d1, d1);
						__m128 d3x5 = _mm_add_ps(d3, d3);
						d3x5 = _mm_add_ps(d3x5, d3x5);
						d3x5 = _mm_add_ps(d3, d3x5);
						_mm_store_ps(gz + 5 * dimCx4, _mm_sub_ps(_mm_add_ps(d1, d5), d3x5));
					} unroll_endfor
					// move to the next channel
					g += 4;
				}
				const float* wpz = gwtg;
				for (k = 0; k < w->info.dim[0]; k += 4)
				{
					float q[36 * 4] __attribute__ ((__aligned__(16)));
#if FOR_IS_PARALLEL
					g = btdb + i * 36 * dimCx4;
#else
					g = btdb;
#endif
					for (j = 0; j < 36; j++)
					{
						__m128 v40 = _mm_setzero_ps();
						__m128 v41 = _mm_setzero_ps();
						__m128 v42 = _mm_setzero_ps();
						__m128 v43 = _mm_setzero_ps();
						for (c = 0; c < adim[2]; c += 4)
						{
							__m128 g4 = _mm_load_ps(g);
							__m128 w40 = _mm_load_ps(wpz);
							__m128 w41 = _mm_load_ps(wpz + 4);
							__m128 w42 = _mm_load_ps(wpz + 8);
							__m128 w43 = _mm_load_ps(wpz + 12);
							__m128 g40 = _mm_shuffle_ps(g4, g4, 0x00);
							__m128 g41 = _mm_shuffle_ps(g4, g4, 0x55);
							__m128 g42 = _mm_shuffle_ps(g4, g4, 0xAA);
							__m128 g43 = _mm_shuffle_ps(g4, g4, 0xFF);
							v40 = _mm_add_ps(_mm_mul_ps(w40, g40), v40);
							v41 = _mm_add_ps(_mm_mul_ps(w41, g41), v41);
							v42 = _mm_add_ps(_mm_mul_ps(w42, g42), v42);
							v43 = _mm_add_ps(_mm_mul_ps(w43, g43), v43);
							g += 4;
							wpz += 16;
						}
						v40 = _mm_add_ps(v40, v41);
						v42 = _mm_add_ps(v42, v43);
						_mm_store_ps(q + j * 4, _mm_add_ps(v40, v42));
					}
					float d[24 * 4] __attribute__ ((__aligned__(16)));
					unroll_for(j, 6) {
						const float* const qz = q + j * 4;
						float* const dz = d + j * 4;
						__m128 q0 = _mm_load_ps(qz);
						__m128 q6 = _mm_load_ps(qz + 24);
						__m128 q12 = _mm_load_ps(qz + 48);
						__m128 q18 = _mm_load_ps(qz + 72);
						__m128 q24 = _mm_load_ps(qz + 96);
						__m128 qs6x12 = _mm_add_ps(q6, q12);
						__m128 qs18x24 = _mm_add_ps(q18, q24);
						__m128 qss = _mm_add_ps(qs6x12, q0);
						/* row 1 */
						_mm_store_ps(dz, _mm_add_ps(qss, qs18x24));
						__m128 qn6x12 = _mm_sub_ps(q6, q12);
						__m128 qn18x24 = _mm_sub_ps(q18, q24);
						qn18x24 = _mm_add_ps(qn18x24, qn18x24);
						/* row 2 */
						_mm_store_ps(dz + 24, _mm_add_ps(qn6x12, qn18x24));
						qs18x24 = _mm_add_ps(qs18x24, qs18x24);
						qs18x24 = _mm_add_ps(qs18x24, qs18x24);
						/* row 3 */
						_mm_store_ps(dz + 48, _mm_add_ps(qs6x12, qs18x24));
						qn18x24 = _mm_add_ps(qn18x24, qn18x24);
						qn18x24 = _mm_add_ps(qn18x24, qn18x24);
						__m128 q30 = _mm_load_ps(qz + 120);
						/* row 4 */
						_mm_store_ps(dz + 72, _mm_add_ps(_mm_add_ps(qn6x12, q30), qn18x24));
					} unroll_endfor
					float* bpz = bp + x * binc[2] + k;
					switch (z[1]) {
						case 1:
							unroll_for(dy, z[0], 4) {
								const float* const dz = d + dy * 24;
								__m128 d0 = _mm_load_ps(dz);
								__m128 d1 = _mm_load_ps(dz + 4);
								__m128 d2 = _mm_load_ps(dz + 8);
								__m128 d3 = _mm_load_ps(dz + 12);
								__m128 d4 = _mm_load_ps(dz + 16);
								__m128 ds1x2 = _mm_add_ps(d1, d2);
								__m128 ds3x4 = _mm_add_ps(d3, d4);
								_mm_stream_ps(bpz, _mm_add_ps(ds1x2, _mm_add_ps(d0, ds3x4)));
								bpz += binc[1] * binc[2];
							} unroll_endfor
							break;
						case 2:
							unroll_for(dy, z[0], 4) {
								const float* const dz = d + dy * 24;
								__m128 d0 = _mm_load_ps(dz);
								__m128 d1 = _mm_load_ps(dz + 4);
								__m128 d2 = _mm_load_ps(dz + 8);
								__m128 d3 = _mm_load_ps(dz + 12);
								__m128 d4 = _mm_load_ps(dz + 16);
								__m128 ds1x2 = _mm_add_ps(d1, d2);
								__m128 ds3x4 = _mm_add_ps(d3, d4);
								_mm_stream_ps(bpz, _mm_add_ps(ds1x2, _mm_add_ps(d0, ds3x4)));
								__m128 dn1x2 = _mm_sub_ps(d1, d2);
								__m128 dn3x4 = _mm_sub_ps(d3, d4);
								dn3x4 = _mm_add_ps(dn3x4, dn3x4);
								_mm_stream_ps(bpz + binc[2], _mm_add_ps(dn1x2, dn3x4));
								bpz += binc[1] * binc[2];
							} unroll_endfor
							break;
						case 3:
							unroll_for(dy, z[0], 4) {
								const float* const dz = d + dy * 24;
								__m128 d0 = _mm_load_ps(dz);
								__m128 d1 = _mm_load_ps(dz + 4);
								__m128 d2 = _mm_load_ps(dz + 8);
								__m128 d3 = _mm_load_ps(dz + 12);
								__m128 d4 = _mm_load_ps(dz + 16);
								__m128 ds1x2 = _mm_add_ps(d1, d2);
								__m128 ds3x4 = _mm_add_ps(d3, d4);
								_mm_stream_ps(bpz, _mm_add_ps(ds1x2, _mm_add_ps(d0, ds3x4)));
								__m128 dn1x2 = _mm_sub_ps(d1, d2);
								__m128 dn3x4 = _mm_sub_ps(d3, d4);
								dn3x4 = _mm_add_ps(dn3x4, dn3x4);
								_mm_stream_ps(bpz + binc[2], _mm_add_ps(dn1x2, dn3x4));
								ds3x4 = _mm_add_ps(ds3x4, ds3x4);
								ds3x4 = _mm_add_ps(ds3x4, ds3x4);
								_mm_stream_ps(bpz + 2 * binc[2], _mm_add_ps(ds1x2, ds3x4));
								bpz += binc[1] * binc[2];
							} unroll_endfor
							break;
						case 4:
							unroll_for(dy, z[0], 4) {
								const float* const dz = d + dy * 24;
								__m128 d0 = _mm_load_ps(dz);
								__m128 d1 = _mm_load_ps(dz + 4);
								__m128 d2 = _mm_load_ps(dz + 8);
								__m128 d3 = _mm_load_ps(dz + 12);
								__m128 d4 = _mm_load_ps(dz + 16);
								__m128 ds1x2 = _mm_add_ps(d1, d2);
								__m128 ds3x4 = _mm_add_ps(d3, d4);
								_mm_stream_ps(bpz, _mm_add_ps(ds1x2, _mm_add_ps(d0, ds3x4)));
								__m128 dn1x2 = _mm_sub_ps(d1, d2);
								__m128 dn3x4 = _mm_sub_ps(d3, d4);
								dn3x4 = _mm_add_ps(dn3x4, dn3x4);
								_mm_stream_ps(bpz + binc[2], _mm_add_ps(dn1x2, dn3x4));
								ds3x4 = _mm_add_ps(ds3x4, ds3x4);
								ds3x4 = _mm_add_ps(ds3x4, ds3x4);
								_mm_stream_ps(bpz + 2 * binc[2], _mm_add_ps(ds1x2, ds3x4));
								__m128 d5 = _mm_load_ps(dz + 20);
								dn3x4 = _mm_add_ps(dn3x4, dn3x4);
								dn3x4 = _mm_add_ps(dn3x4, dn3x4);
								_mm_stream_ps(bpz + 3 * binc[2], _mm_add_ps(_mm_add_ps(dn1x2, d5), dn3x4));
								bpz += binc[1] * binc[2];
							} unroll_endfor
							break;
					};
				}
			}
		} parallel_endfor
	}
	ccfree(workmem);
	return CCV_NNC_EXEC_SUCCESS;
}
#endif

#ifdef HAVE_NEON
inline static void _ccv_nnc_winograd_4x4_3x3_gwtg_neon(const float* const w, const int* const dim, float* const gwtg)
{
	const int jump_dim = dim[0] / 4;
	const int dimCx4 = (dim[3] + 3) & -4;
	parallel_for(k, jump_dim) {
		int i, j;
		float* gwtgz = gwtg + k * 4 * 36 * dimCx4;
		const float* wz[] = {
			w + (k * 4) * 9 * dim[3],
			w + (k * 4 + 1) * 9 * dim[3],
			w + (k * 4 + 2) * 9 * dim[3],
			w + (k * 4 + 3) * 9 * dim[3],
		};
		for (i = 0; i < dim[3]; i++)
		{
			float x9w[9 * 4] __attribute__ ((__aligned__(16)));
			unroll_for(j, 9) {
				x9w[j * 4] = wz[0][j * dim[3] + i];
				x9w[j * 4 + 1] = wz[1][j * dim[3] + i];
				x9w[j * 4 + 2] = wz[2][j * dim[3] + i];
				x9w[j * 4 + 3] = wz[3][j * dim[3] + i];
			} unroll_endfor
			float g[18 * 4] __attribute__ ((__aligned__(16)));
			float32x4_t x9w0 = vld1q_f32(x9w);
			float32x4_t x9w1 = vld1q_f32(x9w + 4);
			float32x4_t x9w2 = vld1q_f32(x9w + 8);
			float32x4_t x9w3 = vld1q_f32(x9w + 12);
			float32x4_t x9w4 = vld1q_f32(x9w + 16);
			float32x4_t x9w5 = vld1q_f32(x9w + 20);
			float32x4_t x9w6 = vld1q_f32(x9w + 24);
			float32x4_t x9w7 = vld1q_f32(x9w + 28);
			float32x4_t x9w8 = vld1q_f32(x9w + 32);
			/* row 1 */
			float32x4_t c1_4 = vdupq_n_f32(1.0 / 4);
			vst1q_f32(g, vmulq_f32(x9w0, c1_4));
			vst1q_f32(g + 4, vmulq_f32(x9w1, c1_4));
			vst1q_f32(g + 8, vmulq_f32(x9w2, c1_4));
			/* row 2 */
			float32x4_t cn1_6 = vdupq_n_f32(-1.0 / 6);
			vst1q_f32(g + 12, vmulq_f32(vaddq_f32(vaddq_f32(x9w0, x9w6), x9w3), cn1_6));
			vst1q_f32(g + 16, vmulq_f32(vaddq_f32(vaddq_f32(x9w1, x9w7), x9w4), cn1_6));
			vst1q_f32(g + 20, vmulq_f32(vaddq_f32(vaddq_f32(x9w2, x9w8), x9w5), cn1_6));
			/* row 3 */
			vst1q_f32(g + 24, vmulq_f32(vsubq_f32(vaddq_f32(x9w0, x9w6), x9w3), cn1_6));
			vst1q_f32(g + 28, vmulq_f32(vsubq_f32(vaddq_f32(x9w1, x9w7), x9w4), cn1_6));
			vst1q_f32(g + 32, vmulq_f32(vsubq_f32(vaddq_f32(x9w2, x9w8), x9w5), cn1_6));
			/* row 6 */
			vst1q_f32(g + 60, x9w6);
			vst1q_f32(g + 64, x9w7);
			vst1q_f32(g + 68, x9w8);
			/* w[x] * 2 */
			x9w3 = vaddq_f32(x9w3, x9w3);
			x9w4 = vaddq_f32(x9w4, x9w4);
			x9w5 = vaddq_f32(x9w5, x9w5);
			/* w[x] * 4 */
			x9w6 = vaddq_f32(x9w6, x9w6);
			x9w6 = vaddq_f32(x9w6, x9w6);
			x9w7 = vaddq_f32(x9w7, x9w7);
			x9w7 = vaddq_f32(x9w7, x9w7);
			x9w8 = vaddq_f32(x9w8, x9w8);
			x9w8 = vaddq_f32(x9w8, x9w8);
			/* row 4 */
			float32x4_t c1_24 = vdupq_n_f32(1.0 / 24);
			vst1q_f32(g + 36, vmulq_f32(vaddq_f32(vaddq_f32(x9w0, x9w6), x9w3), c1_24));
			vst1q_f32(g + 40, vmulq_f32(vaddq_f32(vaddq_f32(x9w1, x9w7), x9w4), c1_24));
			vst1q_f32(g + 44, vmulq_f32(vaddq_f32(vaddq_f32(x9w2, x9w8), x9w5), c1_24));
			/* row 5 */
			vst1q_f32(g + 48, vmulq_f32(vsubq_f32(vaddq_f32(x9w0, x9w6), x9w3), c1_24));
			vst1q_f32(g + 52, vmulq_f32(vsubq_f32(vaddq_f32(x9w1, x9w7), x9w4), c1_24));
			vst1q_f32(g + 56, vmulq_f32(vsubq_f32(vaddq_f32(x9w2, x9w8), x9w5), c1_24));
			unroll_for(j, 6) {
				const float* const gz = g + j * 12;
				float* const gwtgzu = gwtgz + j * 24 * dimCx4;
				float32x4_t g0 = vld1q_f32(gz);
				float32x4_t g1 = vld1q_f32(gz + 4);
				float32x4_t g2 = vld1q_f32(gz + 8);
				vst1q_f32(gwtgzu, vmulq_f32(g0, c1_4));
				vst1q_f32(gwtgzu + 4 * dimCx4, vmulq_f32(vaddq_f32(vaddq_f32(g0, g2), g1), cn1_6));
				vst1q_f32(gwtgzu + 8 * dimCx4, vmulq_f32(vsubq_f32(vaddq_f32(g0, g2), g1), cn1_6));
				vst1q_f32(gwtgzu + 20 * dimCx4, g2);
				/* g[1] * 2 */
				g1 = vaddq_f32(g1, g1);
				/* g[2] * 4 */
				g2 = vaddq_f32(g2, g2);
				g2 = vaddq_f32(g2, g2);
				vst1q_f32(gwtgzu + 12 * dimCx4, vmulq_f32(vaddq_f32(vaddq_f32(g0, g2), g1), c1_24));
				vst1q_f32(gwtgzu + 16 * dimCx4, vmulq_f32(vsubq_f32(vaddq_f32(g0, g2), g1), c1_24));
			} unroll_endfor
			gwtgz += 4;
		}
	} parallel_endfor
}

static int _ccv_nnc_conv_forw_4x4_3x3_winograd_neon(const ccv_nnc_tensor_view_t* const a, const ccv_nnc_tensor_t* const w, const ccv_nnc_tensor_t* const bias, const ccv_nnc_hint_t hint, ccv_nnc_tensor_view_t* const b)
{
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	assert(a_nd == CCV_NNC_MAX_DIM + 1 || a_nd == CCV_NNC_MAX_DIM + 2);
	const int* adim = (a_nd == CCV_NNC_MAX_DIM + 1) ? a->info.dim : a->info.dim + 1;
	const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
	assert(b_nd == CCV_NNC_MAX_DIM + 1 || b_nd == CCV_NNC_MAX_DIM + 2);
	const int* bdim = (b_nd == CCV_NNC_MAX_DIM + 1) ? b->info.dim : b->info.dim + 1;
	const int* ainc = CCV_IS_TENSOR_VIEW(a) ? ((a_nd == CCV_NNC_MAX_DIM + 1) ? a->inc : a->inc + 1) : adim;
	const int* binc = CCV_IS_TENSOR_VIEW(b) ? ((b_nd == CCV_NNC_MAX_DIM + 1) ? b->inc : b->inc + 1) : bdim;
	assert(hint.border.begin[0] <= 1);
	assert(hint.border.begin[1] <= 1);
	assert(w->info.dim[0] % 4 == 0);
	assert(w->info.dim[1] == 3);
	assert(w->info.dim[2] == 3);
	const int jump_dim = (bdim[0] + 3) / 4;
	const int dimCx4 = (adim[2] + 3) & -4;
	// allocating workspace memory for kernel reshaping and input reshaping.
	float* workmem = 0;
#if FOR_IS_PARALLEL
	// If we do parallel for, we need to allocate input reshaping for each block.
	ccmemalign((void **)&workmem, 16, sizeof(float) * (36 * dimCx4 * jump_dim + 36 * dimCx4 * w->info.dim[0]));
#else
	// Otherwise, just one block.
	ccmemalign((void **)&workmem, 16, sizeof(float) * (36 * dimCx4 + 36 * dimCx4 * w->info.dim[0]));
#endif
	if (!workmem)
		return CCV_NNC_EXEC_OOM;
	// Convert w to a 6x6 matrix, by computing G.w.T(G) // T for transpose.
	float* const gwtg = workmem;
	float* const btdb = workmem + 36 * dimCx4 * w->info.dim[0];
	memset(gwtg, 0, sizeof(float) * 36 * dimCx4 * w->info.dim[0]);
	_ccv_nnc_winograd_4x4_3x3_gwtg_neon(w->data.f32, w->info.dim, gwtg);
	// kernel weight for one dim.
	// Workaround issues of dispatch_apply (cannot reference to on-stack array)
	const int tile_dim_s[CCV_NNC_MAX_DIM_ALLOC] = {
		w->info.dim[0], 6, 6, w->info.dim[3]
	};
	const int* const tile_dim = tile_dim_s;
	if (bias)
	{
		const float* const biasval = bias->data.f32;
		// This block will be cause in each for-loop, therefore, you can use it to generate some temporary variables.
		parallel_for(i, jump_dim) {
			const int y = i * 4; // i is unsigned.
			int j, x, k, c;
			int n[CCV_NNC_MAX_DIM];
			int m[CCV_NNC_MAX_DIM];
			int z[CCV_NNC_MAX_DIM];
			set_n_m_dim(y, 0, tile_dim, adim);
			z[0] = ccv_min(y + 4, bdim[0]) - y;
			const float* ap = a->data.f32 + ccv_max(y - hint.border.begin[0], 0) * ainc[1] * ainc[2];
			float* bp = b->data.f32 + y * binc[1] * binc[2];
			for (x = 0; x < bdim[1]; x += 4)
			{
				set_n_m_dim(x, 1, tile_dim, adim);
				z[1] = ccv_min(x + 4, bdim[1]) - x;
#if FOR_IS_PARALLEL
				float* g = btdb + i * 36 * dimCx4;
#else
				float* g = btdb;
#endif
				// zero g such that we can have zero-padding.
				memset(g, 0, sizeof(float) * 36 * dimCx4);
				int dx, dy;
				const float* apz = ap + ccv_max(x - hint.border.begin[1], 0) * ainc[2];
				float* gz = g + (n[0] * 6 + n[1]) * dimCx4;
				unroll_for(dy, m[0], 6) {
					unroll_for(dx, m[1], 6) {
						float* const gzu = gz + (dy * 6 + dx) * dimCx4;
						for (c = 0; c < adim[2]; c++)
							gzu[c] = apz[dx * ainc[2] + c];
					} unroll_endfor
					apz += ainc[1] * ainc[2];
				} unroll_endfor
				for (c = 0; c < adim[2]; c += 4)
				{
					float d[36 * 4]  __attribute__ ((__aligned__(16)));
					/* BT.d */
					unroll_for(j, 6) {
						/* row 1 */
						const float* const gz = g + j * dimCx4;
						float* dz = d + j * 4;
						float32x4_t g0 = vld1q_f32(gz);
						float32x4_t g12 = vld1q_f32(gz + 12 * dimCx4);
						float32x4_t g18 = vld1q_f32(gz + 18 * dimCx4);
						float32x4_t g24 = vld1q_f32(gz + 24 * dimCx4);
						g0 = vaddq_f32(g0, g0);
						g0 = vaddq_f32(g0, g0);
						float32x4_t g12x2 = vaddq_f32(g12, g12);
						g12x2 = vaddq_f32(g12x2, g12x2);
						g12x2 = vaddq_f32(g12x2, g12);
						vst1q_f32(dz, vsubq_f32(vaddq_f32(g0, g24), g12x2));
						/* row 2 */
						float32x4_t g6 = vld1q_f32(gz + 6 * dimCx4);
						float32x4_t g6x12 = vaddq_f32(g6, g12);
						g6x12 = vaddq_f32(g6x12, g6x12);
						g6x12 = vaddq_f32(g6x12, g6x12);
						vst1q_f32(dz + 24, vsubq_f32(vaddq_f32(g18, g24), g6x12));
						/* row 3 */
						g6x12 = vsubq_f32(g6, g12);
						g6x12 = vaddq_f32(g6x12, g6x12);
						g6x12 = vaddq_f32(g6x12, g6x12);
						vst1q_f32(dz + 48, vaddq_f32(vsubq_f32(g24, g18), g6x12));
						/* row 4 */
						float32x4_t g18x6 = vsubq_f32(g18, g6);
						g18x6 = vaddq_f32(g18x6, g18x6);
						vst1q_f32(dz + 72, vaddq_f32(vsubq_f32(g24, g12), g18x6));
						/* row 5 */
						vst1q_f32(dz + 96, vsubq_f32(vsubq_f32(g24, g12), g18x6));
						/* row 6 */
						float32x4_t g30 = vld1q_f32(gz + 30 * dimCx4);
						float32x4_t g18x2 = vaddq_f32(g18, g18);
						g18x2 = vaddq_f32(g18x2, g18x2);
						g18x2 = vaddq_f32(g18, g18x2);
						g6 = vaddq_f32(g6, g6);
						g6 = vaddq_f32(g6, g6);
						vst1q_f32(dz + 120, vsubq_f32(vaddq_f32(g6, g30), g18x2));
					} unroll_endfor
					/* BT.d.B */
					unroll_for(j, 6) {
						float* gz = g + j * 6 * dimCx4;
						const float* const dz = d + j * 24;
						float32x4_t d0 = vld1q_f32(dz);
						float32x4_t d1 = vld1q_f32(dz + 4);
						float32x4_t d2 = vld1q_f32(dz + 8);
						float32x4_t d3 = vld1q_f32(dz + 12);
						float32x4_t d4 = vld1q_f32(dz + 16);
						float32x4_t d5 = vld1q_f32(dz + 20);
						d0 = vaddq_f32(d0, d0);
						d0 = vaddq_f32(d0, d0);
						float32x4_t d2x5 = vaddq_f32(d2, d2);
						d2x5 = vaddq_f32(d2x5, d2x5);
						d2x5 = vaddq_f32(d2, d2x5);
						vst1q_f32(gz, vsubq_f32(vaddq_f32(d0, d4), d2x5));
						float32x4_t d1x2 = vaddq_f32(d1, d2);
						d1x2 = vaddq_f32(d1x2, d1x2);
						d1x2 = vaddq_f32(d1x2, d1x2);
						vst1q_f32(gz + dimCx4, vsubq_f32(vaddq_f32(d3, d4), d1x2));
						d1x2 = vsubq_f32(d1, d2);
						d1x2 = vaddq_f32(d1x2, d1x2);
						d1x2 = vaddq_f32(d1x2, d1x2);
						vst1q_f32(gz + 2 * dimCx4, vaddq_f32(vsubq_f32(d4, d3), d1x2));
						float32x4_t d3x1 = vsubq_f32(d3, d1);
						d3x1 = vaddq_f32(d3x1, d3x1);
						vst1q_f32(gz + 3 * dimCx4, vaddq_f32(vsubq_f32(d4, d2), d3x1));
						vst1q_f32(gz + 4 * dimCx4, vsubq_f32(vsubq_f32(d4, d2), d3x1));
						d1 = vaddq_f32(d1, d1);
						d1 = vaddq_f32(d1, d1);
						float32x4_t d3x5 = vaddq_f32(d3, d3);
						d3x5 = vaddq_f32(d3x5, d3x5);
						d3x5 = vaddq_f32(d3, d3x5);
						vst1q_f32(gz + 5 * dimCx4, vsubq_f32(vaddq_f32(d1, d5), d3x5));
					} unroll_endfor
					// move to the next channel
					g += 4;
				}
				const float* wpz = gwtg;
				for (k = 0; k < w->info.dim[0]; k += 4)
				{
					float q[36 * 4] __attribute__ ((__aligned__(16)));
#if FOR_IS_PARALLEL
					g = btdb + i * 36 * dimCx4;
#else
					g = btdb;
#endif
					for (j = 0; j < 36; j++)
					{
						float32x4_t v40 = vmovq_n_f32(0);
						float32x4_t v41 = vmovq_n_f32(0);
						float32x4_t v42 = vmovq_n_f32(0);
						float32x4_t v43 = vmovq_n_f32(0);
						for (c = 0; c < adim[2]; c += 4)
						{
							float32x2x2_t g4 = vld2_f32(g);
							float32x4_t w40 = vld1q_f32(wpz);
							float32x4_t w41 = vld1q_f32(wpz + 4);
							float32x4_t w42 = vld1q_f32(wpz + 8);
							float32x4_t w43 = vld1q_f32(wpz + 12);
							float32x4_t g40 = vdupq_lane_f32(g4.val[0], 0);
							float32x4_t g41 = vdupq_lane_f32(g4.val[1], 0);
							float32x4_t g42 = vdupq_lane_f32(g4.val[0], 1);
							float32x4_t g43 = vdupq_lane_f32(g4.val[1], 1);
							v40 = vmlaq_f32(v40, w40, g40);
							v41 = vmlaq_f32(v41, w41, g41);
							v42 = vmlaq_f32(v42, w42, g42);
							v43 = vmlaq_f32(v43, w43, g43);
							g += 4;
							wpz += 16;
						}
						v40 = vaddq_f32(v40, v41);
						v42 = vaddq_f32(v42, v43);
						vst1q_f32(q + j * 4, vaddq_f32(v40, v42));
					}
					float d[24 * 4] __attribute__ ((__aligned__(16)));
					unroll_for(j, 6) {
						const float* const qz = q + j * 4;
						float* const dz = d + j * 4;
						float32x4_t q0 = vld1q_f32(qz);
						float32x4_t q6 = vld1q_f32(qz + 24);
						float32x4_t q12 = vld1q_f32(qz + 48);
						float32x4_t q18 = vld1q_f32(qz + 72);
						float32x4_t q24 = vld1q_f32(qz + 96);
						float32x4_t qs6x12 = vaddq_f32(q6, q12);
						float32x4_t qs18x24 = vaddq_f32(q18, q24);
						float32x4_t qss = vaddq_f32(qs6x12, q0);
						/* row 1 */
						vst1q_f32(dz, vaddq_f32(qss, qs18x24));
						float32x4_t qn6x12 = vsubq_f32(q6, q12);
						float32x4_t qn18x24 = vsubq_f32(q18, q24);
						qn18x24 = vaddq_f32(qn18x24, qn18x24);
						/* row 2 */
						vst1q_f32(dz + 24, vaddq_f32(qn6x12, qn18x24));
						qs18x24 = vaddq_f32(qs18x24, qs18x24);
						qs18x24 = vaddq_f32(qs18x24, qs18x24);
						/* row 3 */
						vst1q_f32(dz + 48, vaddq_f32(qs6x12, qs18x24));
						qn18x24 = vaddq_f32(qn18x24, qn18x24);
						qn18x24 = vaddq_f32(qn18x24, qn18x24);
						float32x4_t q30 = vld1q_f32(qz + 120);
						/* row 4 */
						vst1q_f32(dz + 72, vaddq_f32(vaddq_f32(qn6x12, q30), qn18x24));
					} unroll_endfor
					float* bpz = bp + x * binc[2] + k;
					float32x4_t bias4 = vld1q_f32(biasval + k);
					switch (z[1]) {
						case 1:
							unroll_for(dy, z[0], 4) {
								const float* const dz = d + dy * 24;
								float32x4_t d0 = vld1q_f32(dz);
								float32x4_t d1 = vld1q_f32(dz + 4);
								float32x4_t d2 = vld1q_f32(dz + 8);
								float32x4_t d3 = vld1q_f32(dz + 12);
								float32x4_t d4 = vld1q_f32(dz + 16);
								float32x4_t ds1x2 = vaddq_f32(d1, d2);
								float32x4_t ds3x4 = vaddq_f32(d3, d4);
								ds1x2 = vaddq_f32(ds1x2, bias4);
								vst1q_f32(bpz, vaddq_f32(ds1x2, vaddq_f32(d0, ds3x4)));
								bpz += binc[1] * binc[2];
							} unroll_endfor
							break;
						case 2:
							unroll_for(dy, z[0], 4) {
								const float* const dz = d + dy * 24;
								float32x4_t d0 = vld1q_f32(dz);
								float32x4_t d1 = vld1q_f32(dz + 4);
								float32x4_t d2 = vld1q_f32(dz + 8);
								float32x4_t d3 = vld1q_f32(dz + 12);
								float32x4_t d4 = vld1q_f32(dz + 16);
								float32x4_t ds1x2 = vaddq_f32(d1, d2);
								float32x4_t ds3x4 = vaddq_f32(d3, d4);
								ds1x2 = vaddq_f32(ds1x2, bias4);
								vst1q_f32(bpz, vaddq_f32(ds1x2, vaddq_f32(d0, ds3x4)));
								float32x4_t dn1x2 = vsubq_f32(d1, d2);
								float32x4_t dn3x4 = vsubq_f32(d3, d4);
								dn3x4 = vaddq_f32(dn3x4, dn3x4);
								dn1x2 = vaddq_f32(dn1x2, bias4);
								vst1q_f32(bpz + binc[2], vaddq_f32(dn1x2, dn3x4));
								bpz += binc[1] * binc[2];
							} unroll_endfor
							break;
						case 3:
							unroll_for(dy, z[0], 4) {
								const float* const dz = d + dy * 24;
								float32x4_t d0 = vld1q_f32(dz);
								float32x4_t d1 = vld1q_f32(dz + 4);
								float32x4_t d2 = vld1q_f32(dz + 8);
								float32x4_t d3 = vld1q_f32(dz + 12);
								float32x4_t d4 = vld1q_f32(dz + 16);
								float32x4_t ds1x2 = vaddq_f32(d1, d2);
								float32x4_t ds3x4 = vaddq_f32(d3, d4);
								ds1x2 = vaddq_f32(ds1x2, bias4);
								vst1q_f32(bpz, vaddq_f32(ds1x2, vaddq_f32(d0, ds3x4)));
								float32x4_t dn1x2 = vsubq_f32(d1, d2);
								float32x4_t dn3x4 = vsubq_f32(d3, d4);
								dn3x4 = vaddq_f32(dn3x4, dn3x4);
								dn1x2 = vaddq_f32(dn1x2, bias4);
								vst1q_f32(bpz + binc[2], vaddq_f32(dn1x2, dn3x4));
								ds3x4 = vaddq_f32(ds3x4, ds3x4);
								ds3x4 = vaddq_f32(ds3x4, ds3x4);
								vst1q_f32(bpz + 2 * binc[2], vaddq_f32(ds1x2, ds3x4));
								bpz += binc[1] * binc[2];
							} unroll_endfor
							break;
						case 4:
							unroll_for(dy, z[0], 4) {
								const float* const dz = d + dy * 24;
								float32x4_t d0 = vld1q_f32(dz);
								float32x4_t d1 = vld1q_f32(dz + 4);
								float32x4_t d2 = vld1q_f32(dz + 8);
								float32x4_t d3 = vld1q_f32(dz + 12);
								float32x4_t d4 = vld1q_f32(dz + 16);
								float32x4_t ds1x2 = vaddq_f32(d1, d2);
								float32x4_t ds3x4 = vaddq_f32(d3, d4);
								ds1x2 = vaddq_f32(ds1x2, bias4);
								vst1q_f32(bpz, vaddq_f32(ds1x2, vaddq_f32(d0, ds3x4)));
								float32x4_t dn1x2 = vsubq_f32(d1, d2);
								float32x4_t dn3x4 = vsubq_f32(d3, d4);
								dn3x4 = vaddq_f32(dn3x4, dn3x4);
								dn1x2 = vaddq_f32(dn1x2, bias4);
								vst1q_f32(bpz + binc[2], vaddq_f32(dn1x2, dn3x4));
								ds3x4 = vaddq_f32(ds3x4, ds3x4);
								ds3x4 = vaddq_f32(ds3x4, ds3x4);
								vst1q_f32(bpz + 2 * binc[2], vaddq_f32(ds1x2, ds3x4));
								float32x4_t d5 = vld1q_f32(dz + 20);
								dn3x4 = vaddq_f32(dn3x4, dn3x4);
								dn3x4 = vaddq_f32(dn3x4, dn3x4);
								vst1q_f32(bpz + 3 * binc[2], vaddq_f32(vaddq_f32(dn1x2, d5), dn3x4));
								bpz += binc[1] * binc[2];
							} unroll_endfor
							break;
					};
				}
			}
		} parallel_endfor
	} else {
		// This block will be cause in each for-loop, therefore, you can use it to generate some temporary variables.
		parallel_for(i, jump_dim) {
			const int y = i * 4; // i is unsigned.
			int j, x, k, c;
			int n[CCV_NNC_MAX_DIM];
			int m[CCV_NNC_MAX_DIM];
			int z[CCV_NNC_MAX_DIM];
			set_n_m_dim(y, 0, tile_dim, adim);
			z[0] = ccv_min(y + 4, bdim[0]) - y;
			const float* ap = a->data.f32 + ccv_max(y - hint.border.begin[0], 0) * ainc[1] * ainc[2];
			float* bp = b->data.f32 + y * binc[1] * binc[2];
			for (x = 0; x < bdim[1]; x += 4)
			{
				set_n_m_dim(x, 1, tile_dim, adim);
				z[1] = ccv_min(x + 4, bdim[1]) - x;
#if FOR_IS_PARALLEL
				float* g = btdb + i * 36 * dimCx4;
#else
				float* g = btdb;
#endif
				// zero g such that we can have zero-padding.
				memset(g, 0, sizeof(float) * 36 * dimCx4);
				int dx, dy;
				const float* apz = ap + ccv_max(x - hint.border.begin[1], 0) * ainc[2];
				float* gz = g + (n[0] * 6 + n[1]) * dimCx4;
				unroll_for(dy, m[0], 6) {
					unroll_for(dx, m[1], 6) {
						float* const gzu = gz + (dy * 6 + dx) * dimCx4;
						for (c = 0; c < adim[2]; c++)
							gzu[c] = apz[dx * ainc[2] + c];
					} unroll_endfor
					apz += ainc[1] * ainc[2];
				} unroll_endfor
				for (c = 0; c < adim[2]; c += 4)
				{
					float d[36 * 4]  __attribute__ ((__aligned__(16)));
					/* BT.d */
					unroll_for(j, 6) {
						/* row 1 */
						const float* const gz = g + j * dimCx4;
						float* dz = d + j * 4;
						float32x4_t g0 = vld1q_f32(gz);
						float32x4_t g12 = vld1q_f32(gz + 12 * dimCx4);
						float32x4_t g18 = vld1q_f32(gz + 18 * dimCx4);
						float32x4_t g24 = vld1q_f32(gz + 24 * dimCx4);
						g0 = vaddq_f32(g0, g0);
						g0 = vaddq_f32(g0, g0);
						float32x4_t g12x2 = vaddq_f32(g12, g12);
						g12x2 = vaddq_f32(g12x2, g12x2);
						g12x2 = vaddq_f32(g12x2, g12);
						vst1q_f32(dz, vsubq_f32(vaddq_f32(g0, g24), g12x2));
						/* row 2 */
						float32x4_t g6 = vld1q_f32(gz + 6 * dimCx4);
						float32x4_t g6x12 = vaddq_f32(g6, g12);
						g6x12 = vaddq_f32(g6x12, g6x12);
						g6x12 = vaddq_f32(g6x12, g6x12);
						vst1q_f32(dz + 24, vsubq_f32(vaddq_f32(g18, g24), g6x12));
						/* row 3 */
						g6x12 = vsubq_f32(g6, g12);
						g6x12 = vaddq_f32(g6x12, g6x12);
						g6x12 = vaddq_f32(g6x12, g6x12);
						vst1q_f32(dz + 48, vaddq_f32(vsubq_f32(g24, g18), g6x12));
						/* row 4 */
						float32x4_t g18x6 = vsubq_f32(g18, g6);
						g18x6 = vaddq_f32(g18x6, g18x6);
						vst1q_f32(dz + 72, vaddq_f32(vsubq_f32(g24, g12), g18x6));
						/* row 5 */
						vst1q_f32(dz + 96, vsubq_f32(vsubq_f32(g24, g12), g18x6));
						/* row 6 */
						float32x4_t g30 = vld1q_f32(gz + 30 * dimCx4);
						float32x4_t g18x2 = vaddq_f32(g18, g18);
						g18x2 = vaddq_f32(g18x2, g18x2);
						g18x2 = vaddq_f32(g18, g18x2);
						g6 = vaddq_f32(g6, g6);
						g6 = vaddq_f32(g6, g6);
						vst1q_f32(dz + 120, vsubq_f32(vaddq_f32(g6, g30), g18x2));
					} unroll_endfor
					/* BT.d.B */
					unroll_for(j, 6) {
						float* gz = g + j * 6 * dimCx4;
						const float* const dz = d + j * 24;
						float32x4_t d0 = vld1q_f32(dz);
						float32x4_t d1 = vld1q_f32(dz + 4);
						float32x4_t d2 = vld1q_f32(dz + 8);
						float32x4_t d3 = vld1q_f32(dz + 12);
						float32x4_t d4 = vld1q_f32(dz + 16);
						float32x4_t d5 = vld1q_f32(dz + 20);
						d0 = vaddq_f32(d0, d0);
						d0 = vaddq_f32(d0, d0);
						float32x4_t d2x5 = vaddq_f32(d2, d2);
						d2x5 = vaddq_f32(d2x5, d2x5);
						d2x5 = vaddq_f32(d2, d2x5);
						vst1q_f32(gz, vsubq_f32(vaddq_f32(d0, d4), d2x5));
						float32x4_t d1x2 = vaddq_f32(d1, d2);
						d1x2 = vaddq_f32(d1x2, d1x2);
						d1x2 = vaddq_f32(d1x2, d1x2);
						vst1q_f32(gz + dimCx4, vsubq_f32(vaddq_f32(d3, d4), d1x2));
						d1x2 = vsubq_f32(d1, d2);
						d1x2 = vaddq_f32(d1x2, d1x2);
						d1x2 = vaddq_f32(d1x2, d1x2);
						vst1q_f32(gz + 2 * dimCx4, vaddq_f32(vsubq_f32(d4, d3), d1x2));
						float32x4_t d3x1 = vsubq_f32(d3, d1);
						d3x1 = vaddq_f32(d3x1, d3x1);
						vst1q_f32(gz + 3 * dimCx4, vaddq_f32(vsubq_f32(d4, d2), d3x1));
						vst1q_f32(gz + 4 * dimCx4, vsubq_f32(vsubq_f32(d4, d2), d3x1));
						d1 = vaddq_f32(d1, d1);
						d1 = vaddq_f32(d1, d1);
						float32x4_t d3x5 = vaddq_f32(d3, d3);
						d3x5 = vaddq_f32(d3x5, d3x5);
						d3x5 = vaddq_f32(d3, d3x5);
						vst1q_f32(gz + 5 * dimCx4, vsubq_f32(vaddq_f32(d1, d5), d3x5));
					} unroll_endfor
					// move to the next channel
					g += 4;
				}
				const float* wpz = gwtg;
				for (k = 0; k < w->info.dim[0]; k += 4)
				{
					float q[36 * 4] __attribute__ ((__aligned__(16)));
#if FOR_IS_PARALLEL
					g = btdb + i * 36 * dimCx4;
#else
					g = btdb;
#endif
					for (j = 0; j < 36; j++)
					{
						float32x4_t v40 = vmovq_n_f32(0);
						float32x4_t v41 = vmovq_n_f32(0);
						float32x4_t v42 = vmovq_n_f32(0);
						float32x4_t v43 = vmovq_n_f32(0);
						for (c = 0; c < adim[2]; c += 4)
						{
							float32x2x2_t g4 = vld2_f32(g);
							float32x4_t w40 = vld1q_f32(wpz);
							float32x4_t w41 = vld1q_f32(wpz + 4);
							float32x4_t w42 = vld1q_f32(wpz + 8);
							float32x4_t w43 = vld1q_f32(wpz + 12);
							float32x4_t g40 = vdupq_lane_f32(g4.val[0], 0);
							float32x4_t g41 = vdupq_lane_f32(g4.val[1], 0);
							float32x4_t g42 = vdupq_lane_f32(g4.val[0], 1);
							float32x4_t g43 = vdupq_lane_f32(g4.val[1], 1);
							v40 = vmlaq_f32(v40, w40, g40);
							v41 = vmlaq_f32(v41, w41, g41);
							v42 = vmlaq_f32(v42, w42, g42);
							v43 = vmlaq_f32(v43, w43, g43);
							g += 4;
							wpz += 16;
						}
						v40 = vaddq_f32(v40, v41);
						v42 = vaddq_f32(v42, v43);
						vst1q_f32(q + j * 4, vaddq_f32(v40, v42));
					}
					float d[24 * 4] __attribute__ ((__aligned__(16)));
					unroll_for(j, 6) {
						const float* const qz = q + j * 4;
						float* const dz = d + j * 4;
						float32x4_t q0 = vld1q_f32(qz);
						float32x4_t q6 = vld1q_f32(qz + 24);
						float32x4_t q12 = vld1q_f32(qz + 48);
						float32x4_t q18 = vld1q_f32(qz + 72);
						float32x4_t q24 = vld1q_f32(qz + 96);
						float32x4_t qs6x12 = vaddq_f32(q6, q12);
						float32x4_t qs18x24 = vaddq_f32(q18, q24);
						float32x4_t qss = vaddq_f32(qs6x12, q0);
						/* row 1 */
						vst1q_f32(dz, vaddq_f32(qss, qs18x24));
						float32x4_t qn6x12 = vsubq_f32(q6, q12);
						float32x4_t qn18x24 = vsubq_f32(q18, q24);
						qn18x24 = vaddq_f32(qn18x24, qn18x24);
						/* row 2 */
						vst1q_f32(dz + 24, vaddq_f32(qn6x12, qn18x24));
						qs18x24 = vaddq_f32(qs18x24, qs18x24);
						qs18x24 = vaddq_f32(qs18x24, qs18x24);
						/* row 3 */
						vst1q_f32(dz + 48, vaddq_f32(qs6x12, qs18x24));
						qn18x24 = vaddq_f32(qn18x24, qn18x24);
						qn18x24 = vaddq_f32(qn18x24, qn18x24);
						float32x4_t q30 = vld1q_f32(qz + 120);
						/* row 4 */
						vst1q_f32(dz + 72, vaddq_f32(vaddq_f32(qn6x12, q30), qn18x24));
					} unroll_endfor
					float* bpz = bp + x * binc[2] + k;
					switch (z[1]) {
						case 1:
							unroll_for(dy, z[0], 4) {
								const float* const dz = d + dy * 24;
								float32x4_t d0 = vld1q_f32(dz);
								float32x4_t d1 = vld1q_f32(dz + 4);
								float32x4_t d2 = vld1q_f32(dz + 8);
								float32x4_t d3 = vld1q_f32(dz + 12);
								float32x4_t d4 = vld1q_f32(dz + 16);
								float32x4_t ds1x2 = vaddq_f32(d1, d2);
								float32x4_t ds3x4 = vaddq_f32(d3, d4);
								vst1q_f32(bpz, vaddq_f32(ds1x2, vaddq_f32(d0, ds3x4)));
								bpz += binc[1] * binc[2];
							} unroll_endfor
							break;
						case 2:
							unroll_for(dy, z[0], 4) {
								const float* const dz = d + dy * 24;
								float32x4_t d0 = vld1q_f32(dz);
								float32x4_t d1 = vld1q_f32(dz + 4);
								float32x4_t d2 = vld1q_f32(dz + 8);
								float32x4_t d3 = vld1q_f32(dz + 12);
								float32x4_t d4 = vld1q_f32(dz + 16);
								float32x4_t ds1x2 = vaddq_f32(d1, d2);
								float32x4_t ds3x4 = vaddq_f32(d3, d4);
								vst1q_f32(bpz, vaddq_f32(ds1x2, vaddq_f32(d0, ds3x4)));
								float32x4_t dn1x2 = vsubq_f32(d1, d2);
								float32x4_t dn3x4 = vsubq_f32(d3, d4);
								dn3x4 = vaddq_f32(dn3x4, dn3x4);
								vst1q_f32(bpz + binc[2], vaddq_f32(dn1x2, dn3x4));
								bpz += binc[1] * binc[2];
							} unroll_endfor
							break;
						case 3:
							unroll_for(dy, z[0], 4) {
								const float* const dz = d + dy * 24;
								float32x4_t d0 = vld1q_f32(dz);
								float32x4_t d1 = vld1q_f32(dz + 4);
								float32x4_t d2 = vld1q_f32(dz + 8);
								float32x4_t d3 = vld1q_f32(dz + 12);
								float32x4_t d4 = vld1q_f32(dz + 16);
								float32x4_t ds1x2 = vaddq_f32(d1, d2);
								float32x4_t ds3x4 = vaddq_f32(d3, d4);
								vst1q_f32(bpz, vaddq_f32(ds1x2, vaddq_f32(d0, ds3x4)));
								float32x4_t dn1x2 = vsubq_f32(d1, d2);
								float32x4_t dn3x4 = vsubq_f32(d3, d4);
								dn3x4 = vaddq_f32(dn3x4, dn3x4);
								vst1q_f32(bpz + binc[2], vaddq_f32(dn1x2, dn3x4));
								ds3x4 = vaddq_f32(ds3x4, ds3x4);
								ds3x4 = vaddq_f32(ds3x4, ds3x4);
								vst1q_f32(bpz + 2 * binc[2], vaddq_f32(ds1x2, ds3x4));
								bpz += binc[1] * binc[2];
							} unroll_endfor
							break;
						case 4:
							unroll_for(dy, z[0], 4) {
								const float* const dz = d + dy * 24;
								float32x4_t d0 = vld1q_f32(dz);
								float32x4_t d1 = vld1q_f32(dz + 4);
								float32x4_t d2 = vld1q_f32(dz + 8);
								float32x4_t d3 = vld1q_f32(dz + 12);
								float32x4_t d4 = vld1q_f32(dz + 16);
								float32x4_t ds1x2 = vaddq_f32(d1, d2);
								float32x4_t ds3x4 = vaddq_f32(d3, d4);
								vst1q_f32(bpz, vaddq_f32(ds1x2, vaddq_f32(d0, ds3x4)));
								float32x4_t dn1x2 = vsubq_f32(d1, d2);
								float32x4_t dn3x4 = vsubq_f32(d3, d4);
								dn3x4 = vaddq_f32(dn3x4, dn3x4);
								vst1q_f32(bpz + binc[2], vaddq_f32(dn1x2, dn3x4));
								ds3x4 = vaddq_f32(ds3x4, ds3x4);
								ds3x4 = vaddq_f32(ds3x4, ds3x4);
								vst1q_f32(bpz + 2 * binc[2], vaddq_f32(ds1x2, ds3x4));
								float32x4_t d5 = vld1q_f32(dz + 20);
								dn3x4 = vaddq_f32(dn3x4, dn3x4);
								dn3x4 = vaddq_f32(dn3x4, dn3x4);
								vst1q_f32(bpz + 3 * binc[2], vaddq_f32(vaddq_f32(dn1x2, d5), dn3x4));
								bpz += binc[1] * binc[2];
							} unroll_endfor
							break;
					};
				}
			}
		} parallel_endfor
	}
	ccfree(workmem);
	return CCV_NNC_EXEC_SUCCESS;
}
#endif

int _ccv_nnc_conv_forw_4x4_3x3_winograd_cpu_opt(const ccv_nnc_tensor_view_t* const a, const ccv_nnc_tensor_t* const w, const ccv_nnc_tensor_t* const bias, const ccv_nnc_hint_t hint, ccv_nnc_tensor_view_t* const b)
{
#if defined(HAVE_SSE2)
	if (w->info.dim[0] % 4 == 0)
		return _ccv_nnc_conv_forw_4x4_3x3_winograd_sse2(a, w, bias, hint, b);
#elif defined(HAVE_NEON)
	if (w->info.dim[0] % 4 == 0)
		return _ccv_nnc_conv_forw_4x4_3x3_winograd_neon(a, w, bias, hint, b);
#endif
	return _ccv_nnc_conv_forw_4x4_3x3_winograd_ref(a, w, bias, hint, b);
}
