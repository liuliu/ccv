#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#include <math.h>

static int _ccv_nnc_grid_sample_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 2);
	assert(output_size == 1);
	const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* const grid = (const ccv_nnc_tensor_view_t*)inputs[1];
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(a->info.datatype == CCV_32F);
	assert(grid->info.datatype == CCV_32F);
	assert(b->info.datatype == CCV_32F);
	assert(a->info.format == b->info.format);
	assert(a->info.format == CCV_TENSOR_FORMAT_NCHW || a->info.format == CCV_TENSOR_FORMAT_NHWC);
	assert(grid->info.format == CCV_TENSOR_FORMAT_NHWC);
	int adim[CCV_NNC_MAX_DIM_ALLOC];
	int bdim[CCV_NNC_MAX_DIM_ALLOC];
	int griddim[CCV_NNC_MAX_DIM_ALLOC];
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
	int gridstride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(a, adim);
	ccv_nnc_tensor_view_get_dim(b, bdim);
	ccv_nnc_tensor_view_get_dim(grid, griddim);
	ccv_nnc_tensor_view_get_stride(a, astride);
	ccv_nnc_tensor_view_get_stride(b, bstride);
	ccv_nnc_tensor_view_get_stride(grid, gridstride);

	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
	const int grid_nd = ccv_nnc_tensor_nd(grid->info.dim);
	assert(a_nd == 3 || a_nd == 4);
	assert(b_nd == 3 || b_nd == 4);
	assert(grid_nd == 3 || grid_nd == 4);

	const int ahw = ccv_nnc_tensor_hw(a->info, a_nd, CCV_NNC_MAX_DIM);
	const int bhw = ccv_nnc_tensor_hw(b->info, b_nd, CCV_NNC_MAX_DIM);
	const int ghw = ccv_nnc_tensor_hw(grid->info, grid_nd, CCV_NNC_MAX_DIM);
	assert(ahw >= 0);
	assert(bhw >= 0);
	assert(ghw >= 0);

	const int N = ccv_nnc_tensor_get_n(a->info);
	const int C = ccv_nnc_tensor_get_c(a->info);
	const int H_in = adim[ahw];
	const int W_in = adim[ahw + 1];
	const int N_out = ccv_nnc_tensor_get_n(b->info);
	const int C_out = ccv_nnc_tensor_get_c(b->info);
	const int H_out = bdim[bhw];
	const int W_out = bdim[bhw + 1];
	assert(N_out == N);
	assert(C_out == C);
	const int N_grid = ccv_nnc_tensor_get_n(grid->info);
	assert(griddim[ghw + 2] == 2);
	assert(N_grid == N);
	assert(griddim[ghw] == H_out);
	assert(griddim[ghw + 1] == W_out);

	const int align_corners = cmd.info.grid_sample.align_corners;
	const float* const ap = a->data.f32;
	const float* const gridp = grid->data.f32;
	float* const bp = b->data.f32;
	const int anstride = (a_nd == CCV_NNC_MAX_DIM + 2) ? astride[0] : 0;
	const int ahstride = astride[ahw];
	const int awstride = astride[ahw + 1];
	const int acstride = (a->info.format == CCV_TENSOR_FORMAT_NCHW) ? astride[ahw - 1] : astride[ahw + 2];
	const int bnstride = (b_nd == CCV_NNC_MAX_DIM + 2) ? bstride[0] : 0;
	const int bhstride = bstride[bhw];
	const int bwstride = bstride[bhw + 1];
	const int bcstride = (b->info.format == CCV_TENSOR_FORMAT_NCHW) ? bstride[bhw - 1] : bstride[bhw + 2];
	const int gnstride = (grid_nd == CCV_NNC_MAX_DIM + 2) ? gridstride[0] : 0;
	const int ghstride = gridstride[ghw];
	const int gwstride = gridstride[ghw + 1];
	const int gcstride = gridstride[ghw + 2];

	for (int n = 0; n < N; n++)
		for (int y = 0; y < H_out; y++)
			for (int x = 0; x < W_out; x++)
			{
				const int grid_offset = n * gnstride + y * ghstride + x * gwstride;
				const float gx = gridp[grid_offset];
				const float gy = gridp[grid_offset + gcstride];
				const float ix = align_corners ? (gx + 1) * (W_in - 1) * 0.5f : ((gx + 1) * W_in - 1) * 0.5f;
				const float iy = align_corners ? (gy + 1) * (H_in - 1) * 0.5f : ((gy + 1) * H_in - 1) * 0.5f;
				const int x0 = (int)floorf(ix);
				const int y0 = (int)floorf(iy);
				const int x1 = x0 + 1;
				const int y1 = y0 + 1;
				const float wx1 = ix - x0;
				const float wy1 = iy - y0;
				const float wx0 = 1.0f - wx1;
				const float wy0 = 1.0f - wy1;
				const int a_offset_nc = n * anstride;
				const int b_offset_nc = n * bnstride;
				for (int c = 0; c < C; c++)
				{
					float v00 = 0, v01 = 0, v10 = 0, v11 = 0;
					const int a_offset_ncc = a_offset_nc + c * acstride;
					if (y0 >= 0 && y0 < H_in && x0 >= 0 && x0 < W_in)
						v00 = ap[a_offset_ncc + y0 * ahstride + x0 * awstride];
					if (y0 >= 0 && y0 < H_in && x1 >= 0 && x1 < W_in)
						v01 = ap[a_offset_ncc + y0 * ahstride + x1 * awstride];
					if (y1 >= 0 && y1 < H_in && x0 >= 0 && x0 < W_in)
						v10 = ap[a_offset_ncc + y1 * ahstride + x0 * awstride];
					if (y1 >= 0 && y1 < H_in && x1 >= 0 && x1 < W_in)
						v11 = ap[a_offset_ncc + y1 * ahstride + x1 * awstride];
					const float v = v00 * wy0 * wx0 + v01 * wy0 * wx1 + v10 * wy1 * wx0 + v11 * wy1 * wx1;
					bp[b_offset_nc + c * bcstride + y * bhstride + x * bwstride] = v;
				}
			}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_GRID_SAMPLE_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_grid_sample_forw;
}
