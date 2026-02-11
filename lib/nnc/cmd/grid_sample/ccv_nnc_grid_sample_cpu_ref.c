#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#include <math.h>

static inline float _ccv_nnc_grid_sample_unnormalize(const float v, const int size, const int align_corners)
{
	if (align_corners)
		return (v + 1) * (size - 1) * 0.5f;
	return ((v + 1) * size - 1) * 0.5f;
}

static inline int _ccv_nnc_grid_sample_offset(const int n, const int c, const int y, const int x, const int n_axis, const int c_axis, const int y_axis, const int x_axis, const int stride[CCV_NNC_MAX_DIM_ALLOC])
{
	int offset = c * stride[c_axis] + y * stride[y_axis] + x * stride[x_axis];
	if (n_axis >= 0)
		offset += n * stride[n_axis];
	return offset;
}

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

	const int and = ccv_nnc_tensor_nd(a->info.dim);
	const int bnd = ccv_nnc_tensor_nd(b->info.dim);
	const int gnd = ccv_nnc_tensor_nd(grid->info.dim);

	const int an_axis = (and == 4) ? 0 : -1;
	const int bn_axis = (bnd == 4) ? 0 : -1;
	const int gn_axis = (gnd == 4) ? 0 : -1;

	int ac_axis = 0, ah_axis = 0, aw_axis = 0;
	int bc_axis = 0, bh_axis = 0, bw_axis = 0;
	if (a->info.format == CCV_TENSOR_FORMAT_NCHW)
	{
		assert(and == 3 || and == 4);
		assert(bnd == 3 || bnd == 4);
		ac_axis = (and == 4) ? 1 : 0;
		ah_axis = (and == 4) ? 2 : 1;
		aw_axis = (and == 4) ? 3 : 2;
		bc_axis = (bnd == 4) ? 1 : 0;
		bh_axis = (bnd == 4) ? 2 : 1;
		bw_axis = (bnd == 4) ? 3 : 2;
	} else {
		assert(and == 3 || and == 4);
		assert(bnd == 3 || bnd == 4);
		ah_axis = (and == 4) ? 1 : 0;
		aw_axis = (and == 4) ? 2 : 1;
		ac_axis = (and == 4) ? 3 : 2;
		bh_axis = (bnd == 4) ? 1 : 0;
		bw_axis = (bnd == 4) ? 2 : 1;
		bc_axis = (bnd == 4) ? 3 : 2;
	}

	const int N = (an_axis >= 0) ? adim[an_axis] : 1;
	const int C = adim[ac_axis];
	const int H_in = adim[ah_axis];
	const int W_in = adim[aw_axis];
	const int N_out = (bn_axis >= 0) ? bdim[bn_axis] : 1;
	const int C_out = bdim[bc_axis];
	const int H_out = bdim[bh_axis];
	const int W_out = bdim[bw_axis];
	assert(N_out == N);
	assert(C_out == C);

	assert(gnd == 3 || gnd == 4);
	const int gh_axis = (gnd == 4) ? 1 : 0;
	const int gw_axis = (gnd == 4) ? 2 : 1;
	const int gc_axis = (gnd == 4) ? 3 : 2;
	const int N_grid = (gn_axis >= 0) ? griddim[gn_axis] : 1;
	assert(griddim[gc_axis] == 2);
	assert(N_grid == N);
	assert(griddim[gh_axis] == H_out);
	assert(griddim[gw_axis] == W_out);

	const int align_corners = cmd.info.grid_sample.align_corners;
	const float* const ap = a->data.f32;
	const float* const gridp = grid->data.f32;
	float* const bp = b->data.f32;

	for (int n = 0; n < N; n++)
		for (int y = 0; y < H_out; y++)
			for (int x = 0; x < W_out; x++)
			{
				int grid_offset = y * gridstride[gh_axis] + x * gridstride[gw_axis];
				if (gn_axis >= 0)
					grid_offset += n * gridstride[gn_axis];
				const float gx = gridp[grid_offset];
				const float gy = gridp[grid_offset + gridstride[gc_axis]];
				const float ix = _ccv_nnc_grid_sample_unnormalize(gx, W_in, align_corners);
				const float iy = _ccv_nnc_grid_sample_unnormalize(gy, H_in, align_corners);
				const int x0 = (int)floorf(ix);
				const int y0 = (int)floorf(iy);
				const int x1 = x0 + 1;
				const int y1 = y0 + 1;
				const float wx1 = ix - x0;
				const float wy1 = iy - y0;
				const float wx0 = 1.0f - wx1;
				const float wy0 = 1.0f - wy1;
				for (int c = 0; c < C; c++)
				{
					float v00 = 0, v01 = 0, v10 = 0, v11 = 0;
					if (y0 >= 0 && y0 < H_in && x0 >= 0 && x0 < W_in)
					{
						const int offset = _ccv_nnc_grid_sample_offset(n, c, y0, x0, an_axis, ac_axis, ah_axis, aw_axis, astride);
						v00 = ap[offset];
					}
					if (y0 >= 0 && y0 < H_in && x1 >= 0 && x1 < W_in)
					{
						const int offset = _ccv_nnc_grid_sample_offset(n, c, y0, x1, an_axis, ac_axis, ah_axis, aw_axis, astride);
						v01 = ap[offset];
					}
					if (y1 >= 0 && y1 < H_in && x0 >= 0 && x0 < W_in)
					{
						const int offset = _ccv_nnc_grid_sample_offset(n, c, y1, x0, an_axis, ac_axis, ah_axis, aw_axis, astride);
						v10 = ap[offset];
					}
					if (y1 >= 0 && y1 < H_in && x1 >= 0 && x1 < W_in)
					{
						const int offset = _ccv_nnc_grid_sample_offset(n, c, y1, x1, an_axis, ac_axis, ah_axis, aw_axis, astride);
						v11 = ap[offset];
					}
					const float v = v00 * wy0 * wx0 + v01 * wy0 * wx1 + v10 * wy1 * wx0 + v11 * wy1 * wx1;
					const int boffset = _ccv_nnc_grid_sample_offset(n, c, y, x, bn_axis, bc_axis, bh_axis, bw_axis, bstride);
					bp[boffset] = v;
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
