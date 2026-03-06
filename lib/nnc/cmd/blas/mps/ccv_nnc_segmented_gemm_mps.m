#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#include <Foundation/Foundation.h>
#ifdef HAVE_MPS
#include "nnc/mps/ccv_nnc_mps.h"
#endif
#ifdef USE_OPENMP
#include <omp.h>
#endif
#ifdef USE_DISPATCH
#include <dispatch/dispatch.h>
#endif

static int _ccv_nnc_segmented_gemm_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 4);
	const ccv_nnc_tensor_view_t* a = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* indices = (const ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* counts = (const ccv_nnc_tensor_view_t*)inputs[2];
	const ccv_nnc_tensor_view_t* w = (const ccv_nnc_tensor_view_t*)inputs[3];
	const ccv_nnc_tensor_view_t* bias = input_size > 4 ? (const ccv_nnc_tensor_view_t*)inputs[4] : 0;
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(!bias || (bias->info.dim[1] == 0 || bias->info.dim[2] == 0 || bias->info.dim[3] == 0)); // It is a 1-d array
	int a_batch_size, a_rows, a_cols, a_batch_inc, a_rows_inc, a_cols_inc;
	int w_batch_size, w_rows, w_cols, w_batch_inc, w_rows_inc, w_cols_inc;
	int b_batch_size, b_rows, b_cols, b_batch_inc, b_rows_inc, b_cols_inc;
	const static int no_transpose[2] = {};
	ccv_nnc_tensor_get_matrix_params(a->info, CCV_IS_TENSOR_VIEW(a) ? a->stride : 0, a->info.dim, cmd.info.blas.transpose_a, &a_batch_size, &a_rows, &a_cols, &a_batch_inc, &a_rows_inc, &a_cols_inc);
	ccv_nnc_tensor_get_matrix_params(w->info, CCV_IS_TENSOR_VIEW(w) ? w->stride : 0, w->info.dim, cmd.info.blas.transpose_b, &w_batch_size, &w_rows, &w_cols, &w_batch_inc, &w_rows_inc, &w_cols_inc);
	ccv_nnc_tensor_get_matrix_params(b->info, CCV_IS_TENSOR_VIEW(b) ? b->stride : 0, b->info.dim, no_transpose, &b_batch_size, &b_rows, &b_cols, &b_batch_inc, &b_rows_inc, &b_cols_inc);
	assert(a_batch_size == 1); // Currently, a cannot be batched (no broadcast support too).
	assert(a_batch_size == b_batch_size);
	assert(a_rows == b_rows);
	assert(a_cols == w_rows);
	assert(w_cols == b_cols);
	int adim[CCV_NNC_MAX_DIM_ALLOC];
	int astride[CCV_NNC_MAX_DIM_ALLOC] = {0};
	memcpy(adim, a->info.dim, sizeof(adim));
	if (CCV_IS_TENSOR_VIEW(a))
		memcpy(astride, a->stride, sizeof(astride));
	assert(ccv_nnc_tensor_nd(w->info.dim) >= 2);
	const int is_transpose_a = ccv_nnc_is_matrix_transpose(a->info, cmd.info.blas.transpose_a);
	if (ccv_nnc_tensor_nd(adim) < 2)
	{
		if (is_transpose_a)
		{
			adim[1] = 1;
			astride[1] = astride[0];
		} else {
			adim[1] = adim[0];
			astride[1] = astride[0];
			adim[0] = 1;
			astride[0] = astride[1];
		}
	}
	int bstride[CCV_NNC_MAX_DIM_ALLOC] = {0};
	if (CCV_IS_TENSOR_VIEW(b))
		memcpy(bstride, b->stride, sizeof(bstride));
	const int is_transpose_w = ccv_nnc_is_matrix_transpose(w->info, cmd.info.blas.transpose_b);
	int biasdim[CCV_NNC_MAX_DIM_ALLOC] = {0};
	int biasstride[CCV_NNC_MAX_DIM_ALLOC] = {0};
	int bias_batch_size = 1;
	const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
	if (bias)
	{
		assert(CCV_GET_DATA_TYPE(bias->info.datatype) != CCV_QX);
		const int bias_nd = ccv_nnc_tensor_nd(bias->info.dim);
		// Align bias to this.
		assert(bias_nd <= 2 || bias_nd == b_nd);
		int i;
		if (bias_nd == b_nd)
		{
			memcpy(biasdim, bias->info.dim, sizeof(biasdim));
			if (CCV_IS_TENSOR_VIEW(bias))
				memcpy(biasstride, bias->stride, sizeof(biasstride));
			for (i = 0; i < bias_nd - 2; i++)
				bias_batch_size *= biasdim[i];
		} else if (bias_nd == 2) {
			biasdim[0] = bias->info.dim[0];
			for (i = 1; i < b_nd - 1; i++)
				biasdim[i] = 1;
			biasdim[b_nd - 1] = bias->info.dim[1];
			if (CCV_IS_TENSOR_VIEW(bias))
			{
				biasstride[0] = bias->stride[0];
				for (i = 1; i < b_nd - 1; i++)
					biasstride[i] = biasstride[0];
				biasstride[b_nd - 1] = bias->stride[1];
			}
			for (i = 0; i < bias_nd - 1; i++)
				bias_batch_size *= biasdim[i];
		} else {
			for (i = 0; i < b_nd - 1; i++)
				biasdim[i] = 1;
			biasdim[b_nd - 1] = bias->info.dim[0];
			if (CCV_IS_TENSOR_VIEW(bias))
			{
				for (i = 0; i < b_nd - 1; i++)
					biasstride[i] = bias->info.dim[0] * bias->stride[0];
				biasstride[b_nd - 1] = bias->stride[0];
			}
			for (i = 0; i < bias_nd - 1; i++)
				bias_batch_size *= biasdim[i];
		}
	}
	int* adim_r = adim;
	int* astride_r = astride;
	int* biasdim_r = biasdim;
	int* biasstride_r = biasstride;
	const int a_nd = ccv_nnc_tensor_nd(adim);
	const int w_nd = ccv_nnc_tensor_nd(w->info.dim);
	a_batch_size = a_nd < 3 ? 1 : adim[a_nd - 3];
	int i;
	for (i = 0; i < a_nd - 3; i++)
		a_batch_size *= adim[i];
	w_batch_size = w_nd < 3 ? 1 : w->info.dim[w_nd - 3];
	for (i = 0; i < w_nd - 3; i++)
		w_batch_size *= w->info.dim[i];
	b_batch_size = b_nd < 3 ? 1 : b->info.dim[b_nd - 3];
	for (i = 0; i < b_nd - 3; i++)
		b_batch_size *= b->info.dim[i];
	if (w_batch_size == 1 && b_batch_size > 1)
		w_batch_inc = 0;
	@autoreleasepool {
		// Fake the astride at a_nd - 3. For this one, we have flexibility to change fo kernel GEMM kernels.
		const int a_batch_stride = astride[a_nd - 3];
		// Only fake it if it is larger than the expected compact stride.
		if (a_batch_stride > astride[a_nd - 2] * adim[a_nd - 2])
			astride[a_nd - 3] = astride[a_nd - 2] * adim[a_nd - 2];
		const int b_batch_stride = bstride[b_nd - 3];
		// Only fake it if it is larger than the expected compact stride.
		if (b_batch_stride > bstride[b_nd - 2] * b->info.dim[b_nd - 2])
			bstride[b_nd - 3] = bstride[b_nd - 2] * b->info.dim[b_nd - 2];
		const int is_contiguous =
			(!CCV_IS_TENSOR_VIEW(a) || ccv_nnc_tensor_view_is_contiguous(adim, astride)) &&
			(!CCV_IS_TENSOR_VIEW(w) || ccv_nnc_tensor_view_is_contiguous(w->info.dim, w->stride)) &&
			(!CCV_IS_TENSOR_VIEW(b) || ccv_nnc_tensor_view_is_contiguous(b->info.dim, bstride)) &&
			(bias ? (!CCV_IS_TENSOR_VIEW(bias) || ccv_nnc_tensor_view_is_contiguous(bias->info.dim, bias->stride)) : 1);
		astride[a_nd - 3] = a_batch_stride;
		bstride[b_nd - 3] = b_batch_stride;

		const int a_datatype = CCV_GET_DATA_TYPE(a->info.datatype) == CCV_QX ? ((a->info.datatype & 0xff) << 12) : a->info.datatype;
		const int w_datatype = CCV_GET_DATA_TYPE(w->info.datatype) == CCV_QX ? ((w->info.datatype & 0xff) << 12) : w->info.datatype;
		const int is_same_dtype =
			(a_datatype == w_datatype) &&
			(a_datatype == b->info.datatype) &&
			(bias ? (a_datatype == bias->info.datatype) : 1);

		int is_supported_dtype = 0;
		uint32_t mtl_data_type = UINT32_MAX;
		switch (a_datatype) {
			case CCV_16F: {
				is_supported_dtype = 1;
				mtl_data_type = 16;
				break;
			}
			case CCV_16BF: {
				is_supported_dtype = 1;
				mtl_data_type = 121;
				break;
			}
			case CCV_32F: {
				is_supported_dtype = 1;
				mtl_data_type = 3;
				break;
			}
			default: {
				break;
			}
		}

		// NNC uses the convention B = A * W.
		// MFA uses the convention C = A * B.
		ccv_nnc_mfa_context_t* context = ccv_nnc_default_mfa_context();
		const int is_downcast = ((cmd.info.blas.flags & CCV_NNC_GEMM_16F) && a_datatype == CCV_16F);
		const int is_mfa_supported =
			ccv_nnc_mfa_context_supported(context) && is_contiguous && is_same_dtype && is_supported_dtype && !(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA);

		size_t a_data_size = 0;
		if (CCV_GET_DATA_TYPE(a->info.datatype) == CCV_QX)
		{
			ccv_nnc_tensor_param_t a_params = a->info;
			const int palette_datatype = (a_params.datatype & 0xff) << 12;
			ccv_nnc_tensor_param_t depalettize_a_params = a_params;
			depalettize_a_params.datatype = palette_datatype;
			depalettize_a_params.reserved = 0;
			a_data_size = ccv_nnc_tensor_data_size(depalettize_a_params);
		}
		size_t w_data_size = 0;
		if (CCV_GET_DATA_TYPE(w->info.datatype) == CCV_QX)
		{
			ccv_nnc_tensor_param_t w_params = w->info;
			const int palette_datatype = (w_params.datatype & 0xff) << 12;
			ccv_nnc_tensor_param_t depalettize_w_params = w_params;
			depalettize_w_params.datatype = palette_datatype;
			depalettize_w_params.reserved = 0;
			w_data_size = ccv_nnc_tensor_data_size(depalettize_w_params);
		}

		if (METAL_LOG_LEVEL(context) >= 3)
		{
			if (is_mfa_supported)
			{
				ccv_nnc_mfa_log_message("Compatible GEMM found.");
			} else {
				ccv_nnc_mfa_log_message("Incompatible GEMM found. Incompatible because:");
				if (!is_contiguous)
				{
					ccv_nnc_mfa_log_message("  Strided.");
				}
				if (!is_same_dtype)
				{
					ccv_nnc_mfa_log_message("  Mixed precision.");
				}
				if (!is_same_dtype)
				{
					ccv_nnc_mfa_log_message("  Unsupported data type.");
				}
			}
		}

		assert(is_mfa_supported);
		// On supported devices, use Metal directly.
		ccv_nnc_mfa_segmented_gemm_params_t params = {
			.data_type = mtl_data_type,
			.M = (uint32_t)(b_rows + ccv_max(w_batch_size - 2, 0)) / ccv_max(w_batch_size - 1, 1), // C_rows, this is estimated. We estimate it to be b rows / segments.
			.N = (uint32_t)b_cols, // C_cols
			.K = (uint32_t)w_rows, // B_rows
			.originalM = (uint32_t)b_rows, // C_rows
			.A_trans = (is_transpose_a ? 1 : 0),
			.B_trans = (is_transpose_w ? 1 : 0),
			.D_trans = 0,
			.fused_bias = (bias ? 1 : 0),
			.register_float = (is_downcast ? 0 : 1),
			.use_neural_accelerators = !(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS) && ccv_nnc_mfa_has_neural_accelerators(context) && (mtl_data_type != 121 || ccv_nnc_mfa_neural_accelerators_support_bfloat(context)),

			.segments = w_batch_size,
		};
		mtl_buffer_t* scratch = 0;
		const size_t scratch_offset = ccv_nnc_mfa_segmented_gemm_reserved_scratch_size(params);
		if (a_data_size + w_data_size > 0)
			scratch = ccv_nnc_mfa_request_scratch(context, scratch_offset + a_data_size + w_data_size);
		mtl_buffer_t* a_data = mpgetbuffer((ccv_nnc_tensor_t*)a);
		size_t a_dataof = (size_t)mpgetoffset((ccv_nnc_tensor_t*)a);
		ccv_nnc_mfa_depalettize_params_t a_depalettize_params;
		if (CCV_GET_DATA_TYPE(a->info.datatype) == CCV_QX)
		{
			ccv_nnc_tensor_param_t a_params = a->info;
			const size_t count = ccv_nnc_tensor_count(a_params);
			const int qbits = (a_params.datatype & 0xf00) >> 8;
			const int number_in_blocks = a_params.reserved;
			a_depalettize_params = (ccv_nnc_mfa_depalettize_params_t){
				.data_type = mtl_data_type,
				.qbits = (uint32_t)qbits,
				.number_in_blocks = (uint32_t)number_in_blocks,
				.length = (uint64_t)count,
			};
			ccv_nnc_mfa_prepare_depalettize(context, a_depalettize_params);
			a_data = scratch;
			a_dataof = scratch_offset;
		}
		mtl_buffer_t* w_data = mpgetbuffer((ccv_nnc_tensor_t*)w);
		size_t w_dataof = (size_t)mpgetoffset((ccv_nnc_tensor_t*)w);
		ccv_nnc_mfa_depalettize_params_t w_depalettize_params;
		if (CCV_GET_DATA_TYPE(w->info.datatype) == CCV_QX)
		{
			ccv_nnc_tensor_param_t w_params = w->info;
			const size_t count = ccv_nnc_tensor_count(w_params);
			const int qbits = (w_params.datatype & 0xf00) >> 8;
			const int number_in_blocks = w_params.reserved;
			w_depalettize_params = (ccv_nnc_mfa_depalettize_params_t){
				.data_type = mtl_data_type,
				.qbits = (uint32_t)qbits,
				.number_in_blocks = (uint32_t)number_in_blocks,
				.length = (uint64_t)count,
			};
			ccv_nnc_mfa_prepare_depalettize(context, w_depalettize_params);
			w_data = scratch;
			w_dataof = a_data_size + scratch_offset;
		}

		mtl_command_batch_t* command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
		if (CCV_GET_DATA_TYPE(a->info.datatype) == CCV_QX)
		{
			mtl_buffer_t* tensors[3] = {
				mpgetbuffer((ccv_nnc_tensor_t*)a), // A
				(mtl_buffer_t*)scratch, // B
				NULL,
			};
			size_t tensor_offsets[2] = {
				a->dataof, // A offset
				scratch_offset, // B offset
			};
			ccv_nnc_mfa_encode_depalettize(context, a_depalettize_params, command_batch, tensors, tensor_offsets);
		}
		if (CCV_GET_DATA_TYPE(w->info.datatype) == CCV_QX)
		{
			mtl_buffer_t* tensors[3] = {
				mpgetbuffer((ccv_nnc_tensor_t*)w), // A
				(mtl_buffer_t*)scratch, // B
				NULL,
			};
			size_t tensor_offsets[2] = {
				w->dataof, // A offset
				a_data_size + scratch_offset, // B offset
			};
			ccv_nnc_mfa_encode_depalettize(context, w_depalettize_params, command_batch, tensors, tensor_offsets);
		}
		mtl_buffer_t* bias_buffer = NULL;
		if (bias) {
			bias_buffer = mpgetbuffer((ccv_nnc_tensor_t*)bias);
		}
		mtl_buffer_t* tensors[7] = {
			a_data, // A
			mpgetbuffer((ccv_nnc_tensor_t*)indices),
			mpgetbuffer((ccv_nnc_tensor_t*)counts),
			w_data, // B
			mpgetbuffer((ccv_nnc_tensor_t*)b), // C
			bias_buffer, // D
			NULL,
		};
		size_t tensor_offsets[6] = {
			a_dataof, // A offset
			indices->dataof, // indices offset
			counts->dataof, // counts offset
			w_dataof, // B offset
			b->dataof, // C offset
			bias ? bias->dataof : 0, // D offset
		};
		ccv_nnc_mfa_encode_segmented_gemm(context, params, command_batch, tensors, tensor_offsets);
		ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_QX | CCV_32S | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_segmented_gemm_forw;
}
