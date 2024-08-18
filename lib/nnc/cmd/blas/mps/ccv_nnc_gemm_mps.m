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

static int _ccv_nnc_gemm_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 2);
	const ccv_nnc_tensor_view_t* a = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* w = (const ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* bias = input_size > 2 ? (const ccv_nnc_tensor_view_t*)inputs[2] : 0;
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
	assert(a_rows == b_rows);
	assert(a_cols == w_rows);
	assert(w_cols == b_cols);
	int adim[CCV_NNC_MAX_DIM_ALLOC];
	int astride[CCV_NNC_MAX_DIM_ALLOC];
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
	if (a_batch_size == 1 && b_batch_size > 1)
		a_batch_inc = 0;
	if (w_batch_size == 1 && b_batch_size > 1)
		w_batch_inc = 0;
	@autoreleasepool {
		// Fake the astride at a_nd - 3. For this one, we have flexibility to change fo v2 GEMM kernels.
		const int astride_a_nd_3 = astride[a_nd - 3];
		// Only fake it if it is larger than the expected compact stride.
		if (astride_a_nd_3 > astride[a_nd - 2] * adim[a_nd - 2])
			astride[a_nd - 3] = astride[a_nd - 2] * adim[a_nd - 2];
		const int is_contiguous =
			(!CCV_IS_TENSOR_VIEW(a) || ccv_nnc_tensor_view_is_contiguous(adim, astride)) &&
			(!CCV_IS_TENSOR_VIEW(w) || ccv_nnc_tensor_view_is_contiguous(w->info.dim, w->stride)) &&
			(!CCV_IS_TENSOR_VIEW(b) || ccv_nnc_tensor_view_is_contiguous(b->info.dim, b->stride)) &&
			(bias ? (!CCV_IS_TENSOR_VIEW(bias) || ccv_nnc_tensor_view_is_contiguous(bias->info.dim, bias->stride)) : 1);
		astride[a_nd - 3] = astride_a_nd_3;

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
			case CCV_32F: {
				is_supported_dtype = 1;
				mtl_data_type = 3;
				break;
			}
			default: {
				break;
			}
		}

		const int is_same_batch =
			(a_batch_size == w_batch_size) &&
			(a_batch_size == b_batch_size);

		// NNC uses the convention B = A * W.
		// MFA uses the convention C = A * B.
		int is_batched = 0;
		int is_mfa_compatible_batch = 0;
		int A_batch_size = a_batch_size;
		int B_batch_size = w_batch_size;
		int C_batch_size = b_batch_size;
		if (A_batch_size == 1 && B_batch_size == 1 && C_batch_size == 1) {
			// Not batched.
		} else if (A_batch_size <= 0 || B_batch_size <= 0 || C_batch_size <= 0) {
			// Invalid batch size.
		} else {
			// This does not check whether the D batch size matches the others. If it
			// does not match, it will crash when encoding the GEMM command.
			is_batched = 1;
			if (A_batch_size == C_batch_size) {
				if (A_batch_size == B_batch_size) {
					is_mfa_compatible_batch = 1;
				} else if (B_batch_size == 1) {
					is_mfa_compatible_batch = 1;
				}
			}
		}

		ccv_nnc_mfa_context_t* context = ccv_nnc_default_mfa_context();
		const int is_mfa_gemv = !is_batched && ((a_rows == 1 && is_transpose_w && (w_rows % 4) == 0) || (!is_transpose_a && w_cols == 1 && (a_cols % 4) == 0));
		int is_upcast = ((cmd.info.blas.flags & CCV_NNC_GEMM_32F) && a_datatype == CCV_16F);
		const int is_mfa_supported =
			ccv_nnc_mfa_context_supported(context) && is_contiguous && is_same_dtype && is_supported_dtype && (!is_batched || is_mfa_compatible_batch) && !(ccv_nnc_flags() & CCV_NNC_DISABLE_METAL_FLASH_ATTENTION) && (is_mfa_gemv || !(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA_GEMM));

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
				if (is_batched && !is_mfa_compatible_batch)
				{
					ccv_nnc_mfa_log_message("  Unsupported batch.");
				}
			}
		}

		if (is_mfa_supported)
		{
			mtl_buffer_t* scratch = 0;
			if (a_data_size + w_data_size > 0)
				scratch = ccv_nnc_mfa_request_scratch(context, a_data_size + w_data_size);
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
				a_dataof = 0;
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
				w_dataof = a_data_size;
			}
			if (is_mfa_gemv)
			{
				// This is GEMV, use GEMV kernel.
				ccv_nnc_mfa_gemv_params_t params;
				if (a_rows == 1 && is_transpose_w)
				{
					params = (ccv_nnc_mfa_gemv_params_t){
						.data_type = mtl_data_type,
						.ncols = w_rows,
						.nrows = w_cols,
						.fused_bias = bias ? 1 : 0,
					};
				} else {
					params = (ccv_nnc_mfa_gemv_params_t){
						.data_type = mtl_data_type,
						.ncols = a_cols,
						.nrows = a_rows,
						.fused_bias = bias ? 1 : 0,
					};
				}
				ccv_nnc_mfa_prepare_gemv(context, params);

				// Creating a new command buffer has a >10 µs penalty CPU-side. Still
				// faster the >50 µs penalty for MPSGraph (probably why
				// MPSMatrixMultiplication is faster for GEMM).
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
						0, // B offset
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
						a_data_size, // B offset
					};
					ccv_nnc_mfa_encode_depalettize(context, w_depalettize_params, command_batch, tensors, tensor_offsets);
				}
				mtl_buffer_t* bias_buffer = NULL;
				if (bias) {
					bias_buffer = mpgetbuffer((ccv_nnc_tensor_t*)bias);
				}
				mtl_buffer_t* tensors[5] = {
					NULL,
					NULL,
					mpgetbuffer((ccv_nnc_tensor_t*)b), // C
					bias_buffer, // D
					NULL,
				};
				size_t tensor_offsets[4] = {
					0,
					0,
					b->dataof, // C offset
					bias ? bias->dataof : 0, // D offset
				};
				if (a_rows == 1 && is_transpose_w)
				{
					tensors[0] = w_data;
					tensors[1] = a_data;
					tensor_offsets[0] = w_dataof;
					tensor_offsets[1] = a_dataof;
				} else {
					tensors[0] = a_data;
					tensors[1] = w_data;
					tensor_offsets[0] = a_dataof;
					tensor_offsets[1] = w_dataof;
				}
				ccv_nnc_mfa_encode_gemv(context, params, command_batch, tensors, tensor_offsets);
				ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
				return CCV_NNC_EXEC_SUCCESS;
			}
			// On supported devices, use Metal directly.
			ccv_nnc_mfa_gemm_params_t params = {
				.data_type = mtl_data_type,
				.M = (uint32_t)b_rows, // C_rows
				.N = (uint32_t)b_cols, // C_cols
				.K = (uint32_t)w_rows, // B_rows
				.A_trans = (is_transpose_a ? 1 : 0),
				.B_trans = (is_transpose_w ? 1 : 0),
				.D_trans = 0,
				.fused_bias = (bias ? 1 : 0),
				.register_float = (is_upcast ? 1 : 0),

				.batch_dimension = b_batch_size,
				.batch_stride_a = a_batch_size > 1 ? ccv_max(astride_a_nd_3, b_rows * w_rows) : 0,
				.batch_stride_b = w_batch_size > 1 ? b_cols * w_rows : 0,
				.batch_stride_c = b_batch_size > 1 ? b_rows * b_cols : 0,
				.batch_stride_d = bias_batch_size > 1 ? b_cols : 0,
			};
			ccv_nnc_mfa_prepare_gemm(context, params);

			// Creating a new command buffer has a >10 µs penalty CPU-side. Still
			// faster the >50 µs penalty for MPSGraph (probably why
			// MPSMatrixMultiplication is faster for GEMM).
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
					0, // B offset
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
					a_data_size, // B offset
				};
				ccv_nnc_mfa_encode_depalettize(context, w_depalettize_params, command_batch, tensors, tensor_offsets);
			}
			mtl_buffer_t* bias_buffer = NULL;
			if (bias) {
				bias_buffer = mpgetbuffer((ccv_nnc_tensor_t*)bias);
			}
			mtl_buffer_t* tensors[5] = {
				a_data, // A
				w_data, // B
				mpgetbuffer((ccv_nnc_tensor_t*)b), // C
				bias_buffer, // D
				NULL,
			};
			size_t tensor_offsets[4] = {
				a_dataof, // A offset
				w_dataof, // B offset
				b->dataof, // C offset
				bias ? bias->dataof : 0, // D offset
			};
			ccv_nnc_mfa_encode_gemm(context, params, command_batch, tensors, tensor_offsets);
			ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
		} else {
			mtl_buffer_t* a_data = mpgetbuffer((ccv_nnc_tensor_t*)a);
			size_t a_dataof = (size_t)mpgetoffset((ccv_nnc_tensor_t*)a);
			mtl_buffer_t* w_data = mpgetbuffer((ccv_nnc_tensor_t*)w);
			size_t w_dataof = (size_t)mpgetoffset((ccv_nnc_tensor_t*)w);
			MPSCommandBuffer* command_buffer;
			if (CCV_GET_DATA_TYPE(a->info.datatype) == CCV_QX || CCV_GET_DATA_TYPE(w->info.datatype) == CCV_QX)
			{
				mtl_buffer_t* scratch = 0;
				if (a_data_size + w_data_size > 0)
					scratch = ccv_nnc_mfa_request_scratch(context, a_data_size + w_data_size);
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
					a_dataof = 0;
				}
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
					w_dataof = a_data_size;
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
						0, // B offset
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
						a_data_size, // B offset
					};
					ccv_nnc_mfa_encode_depalettize(context, w_depalettize_params, command_batch, tensors, tensor_offsets);
				}
				command_buffer = ccv_nnc_stream_context_finish_command_batch_encoding_and_return_mps_command_buffer(stream_context, command_batch);
			} else // Otherwise, incur the ~10-50 microsecond latency of MPS.
				command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);

			// If all conditions are met, use MPSMatrixMultiplication. Note that the bias only supported for Float32 and this has to be added because MPSGraph on Float32 won't do the computation properly on Intel.
			if (is_contiguous && is_same_dtype && is_same_batch && !(ccv_nnc_flags() & CCV_NNC_DISABLE_MIXED_MPS_GEMM) && (!bias || bias->info.datatype == CCV_32F))
			{
				id<MTLBuffer> a_buffer = (id<MTLBuffer>)a_data;
				MPSMatrix* leftMatrix = [[MPSMatrix alloc] initWithBuffer:a_buffer offset:a_dataof descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:(is_transpose_a ? a_cols : a_rows) columns:(is_transpose_a ? a_rows : a_cols) matrices:b_batch_size rowBytes:CCV_GET_DATA_TYPE_SIZE(a_datatype) * (is_transpose_a ? a_cols_inc : a_rows_inc) matrixBytes:CCV_GET_DATA_TYPE_SIZE(a_datatype) * a_batch_inc dataType:ccv_nnc_mps_datatype(a->info.datatype)]];
				id<MTLBuffer> w_buffer = (id<MTLBuffer>)w_data;
				MPSMatrix* rightMatrix = [[MPSMatrix alloc] initWithBuffer:w_buffer offset:w_dataof descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:(is_transpose_w ? w_cols : w_rows) columns:(is_transpose_w ? w_rows : w_cols) matrices:b_batch_size rowBytes:CCV_GET_DATA_TYPE_SIZE(w_datatype) * (is_transpose_w ? w_cols_inc : w_rows_inc) matrixBytes:CCV_GET_DATA_TYPE_SIZE(w_datatype) * w_batch_inc dataType:ccv_nnc_mps_datatype(w->info.datatype)]];
				id<MTLBuffer> b_buffer = mpgetbuffer((ccv_nnc_tensor_t*)b);
				MPSMatrix* resultMatrix = [[MPSMatrix alloc] initWithBuffer:b_buffer offset:b->dataof descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:b_rows columns:b_cols matrices:b_batch_size rowBytes:CCV_GET_DATA_TYPE_SIZE(b->info.datatype) * b_rows_inc matrixBytes:CCV_GET_DATA_TYPE_SIZE(b->info.datatype) * b_batch_inc dataType:ccv_nnc_mps_datatype(b->info.datatype)]];
				MPSMatrixMultiplication* matrixMultiplication = [[MPSMatrixMultiplication alloc] initWithDevice:ccv_nnc_default_device() transposeLeft:(is_transpose_a ? YES : NO) transposeRight:(is_transpose_w ? YES : NO) resultRows:b_rows resultColumns:b_cols interiorColumns:a_cols alpha:1 beta:0];
				[leftMatrix synchronizeOnCommandBuffer:command_buffer];
				[rightMatrix synchronizeOnCommandBuffer:command_buffer];
				[matrixMultiplication encodeToCommandBuffer:command_buffer leftMatrix:leftMatrix rightMatrix:rightMatrix resultMatrix:resultMatrix];
				[resultMatrix synchronizeOnCommandBuffer:command_buffer];
				[matrixMultiplication release];
				[leftMatrix release];
				[rightMatrix release];
				if (bias)
				{
					id<MTLBuffer> bias_buffer = mpgetbuffer((ccv_nnc_tensor_t*)bias);
					size_t bias_dataof = (size_t)mpgetoffset((ccv_nnc_tensor_t*)bias);
					MPSVector* biasVector = [[MPSVector alloc] initWithBuffer:bias_buffer offset:bias_dataof descriptor:[MPSVectorDescriptor vectorDescriptorWithLength:ccv_nnc_tensor_count(bias->info) dataType:ccv_nnc_mps_datatype(bias->info.datatype)]];
					[biasVector synchronizeOnCommandBuffer:command_buffer];
					MPSMatrixNeuron* biasAdd = [[MPSMatrixNeuron alloc] initWithDevice:ccv_nnc_default_device()];
					[biasAdd encodeToCommandBuffer:command_buffer inputMatrix:resultMatrix biasVector:biasVector resultMatrix:resultMatrix];
					[biasVector synchronizeOnCommandBuffer:command_buffer];
					[biasAdd release];
					[biasVector release];
				}
				[resultMatrix release];
			} else {
				// Otherwise, use MPSGraph.
				ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
				// Key will be consumed by the next method, therefore, no need to free.
				int indices[3];
				MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
					MPSGraphTensor* mps_input_a;
					MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, adim_r, astride_r, &mps_input_a);
					MPSGraphTensor* mps_input_w;
					MPSGraphTensor* mps_w = ccv_nnc_mps_graph_tensor_input(graph, w, w->info.dim, w->stride, &mps_input_w);
					MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, adim_r, astride_r);
					MPSGraphShapedType* mps_w_shape = ccv_nnc_mps_graph_tensor_input_shape(w, w->info.dim, w->stride);
					if (is_transpose_a)
						mps_a = [graph transposeTensor:mps_a dimension:-2 withDimension:-1 name:nil];
					if (is_transpose_w)
						mps_w = [graph transposeTensor:mps_w dimension:-2 withDimension:-1 name:nil];
					MPSGraphTensor* mps_b = [graph matrixMultiplicationWithPrimaryTensor:mps_a secondaryTensor:mps_w name:nil];
					[inputTensors addObject:mps_input_a];
					[inputShapedTypes addObject:mps_a_shape];
					[inputTensors addObject:mps_input_w];
					[inputShapedTypes addObject:mps_w_shape];
					if (bias)
					{
						MPSGraphTensor* mps_input_bias;
						MPSGraphTensor* mps_bias = ccv_nnc_mps_graph_tensor_input(graph, bias, biasdim_r, biasstride_r, &mps_input_bias);
						MPSGraphShapedType* mps_bias_shape = ccv_nnc_mps_graph_tensor_input_shape(bias, biasdim_r, biasstride_r);
						// Add support broadcast directly.
						mps_b = [graph additionWithPrimaryTensor:mps_b secondaryTensor:mps_bias name:nil];
						[inputTensors addObject:mps_input_bias];
						[inputShapedTypes addObject:mps_bias_shape];
					}
					[resultTensors addObject:mps_b];
				});
				MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data_with_buffer(a, adim, astride, a_data, a_dataof);
				MPSGraphTensorData* data_w = ccv_nnc_mps_graph_tensor_data_with_buffer(w, w->info.dim, w->stride, w_data, w_dataof);
				if (bias)
				{
					MPSGraphTensorData* data_bias = ccv_nnc_mps_graph_tensor_data(bias, biasdim, biasstride);
					MPSGraphTensorData* data[] = {data_a, data_w, data_bias};
					ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]], data[indices[2]]], &b, (int*[]){ b->info.dim }, (int*[]){ b->stride }, 1, 0);
				} else {
					MPSGraphTensorData* data[] = {data_a, data_w};
					ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]]], &b, (int*[]){ b->info.dim }, (int*[]){ b->stride }, 1, 0);
				}
			}
			ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_gemm_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	// inputs: gradient g, forw prop input a, [w]
	// outputs: output gradient h, weight updates dw, bias updates bias
	assert(input_size >= 2 && output_size >= 1);
	const ccv_nnc_tensor_view_t* g = (const ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* dw = output_size > 1 ? (ccv_nnc_tensor_view_t*)outputs[1] : 0;
	ccv_nnc_tensor_view_t* bias = output_size > 2 ? (ccv_nnc_tensor_view_t*)outputs[2] : 0;

	const ccv_nnc_tensor_view_t* a = input_size > 1 ? (const ccv_nnc_tensor_view_t*)inputs[1] : 0;
	ccv_nnc_tensor_view_t* h = (ccv_nnc_tensor_view_t*)outputs[0];
	const ccv_nnc_tensor_view_t* w = input_size > 2 ? (const ccv_nnc_tensor_view_t*)inputs[2] : 0;

	assert(!bias || (bias->info.dim[1] == 0 || bias->info.dim[2] == 0 || bias->info.dim[3] == 0)); // // It is a 2-d or 3-d array
	const int is_transpose_a = a || h ? ccv_nnc_is_matrix_transpose(a ? a->info : h->info, cmd.info.blas.transpose_a) : 0;
	const int is_transpose_w = w || dw ? ccv_nnc_is_matrix_transpose(w ? w->info : dw->info, cmd.info.blas.transpose_b) : 0;
	const int a_datatype = a ? (CCV_GET_DATA_TYPE(a->info.datatype) == CCV_QX ? ((a->info.datatype & 0xff) << 12) : a->info.datatype) : 0;
	const int w_datatype = w ? (CCV_GET_DATA_TYPE(w->info.datatype) == CCV_QX ? ((w->info.datatype & 0xff) << 12) : w->info.datatype) : 0;

	@autoreleasepool {
		const int is_contiguous =
			(!CCV_IS_TENSOR_VIEW(g) || ccv_nnc_tensor_view_is_contiguous(g->info.dim, g->stride)) &&
			(w ? (!CCV_IS_TENSOR_VIEW(w) || ccv_nnc_tensor_view_is_contiguous(w->info.dim, w->stride)) : 1) &&
			(dw ? (!CCV_IS_TENSOR_VIEW(dw) || ccv_nnc_tensor_view_is_contiguous(dw->info.dim, dw->stride)) : 1) &&
			(a ? (!CCV_IS_TENSOR_VIEW(a) || ccv_nnc_tensor_view_is_contiguous(a->info.dim, a->stride)) : 1) &&
			(h ? (!CCV_IS_TENSOR_VIEW(h) || ccv_nnc_tensor_view_is_contiguous(h->info.dim, h->stride)) : 1);

		const int is_same_dtype =
			(w ? (g->info.datatype == w_datatype) : 1) &&
			(dw ? (g->info.datatype == dw->info.datatype) : 1) &&
			(a ? (g->info.datatype == a_datatype) : 1) &&
			(h ? (g->info.datatype == h->info.datatype) : 1);

		int is_supported_dtype = 0;
		uint32_t mtl_data_type = UINT32_MAX;
		switch (g->info.datatype) {
			case CCV_16F: {
				is_supported_dtype = 1;
				mtl_data_type = 16;
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
		int i;
		int w_batch_size, w_rows, w_cols, w_batch_inc, w_rows_inc, w_cols_inc;
		int dw_batch_size, dw_rows, dw_cols, dw_batch_inc, dw_rows_inc, dw_cols_inc;
		int g_batch_size, g_rows, g_cols, g_batch_inc, g_rows_inc, g_cols_inc;
		const static int no_transpose[2] = {};
		if (w)
			ccv_nnc_tensor_get_matrix_params(w->info, CCV_IS_TENSOR_VIEW(w) ? w->stride : 0, w->info.dim, cmd.info.blas.transpose_b, &w_batch_size, &w_rows, &w_cols, &w_batch_inc, &w_rows_inc, &w_cols_inc);
		if (dw)
			ccv_nnc_tensor_get_matrix_params(dw->info, CCV_IS_TENSOR_VIEW(dw) ? dw->stride : 0, dw->info.dim, cmd.info.blas.transpose_b, &dw_batch_size, &dw_rows, &dw_cols, &dw_batch_inc, &dw_rows_inc, &dw_cols_inc);
		ccv_nnc_tensor_get_matrix_params(g->info, CCV_IS_TENSOR_VIEW(g) ? g->stride : 0, g->info.dim, no_transpose, &g_batch_size, &g_rows, &g_cols, &g_batch_inc, &g_rows_inc, &g_cols_inc);
		const int g_nd = ccv_nnc_tensor_nd(g->info.dim);
		g_batch_size = g_nd < 3 ? 1 : g->info.dim[g_nd - 3];
		for (i = 0; i < g_nd - 3; i++)
			g_batch_size *= g->info.dim[i];
		const int w_nd = w ? ccv_nnc_tensor_nd(w->info.dim) : 1;
		w_batch_size = w_nd < 3 ? 1 : w->info.dim[w_nd - 3];
		for (i = 0; i < w_nd - 3; i++)
			w_batch_size *= w->info.dim[i];
		const int dw_nd = dw ? ccv_nnc_tensor_nd(dw->info.dim) : 1;
		dw_batch_size = dw_nd < 3 ? 1 : dw->info.dim[dw_nd - 3];
		for (i = 0; i < dw_nd - 3; i++)
			dw_batch_size *= dw->info.dim[i];
		const int a_nd = a ? ccv_nnc_tensor_nd(a->info.dim) : 1;
		int a_batch_size = a_nd < 3 ? 1 : a->info.dim[a_nd - 3];
		for (i = 0; i < a_nd - 3; i++)
			a_batch_size *= a->info.dim[i];
		const int h_nd = h ? ccv_nnc_tensor_nd(h->info.dim) : 1;
		int h_batch_size = h_nd < 3 ? 1 : h->info.dim[h_nd - 3];
		for (i = 0; i < h_nd - 3; i++)
			h_batch_size *= h->info.dim[i];

		const int is_same_batch =
			(w ? g_batch_size == w_batch_size : 1) &&
			(a ? g_batch_size == a_batch_size : 1) &&
			(dw ? g_batch_size == dw_batch_size : 1) &&
			(h ? g_batch_size == h_batch_size : 1);

		// NNC uses the convention B = A * W.
		// MFA uses the convention C = A * B.

		ccv_nnc_mfa_context_t* context = ccv_nnc_default_mfa_context();
		const int is_mfa_supported =
			ccv_nnc_mfa_context_supported(context) && is_contiguous && is_same_dtype && is_supported_dtype && is_same_batch && !bias && !(ccv_nnc_flags() & CCV_NNC_DISABLE_METAL_FLASH_ATTENTION) && !(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA_GEMM);

		size_t a_data_size = 0;
		if (a && dw && CCV_GET_DATA_TYPE(a->info.datatype) == CCV_QX)
		{
			ccv_nnc_tensor_param_t a_params = a->info;
			const int palette_datatype = (a_params.datatype & 0xff) << 12;
			ccv_nnc_tensor_param_t depalettize_a_params = a_params;
			depalettize_a_params.datatype = palette_datatype;
			depalettize_a_params.reserved = 0;
			a_data_size = ccv_nnc_tensor_data_size(depalettize_a_params);
		}
		size_t w_data_size = 0;
		if (w && h && CCV_GET_DATA_TYPE(w->info.datatype) == CCV_QX)
		{
			ccv_nnc_tensor_param_t w_params = w->info;
			const int palette_datatype = (w_params.datatype & 0xff) << 12;
			ccv_nnc_tensor_param_t depalettize_w_params = w_params;
			depalettize_w_params.datatype = palette_datatype;
			depalettize_w_params.reserved = 0;
			w_data_size = ccv_nnc_tensor_data_size(depalettize_w_params);
		}
		if (is_mfa_supported)
		{
			mtl_buffer_t* scratch = 0;
			if (a_data_size + w_data_size > 0)
				scratch = ccv_nnc_mfa_request_scratch(context, a_data_size + w_data_size);
			mtl_buffer_t* a_data = 0;
			size_t a_dataof = 0;
			ccv_nnc_mfa_depalettize_params_t a_depalettize_params;
			if (a && dw)
			{
				a_data = mpgetbuffer((ccv_nnc_tensor_t*)a);
				a_dataof = (size_t)mpgetoffset((ccv_nnc_tensor_t*)a);
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
					a_dataof = 0;
				}
			}
			mtl_buffer_t* w_data = 0;
			size_t w_dataof = 0;
			ccv_nnc_mfa_depalettize_params_t w_depalettize_params;
			if (w && h)
			{
				w_data = mpgetbuffer((ccv_nnc_tensor_t*)w);
				w_dataof = (size_t)mpgetoffset((ccv_nnc_tensor_t*)w);
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
					w_dataof = a_data_size;
				}
			}
			ccv_nnc_mfa_gemm_params_t h_params;
			// On supported devices, use Metal directly.
			if (h)
			{
				if (is_transpose_a)
				{
					ccv_nnc_mfa_gemm_params_t params = {
						.data_type = mtl_data_type,
						.M = (uint32_t)w_rows, // C_rows
						.N = (uint32_t)g_rows, // C_cols
						.K = (uint32_t)w_cols, // B_rows
						.A_trans = 1,
						.B_trans = (is_transpose_w ? 1 : 0),
						.D_trans = 0,
						.fused_bias = 0,

						.batch_dimension = g_batch_size,
						.batch_stride_a = w_batch_size > 1 ? w_rows * w_cols : 0,
						.batch_stride_b = g_batch_size > 1 ? g_rows * w_cols : 0,
						.batch_stride_c = h_batch_size > 1 ? w_rows * g_rows : 0,
						.batch_stride_d = 0,
					};
					ccv_nnc_mfa_prepare_gemm(context, params);
					h_params = params;
				} else {
					ccv_nnc_mfa_gemm_params_t params = {
						.data_type = mtl_data_type,
						.M = (uint32_t)g_rows, // C_rows
						.N = (uint32_t)w_rows, // C_cols
						.K = (uint32_t)w_cols, // B_rows
						.A_trans = 0,
						.B_trans = (is_transpose_w ? 0 : 1),
						.D_trans = 0,
						.fused_bias = 0,

						.batch_dimension = g_batch_size,
						.batch_stride_a = g_batch_size > 1 ? g_rows * w_cols : 0,
						.batch_stride_b = w_batch_size > 1 ? w_rows * w_cols : 0,
						.batch_stride_c = h_batch_size > 1 ? g_rows * w_rows : 0,
						.batch_stride_d = 0,
					};
					ccv_nnc_mfa_prepare_gemm(context, params);
					h_params = params;
				}
			}

			ccv_nnc_mfa_gemm_params_t dw_params;
			// On supported devices, use Metal directly.
			if (dw)
			{
				if (is_transpose_w)
				{
					ccv_nnc_mfa_gemm_params_t params = {
						.data_type = mtl_data_type,
						.M = (uint32_t)dw_cols, // C_rows
						.N = (uint32_t)dw_rows, // C_cols
						.K = (uint32_t)g_rows, // B_rows
						.A_trans = 1,
						.B_trans = (is_transpose_a ? 1 : 0),
						.D_trans = 0,
						.fused_bias = 0,

						.batch_dimension = g_batch_size,
						.batch_stride_a = g_batch_size > 1 ? dw_cols * g_rows : 0,
						.batch_stride_b = a_batch_size > 1 ? dw_rows * g_rows : 0,
						.batch_stride_c = dw_batch_size > 1 ? dw_cols * dw_rows : 0,
						.batch_stride_d = 0,
					};
					ccv_nnc_mfa_prepare_gemm(context, params);
					dw_params = params;
				} else {
					ccv_nnc_mfa_gemm_params_t params = {
						.data_type = mtl_data_type,
						.M = (uint32_t)dw_rows, // C_rows
						.N = (uint32_t)dw_cols, // C_cols
						.K = (uint32_t)g_rows, // B_rows
						.A_trans = (is_transpose_a ? 0 : 1),
						.B_trans = 0,
						.D_trans = 0,
						.fused_bias = 0,

						.batch_dimension = g_batch_size,
						.batch_stride_a = a_batch_size > 1 ? dw_rows * g_rows : 0,
						.batch_stride_b = g_batch_size > 1 ? dw_cols * g_rows : 0,
						.batch_stride_c = dw_batch_size > 1 ? dw_rows * dw_cols : 0,
						.batch_stride_d = 0,
					};
					ccv_nnc_mfa_prepare_gemm(context, params);
					dw_params = params;
				}
			}

			// Creating a new command buffer has a >10 µs penalty CPU-side. Still
			// faster the >50 µs penalty for MPSGraph (probably why
			// MPSMatrixMultiplication is faster for GEMM).
			mtl_command_batch_t* command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);

			if (h)
			{
				if (CCV_GET_DATA_TYPE(w->info.datatype) == CCV_QX)
				{
					mtl_buffer_t* tensors[3] = {
						mpgetbuffer((ccv_nnc_tensor_t*)w), // A
						(mtl_buffer_t*)scratch, // B
						NULL,
					};
					size_t tensor_offsets[2] = {
						w->dataof, // A offset
						a_data_size, // B offset
					};
					ccv_nnc_mfa_encode_depalettize(context, w_depalettize_params, command_batch, tensors, tensor_offsets);
				}
				if (is_transpose_a)
				{
					mtl_buffer_t* tensors[4] = {
						w_data, // A
						mpgetbuffer((ccv_nnc_tensor_t*)g), // B
						mpgetbuffer((ccv_nnc_tensor_t*)h), // C
						NULL,
					};
					size_t tensor_offsets[4] = {
						w_dataof, // A offset
						g->dataof, // B offset
						h->dataof, // C offset
						0, // D offset
					};
					ccv_nnc_mfa_encode_gemm(context, h_params, command_batch, tensors, tensor_offsets);
				} else {
					mtl_buffer_t* tensors[4] = {
						mpgetbuffer((ccv_nnc_tensor_t*)g), // A
						w_data, // B
						mpgetbuffer((ccv_nnc_tensor_t*)h), // C
						NULL,
					};
					size_t tensor_offsets[4] = {
						g->dataof, // A offset
						w_dataof, // B offset
						h->dataof, // C offset
						0, // D offset
					};
					ccv_nnc_mfa_encode_gemm(context, h_params, command_batch, tensors, tensor_offsets);
				}
			}
			if (dw)
			{
				if (CCV_GET_DATA_TYPE(a->info.datatype) == CCV_QX)
				{
					mtl_buffer_t* tensors[3] = {
						mpgetbuffer((ccv_nnc_tensor_t*)a), // A
						(mtl_buffer_t*)scratch, // B
						NULL,
					};
					size_t tensor_offsets[2] = {
						a->dataof, // A offset
						0, // B offset
					};
					ccv_nnc_mfa_encode_depalettize(context, a_depalettize_params, command_batch, tensors, tensor_offsets);
				}
				if (is_transpose_w)
				{
					mtl_buffer_t* tensors[4] = {
						mpgetbuffer((ccv_nnc_tensor_t*)g), // A
						a_data, // B
						mpgetbuffer((ccv_nnc_tensor_t*)dw), // C
						NULL,
					};
					size_t tensor_offsets[4] = {
						g->dataof, // A offset
						a_dataof, // B offset
						dw->dataof, // C offset
						0, // D offset
					};
					ccv_nnc_mfa_encode_gemm(context, dw_params, command_batch, tensors, tensor_offsets);
				} else {
					mtl_buffer_t* tensors[4] = {
						a_data, // A
						mpgetbuffer((ccv_nnc_tensor_t*)g), // B
						mpgetbuffer((ccv_nnc_tensor_t*)dw), // C
						NULL,
					};
					size_t tensor_offsets[4] = {
						a_dataof, // A offset
						g->dataof, // B offset
						dw->dataof, // C offset
						0, // D offset
					};
					ccv_nnc_mfa_encode_gemm(context, dw_params, command_batch, tensors, tensor_offsets);
				}
			}
			ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
		} else {
			mtl_buffer_t* a_data = 0;
			size_t a_dataof = 0;
			if (a && dw)
			{
				a_data = mpgetbuffer((ccv_nnc_tensor_t*)a);
				a_dataof = (size_t)mpgetoffset((ccv_nnc_tensor_t*)a);
			}
			mtl_buffer_t* w_data = 0;
			size_t w_dataof = 0;
			if (w && h)
			{
				w_data = mpgetbuffer((ccv_nnc_tensor_t*)w);
				w_dataof = (size_t)mpgetoffset((ccv_nnc_tensor_t*)w);
			}
			MPSCommandBuffer* command_buffer;
			if ((a && dw && CCV_GET_DATA_TYPE(a->info.datatype) == CCV_QX) || (w && h && CCV_GET_DATA_TYPE(w->info.datatype) == CCV_QX))
			{
				mtl_buffer_t* scratch = 0;
				if (a_data_size + w_data_size > 0)
					scratch = ccv_nnc_mfa_request_scratch(context, a_data_size + w_data_size);
				ccv_nnc_mfa_depalettize_params_t a_depalettize_params;
				if (a && dw && CCV_GET_DATA_TYPE(a->info.datatype) == CCV_QX)
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
					a_dataof = 0;
				}
				ccv_nnc_mfa_depalettize_params_t w_depalettize_params;
				if (w && h && CCV_GET_DATA_TYPE(w->info.datatype) == CCV_QX)
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
					w_dataof = a_data_size;
				}
				mtl_command_batch_t* command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
				if (a && dw && CCV_GET_DATA_TYPE(a->info.datatype) == CCV_QX)
				{
					mtl_buffer_t* tensors[3] = {
						mpgetbuffer((ccv_nnc_tensor_t*)a), // A
						(mtl_buffer_t*)scratch, // B
						NULL,
					};
					size_t tensor_offsets[2] = {
						a->dataof, // A offset
						0, // B offset
					};
					ccv_nnc_mfa_encode_depalettize(context, a_depalettize_params, command_batch, tensors, tensor_offsets);
				}
				if (w && h && CCV_GET_DATA_TYPE(w->info.datatype) == CCV_QX)
				{
					mtl_buffer_t* tensors[3] = {
						mpgetbuffer((ccv_nnc_tensor_t*)w), // A
						(mtl_buffer_t*)scratch, // B
						NULL,
					};
					size_t tensor_offsets[2] = {
						w->dataof, // A offset
						a_data_size, // B offset
					};
					ccv_nnc_mfa_encode_depalettize(context, w_depalettize_params, command_batch, tensors, tensor_offsets);
				}
				command_buffer = ccv_nnc_stream_context_finish_command_batch_encoding_and_return_mps_command_buffer(stream_context, command_batch);
			} else // Otherwise, incur the ~10-50 microsecond latency of MPS.
				command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);

			if (h) {
				assert(w); // when calculate h, w must exist
				// [output gradient]
				ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
				int indices[2];

				MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
					MPSGraphTensor* mps_input_g;
					MPSGraphTensor* mps_g = ccv_nnc_mps_graph_tensor_input(graph, g, g->info.dim, g->stride, &mps_input_g);
					[inputTensors addObject:mps_input_g];
					MPSGraphShapedType* mps_g_shape = ccv_nnc_mps_graph_tensor_input_shape(g, g->info.dim, g->stride);
					[inputShapedTypes addObject:mps_g_shape];

					MPSGraphTensor* mps_input_w;
					MPSGraphTensor* mps_w = ccv_nnc_mps_graph_tensor_input(graph, w, w->info.dim, w->stride, &mps_input_w);
					[inputTensors addObject:mps_input_w];
					MPSGraphShapedType* mps_w_shape = ccv_nnc_mps_graph_tensor_input_shape(w, w->info.dim, w->stride);
					[inputShapedTypes addObject:mps_w_shape];

					if (!is_transpose_w)
						mps_w = [graph transposeTensor:mps_w dimension:-2 withDimension:-1 name:nil];

					MPSGraphTensor* mps_h = [graph matrixMultiplicationWithPrimaryTensor:mps_g secondaryTensor:mps_w name:nil];
					if (is_transpose_a)
						mps_h = [graph transposeTensor:mps_h dimension:-2 withDimension:-1 name:nil];

					const NSUInteger mps_h_nd = mps_h.shape.count;
					int flag = (h_nd < mps_h_nd);
					int i;
					for (i = 0; !flag && i < mps_h_nd; i++)
						if (mps_h.shape[i].integerValue != h->info.dim[i])
							flag = 1;

					// if target h nd smaller than current mps_h_nd (for example, doing batch), mps_h needs to be reduced
					if (flag) {
						NSMutableArray<NSNumber*>* axes = [NSMutableArray new];
						for (i = 0; i < mps_h_nd - h_nd; i++)
							if (mps_h.shape[i].integerValue != 1)
								[axes addObject:@(i)];
						const int h_start = mps_h_nd - h_nd;
						assert(h_start >= 0);
						for (i = h_start; i < mps_h_nd; i++)
							if (mps_h.shape[i].integerValue != h->info.dim[i - h_start])
								[axes addObject:@(i)];
						if (axes.count > 0)
							mps_h = [graph reductionSumWithTensor:mps_h axes:axes name:nil];
						[axes release];
					}
					[resultTensors addObject:mps_h];
				});
				MPSGraphTensorData* data_g = ccv_nnc_mps_graph_tensor_data(g, g->info.dim, g->stride);
				MPSGraphTensorData* data_w = ccv_nnc_mps_graph_tensor_data_with_buffer(w, w->info.dim, w->stride, w_data, w_dataof);
				MPSGraphTensorData* data[] = {data_g, data_w};
				ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]]], &h, (int*[]){ h->info.dim }, (int*[]){ h->stride }, 1, 0);
			}

			if (dw) {
				assert(a); // when calculate dw, a must exist

				// [weight updates]
				ccv_nnc_mps_graph_key_t dw_key = ccv_nnc_mps_graph_key_new(cmd, 1, hint, flags, inputs, input_size, outputs, output_size);
				int dw_indices[2];

				MPSGraphExecutable* executable_dw = ccv_nnc_mps_graph_executable_cache(dw_key, dw_indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
					MPSGraphTensor* mps_input_g;
					MPSGraphTensor* mps_g = ccv_nnc_mps_graph_tensor_input(graph, g, g->info.dim, g->stride, &mps_input_g);
					[inputTensors addObject:mps_input_g];
					MPSGraphShapedType* mps_g_shape = ccv_nnc_mps_graph_tensor_input_shape(g, g->info.dim, g->stride);
					[inputShapedTypes addObject:mps_g_shape];

					MPSGraphTensor* mps_input_a;
					MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
					[inputTensors addObject:mps_input_a];
					MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, a->info.dim, a->stride);
					[inputShapedTypes addObject:mps_a_shape];
					if (!is_transpose_a)
						mps_a = [graph transposeTensor:mps_a dimension:-2 withDimension:-1 name:nil];

					MPSGraphTensor* mps_dw = [graph matrixMultiplicationWithPrimaryTensor:mps_a secondaryTensor:mps_g name:nil];
					if (is_transpose_w)
						mps_dw = [graph transposeTensor:mps_dw dimension:-2 withDimension:-1 name:nil];

					const NSUInteger mps_dw_nd = mps_dw.shape.count;
					int flag = (dw_nd < mps_dw_nd);
					int i;
					for (i = 0; !flag && i < mps_dw_nd; i++)
						if (mps_dw.shape[i].integerValue != dw->info.dim[i])
							flag = 1;

					// if target dw nd smaller than current mupltiplication nd (like we are doing batch), mps_dw needs to be reduced
					if (flag) {
						NSMutableArray<NSNumber*>* axes = [NSMutableArray new];
						for (i = 0; i < mps_dw_nd - dw_nd; i++)
							if (mps_dw.shape[i].integerValue != 1)
								[axes addObject:@(i)];
						const int dw_start = mps_dw_nd - dw_nd;
						assert(dw_start >= 0);
						for (i = dw_start; i < mps_dw_nd; i++)
							if (mps_dw.shape[i].integerValue != dw->info.dim[i - dw_start])
								[axes addObject:@(i)];
						if (axes.count > 0)
							mps_dw = [graph reductionSumWithTensor:mps_dw axes:axes name:nil];
						[axes release];
					}

					[resultTensors addObject:mps_dw];

				});
				MPSGraphTensorData* data_g = ccv_nnc_mps_graph_tensor_data(g, g->info.dim, g->stride);
				MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data_with_buffer(a, a->info.dim, a->stride, a_data, a_dataof);
				MPSGraphTensorData* data[] = {data_g, data_a};
				ccv_nnc_mps_graph_executable_result(executable_dw, command_buffer, @[data[dw_indices[0]], data[dw_indices[1]]], &dw , (int*[]){ dw->info.dim }, (int*[]){ dw->stride }, 1, 0);
			}

			if (bias) {
				// [bias updates]
				ccv_nnc_mps_graph_key_t db_key = ccv_nnc_mps_graph_key_new(cmd, 2, hint, flags, inputs, input_size, outputs, output_size);
				int db_indices[1];

				MPSGraphExecutable* executable_db = ccv_nnc_mps_graph_executable_cache(db_key, db_indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
					MPSGraphTensor* mps_input_g;
					MPSGraphTensor* mps_g = ccv_nnc_mps_graph_tensor_input(graph, g, g->info.dim, g->stride, &mps_input_g);
					[inputTensors addObject:mps_input_g];
					MPSGraphShapedType* mps_g_shape = ccv_nnc_mps_graph_tensor_input_shape(g, g->info.dim, g->stride);
					[inputShapedTypes addObject:mps_g_shape];

					MPSGraphShapedType* mps_bias_shape = ccv_nnc_mps_graph_tensor_input_shape(bias, bias->info.dim, bias->stride);

					NSMutableArray<NSNumber*>* bias_target_shape = mps_bias_shape.shape.mutableCopy;
					NSMutableArray<NSNumber*>* axes = [NSMutableArray new];
					const int g_nd = ccv_nnc_tensor_nd(g->info.dim);
					const int bias_nd = ccv_nnc_tensor_nd(bias->info.dim);
					int i;

					// make bias_target_shape has same dim as g before finding reduce axis
					for (i = 0; i < g_nd - bias_nd; i++)
						[bias_target_shape insertObject:@(1) atIndex:0]; // [1,..,1,N]

					for (i = 0; i < g_nd; i++)
						if (g->info.dim[i] != bias_target_shape[i].integerValue)
							[axes addObject:@(i)];
					MPSGraphTensor* mps_db = [graph reductionSumWithTensor:mps_g axes:axes name:nil];
					[bias_target_shape release];
					[axes release];
					[resultTensors addObject:mps_db];
				});
				MPSGraphTensorData* data_g = ccv_nnc_mps_graph_tensor_data(g, g->info.dim, g->stride);
				ccv_nnc_mps_graph_executable_result(executable_db, command_buffer, @[data_g], &bias , (int*[]){ bias->info.dim  }, (int*[]){ bias->info.dim }, 1, 0);
			}
			ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
		}
	}

	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_QX;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_gemm_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_gemm_back;
}
