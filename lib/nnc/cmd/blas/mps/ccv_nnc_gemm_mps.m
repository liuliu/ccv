#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
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
	const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
	if (bias)
	{
		const int bias_nd = ccv_nnc_tensor_nd(bias->info.dim);
		// Align bias to this.
		assert(bias_nd <= 2 || bias_nd == b_nd);
		int i;
		if (bias_nd == b_nd)
		{
			memcpy(biasdim, bias->info.dim, sizeof(biasdim));
			if (CCV_IS_TENSOR_VIEW(bias))
				memcpy(biasstride, bias->stride, sizeof(biasstride));
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
	assert(a_batch_size == b_batch_size || a_batch_size == 1);
	if (a_batch_size == 1 && b_batch_size > 1)
		a_batch_inc = 0;
	assert(w_batch_size == a_batch_size || w_batch_size == 1);
	if (w_batch_size == 1 && b_batch_size > 1)
		w_batch_inc = 0;
	@autoreleasepool {
		const int is_contiguous =
			(!CCV_IS_TENSOR_VIEW(a) || ccv_nnc_tensor_view_is_contiguous(adim, astride)) &&
			(!CCV_IS_TENSOR_VIEW(w) || ccv_nnc_tensor_view_is_contiguous(w->info.dim, w->stride)) &&
			(!CCV_IS_TENSOR_VIEW(b) || ccv_nnc_tensor_view_is_contiguous(b->info.dim, b->stride));

		const int is_same_dtype =
			(a->info.datatype == w->info.datatype) &&
			(a->info.datatype == b->info.datatype);
		
		int is_supported_dtype = 0;
		uint32_t mtl_data_type = UINT32_MAX;
		switch (a->info.datatype) {
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
		const int is_mfa_supported =
			ccv_nnc_mfa_context_supported(context) && is_contiguous && is_same_dtype && is_supported_dtype && (is_mfa_compatible_batch || !is_batched) && !bias;

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
				if (!(is_mfa_compatible_batch || !is_batched))
				{
					ccv_nnc_mfa_log_message("  Unsupported batch.");
				}
				if (!(!bias))
				{
					ccv_nnc_mfa_log_message("  Requires fused activations.");
				}
			}
		}

		if (is_mfa_supported)
		{
			// On supported devices, use Metal directly.
			ccv_nnc_mfa_gemm_params_t params = {
				.data_type = mtl_data_type,
				.M = (uint32_t)b_rows, // C_rows
				.N = (uint32_t)b_cols, // C_cols
				.K = (uint32_t)w_rows, // B_rows
				.A_trans = (is_transpose_a ? 1 : 0),
				.B_trans = (is_transpose_w ? 1 : 0),
				.alpha = (float)1.0,
				.beta = (float)0.0,
				.batched = is_batched,
				.fused_activation = 0,
				
				.batch_dims_a = { 0 },
				.batch_dims_b = { 0 },
			};
			if (is_batched) {
				// Create a null-terminated list of batch dimensions.
				int A_batch_dim = a_nd - 2;
				for (int i = 0; i < A_batch_dim; ++i) {
					params.batch_dims_a[i] = adim[i];
				}
				if (A_batch_dim < CCV_NNC_MAX_DIM_ALLOC) {
					params.batch_dims_a[A_batch_dim] = 0;
				}
				
				int B_batch_dim = w_nd - 2;
				for (int i = 0; i < B_batch_dim; ++i) {
					params.batch_dims_b[i] = w->info.dim[i];
				}
				if (B_batch_dim < CCV_NNC_MAX_DIM_ALLOC) {
					params.batch_dims_b[B_batch_dim] = 0;
				}
			}
			ccv_nnc_mfa_sync_prepare_gemm(context, params);

			// Creating a new command buffer has a >10 µs penalty CPU-side. Still
			// faster the >50 µs penalty for MPSGraph (probably why
			// MPSMatrixMultiplication is faster for GEMM).
			mtl_command_batch_t* command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
			mtl_buffer_t* tensors[4] = {
				mpgetbuffer((ccv_nnc_tensor_t*)a), // A
				mpgetbuffer((ccv_nnc_tensor_t*)w), // B
				mpgetbuffer((ccv_nnc_tensor_t*)b), // C
				NULL
			};
			size_t tensor_offsets[3] = {
				a->dataof, // A offset
				w->dataof, // B offset
				b->dataof, // C offset
			};
			ccv_nnc_mfa_encode_gemm(context, params, command_batch, tensors, tensor_offsets);

			// TODO: Add this diagnostic once we consistently capture >>1 commands/batch.
//			if (METAL_LOG_LEVEL(context) >= 3) {
//				if (command_batch->batched_command_count == 0) {
//					ccv_nnc_mfa_log_message("Encoded 0 commands in the batch.");
//				} else if (command_batch->batched_command_count == 1) {
//					ccv_nnc_mfa_log_message("Encoded 1 command in the batch.");
//				} else {
//					ccv_nnc_mfa_log_message("Encoded >1 commands in the batch.");
//				}
//			}
			ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
			// TODO: Try to use `fused_activation` for with bias case.
		} else {
			// Otherwise, incur the ~10-50 microsecond latency of MPS.
			MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
			
			// If all conditions are met, use MPSMatrixMultiplication.
			if (is_contiguous && is_same_dtype && is_same_batch && !(ccv_nnc_flags() & CCV_NNC_DISABLE_MIXED_MPS_GEMM) && !bias)
			{
				id<MTLBuffer> a_buffer = mpgetbuffer((ccv_nnc_tensor_t*)a);
				MPSMatrix* leftMatrix = [[MPSMatrix alloc] initWithBuffer:a_buffer offset:a->dataof descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:(is_transpose_a ? a_cols : a_rows) columns:(is_transpose_a ? a_rows : a_cols) matrices:b_batch_size rowBytes:CCV_GET_DATA_TYPE_SIZE(a->info.datatype) * (is_transpose_a ? a_cols_inc : a_rows_inc) matrixBytes:CCV_GET_DATA_TYPE_SIZE(a->info.datatype) * a_batch_inc dataType:ccv_nnc_mps_datatype(a->info.datatype)]];
				id<MTLBuffer> w_buffer = mpgetbuffer((ccv_nnc_tensor_t*)w);
				MPSMatrix* rightMatrix = [[MPSMatrix alloc] initWithBuffer:w_buffer offset:w->dataof descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:(is_transpose_w ? w_cols : w_rows) columns:(is_transpose_w ? w_rows : w_cols) matrices:b_batch_size rowBytes:CCV_GET_DATA_TYPE_SIZE(w->info.datatype) * (is_transpose_w ? w_cols_inc : w_rows_inc) matrixBytes:CCV_GET_DATA_TYPE_SIZE(w->info.datatype) * w_batch_inc dataType:ccv_nnc_mps_datatype(w->info.datatype)]];
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
				[resultMatrix release];
			} else {
				// Otherwise, use MPSGraph.
				ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, hint, flags, inputs, input_size, outputs, output_size);
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
				MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, adim, astride);
				MPSGraphTensorData* data_w = ccv_nnc_mps_graph_tensor_data(w, w->info.dim, w->stride);
				if (bias)
				{
					MPSGraphTensorData* data_bias = ccv_nnc_mps_graph_tensor_data(bias, biasdim, biasstride);
					MPSGraphTensorData* data[] = {data_a, data_w, data_bias};
					ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]], data[indices[2]]], &b, (int*[]){ b->info.dim }, (int*[]){ b->stride }, 1);
				} else {
					MPSGraphTensorData* data[] = {data_a, data_w};
					ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]]], &b, (int*[]){ b->info.dim }, (int*[]){ b->stride }, 1);
				}
			}
			ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_gemm_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
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
