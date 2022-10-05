extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

#if defined(HAVE_CUDNN) && CUDNN_VERSION >= 8100 // _v8 API were added in 8.0.2.

static inline int use_persist_common_heuristics(const int num_layers, const int hidden_size, const int bidirectional, const int input_size)
{
	return num_layers == 1 && hidden_size <= 1024 && hidden_size % 128 == 0 && input_size % 128 == 0 && bidirectional == 0;
}

static size_t _ccv_nnc_lstm_reserve_space_size(const ccv_nnc_cmd_t cmd, const int datatype, const int feature_size, const int batch_count, const int max_seq_count)
{
	if (cmd.info.rnn.is_test)
		return 0; // No need to use any reserve space when testing.
	assert(datatype == CCV_16F || datatype == CCV_32F);
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(0);
	cudnnRNNDescriptor_t rnn = ccv_nnc_stream_context_get_rnn_descriptor(0);
	cudnnRNNAlgo_t rnn_algo = use_persist_common_heuristics(cmd.info.rnn.num_layers, cmd.info.rnn.hidden_size, cmd.info.rnn.bidirectional, feature_size) ? CUDNN_RNN_ALGO_PERSIST_DYNAMIC : CUDNN_RNN_ALGO_STANDARD;
	const cudnnDataType_t data_type = datatype == CCV_16F ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;
	const cudnnMathType_t math_type = datatype == CCV_16F ? CUDNN_TENSOR_OP_MATH : CUDNN_DEFAULT_MATH;
	const int proj_size = cmd.info.rnn.proj_size == 0 ? cmd.info.rnn.hidden_size : cmd.info.rnn.proj_size;
	cudnnDropoutDescriptor_t dropout_desc = cmd.info.rnn.dropout == 0 ? 0 : ccv_nnc_stream_context_get_dropout_descriptor(0, cmd.info.rnn.dropout);
	CUDNN_ENFORCE(cudnnSetRNNDescriptor_v8(rnn, rnn_algo, CUDNN_LSTM, cmd.info.rnn.bias ? CUDNN_RNN_DOUBLE_BIAS : CUDNN_RNN_NO_BIAS, cmd.info.rnn.bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL, CUDNN_LINEAR_INPUT, data_type, data_type, math_type, feature_size, cmd.info.rnn.hidden_size, proj_size, cmd.info.rnn.num_layers, dropout_desc, CUDNN_RNN_PADDED_IO_ENABLED));
	size_t workspace_size = 0;
	size_t reserve_size = 0;
	const cudnnForwardMode_t fwd_mode = CUDNN_FWD_MODE_TRAINING;
	cudnnRNNDataDescriptor_t x = ccv_nnc_stream_context_get_rnn_data_descriptor(0);
	int* const seq_lengths = (int*)ccv_nnc_stream_context_get_workspace(0, sizeof(int) * batch_count, CCV_TENSOR_CPU_MEMORY);
	int i;
	for (i = 0; i < batch_count; i++)
		seq_lengths[i] = max_seq_count;
	const cudnnRNNDataLayout_t data_layout = cmd.info.rnn.batch_first ? CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED : CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED;
	CUDNN_ENFORCE(cudnnSetRNNDataDescriptor(x, data_type, data_layout, max_seq_count, batch_count, feature_size, seq_lengths, 0));
	cudnnGetRNNTempSpaceSizes(cudnn, rnn, fwd_mode, x, &workspace_size, &reserve_size);
	ccv_nnc_stream_context_return_rnn_data_descriptor(0, x);
	ccv_nnc_stream_context_return_rnn_descriptor(0, rnn);
	if (dropout_desc)
		ccv_nnc_stream_context_return_dropout_descriptor(0, dropout_desc);
	return reserve_size;
}

static int _ccv_nnc_lstm_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 5);
	assert(output_size >= 3);
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	const int x_nd = ccv_nnc_tensor_nd(inputs[0]->info.dim);
	assert(x_nd == 3 || x_nd == 2);
	const int batch_count = x_nd == 3 ? (cmd.info.rnn.batch_first ? inputs[0]->info.dim[0] : inputs[0]->info.dim[1]) : 1;
	assert(batch_count > 0);
	const int max_seq_count = x_nd == 3 ? (cmd.info.rnn.batch_first ? inputs[0]->info.dim[1] : inputs[0]->info.dim[0]) : inputs[0]->info.dim[0];
	assert(max_seq_count > 0);
	const int feature_count = inputs[0]->info.dim[x_nd - 1];
	assert(feature_count > 0);
	const ccv_nnc_tensor_t* const ix = inputs[1];
	int* const seq_lengths = ix ? ix->data.i32 : (int*)ccv_nnc_stream_context_get_workspace(stream_context, sizeof(int) * batch_count, CCV_TENSOR_CPU_MEMORY);
	int i;
	if (ix)
	{
		assert(ix->info.datatype == CCV_32S);
		assert(ccv_nnc_tensor_nd(ix->info.dim) == 1);
		assert(batch_count == ix->info.dim[0]);
		assert(CCV_TENSOR_GET_MEMORY(ix->info.type) == CCV_TENSOR_GPU_MEMORY);
	} else {
		for (i = 0; i < batch_count; i++)
			seq_lengths[i] = max_seq_count;
	}
	const cudnnRNNDataLayout_t data_layout = cmd.info.rnn.batch_first ? CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED : CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED;
	assert(inputs[0]->info.datatype == CCV_16F || inputs[0]->info.datatype == CCV_32F);
	cudnnRNNDataDescriptor_t x = ccv_nnc_stream_context_get_rnn_data_descriptor(stream_context);
	CUDNN_ENFORCE(cudnnSetRNNDataDescriptor(x, inputs[0]->info.datatype == CCV_16F ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT, data_layout, max_seq_count, batch_count, feature_count, seq_lengths, 0));
	const int proj_size = cmd.info.rnn.proj_size == 0 ? cmd.info.rnn.hidden_size : cmd.info.rnn.proj_size;
	cudnnTensorDescriptor_t h = ccv_nnc_stream_context_get_tensor_descriptor(stream_context);
	const int bidirectional_num_layers = cmd.info.rnn.num_layers * (!!cmd.info.rnn.bidirectional + 1);
	CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(h, ccv_nnc_cudnn_datatype(inputs[0]->info.datatype), bidirectional_num_layers, batch_count, proj_size, 1, batch_count * proj_size, proj_size, 1, 1));
	assert(!inputs[2] || (inputs[2]->info.dim[0] == bidirectional_num_layers && inputs[2]->info.dim[1] == batch_count && inputs[2]->info.dim[2] == proj_size));
	cudnnTensorDescriptor_t c = ccv_nnc_stream_context_get_tensor_descriptor(stream_context);
	CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(c, ccv_nnc_cudnn_datatype(inputs[0]->info.datatype), bidirectional_num_layers, batch_count, cmd.info.rnn.hidden_size, 1, batch_count * cmd.info.rnn.hidden_size, cmd.info.rnn.hidden_size, 1, 1));
	assert(!inputs[3] || (inputs[3]->info.dim[0] == bidirectional_num_layers && inputs[3]->info.dim[1] == batch_count && inputs[3]->info.dim[2] == cmd.info.rnn.hidden_size));
	const ccv_nnc_tensor_t* const w = inputs[4];
	assert(!inputs[2] || inputs[0]->info.datatype == inputs[2]->info.datatype);
	assert(!inputs[3] || inputs[0]->info.datatype == inputs[3]->info.datatype);
	assert(inputs[0]->info.datatype == inputs[4]->info.datatype);
	const int y_nd = ccv_nnc_tensor_nd(outputs[0]->info.dim);
	const int output_feature_count = outputs[0]->info.dim[y_nd - 1];
	cudnnRNNDataDescriptor_t y = ccv_nnc_stream_context_get_rnn_data_descriptor(stream_context);
	// Note that the paddingFill is NULL, meaning what is in the padding is undefined.
	CUDNN_ENFORCE(cudnnSetRNNDataDescriptor(y, outputs[0]->info.datatype == CCV_16F ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT, data_layout, max_seq_count, batch_count, output_feature_count, seq_lengths, 0));
	assert(outputs[0]->info.datatype == inputs[0]->info.datatype);
	assert(!inputs[2] || !outputs[1] || inputs[2]->info.datatype == outputs[1]->info.datatype);
	assert(!inputs[3] || !outputs[2] || inputs[3]->info.datatype == outputs[2]->info.datatype);
	ccv_nnc_tensor_t* const r = output_size >= 4 ? outputs[3] : 0;
	if (r)
		{ assert(outputs[0]->info.datatype == outputs[3]->info.datatype); }
	cudnnRNNDescriptor_t rnn = ccv_nnc_stream_context_get_rnn_descriptor(stream_context);
	const int is_test = cmd.info.rnn.is_test;
	const cudnnDataType_t data_type = inputs[0]->info.datatype == CCV_16F ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;
	const cudnnMathType_t math_type = inputs[0]->info.datatype == CCV_16F ? CUDNN_TENSOR_OP_MATH : CUDNN_DEFAULT_MATH;
	assert(output_feature_count == proj_size * (!!cmd.info.rnn.bidirectional + 1));
	cudnnRNNAlgo_t rnn_algo = use_persist_common_heuristics(cmd.info.rnn.num_layers, cmd.info.rnn.hidden_size, cmd.info.rnn.bidirectional, feature_count) ? CUDNN_RNN_ALGO_PERSIST_DYNAMIC : CUDNN_RNN_ALGO_STANDARD;
	cudnnDropoutDescriptor_t dropout_desc = cmd.info.rnn.dropout == 0 ? 0 : ccv_nnc_stream_context_get_dropout_descriptor(stream_context, cmd.info.rnn.dropout);
	CUDNN_ENFORCE(cudnnSetRNNDescriptor_v8(rnn, rnn_algo, CUDNN_LSTM, cmd.info.rnn.bias ? CUDNN_RNN_DOUBLE_BIAS : CUDNN_RNN_NO_BIAS, cmd.info.rnn.bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL, CUDNN_LINEAR_INPUT, data_type, data_type, math_type, feature_count, cmd.info.rnn.hidden_size, proj_size, cmd.info.rnn.num_layers, dropout_desc, CUDNN_RNN_PADDED_IO_ENABLED));
	size_t workspace_size = 0;
	size_t reserve_size = 0;
	void* workspace = 0;
	const cudnnForwardMode_t fwd_mode = is_test ? CUDNN_FWD_MODE_INFERENCE : CUDNN_FWD_MODE_TRAINING;
	cudnnGetRNNTempSpaceSizes(cudnn, rnn, fwd_mode, x, &workspace_size, &reserve_size);
	if (!is_test)
	{
		assert(reserve_size > 0);
		assert(r);
		assert(CCV_GET_DATA_TYPE_SIZE(r->info.datatype) * ccv_nnc_tensor_count(r->info) >= reserve_size);
	}
	int* dev_seq_lengths;
	// TODO: If error, return OOM
	if (workspace_size)
	{
		workspace = ccv_nnc_stream_context_get_workspace(stream_context, workspace_size + (!ix ? sizeof(int) * batch_count : 0), CCV_TENSOR_GPU_MEMORY);
		dev_seq_lengths = (int*)((uint8_t*)workspace + workspace_size);
	} else if (!ix)
		dev_seq_lengths = (int*)ccv_nnc_stream_context_get_workspace(stream_context, sizeof(int) * batch_count, CCV_TENSOR_GPU_MEMORY);
	if (!ix) // Copy from host to device.
	{
		cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
		CUDA_ENFORCE(cudaMemcpyAsync(dev_seq_lengths, seq_lengths, sizeof(int) * batch_count, cudaMemcpyHostToDevice, stream));
	}
	size_t weight_size = 0;
	cudnnGetRNNWeightSpaceSize(cudnn, rnn, &weight_size);
	assert(CCV_GET_DATA_TYPE_SIZE(w->info.datatype) * ccv_nnc_tensor_count(w->info) >= weight_size);
	assert(weight_size > 0);
	CUDNN_ENFORCE(cudnnRNNForward(cudnn, rnn, fwd_mode, dev_seq_lengths, x, inputs[0]->data.u8, y, outputs[0]->data.u8, h, inputs[2] ? inputs[2]->data.u8 : 0, outputs[1] ? outputs[1]->data.u8 : 0, c, inputs[3] ? inputs[3]->data.u8 : 0, outputs[2] ? outputs[2]->data.u8 : 0, weight_size, w->data.u8, workspace_size, workspace, reserve_size, r ? r->data.u8 : 0));
	ccv_nnc_stream_context_return_rnn_data_descriptor(stream_context, x);
	ccv_nnc_stream_context_return_rnn_data_descriptor(stream_context, y);
	ccv_nnc_stream_context_return_tensor_descriptor(stream_context, h);
	ccv_nnc_stream_context_return_tensor_descriptor(stream_context, c);
	ccv_nnc_stream_context_return_rnn_descriptor(stream_context, rnn);
	if (dropout_desc)
		ccv_nnc_stream_context_return_dropout_descriptor(stream_context, dropout_desc);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_lstm_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 13);
	assert(output_size >= 3);
	assert(outputs[0] || (outputs[0] && outputs[2] && outputs[3]));
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	cudnnRNNDescriptor_t rnn = ccv_nnc_stream_context_get_rnn_descriptor(stream_context);
	assert(!cmd.info.rnn.is_test);
	const cudnnDataType_t data_type = inputs[0]->info.datatype == CCV_16F ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;
	const cudnnMathType_t math_type = inputs[0]->info.datatype == CCV_16F ? CUDNN_TENSOR_OP_MATH : CUDNN_DEFAULT_MATH;
	const int proj_size = cmd.info.rnn.proj_size == 0 ? cmd.info.rnn.hidden_size : cmd.info.rnn.proj_size;
	// We can get from either dx or x.
	const ccv_nnc_tensor_t* const x_or_dx = outputs[0] ? outputs[0] : inputs[4];
	const int x_nd = ccv_nnc_tensor_nd(x_or_dx->info.dim);
	assert(x_nd == 3 || x_nd == 2);
	const int batch_count = x_nd == 3 ? (cmd.info.rnn.batch_first ? x_or_dx->info.dim[0] : x_or_dx->info.dim[1]) : 1;
	assert(batch_count > 0);
	const int max_seq_count = x_nd == 3 ? (cmd.info.rnn.batch_first ? x_or_dx->info.dim[1] : x_or_dx->info.dim[0]) : x_or_dx->info.dim[0];
	assert(max_seq_count > 0);
	const int feature_count = x_or_dx->info.dim[x_nd - 1];
	assert(feature_count > 0);
	const int dy_nd = ccv_nnc_tensor_nd(inputs[0]->info.dim);
	const int output_feature_count = inputs[0]->info.dim[dy_nd - 1];
	assert(output_feature_count == proj_size * (!!cmd.info.rnn.bidirectional + 1));
	cudnnRNNAlgo_t rnn_algo = use_persist_common_heuristics(cmd.info.rnn.num_layers, cmd.info.rnn.hidden_size, cmd.info.rnn.bidirectional, feature_count) ? CUDNN_RNN_ALGO_PERSIST_DYNAMIC : CUDNN_RNN_ALGO_STANDARD;
	cudnnDropoutDescriptor_t dropout_desc = cmd.info.rnn.dropout == 0 ? 0 : ccv_nnc_stream_context_get_dropout_descriptor(stream_context, cmd.info.rnn.dropout);
	assert(inputs[0]->info.datatype == CCV_16F || inputs[0]->info.datatype == CCV_32F);
	CUDNN_ENFORCE(cudnnSetRNNDescriptor_v8(rnn, rnn_algo, CUDNN_LSTM, cmd.info.rnn.bias ? CUDNN_RNN_DOUBLE_BIAS : CUDNN_RNN_NO_BIAS, cmd.info.rnn.bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL, CUDNN_LINEAR_INPUT, data_type, data_type, math_type, feature_count, cmd.info.rnn.hidden_size, proj_size, cmd.info.rnn.num_layers, dropout_desc, CUDNN_RNN_PADDED_IO_ENABLED));
	const ccv_nnc_tensor_t* const ix = inputs[5];
	int* const seq_lengths = ix ? ix->data.i32 : (int*)ccv_nnc_stream_context_get_workspace(stream_context, sizeof(int) * batch_count, CCV_TENSOR_CPU_MEMORY);
	int i;
	if (ix)
	{
		assert(ix->info.datatype == CCV_32S);
		assert(ccv_nnc_tensor_nd(ix->info.dim) == 1);
		assert(batch_count == ix->info.dim[0]);
		assert(CCV_TENSOR_GET_MEMORY(ix->info.type) == CCV_TENSOR_GPU_MEMORY);
	} else {
		for (i = 0; i < batch_count; i++)
			seq_lengths[i] = max_seq_count;
	}
	const cudnnRNNDataLayout_t data_layout = cmd.info.rnn.batch_first ? CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED : CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED;
	cudnnRNNDataDescriptor_t x = ccv_nnc_stream_context_get_rnn_data_descriptor(stream_context);
	CUDNN_ENFORCE(cudnnSetRNNDataDescriptor(x, data_type, data_layout, max_seq_count, batch_count, feature_count, seq_lengths, 0));
	cudnnRNNDataDescriptor_t y = ccv_nnc_stream_context_get_rnn_data_descriptor(stream_context);
	// Note that the paddingFill is NULL, meaning what is in the padding is undefined.
	CUDNN_ENFORCE(cudnnSetRNNDataDescriptor(y, data_type, data_layout, max_seq_count, batch_count, output_feature_count, seq_lengths, 0));
	size_t workspace_size = 0;
	size_t reserve_size = 0;
	void* workspace = 0;
	cudnnGetRNNTempSpaceSizes(cudnn, rnn, CUDNN_FWD_MODE_TRAINING, x, &workspace_size, &reserve_size);
	int* dev_seq_lengths;
	// TODO: If error, return OOM
	if (workspace_size)
	{
		workspace = ccv_nnc_stream_context_get_workspace(stream_context, workspace_size + (!ix ? sizeof(int) * batch_count : 0), CCV_TENSOR_GPU_MEMORY);
		dev_seq_lengths = (int*)((uint8_t*)workspace + workspace_size);
	} else if (!ix)
		dev_seq_lengths = (int*)ccv_nnc_stream_context_get_workspace(stream_context, sizeof(int) * batch_count, CCV_TENSOR_GPU_MEMORY);
	if (!ix) // Copy from host to device.
	{
		cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
		CUDA_ENFORCE(cudaMemcpyAsync(dev_seq_lengths, seq_lengths, sizeof(int) * batch_count, cudaMemcpyHostToDevice, stream));
	}
	assert(!inputs[1] || inputs[0]->info.datatype == inputs[1]->info.datatype);
	assert(!inputs[2] || inputs[0]->info.datatype == inputs[2]->info.datatype);
	assert(!inputs[6] || inputs[0]->info.datatype == inputs[6]->info.datatype);
	assert(!inputs[7] || inputs[0]->info.datatype == inputs[7]->info.datatype);
	assert(inputs[0]->info.datatype == inputs[8]->info.datatype);
	assert(inputs[8]->info.datatype == inputs[9]->info.datatype);
	assert(outputs[0]->info.datatype == inputs[0]->info.datatype);
	assert(!outputs[2] || outputs[0]->info.datatype == outputs[2]->info.datatype);
	assert(!outputs[3] || outputs[0]->info.datatype == outputs[3]->info.datatype);
	cudnnTensorDescriptor_t h = ccv_nnc_stream_context_get_tensor_descriptor(stream_context);
	const int bidirectional_num_layers = cmd.info.rnn.num_layers * (!!cmd.info.rnn.bidirectional + 1);
	CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(h, ccv_nnc_cudnn_datatype(inputs[0]->info.datatype), bidirectional_num_layers, batch_count, proj_size, 1, batch_count * proj_size, proj_size, 1, 1));
	assert(!inputs[2] || (inputs[2]->info.dim[0] == bidirectional_num_layers && inputs[2]->info.dim[1] == batch_count && inputs[2]->info.dim[2] == proj_size));
	cudnnTensorDescriptor_t c = ccv_nnc_stream_context_get_tensor_descriptor(stream_context);
	CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(c, ccv_nnc_cudnn_datatype(inputs[0]->info.datatype), bidirectional_num_layers, batch_count, cmd.info.rnn.hidden_size, 1, batch_count * cmd.info.rnn.hidden_size, cmd.info.rnn.hidden_size, 1, 1));
	size_t weight_size = 0;
	cudnnGetRNNWeightSpaceSize(cudnn, rnn, &weight_size);
	const ccv_nnc_tensor_t* const w = inputs[8];
	assert(CCV_GET_DATA_TYPE_SIZE(w->info.datatype) * ccv_nnc_tensor_count(w->info) >= weight_size);
	assert(weight_size > 0);
	assert(reserve_size > 0);
	const ccv_nnc_tensor_t* const r = inputs[12];
	assert(CCV_GET_DATA_TYPE_SIZE(r->info.datatype) * ccv_nnc_tensor_count(r->info) >= reserve_size);
	CUDNN_ENFORCE(cudnnRNNBackwardData_v8(cudnn, rnn, dev_seq_lengths, y, inputs[9]->data.u8, inputs[0]->data.u8, x, outputs[0]->data.u8, h, inputs[6] ? inputs[6]->data.u8 : 0, inputs[1] ? inputs[1]->data.u8 : 0, outputs[2] ? outputs[2]->data.u8 : 0, c, inputs[7] ? inputs[7]->data.u8 : 0, inputs[2] ? inputs[2]->data.u8 : 0, outputs[3] ? outputs[3]->data.u8 : 0, weight_size, w->data.u8, workspace_size, workspace, reserve_size, r->data.u8));
	if (outputs[4])
	{
		const ccv_nnc_cudnn_tensor_view_descriptor_t dw = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)outputs[4]);
		static const float zero = 0;
		CUDNN_ENFORCE(cudnnSetTensor(cudnn, dw.descriptor, dw.data.u8, &zero));
		assert(outputs[4]->info.datatype == inputs[0]->info.datatype);
		CUDNN_ENFORCE(cudnnRNNBackwardWeights_v8(cudnn, rnn, CUDNN_WGRAD_MODE_ADD, dev_seq_lengths, x, inputs[4]->data.u8, h, inputs[6] ? inputs[6]->data.u8 : 0, y, inputs[9]->data.u8, weight_size, outputs[4]->data.u8, workspace_size, workspace, reserve_size, r->data.u8));
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(dw);
	}
	ccv_nnc_stream_context_return_rnn_data_descriptor(stream_context, x);
	ccv_nnc_stream_context_return_rnn_data_descriptor(stream_context, y);
	ccv_nnc_stream_context_return_tensor_descriptor(stream_context, h);
	ccv_nnc_stream_context_return_tensor_descriptor(stream_context, c);
	ccv_nnc_stream_context_return_rnn_descriptor(stream_context, rnn);
	if (dropout_desc)
		ccv_nnc_stream_context_return_dropout_descriptor(stream_context, dropout_desc);
	return CCV_NNC_EXEC_SUCCESS;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_LSTM_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#if defined(HAVE_CUDNN) && CUDNN_VERSION >= 8100
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->exec = _ccv_nnc_lstm_forw;
	registry->aux = (void*)_ccv_nnc_lstm_reserve_space_size;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_LSTM_BACKWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#if defined(HAVE_CUDNN) && CUDNN_VERSION >= 8100
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->exec = _ccv_nnc_lstm_back;
	registry->aux = (void*)_ccv_nnc_lstm_reserve_space_size;
#endif
}
