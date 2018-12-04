extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

#ifdef HAVE_NCCL

static int _ccv_nnc_allreduce_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == output_size);
	int i, device_count = 0;
	assert(input_size > 0);
	const size_t tensor_count = ccv_nnc_tensor_count(inputs[0]->info);
	for (i = 0; i < input_size; i++)
	{
		assert(!CCV_IS_TENSOR_VIEW(inputs[i]));
		assert(ccv_nnc_tensor_count(inputs[i]->info) == tensor_count);
		assert(!CCV_IS_TENSOR_VIEW(outputs[i]));
		assert(ccv_nnc_tensor_count(outputs[i]->info) == tensor_count);
		assert(CCV_TENSOR_GET_DEVICE(inputs[i]->info.type) == CCV_TENSOR_GET_DEVICE(outputs[i]->info.type));
		const int device_id = CCV_TENSOR_GET_DEVICE_ID(inputs[i]->info.type);
		device_count = ccv_max(device_id + 1, device_count);
	}
	NCCL_ENFORCE(ncclGroupStart());
	for (i = 0; i < input_size; i++)
	{
		ccv_nnc_tensor_t* const a = inputs[i];
		const int device_id = CCV_TENSOR_GET_DEVICE_ID(a->info.type);
		ncclComm_t comm = ccv_nnc_nccl_get_comm(stream_context, device_count, device_id);
		ccv_nnc_stream_context_t* const neighbor_context = stream_context ? ccv_nnc_stream_context_find_neighbor(stream_context, device_id) : 0;
		cudaStream_t stream = ccv_nnc_stream_context_get_stream(neighbor_context);
		ccv_nnc_tensor_t* const b = outputs[i];
		NCCL_ENFORCE(ncclAllReduce(a->data.f32, b->data.f32, tensor_count, ncclFloat, ncclSum, comm, stream));
	}
	NCCL_ENFORCE(ncclGroupEnd());
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_allreduce_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(output_size >= input_size);
	// For allreduce, forward and backward are the same.
	return _ccv_nnc_allreduce_forw(cmd, hint, flags, inputs, output_size, outputs, output_size, stream_context);
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_COMM_ALLREDUCE_FORWARD, CCV_NNC_BACKEND_GPU_NCCL)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_NCCL
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_allreduce_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_COMM_ALLREDUCE_BACKWARD, CCV_NNC_BACKEND_GPU_NCCL)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_NCCL
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_allreduce_back;
#endif
}

#ifdef HAVE_NCCL

static int _ccv_nnc_broadcast_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	assert(output_size >= 1);
	int i, device_count = 0;
	assert(!CCV_IS_TENSOR_VIEW(inputs[0]));
	const size_t tensor_count = ccv_nnc_tensor_count(inputs[0]->info);
	for (i = 0; i < output_size; i++)
	{
		assert(!CCV_IS_TENSOR_VIEW(outputs[i]));
		assert(ccv_nnc_tensor_count(outputs[i]->info) == tensor_count);
		const int device_id = CCV_TENSOR_GET_DEVICE_ID(outputs[i]->info.type);
		device_count = ccv_max(device_id + 1, device_count);
	}
	ccv_nnc_tensor_t* const a = inputs[0];
	ncclComm_t comm = ccv_nnc_nccl_get_comm(stream_context, device_count, CCV_TENSOR_GET_DEVICE_ID(a->info.type));
	int root;
	NCCL_ENFORCE(ncclCommUserRank(comm, &root)); // The root rank.
	NCCL_ENFORCE(ncclGroupStart());
	for (i = 0; i < output_size; i++)
	{
		ccv_nnc_tensor_t* const b = outputs[i];
		const int device_id = CCV_TENSOR_GET_DEVICE_ID(b->info.type);
		ncclComm_t comm = ccv_nnc_nccl_get_comm(stream_context, device_count, device_id);
		ccv_nnc_stream_context_t* const neighbor_context = stream_context ? ccv_nnc_stream_context_find_neighbor(stream_context, device_id) : 0;
		cudaStream_t stream = ccv_nnc_stream_context_get_stream(neighbor_context);
		NCCL_ENFORCE(ncclBroadcast(a->data.f32, b->data.f32, tensor_count, ncclFloat, root, comm, stream));
	}
	NCCL_ENFORCE(ncclGroupEnd());
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_reduce_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 1);
	assert(output_size == 1);
	int i, device_count = 0;
	assert(!CCV_IS_TENSOR_VIEW(outputs[0]));
	const size_t tensor_count = ccv_nnc_tensor_count(outputs[0]->info);
	for (i = 0; i < input_size; i++)
	{
		assert(!CCV_IS_TENSOR_VIEW(inputs[i]));
		assert(ccv_nnc_tensor_count(inputs[i]->info) == tensor_count);
		const int device_id = CCV_TENSOR_GET_DEVICE_ID(inputs[i]->info.type);
		device_count = ccv_max(device_id + 1, device_count);
	}
	ccv_nnc_tensor_t* const b = outputs[0];
	ncclComm_t comm = ccv_nnc_nccl_get_comm(stream_context, device_count, CCV_TENSOR_GET_DEVICE_ID(b->info.type));
	int root;
	NCCL_ENFORCE(ncclCommUserRank(comm, &root)); // The root rank.
	NCCL_ENFORCE(ncclGroupStart());
	for (i = 0; i < input_size; i++)
	{
		ccv_nnc_tensor_t* const a = inputs[i];
		const int device_id = CCV_TENSOR_GET_DEVICE_ID(a->info.type);
		ncclComm_t comm = ccv_nnc_nccl_get_comm(stream_context, device_count, device_id);
		ccv_nnc_stream_context_t* const neighbor_context = stream_context ? ccv_nnc_stream_context_find_neighbor(stream_context, device_id) : 0;
		cudaStream_t stream = ccv_nnc_stream_context_get_stream(neighbor_context);
		NCCL_ENFORCE(ncclReduce(a->data.f32, b->data.f32, tensor_count, ncclFloat, ncclSum, root, comm, stream));
	}
	NCCL_ENFORCE(ncclGroupEnd());
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_broadcast_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(output_size == 1);
	// For allreduce, forward and backward are the same.
	return _ccv_nnc_reduce_forw(cmd, hint, flags, inputs, (input_size - 1) / 2, outputs, output_size, stream_context);
}

static int _ccv_nnc_reduce_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	// For allreduce, forward and backward are the same.
	return _ccv_nnc_broadcast_forw(cmd, hint, flags, inputs, 1, outputs, output_size, stream_context);
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_COMM_BROADCAST_FORWARD, CCV_NNC_BACKEND_GPU_NCCL)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_NCCL
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_broadcast_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_COMM_BROADCAST_BACKWARD, CCV_NNC_BACKEND_GPU_NCCL)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_NCCL
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_broadcast_back;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_COMM_REDUCE_FORWARD, CCV_NNC_BACKEND_GPU_NCCL)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_NCCL
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_reduce_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_COMM_REDUCE_BACKWARD, CCV_NNC_BACKEND_GPU_NCCL)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_NCCL
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_reduce_back;
#endif
}
