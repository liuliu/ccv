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
	assert(input_size >= output_size);
	int i, device_count = 0;
	assert(input_size > 0);
	const size_t tensor_count = ccv_nnc_tensor_count(inputs[0]->info);
	for (i = 0; i < output_size; i++)
	{
		assert(CCV_IS_TENSOR_CONTIGUOUS(inputs[i]));
		assert(ccv_nnc_tensor_count(inputs[i]->info) == tensor_count);
		assert(CCV_IS_TENSOR_CONTIGUOUS(outputs[i]));
		assert(ccv_nnc_tensor_count(outputs[i]->info) == tensor_count);
		assert(CCV_TENSOR_GET_DEVICE(inputs[i]->info.type) == CCV_TENSOR_GET_DEVICE(outputs[i]->info.type));
		const int device_id = CCV_TENSOR_GET_DEVICE_ID(inputs[i]->info.type);
		device_count = ccv_max(device_id + 1, device_count);
	}
	ncclComm_t comms[output_size];
	for (i = 0; i < output_size; i++)
	{
		ccv_nnc_tensor_t* const a = inputs[i];
		const int device_id = CCV_TENSOR_GET_DEVICE_ID(a->info.type);
		comms[i] = ccv_nnc_nccl_get_comm(stream_context, device_count, device_id);
	}
	NCCL_ENFORCE(ncclGroupStart());
	for (i = 0; i < output_size; i++)
	{
		ccv_nnc_tensor_t* const a = inputs[i];
		const int datatype = a->info.datatype;
		const int device_id = CCV_TENSOR_GET_DEVICE_ID(a->info.type);
		ncclComm_t comm = comms[i];
		ccv_nnc_stream_context_t* const neighbor_context = stream_context ? ccv_nnc_stream_context_find_neighbor(stream_context, device_id) : 0;
		cudaStream_t stream = ccv_nnc_stream_context_get_stream(neighbor_context);
		ccv_nnc_tensor_t* const b = outputs[i];
		assert(a->info.datatype == b->info.datatype);
		NCCL_ENFORCE(ncclAllReduce(a->data.f32, b->data.f32, tensor_count, ccv_nnc_nccl_datatype(datatype), ncclSum, comm, stream));
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

#if NCCL_VERSION_CODE >= 2700
static int _ccv_nnc_all_to_all_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == output_size);
	assert(input_size > 0);
	const int rank_count = input_size;
	int i, j, device_count = 0;
	size_t k;
	assert(CCV_IS_TENSOR_CONTIGUOUS(inputs[0]));
	const int datatype = inputs[0]->info.datatype;
	const int tensor_nd = ccv_nnc_tensor_nd(inputs[0]->info.dim);
	const int axis = cmd.info.all_to_all.axis;
	assert(axis >= 0 && axis < tensor_nd);
	assert(inputs[0]->info.dim[axis] % rank_count == 0);
	const size_t datatype_size = CCV_GET_DATA_TYPE_SIZE(datatype);
	size_t inner_count = 1;
	for (i = axis + 1; i < tensor_nd; i++)
		inner_count *= inputs[0]->info.dim[i];
	size_t outer_count = 1;
	for (i = 0; i < axis; i++)
		outer_count *= inputs[0]->info.dim[i];
	const size_t axis_dim_count = inputs[0]->info.dim[axis] * inner_count;
	const size_t chunk_count = inputs[0]->info.dim[axis] / rank_count * inner_count;
	const size_t chunk_size = chunk_count * datatype_size;
	for (i = 0; i < rank_count; i++)
	{
		assert(CCV_IS_TENSOR_CONTIGUOUS(inputs[i]));
		assert(CCV_IS_TENSOR_CONTIGUOUS(outputs[i]));
		assert(inputs[i]->info.format == inputs[0]->info.format);
		assert(outputs[i]->info.format == inputs[0]->info.format);
		assert(inputs[i]->info.datatype == datatype);
		assert(outputs[i]->info.datatype == datatype);
		assert(memcmp(inputs[i]->info.dim, inputs[0]->info.dim, sizeof(inputs[0]->info.dim)) == 0);
		assert(memcmp(outputs[i]->info.dim, inputs[0]->info.dim, sizeof(inputs[0]->info.dim)) == 0);
		const int input_device_id = CCV_TENSOR_GET_DEVICE_ID(inputs[i]->info.type);
		const int output_device_id = CCV_TENSOR_GET_DEVICE_ID(outputs[i]->info.type);
		assert(input_device_id == output_device_id);
		device_count = ccv_max(input_device_id + 1, device_count);
	}
	for (i = 0; i < rank_count; i++)
		for (j = 0; j < rank_count; j++)
			assert(inputs[i] != outputs[j]);
	ncclComm_t comms[rank_count];
	int ranks[rank_count];
	int device_ids[rank_count];
	cudaStream_t streams[rank_count];
	for (i = 0; i < rank_count; i++)
	{
		const int device_id = CCV_TENSOR_GET_DEVICE_ID(inputs[i]->info.type);
		device_ids[i] = device_id;
		comms[i] = ccv_nnc_nccl_get_comm(stream_context, device_count, device_id);
		NCCL_ENFORCE(ncclCommUserRank(comms[i], &ranks[i]));
		cudevice(device_id);
		ccv_nnc_stream_context_t* const neighbor_context = stream_context ? ccv_nnc_stream_context_find_neighbor(stream_context, device_id) : 0;
		streams[i] = ccv_nnc_stream_context_get_stream(neighbor_context);
		for (k = 0; k < outer_count; k++)
			CUDA_ENFORCE(cudaMemcpyAsync(outputs[i]->data.u8 + (k * axis_dim_count + i * chunk_count) * datatype_size, inputs[i]->data.u8 + (k * axis_dim_count + i * chunk_count) * datatype_size, chunk_size, cudaMemcpyDeviceToDevice, streams[i]));
	}
	NCCL_ENFORCE(ncclGroupStart());
	for (i = 0; i < rank_count; i++)
	{
		cudevice(device_ids[i]);
		for (j = 0; j < rank_count; j++)
			if (i != j)
			{
				for (k = 0; k < outer_count; k++)
				{
					NCCL_ENFORCE(ncclSend(inputs[i]->data.u8 + (k * axis_dim_count + j * chunk_count) * datatype_size, chunk_count, ccv_nnc_nccl_datatype(datatype), ranks[j], comms[i], streams[i]));
					NCCL_ENFORCE(ncclRecv(outputs[i]->data.u8 + (k * axis_dim_count + j * chunk_count) * datatype_size, chunk_count, ccv_nnc_nccl_datatype(datatype), ranks[j], comms[i], streams[i]));
				}
			}
	}
	NCCL_ENFORCE(ncclGroupEnd());
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_all_to_all_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return _ccv_nnc_all_to_all_forw(cmd, hint, flags, inputs, input_size, outputs, output_size, stream_context);
}
#endif

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_COMM_ALLREDUCE_FORWARD, CCV_NNC_BACKEND_GPU_NCCL)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_NCCL
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_allreduce_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_COMM_ALLREDUCE_BACKWARD, CCV_NNC_BACKEND_GPU_NCCL)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_NCCL
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_allreduce_back;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_COMM_ALL_TO_ALL_FORWARD, CCV_NNC_BACKEND_GPU_NCCL)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#if defined(HAVE_NCCL) && NCCL_VERSION_CODE >= 2700
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_64F | CCV_32F | CCV_16F | CCV_32S | CCV_8U;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_all_to_all_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_COMM_ALL_TO_ALL_BACKWARD, CCV_NNC_BACKEND_GPU_NCCL)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#if defined(HAVE_NCCL) && NCCL_VERSION_CODE >= 2700
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_64F | CCV_32F | CCV_16F | CCV_32S | CCV_8U;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_all_to_all_back;
#endif
}

#ifdef HAVE_NCCL

static int _ccv_nnc_broadcast_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	assert(output_size >= 1);
	int i, device_count = 0;
	assert(CCV_IS_TENSOR_CONTIGUOUS(inputs[0]));
	const size_t tensor_count = ccv_nnc_tensor_count(inputs[0]->info);
	for (i = 0; i < output_size; i++)
	{
		assert(CCV_IS_TENSOR_CONTIGUOUS(outputs[i]));
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
		assert(a->info.datatype == b->info.datatype);
		const int datatype = a->info.datatype;
		const int device_id = CCV_TENSOR_GET_DEVICE_ID(b->info.type);
		ncclComm_t comm = ccv_nnc_nccl_get_comm(stream_context, device_count, device_id);
		ccv_nnc_stream_context_t* const neighbor_context = stream_context ? ccv_nnc_stream_context_find_neighbor(stream_context, device_id) : 0;
		cudaStream_t stream = ccv_nnc_stream_context_get_stream(neighbor_context);
		NCCL_ENFORCE(ncclBroadcast(a->data.f32, b->data.f32, tensor_count, ccv_nnc_nccl_datatype(datatype), root, comm, stream));
	}
	NCCL_ENFORCE(ncclGroupEnd());
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_reduce_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 1);
	assert(output_size >= 1);
	int i, device_count = 0;
	assert(CCV_IS_TENSOR_CONTIGUOUS(outputs[0]));
	const size_t tensor_count = ccv_nnc_tensor_count(outputs[0]->info);
	for (i = 0; i < input_size; i++)
	{
		assert(CCV_IS_TENSOR_CONTIGUOUS(inputs[i]));
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
		assert(a->info.datatype == b->info.datatype);
		const int datatype = a->info.datatype;
		const int device_id = CCV_TENSOR_GET_DEVICE_ID(a->info.type);
		ncclComm_t comm = ccv_nnc_nccl_get_comm(stream_context, device_count, device_id);
		ccv_nnc_stream_context_t* const neighbor_context = stream_context ? ccv_nnc_stream_context_find_neighbor(stream_context, device_id) : 0;
		cudaStream_t stream = ccv_nnc_stream_context_get_stream(neighbor_context);
		NCCL_ENFORCE(ncclReduce(a->data.f32, b->data.f32, tensor_count, ccv_nnc_nccl_datatype(datatype), ncclSum, root, comm, stream));
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
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_broadcast_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_COMM_BROADCAST_BACKWARD, CCV_NNC_BACKEND_GPU_NCCL)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_NCCL
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_broadcast_back;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_COMM_REDUCE_FORWARD, CCV_NNC_BACKEND_GPU_NCCL)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_NCCL
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_reduce_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_COMM_REDUCE_BACKWARD, CCV_NNC_BACKEND_GPU_NCCL)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_NCCL
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_reduce_back;
#endif
}
