#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#include "nnc/mps/ccv_nnc_mps.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#import <errno.h>
#import <fcntl.h>
#import <limits.h>
#import <stdio.h>
#import <stdatomic.h>
#import <stdint.h>
#import <stdlib.h>
#import <sys/stat.h>
#import <unistd.h>

typedef struct {
	size_t payload_per_expert;
	size_t scale_per_expert;
	size_t scale_offset;
	size_t size;
	ccv_nnc_tensor_param_t info;
} ccv_nnc_moe_weight_layout_t;

typedef struct {
	uint32_t generation;
	uint32_t desired_count;
	uint32_t load_count;
	uint32_t invalid;
} ccv_nnc_moe_gpu_plan_header_t;

@interface MFAMoEWeightsStreamingState : NSObject {
@public
	int resident_slot_count;
	int routing_width;
	int expert_count;
	int prefill;
	uint64_t generation;
	id source_objects[3];
	off_t source_dataofs[3];
	NSString* source_paths[3];
	off_t source_offsets[3];
	size_t source_region_sizes[3];
	int source_fds[3];
	unsigned char source_fd_owners[3];
	ccv_nnc_tensor_param_t source_infos[3];
	ccv_nnc_moe_weight_layout_t source_layouts[3];
	ccv_nnc_moe_weight_layout_t resident_layouts[3];
	id<MTLBuffer> resident_buffers[3];
	id<MTLBuffer> prefill_buffers[3];
	ccv_nnc_mfa_durable_scratch_t* prefill_leases[3];
	size_t gpu_plan_size;
	id<MTLBuffer> gpu_logical_to_slot_buffer;
	id<MTLBuffer> gpu_slot_to_logical_buffer;
	id<MTLBuffer> gpu_last_used_buffer;
	id<MTLBuffer> gpu_plan_buffer;
	id<MTLBuffer> gpu_route_generation_buffer;
	id<MTLBuffer> gpu_ready_generation_buffer;
	uint32_t gpu_last_encoded_generation;
}
@end

@implementation MFAMoEWeightsStreamingState
- (instancetype)init
{
	self = [super init];
	if (self)
	{
		int i;
		for (i = 0; i < 3; i++)
			source_fds[i] = -1;
	}
	return self;
}

- (void)dealloc
{
	int i;
	for (i = 0; i < 3; i++)
	{
		[source_objects[i] release];
		[source_paths[i] release];
		if (source_fd_owners[i] && source_fds[i] >= 0)
			close(source_fds[i]);
		[resident_buffers[i] release];
		if (prefill_leases[i])
			ccv_nnc_mfa_release_durable_scratch(prefill_leases[i]);
	}
	[gpu_logical_to_slot_buffer release];
	[gpu_slot_to_logical_buffer release];
	[gpu_last_used_buffer release];
	[gpu_plan_buffer release];
	[gpu_route_generation_buffer release];
	[gpu_ready_generation_buffer release];
	[super dealloc];
}
@end

static char ccv_nnc_moe_weights_streaming_state_key;
static char ccv_nnc_moe_weights_streaming_projection_keys[3];

static size_t _ccv_nnc_moe_align_up(const size_t value, const size_t alignment)
{
	return (value + alignment - 1) & ~(alignment - 1);
}

static size_t _ccv_nnc_moe_gpu_plan_size(const int expert_count)
{
	if (expert_count <= 0)
		return 0;
	return sizeof(ccv_nnc_moe_gpu_plan_header_t) +
		(size_t)expert_count * 4 * sizeof(int32_t);
}

static int32_t* _ccv_nnc_moe_plan_array(void* const plan, const int expert_count, const int array_index)
{
	return (int32_t*)((unsigned char*)plan + sizeof(ccv_nnc_moe_gpu_plan_header_t)) +
		(size_t)expert_count * array_index;
}

static int _ccv_nnc_moe_weight_layout(const ccv_nnc_tensor_param_t source_info, const int slots, ccv_nnc_moe_weight_layout_t* const layout)
{
	const int nd = ccv_nnc_tensor_nd(source_info.dim);
	if (nd != 3 || slots <= 0 || source_info.dim[0] <= 0 || source_info.dim[1] <= 0 || source_info.dim[2] <= 0)
		return 0;
	const int subtype = source_info.datatype & 0xf00;
	const int scale_datatype = (source_info.datatype & 0xff) << 12;
	const size_t scale_size = CCV_GET_DATA_TYPE_SIZE(scale_datatype);
	if (CCV_GET_DATA_TYPE(source_info.datatype) != CCV_QX || scale_size == 0)
		return 0;
	size_t payload_per_expert = 0;
	if (subtype == CCV_NNC_QX_8I_ROWWISE)
		payload_per_expert = (size_t)source_info.dim[1] * source_info.dim[2];
	else if (subtype == CCV_NNC_QX_8I_ROWWISE_X) {
		const size_t group_size = (size_t)ccv_nnc_8i_rowwise_x_group_size(source_info.reserved);
		const size_t group_bits = (size_t)ccv_nnc_8i_rowwise_x_group_bits(source_info.reserved);
		const size_t groups_per_row = ((size_t)source_info.dim[2] + group_size - 1) / group_size;
		const size_t payload_bits = (size_t)source_info.dim[1] * groups_per_row * group_bits;
		if (payload_bits & 7)
			return 0;
		payload_per_expert = payload_bits >> 3;
	} else
		return 0;
	const size_t scale_per_expert = (size_t)source_info.dim[1] * scale_size;
	const size_t payload_size = (size_t)slots * payload_per_expert;
	const size_t scale_size_total = (size_t)slots * scale_per_expert;
	const size_t scale_offset = _ccv_nnc_moe_align_up(payload_size, 128);
	ccv_nnc_tensor_param_t info = source_info;
	info.dim[0] = slots;
	const size_t size = scale_offset + scale_size_total;
	if (size != ccv_nnc_tensor_data_size_without_padding(info))
		return 0;
	*layout = (ccv_nnc_moe_weight_layout_t){
		.payload_per_expert = payload_per_expert,
		.scale_per_expert = scale_per_expert,
		.scale_offset = scale_offset,
		.size = size,
		.info = info,
	};
	return 1;
}

static int _ccv_nnc_moe_base_datatype(const ccv_nnc_tensor_param_t info)
{
	return CCV_GET_DATA_TYPE(info.datatype) == CCV_QX ?
		((info.datatype & 0xff) << 12) : info.datatype;
}

static int _ccv_nnc_moe_same_dims(const ccv_nnc_tensor_param_t a, const ccv_nnc_tensor_param_t b)
{
	return memcmp(a.dim, b.dim, sizeof(a.dim)) == 0;
}

static int _ccv_nnc_moe_tensor_range_valid(const ccv_nnc_tensor_t* const tensor, const size_t size)
{
	if (!tensor || CCV_TENSOR_GET_MEMORY(tensor->info.type) != CCV_TENSOR_GPU_MEMORY)
		return 0;
	id<MTLBuffer> const buffer = mpgetbuffer(tensor);
	const off_t offset = mpgetoffset(tensor);
	return buffer && offset >= 0 && (uint64_t)offset <= buffer.length &&
		size <= buffer.length - (uint64_t)offset;
}

static int _ccv_nnc_moe_validate(const ccv_nnc_cmd_t cmd,
	ccv_nnc_tensor_t* const* const inputs, const int input_size,
	ccv_nnc_tensor_t* const* const outputs, const int output_size)
{
	if (input_size != 6 || output_size != 6 ||
		cmd.info.moe_weights_streaming.resident_slots <= 0 ||
		cmd.info.moe_weights_streaming.routing_width <= 0 ||
		!ccv_nnc_mfa_context_supported(ccv_nnc_default_mfa_context()) ||
		(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA))
		return 0;
	int i;
	for (i = 0; i < 6; i++)
		if (!inputs[i] || !outputs[i] ||
			CCV_TENSOR_GET_MEMORY(inputs[i]->info.type) != CCV_TENSOR_GPU_MEMORY ||
			CCV_TENSOR_GET_MEMORY(outputs[i]->info.type) != CCV_TENSOR_GPU_MEMORY ||
			!CCV_IS_TENSOR_CONTIGUOUS(inputs[i]) || !CCV_IS_TENSOR_CONTIGUOUS(outputs[i]))
			return 0;
	if (inputs[0]->info.datatype != CCV_32S || inputs[1]->info.datatype != CCV_32S ||
		memcmp(&inputs[0]->info, &outputs[0]->info, sizeof(inputs[0]->info)) != 0 ||
		memcmp(&inputs[1]->info, &outputs[1]->info, sizeof(inputs[1]->info)) != 0 ||
		memcmp(&inputs[2]->info, &outputs[2]->info, sizeof(inputs[2]->info)) != 0)
		return 0;
	const size_t index_count = ccv_nnc_tensor_count(inputs[0]->info);
	const size_t count_count = ccv_nnc_tensor_count(inputs[1]->info);
	const size_t route_weight_count = ccv_nnc_tensor_count(inputs[2]->info);
	const size_t route_weight_size = ccv_nnc_tensor_data_size_without_padding(inputs[2]->info);
	if (index_count == 0 || index_count != count_count || route_weight_count == 0 ||
		index_count > UINT32_MAX || route_weight_count > UINT32_MAX ||
		route_weight_size == 0 || route_weight_size > UINT32_MAX)
		return 0;
	const ccv_nnc_tensor_param_t gate = inputs[3]->info;
	const ccv_nnc_tensor_param_t up = inputs[4]->info;
	const ccv_nnc_tensor_param_t down = inputs[5]->info;
	if (ccv_nnc_tensor_nd(gate.dim) != 3 || ccv_nnc_tensor_nd(up.dim) != 3 ||
		ccv_nnc_tensor_nd(down.dim) != 3 || gate.dim[0] <= 0 ||
		gate.dim[1] <= 0 || gate.dim[2] <= 0 ||
		!_ccv_nnc_moe_same_dims(gate, up) ||
		down.dim[0] != gate.dim[0] || down.dim[1] != gate.dim[2] ||
		down.dim[2] != gate.dim[1] || gate.format != up.format ||
		gate.format != down.format || gate.datatype != up.datatype ||
		gate.reserved != up.reserved ||
		cmd.info.moe_weights_streaming.resident_slots > gate.dim[0] ||
		index_count > (size_t)gate.dim[0])
		return 0;
	const int base_datatype = _ccv_nnc_moe_base_datatype(gate);
	if ((base_datatype != CCV_16F && base_datatype != CCV_16BF && base_datatype != CCV_32F) ||
		_ccv_nnc_moe_base_datatype(up) != base_datatype ||
		_ccv_nnc_moe_base_datatype(down) != base_datatype)
		return 0;
	for (i = 3; i < 6; i++)
	{
		const ccv_nnc_tensor_param_t handle = outputs[i]->info;
		if (!_ccv_nnc_moe_same_dims(inputs[i]->info, handle) ||
			handle.format != inputs[i]->info.format ||
			handle.type != inputs[i]->info.type ||
			CCV_GET_DATA_TYPE(handle.datatype) != CCV_QX ||
			(handle.datatype & 0xf00) != CCV_NNC_QX_EPHERMAL_STAGING ||
			_ccv_nnc_moe_base_datatype(handle) != _ccv_nnc_moe_base_datatype(inputs[i]->info))
			return 0;
	}
	for (i = 0; i < 3; i++)
		if (!_ccv_nnc_moe_tensor_range_valid(inputs[i],
				ccv_nnc_tensor_data_size_without_padding(inputs[i]->info)) ||
			!_ccv_nnc_moe_tensor_range_valid(outputs[i],
				ccv_nnc_tensor_data_size_without_padding(outputs[i]->info)))
			return 0;
	for (i = 3; i < 6; i++)
		if (!_ccv_nnc_moe_tensor_range_valid(outputs[i], 1))
			return 0;
	return 1;
}

static dispatch_queue_t _ccv_nnc_moe_weight_queue(void)
{
	static dispatch_queue_t queue;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		dispatch_queue_attr_t const attr = dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0);
		queue = dispatch_queue_create("ccv.nnc.moe-weights-streaming", attr);
	});
	return queue;
}

static MFAMoEWeightsStreamingState* _ccv_nnc_moe_weights_state(const ccv_nnc_tensor_t* const tensor)
{
	if (!tensor || CCV_TENSOR_GET_MEMORY(tensor->info.type) != CCV_TENSOR_GPU_MEMORY ||
		CCV_GET_DATA_TYPE(tensor->info.datatype) != CCV_QX ||
		(tensor->info.datatype & 0xf00) != CCV_NNC_QX_EPHERMAL_STAGING || !tensor->data.u8)
		return nil;
	id const object = (id)tensor->data.u8;
	int i;
	for (i = 0; i < 3; i++)
	{
		MFAMoEWeightsStreamingState* const state = objc_getAssociatedObject(
			object, &ccv_nnc_moe_weights_streaming_projection_keys[i]);
		if (state)
			return state;
	}
	return nil;
}

static void _ccv_nnc_moe_attach_state(MFAMoEWeightsStreamingState* const state, ccv_nnc_tensor_t* const* const outputs)
{
	int i;
	for (i = 0; i < 3; i++)
	{
		id object = (id)outputs[i + 3]->data.u8;
		int j;
		for (j = 0; j < 3; j++)
			objc_setAssociatedObject(object, &ccv_nnc_moe_weights_streaming_projection_keys[j],
				j == i ? state : nil, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
	}
}

static int _ccv_nnc_moe_state_matches(MFAMoEWeightsStreamingState* const state,
	ccv_nnc_tensor_t* const* const inputs, const int resident_slots, const int routing_width)
{
	if (!state || state->resident_slot_count != resident_slots || state->routing_width != routing_width)
		return 0;
	int i;
	for (i = 0; i < 3; i++)
		if (state->source_objects[i] != (id)inputs[i + 3]->data.u8 ||
			state->source_dataofs[i] != inputs[i + 3]->dataof ||
			memcmp(&state->source_infos[i], &inputs[i + 3]->info, sizeof(inputs[i + 3]->info)) != 0)
			return 0;
	return 1;
}

static MFAMoEWeightsStreamingState* _ccv_nnc_moe_state_for_inputs(ccv_nnc_tensor_t* const* const inputs,
	ccv_nnc_tensor_t* const* const outputs, const int resident_slots, const int routing_width)
{
	id const gate_source = (id)inputs[3]->data.u8;
	MFAMoEWeightsStreamingState* state = objc_getAssociatedObject(
		gate_source, &ccv_nnc_moe_weights_streaming_state_key);
	int i;
	if (!_ccv_nnc_moe_state_matches(state, inputs, resident_slots, routing_width))
	{
		state = [MFAMoEWeightsStreamingState new];
		state->resident_slot_count = resident_slots;
		state->routing_width = routing_width;
		state->expert_count = inputs[3]->info.dim[0];
		int invalid_source = 0;
		for (i = 0; i < 3; i++)
		{
			state->source_objects[i] = [(id)inputs[i + 3]->data.u8 retain];
			state->source_dataofs[i] = inputs[i + 3]->dataof;
			state->source_infos[i] = inputs[i + 3]->info;
			ccv_nnc_mps_file_backed_region_t region;
			const int has_region = ccv_nnc_mps_file_backed_region(inputs[i + 3], &region);
			const int has_source_layout = _ccv_nnc_moe_weight_layout(
				inputs[i + 3]->info, state->expert_count, &state->source_layouts[i]);
			const int has_resident_layout = _ccv_nnc_moe_weight_layout(
				inputs[i + 3]->info, resident_slots, &state->resident_layouts[i]);
			if (!has_region || !has_source_layout || !has_resident_layout ||
				region.size < state->source_layouts[i].size)
			{
				invalid_source = 1;
				break;
			}
			state->source_paths[i] = [region.path copy];
			state->source_offsets[i] = region.offset;
			state->source_region_sizes[i] = region.size;
		}
		if (invalid_source)
		{
			[state release];
			return nil;
		}
		ccv_nnc_mfa_prepare_moe_weights_streaming(ccv_nnc_default_mfa_context());
		id<MTLDevice> const device = ccv_nnc_default_device();
		for (i = 0; i < 3; i++)
			state->resident_buffers[i] = [device newBufferWithLength:state->resident_layouts[i].size options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
		state->gpu_plan_size = _ccv_nnc_moe_gpu_plan_size(state->expert_count);
		state->gpu_logical_to_slot_buffer = [device newBufferWithLength:sizeof(int32_t) * (size_t)state->expert_count options:MTLResourceStorageModePrivate];
		state->gpu_slot_to_logical_buffer = [device newBufferWithLength:sizeof(int32_t) * resident_slots options:MTLResourceStorageModePrivate];
		state->gpu_last_used_buffer = [device newBufferWithLength:sizeof(uint32_t) * resident_slots options:MTLResourceStorageModePrivate];
		state->gpu_plan_buffer = [device newBufferWithLength:state->gpu_plan_size options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
		state->gpu_route_generation_buffer = [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
		state->gpu_ready_generation_buffer = [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
		if (state->gpu_plan_size == 0 ||
			!state->resident_buffers[0] || !state->resident_buffers[1] || !state->resident_buffers[2] ||
			!state->gpu_logical_to_slot_buffer || !state->gpu_slot_to_logical_buffer ||
			!state->gpu_last_used_buffer || !state->gpu_plan_buffer ||
			!state->gpu_route_generation_buffer || !state->gpu_ready_generation_buffer)
		{
			[state release];
			return nil;
		}
		__atomic_store_n((uint32_t*)state->gpu_route_generation_buffer.contents, 0, __ATOMIC_RELEASE);
		__atomic_store_n((uint32_t*)state->gpu_ready_generation_buffer.contents, 0, __ATOMIC_RELEASE);
		objc_setAssociatedObject(gate_source, &ccv_nnc_moe_weights_streaming_state_key,
			state, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
		[state release];
	}
	_ccv_nnc_moe_attach_state(state, outputs);
	return state;
}

static int _ccv_nnc_moe_prepare_prefill_buffers(MFAMoEWeightsStreamingState* const state)
{
	int i;
	for (i = 0; i < 3; i++)
		if (state->prefill_leases[i])
			return 0;
	for (i = 0; i < 3; i++)
	{
		ccv_nnc_mfa_durable_scratch_t* const lease = ccv_nnc_mfa_request_durable_scratch(
			ccv_nnc_default_mfa_context(), state->source_layouts[i].size);
		id<MTLBuffer> const scratch = lease ?
			(__bridge id<MTLBuffer>)ccv_nnc_mfa_durable_scratch_buffer(lease) : nil;
		if (!lease || !scratch || !scratch.contents || scratch.length < state->source_layouts[i].size)
		{
			if (lease)
				ccv_nnc_mfa_release_durable_scratch(lease);
			int j;
			for (j = 0; j < i; j++)
			{
				ccv_nnc_mfa_release_durable_scratch(state->prefill_leases[j]);
				state->prefill_leases[j] = 0;
				state->prefill_buffers[j] = nil;
			}
			fprintf(stderr, "MoE weights durable scratch allocation failed (projection=%d, size=%zu)\n",
				i, state->source_layouts[i].size);
			return 0;
		}
		state->prefill_leases[i] = lease;
		state->prefill_buffers[i] = scratch;
	}
	return 1;
}


static int _ccv_nnc_moe_encode_gpu_plan(MFAMoEWeightsStreamingState* const state, ccv_nnc_tensor_t* const* const inputs, ccv_nnc_tensor_t* const* const outputs, const uint64_t generation, ccv_nnc_stream_context_t* const stream_context)
{
	if (generation == 0 || generation > UINT32_MAX)
		return 0;
	const size_t route_weight_bytes_size = ccv_nnc_tensor_data_size_without_padding(inputs[2]->info);
	if (route_weight_bytes_size > UINT32_MAX)
		return 0;
	const ccv_nnc_mfa_moe_weights_streaming_params_t params = {
		.generation = (uint32_t)generation,
		.index_count = (uint32_t)ccv_nnc_tensor_count(inputs[0]->info),
		.expert_count = (uint32_t)state->expert_count,
		.resident_slots = (uint32_t)state->resident_slot_count,
		.routing_width = (uint32_t)state->routing_width,
		.route_weight_count = (uint32_t)ccv_nnc_tensor_count(inputs[2]->info),
		.route_weight_bytes = (uint32_t)route_weight_bytes_size,
	};
	MTLCommandBatch* const command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
	mtl_buffer_t* tensors[12] = {
		(__bridge mtl_buffer_t*)mpgetbuffer(inputs[0]),
		(__bridge mtl_buffer_t*)mpgetbuffer(inputs[1]),
		(__bridge mtl_buffer_t*)mpgetbuffer(inputs[2]),
		(__bridge mtl_buffer_t*)mpgetbuffer(outputs[0]),
		(__bridge mtl_buffer_t*)mpgetbuffer(outputs[1]),
		(__bridge mtl_buffer_t*)mpgetbuffer(outputs[2]),
		(__bridge mtl_buffer_t*)state->gpu_logical_to_slot_buffer,
		(__bridge mtl_buffer_t*)state->gpu_slot_to_logical_buffer,
		(__bridge mtl_buffer_t*)state->gpu_last_used_buffer,
		(__bridge mtl_buffer_t*)state->gpu_plan_buffer,
		(__bridge mtl_buffer_t*)state->gpu_ready_generation_buffer,
		0,
	};
	size_t tensor_offsets[11] = {
		(size_t)mpgetoffset(inputs[0]),
		(size_t)mpgetoffset(inputs[1]),
		(size_t)mpgetoffset(inputs[2]),
		(size_t)mpgetoffset(outputs[0]),
		(size_t)mpgetoffset(outputs[1]),
		(size_t)mpgetoffset(outputs[2]),
		0, 0, 0, 0, 0,
	};
	ccv_nnc_mfa_encode_moe_weights_streaming(
		ccv_nnc_default_mfa_context(), params, command_batch, tensors, tensor_offsets);
	const int encoded = ccv_nnc_mps_encode_fast_fence_signal_in_command_batch(
		command_batch, state->gpu_route_generation_buffer, 0,
		state->gpu_last_encoded_generation, (uint32_t)generation,
		state->gpu_plan_buffer, 0, state->gpu_plan_size);
	ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
	if (!encoded)
	{
		ccv_nnc_stream_context_wait(stream_context);
		return 0;
	}
	state->gpu_last_encoded_generation = (uint32_t)generation;
	ccv_nnc_stream_context_commit(stream_context);
	return 1;
}
static int _ccv_nnc_moe_pread_all(const int fd, const off_t source_offset, void* const destination, const size_t size)
{
	if (fd < 0 || source_offset < 0 || !destination)
		return 0;
#ifdef F_RDADVISE
	struct radvisory advice = { .ra_offset = source_offset, .ra_count = (int)ccv_min(size, (size_t)INT_MAX) };
	(void)fcntl(fd, F_RDADVISE, &advice);
#endif
	size_t completed = 0;
	while (completed < size)
	{
		const ssize_t result = pread(fd, (unsigned char*)destination + completed, size - completed, source_offset + (off_t)completed);
		if (result > 0)
			completed += (size_t)result;
		else if (result < 0 && errno == EINTR)
			continue;
		else
			return 0;
	}
	return 1;
}

static int _ccv_nnc_moe_open_source_fds(MFAMoEWeightsStreamingState* const state)
{
	int i;
	for (i = 0; i < 3; i++)
		if (state->source_fds[i] < 0)
			break;
	if (i == 3)
		return 1;
	for (i = 0; i < 3; i++)
	{
		if (state->source_fd_owners[i] && state->source_fds[i] >= 0)
			close(state->source_fds[i]);
		state->source_fds[i] = -1;
		state->source_fd_owners[i] = 0;
	}
	for (i = 0; i < 3; i++)
	{
		int matching_projection;
		for (matching_projection = 0; matching_projection < i; matching_projection++)
			if ([state->source_paths[i] isEqualToString:state->source_paths[matching_projection]])
				break;
		const int shared_fd = matching_projection < i;
		const int fd = shared_fd ? state->source_fds[matching_projection] :
			open(state->source_paths[i].fileSystemRepresentation, O_RDONLY);
		if (fd < 0)
			break;
		state->source_fds[i] = fd;
		state->source_fd_owners[i] = !shared_fd;
		struct stat status;
		if (fstat(fd, &status) != 0 || state->source_offsets[i] < 0 ||
			(uint64_t)state->source_offsets[i] > (uint64_t)status.st_size ||
			state->source_layouts[i].size > (uint64_t)status.st_size - (uint64_t)state->source_offsets[i] ||
			state->source_layouts[i].size > state->source_region_sizes[i])
			break;
	}
	if (i == 3)
		return 1;
	for (i = 0; i < 3; i++)
	{
		if (state->source_fd_owners[i] && state->source_fds[i] >= 0)
			close(state->source_fds[i]);
		state->source_fds[i] = -1;
		state->source_fd_owners[i] = 0;
	}
	return 0;
}

static int _ccv_nnc_moe_read_expert_span_to(MFAMoEWeightsStreamingState* const state, const int projection,
	const int expert, const int expert_count, id<MTLBuffer> const destination_buffer,
	const ccv_nnc_moe_weight_layout_t destination_layout, const int destination_slot)
{
	if (state->source_fds[projection] < 0)
		return 0;
	const ccv_nnc_moe_weight_layout_t source_layout = state->source_layouts[projection];
	if (expert < 0 || expert_count <= 0 || destination_slot < 0 ||
		expert > state->expert_count - expert_count ||
		destination_slot > destination_layout.info.dim[0] - expert_count)
	{
		fprintf(stderr, "MoE weights pread destination invalid (projection=%d, expert=%d, count=%d, slot=%d)\n",
			projection, expert, expert_count, destination_slot);
		return 0;
	}
	const size_t payload_size = (size_t)expert_count * destination_layout.payload_per_expert;
	const size_t scale_size = (size_t)expert_count * destination_layout.scale_per_expert;
	const size_t payload_start = (size_t)destination_slot * destination_layout.payload_per_expert;
	const size_t scale_start = destination_layout.scale_offset +
		(size_t)destination_slot * destination_layout.scale_per_expert;
	if (!destination_buffer.contents || payload_start > destination_buffer.length ||
		payload_size > destination_buffer.length - payload_start ||
		scale_start > destination_buffer.length || scale_size > destination_buffer.length - scale_start)
	{
		fprintf(stderr, "MoE weights pread destination invalid (projection=%d, expert=%d, count=%d, slot=%d)\n",
			projection, expert, expert_count, destination_slot);
		return 0;
	}
	const size_t source_payload_delta = (size_t)expert * source_layout.payload_per_expert;
	const size_t source_scale_delta = source_layout.scale_offset +
		(size_t)expert * source_layout.scale_per_expert;
	if (source_payload_delta > INT64_MAX || source_scale_delta > INT64_MAX ||
		(uint64_t)state->source_offsets[projection] > (uint64_t)INT64_MAX - source_payload_delta ||
		(uint64_t)state->source_offsets[projection] > (uint64_t)INT64_MAX - source_scale_delta)
		return 0;
	const off_t source_payload = state->source_offsets[projection] + (off_t)source_payload_delta;
	const off_t source_scale = state->source_offsets[projection] + (off_t)source_scale_delta;
	unsigned char* const destination = destination_buffer.contents;
	if (!_ccv_nnc_moe_pread_all(state->source_fds[projection], source_payload,
		destination + payload_start, payload_size))
	{
		fprintf(stderr, "MoE weights payload pread failed (projection=%d, expert=%d, count=%d, offset=%lld, size=%zu, errno=%d, path=%s)\n",
			projection, expert, expert_count, (long long)source_payload,
			payload_size, errno, state->source_paths[projection].fileSystemRepresentation);
		return 0;
	}
	if (!_ccv_nnc_moe_pread_all(state->source_fds[projection], source_scale,
		destination + scale_start, scale_size))
	{
		fprintf(stderr, "MoE weights scale pread failed (projection=%d, expert=%d, count=%d, offset=%lld, size=%zu, errno=%d, path=%s)\n",
			projection, expert, expert_count, (long long)source_scale,
			scale_size, errno, state->source_paths[projection].fileSystemRepresentation);
		return 0;
	}
	return 1;
}

static int _ccv_nnc_moe_read_expert_to(MFAMoEWeightsStreamingState* const state, const int projection,
	const int expert, id<MTLBuffer> const destination_buffer,
	const ccv_nnc_moe_weight_layout_t destination_layout, const int destination_slot)
{
	return _ccv_nnc_moe_read_expert_span_to(state, projection, expert, 1,
		destination_buffer, destination_layout, destination_slot);
}

static int _ccv_nnc_moe_read_expert(MFAMoEWeightsStreamingState* const state, const int projection, const int expert, const int resident_slot)
{
	return _ccv_nnc_moe_read_expert_to(state, projection, expert,
		state->resident_buffers[projection], state->resident_layouts[projection], resident_slot);
}

static int _ccv_nnc_moe_read_prefill_expert_span(MFAMoEWeightsStreamingState* const state,
	const int projection, const int expert, const int expert_count, const int prefill_slot,
	id<MTLBuffer> const prefill_buffer)
{
	return _ccv_nnc_moe_read_expert_span_to(state, projection, expert, expert_count,
		prefill_buffer, state->source_layouts[projection], prefill_slot);
}

static int _ccv_nnc_moe_copy_resident_to_prefill(MFAMoEWeightsStreamingState* const state,
	const int projection, const int resident_slot, const int prefill_slot,
	id<MTLBuffer> const prefill_buffer)
{
	const ccv_nnc_moe_weight_layout_t resident_layout = state->resident_layouts[projection];
	const ccv_nnc_moe_weight_layout_t prefill_layout = state->source_layouts[projection];
	const unsigned char* const source = state->resident_buffers[projection].contents;
	unsigned char* const destination = prefill_buffer.contents;
	if (!source || !destination || resident_slot < 0 || resident_slot >= state->resident_slot_count ||
		prefill_slot < 0 || prefill_slot >= state->expert_count)
		return 0;
	memcpy(destination + (size_t)prefill_slot * prefill_layout.payload_per_expert,
		source + (size_t)resident_slot * resident_layout.payload_per_expert,
		prefill_layout.payload_per_expert);
	memcpy(destination + prefill_layout.scale_offset + (size_t)prefill_slot * prefill_layout.scale_per_expert,
		source + resident_layout.scale_offset + (size_t)resident_slot * resident_layout.scale_per_expert,
		prefill_layout.scale_per_expert);
	return 1;
}

static void _ccv_nnc_moe_publish_ready_generation(MFAMoEWeightsStreamingState* const state, const uint32_t generation)
{
	uint32_t* const ready_generation = (uint32_t*)state->gpu_ready_generation_buffer.contents;
	uint32_t ready = __atomic_load_n(ready_generation, __ATOMIC_ACQUIRE);
	while (ready < generation && !__atomic_compare_exchange_n(
		ready_generation, &ready, generation, 1, __ATOMIC_RELEASE, __ATOMIC_ACQUIRE)) {}
}

static void _ccv_nnc_moe_load_gpu_plan(MFAMoEWeightsStreamingState* const state, const uint64_t generation,
	const int expert_count, const int prefill, id<MTLBuffer> const prefill_gate_buffer,
	id<MTLBuffer> const prefill_up_buffer, id<MTLBuffer> const prefill_down_buffer)
{
	id<MTLBuffer> const prefill_buffers[3] = {
		prefill_gate_buffer, prefill_up_buffer, prefill_down_buffer,
	};
	uint32_t* const route_generation = (uint32_t*)state->gpu_route_generation_buffer.contents;
	while (__atomic_load_n(route_generation, __ATOMIC_ACQUIRE) < (uint32_t)generation) {}
	uint32_t plan_storage[state->gpu_plan_size / sizeof(uint32_t)];
	memcpy(plan_storage, state->gpu_plan_buffer.contents, state->gpu_plan_size);
	ccv_nnc_moe_gpu_plan_header_t* const plan = (ccv_nnc_moe_gpu_plan_header_t*)plan_storage;
	// A later all-hit generation can reuse this state's single mailbox before an
	// older CPU task runs. A miss cannot be passed because its GPU consumer waits
	// for this worker, so an overtaken task needs no I/O or readiness publication.
	if (generation <= UINT32_MAX && plan->generation > (uint32_t)generation)
		return;
	int32_t* const desired_experts = _ccv_nnc_moe_plan_array(plan_storage, expert_count, 0);
	int32_t* const desired_slots = _ccv_nnc_moe_plan_array(plan_storage, expert_count, 1);
	int32_t* const load_experts = _ccv_nnc_moe_plan_array(plan_storage, expert_count, 2);
	int32_t* const load_slots = _ccv_nnc_moe_plan_array(plan_storage, expert_count, 3);
	int valid = generation <= UINT32_MAX && plan->generation == (uint32_t)generation && !plan->invalid &&
		plan->desired_count <= (uint32_t)(prefill ? expert_count : state->resident_slot_count) &&
		plan->load_count <= plan->desired_count;
	uint32_t i;
	for (i = 0; i < plan->desired_count && valid; i++)
		if (desired_experts[i] < 0 || desired_experts[i] >= expert_count ||
			desired_slots[i] < -1 || desired_slots[i] >= state->resident_slot_count ||
			(!prefill && desired_slots[i] < 0))
			valid = 0;
	for (i = 0; i < plan->load_count && valid; i++)
	{
		if (load_experts[i] < 0 || load_experts[i] >= expert_count ||
			load_slots[i] < 0 || load_slots[i] >= (prefill ? expert_count : state->resident_slot_count))
		{
			valid = 0;
			break;
		}
		uint32_t j;
		for (j = 0; j < i; j++)
			if (load_experts[j] == load_experts[i] || load_slots[j] == load_slots[i])
			{
				valid = 0;
				break;
			}
	}
	if (!valid)
	{
		fprintf(stderr, "MoE weights GPU plan validation failed (generation=%llu)\n",
			(unsigned long long)generation);
		_ccv_nnc_moe_publish_ready_generation(state, (uint32_t)generation);
		return;
	}
	int projection;
	const int source_open_failed = plan->load_count > 0 && !_ccv_nnc_moe_open_source_fds(state);
	dispatch_group_t const read_group = dispatch_group_create();
	dispatch_queue_t const read_queue = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
	__block _Atomic(int) read_failed = source_open_failed;
	int read_task_count = 0;
	if (prefill)
		for (i = 0; i < plan->desired_count; i++)
		{
			const int resident_slot = desired_slots[i];
			if (resident_slot < 0)
				continue;
			for (projection = 0; projection < 3; projection++)
			{
				const int copy_projection = projection;
				const int copy_resident_slot = resident_slot;
				const int prefill_slot = (int)i;
				id<MTLBuffer> const prefill_buffer = prefill_buffers[projection];
				dispatch_group_async(read_group, read_queue, ^{
					if (!_ccv_nnc_moe_copy_resident_to_prefill(
						state, copy_projection, copy_resident_slot, prefill_slot, prefill_buffer))
						atomic_store(&read_failed, 1);
				});
				read_task_count++;
			}
		}
	if (prefill && !source_open_failed)
	{
		i = 0;
		while (i < plan->load_count)
		{
			const int read_expert = load_experts[i];
			const int read_slot = load_slots[i];
			uint32_t run_length = 1;
			while (i + run_length < plan->load_count &&
				load_experts[i + run_length] == read_expert + (int)run_length &&
				load_slots[i + run_length] == read_slot + (int)run_length)
				run_length++;
			for (projection = 0; projection < 3; projection++)
			{
				const int read_projection = projection;
				const int read_count = (int)run_length;
				id<MTLBuffer> const prefill_buffer = prefill_buffers[projection];
				dispatch_group_async(read_group, read_queue, ^{
					if (!_ccv_nnc_moe_read_prefill_expert_span(
						state, read_projection, read_expert, read_count, read_slot, prefill_buffer))
						atomic_store(&read_failed, 1);
				});
				read_task_count++;
			}
			i += run_length;
		}
	} else if (!source_open_failed)
		for (i = 0; i < plan->load_count; i++)
			for (projection = 0; projection < 3; projection++)
			{
				const int read_projection = projection;
				const int read_expert = load_experts[i];
				const int read_slot = load_slots[i];
				dispatch_group_async(read_group, read_queue, ^{
					if (!_ccv_nnc_moe_read_expert(state, read_projection, read_expert, read_slot))
						atomic_store(&read_failed, 1);
				});
				read_task_count++;
			}
	if (read_task_count > 0)
		dispatch_group_wait(read_group, DISPATCH_TIME_FOREVER);
	dispatch_release(read_group);
	if (atomic_load(&read_failed))
		fprintf(stderr, "MoE weights asynchronous load failed (generation=%llu)\n",
			(unsigned long long)generation);
	const int cpu_must_publish = prefill || plan->load_count > 0;
	if (cpu_must_publish)
		_ccv_nnc_moe_publish_ready_generation(state, (uint32_t)generation);
}

static int _ccv_nnc_moe_weights_streaming_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	if (!_ccv_nnc_moe_validate(cmd, inputs, input_size, outputs, output_size))
		return CCV_NNC_EXEC_INVALID;
	MFAMoEWeightsStreamingState* const state = _ccv_nnc_moe_state_for_inputs(
		inputs, outputs, cmd.info.moe_weights_streaming.resident_slots,
		cmd.info.moe_weights_streaming.routing_width);
	if (!state)
		return CCV_NNC_EXEC_INVALID;
	const size_t route_weight_count = ccv_nnc_tensor_count(inputs[2]->info);
	const int prefill = route_weight_count != (size_t)cmd.info.moe_weights_streaming.routing_width;
	if (prefill && !_ccv_nnc_moe_prepare_prefill_buffers(state))
		return CCV_NNC_EXEC_INVALID;
	if (state->generation >= UINT32_MAX)
		return CCV_NNC_EXEC_INVALID;
	const uint64_t generation = ++state->generation;
	state->prefill = prefill;
	if (!_ccv_nnc_moe_encode_gpu_plan(state, inputs, outputs, generation, stream_context))
		return CCV_NNC_EXEC_INVALID;
	id<MTLBuffer> const prefill_gate_buffer = state->prefill_buffers[0];
	id<MTLBuffer> const prefill_up_buffer = state->prefill_buffers[1];
	id<MTLBuffer> const prefill_down_buffer = state->prefill_buffers[2];
	dispatch_async(_ccv_nnc_moe_weight_queue(), ^{
		@autoreleasepool {
			_ccv_nnc_moe_load_gpu_plan(state, generation, state->expert_count, prefill,
				prefill_gate_buffer, prefill_up_buffer, prefill_down_buffer);
		}
	});
	return CCV_NNC_EXEC_SUCCESS;
}

void ccv_nnc_mps_moe_weights_encode_wait(const ccv_nnc_tensor_t* const tensor, MTLCommandBatch* const command_batch)
{
	MFAMoEWeightsStreamingState* const state = _ccv_nnc_moe_weights_state(tensor);
	if (!state)
		return;
	const uint64_t generation = state->generation;
	if (!command_batch || generation == 0 || generation > UINT32_MAX)
		return;
	// The GPU planner publishes ready_generation on an all-hit decode. The CPU
	// loader publishes the same generation after staging misses or prefill, so
	// every consumer can use the same unconditional wait.
	ccv_nnc_mps_encode_fast_fence_wait_in_command_batch(command_batch,
		state->gpu_ready_generation_buffer, 0, (uint32_t)generation);
}

int ccv_nnc_mps_moe_weights_resolve(const ccv_nnc_tensor_t* const tensor, ccv_nnc_mps_moe_weights_view_t* const view)
{
	MFAMoEWeightsStreamingState* const state = _ccv_nnc_moe_weights_state(tensor);
	if (!state)
		return 0;
	if (!view)
		return -1;
	id const object = (id)tensor->data.u8;
	int projection;
	for (projection = 0; projection < 3; projection++)
		if (objc_getAssociatedObject(object, &ccv_nnc_moe_weights_streaming_projection_keys[projection]) == state)
			break;
	if (projection == 3)
		return -1;
	id<MTLBuffer> const buffer = state->prefill ?
		state->prefill_buffers[projection] : state->resident_buffers[projection];
	if (!buffer)
		return -1;
	view->buffer = buffer;
	view->offset = 0;
	view->info = state->prefill ?
		state->source_infos[projection] : state->resident_layouts[projection].info;
	return 1;
}

static ccv_nnc_mfa_durable_scratch_t* _ccv_nnc_moe_take_prefill_lease(const ccv_nnc_tensor_t* const tensor)
{
	MFAMoEWeightsStreamingState* const state = _ccv_nnc_moe_weights_state(tensor);
	if (!state)
		return 0;
	id const object = (id)tensor->data.u8;
	int projection;
	for (projection = 0; projection < 3; projection++)
		if (objc_getAssociatedObject(object, &ccv_nnc_moe_weights_streaming_projection_keys[projection]) == state)
			break;
	if (projection == 3)
		return 0;
	ccv_nnc_mfa_durable_scratch_t* const lease = state->prefill_leases[projection];
	state->prefill_leases[projection] = 0;
	state->prefill_buffers[projection] = nil;
	return lease;
}

void ccv_nnc_mps_moe_weights_finish_command_batch(ccv_nnc_tensor_t* const* const tensors, const int tensor_count, ccv_nnc_stream_context_t* const stream_context, MTLCommandBatch* const command_batch)
{
	ccv_nnc_mfa_durable_scratch_t* failed[tensor_count > 0 ? tensor_count : 1];
	int failed_count = 0;
	int i;
	for (i = 0; i < tensor_count; i++)
	{
		ccv_nnc_mfa_durable_scratch_t* const lease = _ccv_nnc_moe_take_prefill_lease(tensors[i]);
		if (lease && !ccv_nnc_mfa_retire_durable_scratch(lease, command_batch))
			failed[failed_count++] = lease;
	}
	ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
	if (failed_count > 0)
	{
		// A failed retirement did not consume the lease. Wait before returning it
		// because CPU reuse is not ordered by Metal's GPU hazard tracking.
		ccv_nnc_stream_context_wait(stream_context);
		for (i = 0; i < failed_count; i++)
			ccv_nnc_mfa_release_durable_scratch(failed[i]);
	}
}

REGISTER_COMMAND_BACKEND(CCV_NNC_MOE_WEIGHTS_STREAMING_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32S | CCV_16F | CCV_32F | CCV_16BF | CCV_QX;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_moe_weights_streaming_forw;
}
