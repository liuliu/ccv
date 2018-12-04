#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <3rdparty/dsfmt/dSFMT.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_CASE("nccl with allreduce in blocking mode")
{
	const int device_count = ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU);
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_COMM_ALLREDUCE_FORWARD, CCV_NNC_BACKEND_GPU_NCCL) && device_count > 1);
	ccv_nnc_tensor_t* tensors[device_count];
	int i;
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_tensor_param_t info = ONE_GPU_TENSOR(000, 100);
		CCV_TENSOR_SET_DEVICE_ID(info.type, i);
		tensors[i] = ccv_nnc_tensor_new(0, info, 0);
		ccv_nnc_cmd_exec(CMD_SET_FORWARD(i), ccv_nnc_no_hint, 0, 0, 0, &tensors[i], 1, 0);
	}
	ccv_nnc_cmd_exec(CMD_COMM_ALLREDUCE_FORWARD(), ccv_nnc_no_hint, 0, tensors, device_count, tensors, device_count, 0);
	ccv_nnc_tensor_t* cpu_tensors[device_count];
	for (i = 0; i < device_count; i++)
	{
		cpu_tensors[i] = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(100), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, &tensors[i], 1, &cpu_tensors[i], 1, 0);
	}
	ccv_nnc_tensor_t* demo_tensor = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(100), 0);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD((device_count - 1) * device_count / 2), ccv_nnc_no_hint, 0, 0, 0, &demo_tensor, 1, 0);
	for (i = 0; i < device_count; i++)
		REQUIRE_TENSOR_EQ(demo_tensor, cpu_tensors[i], "all values should be summed");
	ccv_nnc_tensor_free(demo_tensor);
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_tensor_free(tensors[i]);
		ccv_nnc_tensor_free(cpu_tensors[i]);
	}
}

TEST_CASE("nccl with broadcast in blocking mode")
{
	const int device_count = ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU);
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_COMM_BROADCAST_FORWARD, CCV_NNC_BACKEND_GPU_NCCL) && device_count > 1);
	ccv_nnc_tensor_t* tensors[device_count];
	int i;
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_tensor_param_t info = ONE_GPU_TENSOR(000, 100);
		CCV_TENSOR_SET_DEVICE_ID(info.type, i);
		tensors[i] = ccv_nnc_tensor_new(0, info, 0);
		ccv_nnc_cmd_exec(CMD_SET_FORWARD(i + 1), ccv_nnc_no_hint, 0, 0, 0, &tensors[i], 1, 0);
	}
	ccv_nnc_cmd_exec(CMD_COMM_BROADCAST_FORWARD(), ccv_nnc_no_hint, 0, tensors, 1, tensors, device_count, 0);
	ccv_nnc_tensor_t* cpu_tensors[device_count];
	for (i = 0; i < device_count; i++)
	{
		cpu_tensors[i] = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(100), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, &tensors[i], 1, &cpu_tensors[i], 1, 0);
	}
	ccv_nnc_tensor_t* demo_tensor = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(100), 0);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD(1), ccv_nnc_no_hint, 0, 0, 0, &demo_tensor, 1, 0);
	for (i = 0; i < device_count; i++)
		REQUIRE_TENSOR_EQ(demo_tensor, cpu_tensors[i], "all values should be summed");
	ccv_nnc_tensor_free(demo_tensor);
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_tensor_free(tensors[i]);
		ccv_nnc_tensor_free(cpu_tensors[i]);
	}
}

TEST_CASE("nccl with reduce in blocking mode")
{
	const int device_count = ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU);
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_COMM_REDUCE_FORWARD, CCV_NNC_BACKEND_GPU_NCCL) && device_count > 1);
	ccv_nnc_tensor_t* tensors[device_count];
	int i;
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_tensor_param_t info = ONE_GPU_TENSOR(000, 100);
		CCV_TENSOR_SET_DEVICE_ID(info.type, i);
		tensors[i] = ccv_nnc_tensor_new(0, info, 0);
		ccv_nnc_cmd_exec(CMD_SET_FORWARD(i + 1), ccv_nnc_no_hint, 0, 0, 0, &tensors[i], 1, 0);
	}
	ccv_nnc_cmd_exec(CMD_COMM_REDUCE_FORWARD(), ccv_nnc_no_hint, 0, tensors, device_count, tensors, 1, 0);
	ccv_nnc_tensor_t* cpu_tensor;
	cpu_tensor = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(100), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, &tensors[0], 1, &cpu_tensor, 1, 0);
	ccv_nnc_tensor_t* demo_tensor = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(100), 0);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD((device_count + 1) * device_count / 2), ccv_nnc_no_hint, 0, 0, 0, &demo_tensor, 1, 0);
	REQUIRE_TENSOR_EQ(demo_tensor, cpu_tensor, "all values should be summed");
	ccv_nnc_tensor_free(demo_tensor);
	ccv_nnc_tensor_free(cpu_tensor);
	for (i = 0; i < device_count; i++)
		ccv_nnc_tensor_free(tensors[i]);
}

static ccv_nnc_stream_context_t* _neighbor_discovery(const int device_id, void* const contexts)
{
	ccv_nnc_stream_context_t** stream_contexts = (ccv_nnc_stream_context_t**)contexts;
	return stream_contexts[device_id];
}

TEST_CASE("nccl with allreduce in non-blocking mode")
{
	const int device_count = ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU);
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_COMM_ALLREDUCE_FORWARD, CCV_NNC_BACKEND_GPU_NCCL) && device_count > 1);
	ccv_nnc_tensor_t* tensors[device_count];
	ccv_nnc_stream_context_t* contexts[device_count];
	int i;
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_tensor_param_t info = ONE_GPU_TENSOR(000, 100);
		CCV_TENSOR_SET_DEVICE_ID(info.type, i);
		tensors[i] = ccv_nnc_tensor_new(0, info, 0);
		ccv_nnc_cmd_exec(CMD_SET_FORWARD(i + 0.5), ccv_nnc_no_hint, 0, 0, 0, &tensors[i], 1, 0);
		int stream_type = CCV_STREAM_CONTEXT_GPU;
		CCV_STREAM_SET_DEVICE_ID(stream_type, i);
		contexts[i] = ccv_nnc_stream_context_new(stream_type);
	}
	ccv_nnc_stream_context_set_neighbor_discovery(contexts[0], _neighbor_discovery, contexts);
	ccv_nnc_cmd_exec(CMD_COMM_ALLREDUCE_FORWARD(), ccv_nnc_no_hint, 0, tensors, device_count, tensors, device_count, contexts[0]);
	ccv_nnc_tensor_t* cpu_tensors[device_count];
	for (i = 0; i < device_count; i++)
	{
		cpu_tensors[i] = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(100), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, &tensors[i], 1, &cpu_tensors[i], 1, contexts[i]);
	}
	ccv_nnc_tensor_t* demo_tensor = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(100), 0);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD((device_count - 1) * device_count / 2 + 0.5 * device_count), ccv_nnc_no_hint, 0, 0, 0, &demo_tensor, 1, 0);
	for (i = 0; i < device_count; i++)
		ccv_nnc_stream_context_wait(contexts[i]);
	for (i = 0; i < device_count; i++)
		REQUIRE_TENSOR_EQ(demo_tensor, cpu_tensors[i], "all values should be summed");
	ccv_nnc_tensor_free(demo_tensor);
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_tensor_free(tensors[i]);
		ccv_nnc_tensor_free(cpu_tensors[i]);
		ccv_nnc_stream_context_free(contexts[i]);
	}
}

TEST_CASE("nccl with broadcast in non-blocking mode")
{
	const int device_count = ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU);
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_COMM_BROADCAST_FORWARD, CCV_NNC_BACKEND_GPU_NCCL) && device_count > 1);
	ccv_nnc_tensor_t* tensors[device_count];
	ccv_nnc_stream_context_t* contexts[device_count];
	int i;
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_tensor_param_t info = ONE_GPU_TENSOR(000, 100);
		CCV_TENSOR_SET_DEVICE_ID(info.type, i);
		tensors[i] = ccv_nnc_tensor_new(0, info, 0);
		ccv_nnc_cmd_exec(CMD_SET_FORWARD(i + 1), ccv_nnc_no_hint, 0, 0, 0, &tensors[i], 1, 0);
		int stream_type = CCV_STREAM_CONTEXT_GPU;
		CCV_STREAM_SET_DEVICE_ID(stream_type, i);
		contexts[i] = ccv_nnc_stream_context_new(stream_type);
	}
	ccv_nnc_stream_context_set_neighbor_discovery(contexts[0], _neighbor_discovery, contexts);
	ccv_nnc_cmd_exec(CMD_COMM_BROADCAST_FORWARD(), ccv_nnc_no_hint, 0, tensors, 1, tensors, device_count, contexts[0]);
	ccv_nnc_tensor_t* cpu_tensors[device_count];
	for (i = 0; i < device_count; i++)
	{
		cpu_tensors[i] = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(100), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, &tensors[i], 1, &cpu_tensors[i], 1, contexts[i]);
	}
	ccv_nnc_tensor_t* demo_tensor = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(100), 0);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD(1), ccv_nnc_no_hint, 0, 0, 0, &demo_tensor, 1, 0);
	for (i = 0; i < device_count; i++)
		ccv_nnc_stream_context_wait(contexts[i]);
	for (i = 0; i < device_count; i++)
		REQUIRE_TENSOR_EQ(demo_tensor, cpu_tensors[i], "all values should be summed");
	ccv_nnc_tensor_free(demo_tensor);
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_tensor_free(tensors[i]);
		ccv_nnc_tensor_free(cpu_tensors[i]);
		ccv_nnc_stream_context_free(contexts[i]);
	}
}

TEST_CASE("nccl with reduce in non-blocking mode")
{
	const int device_count = ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU);
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_COMM_REDUCE_FORWARD, CCV_NNC_BACKEND_GPU_NCCL) && device_count > 1);
	ccv_nnc_tensor_t* tensors[device_count];
	ccv_nnc_stream_context_t* contexts[device_count];
	int i;
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_tensor_param_t info = ONE_GPU_TENSOR(000, 100);
		CCV_TENSOR_SET_DEVICE_ID(info.type, i);
		tensors[i] = ccv_nnc_tensor_new(0, info, 0);
		ccv_nnc_cmd_exec(CMD_SET_FORWARD(i + 1), ccv_nnc_no_hint, 0, 0, 0, &tensors[i], 1, 0);
		int stream_type = CCV_STREAM_CONTEXT_GPU;
		CCV_STREAM_SET_DEVICE_ID(stream_type, i);
		contexts[i] = ccv_nnc_stream_context_new(stream_type);
	}
	ccv_nnc_stream_context_set_neighbor_discovery(contexts[0], _neighbor_discovery, contexts);
	ccv_nnc_cmd_exec(CMD_COMM_REDUCE_FORWARD(), ccv_nnc_no_hint, 0, tensors, device_count, tensors, 1, contexts[0]);
	ccv_nnc_tensor_t* cpu_tensor;
	cpu_tensor = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(100), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, &tensors[0], 1, &cpu_tensor, 1, contexts[0]);
	ccv_nnc_tensor_t* demo_tensor = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(100), 0);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD((device_count + 1) * device_count / 2), ccv_nnc_no_hint, 0, 0, 0, &demo_tensor, 1, 0);
	ccv_nnc_stream_context_wait(contexts[0]);
	REQUIRE_TENSOR_EQ(demo_tensor, cpu_tensor, "all values should be summed");
	ccv_nnc_tensor_free(demo_tensor);
	ccv_nnc_tensor_free(cpu_tensor);
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_tensor_free(tensors[i]);
		ccv_nnc_stream_context_wait(contexts[i]);
		ccv_nnc_stream_context_free(contexts[i]);
	}
}

#include "case_main.h"
