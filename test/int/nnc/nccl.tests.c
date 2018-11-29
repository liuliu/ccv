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

TEST_CASE("nccl with allreduce in block mode")
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

#include "case_main.h"
