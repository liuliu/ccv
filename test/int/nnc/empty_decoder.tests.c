#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_TEARDOWN()
{
}

TEST_CASE("decoder operators infer empty outputs and do not touch storage on CPU or GPU")
{
	ccv_nnc_cmd_t commands[] = {
		CMD_HYPER_CONNECTION_FORWARD(4, 20, 1e-6),
		CMD_HYPER_CONNECTION_FORWARD(4, 20, 1e-6),
		CMD_HYPER_CONNECTION_FORWARD(4, 20, 1e-6),
		CMD_MOE_ROUTING_FORWARD(2, 1, 0),
		CMD_MOE_WEIGHTS_STREAMING_FORWARD(2, 2),
		CMD_SCALED_DOT_PRODUCT_ARG_PARTITION_FORWARD(4, 1, 1, 1, 16),
		CMD_SCALED_DOT_PRODUCT_ARG_PARTITION_FORWARD(4, 1, 1, 1, 16),
		CMD_SPARSE_INDEXED_ATTENTION_FORWARD(1, 1, 1),
		CMD_SWIGLU_FORWARD(10),
		CMD_SEGMENTED_SWIGLU_FORWARD(10),
		CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)),
		CMD_SCATTER_ADD_FORWARD(0, 0),
		CMD_SCATTER_ADD_FORWARD(8, 2),
	};
	const int input_sizes[] = {3, 4, 4, 3, 6, 3, 4, 7, 3, 6, 4, 2, 2};
	const int output_sizes[] = {3, 3, 1, 5, 6, 2, 1, 1, 1, 1, 1, 1, 1};
	commands[5].info.scaled_dot_product_arg_partition.candidate_block_size = 4;
	commands[5].info.scaled_dot_product_arg_partition.candidate_kth = 2;
	commands[6].info.scaled_dot_product_arg_partition.candidate_block_size = 4;
	commands[6].info.scaled_dot_product_arg_partition.candidate_kth = 2;
	const uint32_t backends[] = {CCV_NNC_BACKEND_CPU_REF, CCV_NNC_BACKEND_MPS, CCV_NNC_BACKEND_GPU_CUBLAS};
	int backend, i, j;
	for (backend = 0; backend < sizeof(backends) / sizeof(backends[0]); backend++)
		for (i = 0; i < sizeof(commands) / sizeof(commands[0]); i++)
		{
			ccv_nnc_cmd_t cmd = commands[i];
			cmd.backend = backends[backend];
			if (!ccv_nnc_cmd_ok(cmd.cmd, cmd.backend))
				continue; // Only exercise registered backends, including cuBLAS for segmented GEMM.
			ccv_nnc_tensor_param_t input_params[7], output_params[6];
			ccv_nnc_tensor_t input_headers[7], output_headers[6];
			ccv_nnc_tensor_t* inputs[7];
			ccv_nnc_tensor_t* outputs[6];
			ccv_nnc_tensor_t* storage[6];
			ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 16), 0);
			ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 16), 0);
			for (j = 0; j < 16; j++)
				expected->data.f32[j] = j + 0.25f;
			for (j = 0; j < input_sizes[i]; j++)
			{
				input_params[j] = backend ? GPU_TENSOR_NHWC(000, 32F, 0) : CPU_TENSOR_NHWC(32F, 0);
				// Empty routes must suppress staging even when the expert banks are nonempty.
				if (i == 4 && j >= 3)
					input_params[j] = backend ? GPU_TENSOR_NHWC(000, 32F, 4, 32, 32) : CPU_TENSOR_NHWC(32F, 4, 32, 32);
				input_headers[j] = (ccv_nnc_tensor_t){.info = input_params[j]};
				inputs[j] = &input_headers[j];
			}
			// Inputs deliberately have no storage. Empty ops must return before
			// accessing activations, expert banks, route plans, or attention caches.
			ccv_nnc_hint_tensor_auto(cmd, input_params, input_sizes[i], ccv_nnc_no_hint, output_params, output_sizes[i]);
			for (j = 0; j < output_sizes[i]; j++)
			{
				REQUIRE_EQ(ccv_nnc_tensor_count(output_params[j]), 0, "empty activation must infer an empty output");
				if (i == 3)
					REQUIRE_EQ(output_params[j].datatype, j < 2 ? CCV_32F : CCV_32S, "routing preserves activation/weight/index datatypes");
				if (i == 5 || i == 6)
					REQUIRE_EQ(output_params[j].datatype, CCV_32S, "both selection and candidate pool outputs contain indices");
				storage[j] = ccv_nnc_tensor_new(0, backend ? GPU_TENSOR_NHWC(000, 32F, 16) : CPU_TENSOR_NHWC(32F, 16), 0);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(expected), TENSOR_LIST(storage[j]), 0);
				output_headers[j] = *storage[j];
				output_headers[j].info = output_params[j];
				outputs[j] = &output_headers[j];
			}
			REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, inputs, input_sizes[i], outputs, output_sizes[i], 0), CCV_NNC_EXEC_SUCCESS, "empty forward must succeed");
			for (j = 0; j < output_sizes[i]; j++)
			{
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(storage[j]), TENSOR_LIST(actual), 0);
				REQUIRE_TENSOR_EQ(expected, actual, "empty forward must leave backing storage unchanged");
				ccv_nnc_tensor_free(storage[j]);
			}
			ccv_nnc_tensor_free(expected);
			ccv_nnc_tensor_free(actual);
		}
}

TEST_CASE("empty decoder operators reject invalid arity before accessing outputs")
{
	const ccv_nnc_cmd_t commands[] = {
		CMD_HYPER_CONNECTION_FORWARD(4, 20, 1e-6),
		CMD_MOE_ROUTING_FORWARD(2, 1, 0),
		CMD_MOE_WEIGHTS_STREAMING_FORWARD(2, 2),
	};
	const int input_sizes[] = {3, 3, 6};
	const int output_sizes[] = {3, 5, 6};
	const uint32_t backends[] = {CCV_NNC_BACKEND_CPU_REF, CCV_NNC_BACKEND_MPS};
	int backend, i, j;
	for (backend = 0; backend < sizeof(backends) / sizeof(backends[0]); backend++)
		for (i = 0; i < sizeof(commands) / sizeof(commands[0]); i++)
		{
			ccv_nnc_cmd_t cmd = commands[i];
			cmd.backend = backends[backend];
			if (!ccv_nnc_cmd_ok(cmd.cmd, cmd.backend))
				continue;
			ccv_nnc_tensor_t empty = {.info = backend ? GPU_TENSOR_NHWC(000, 32F, 0) : CPU_TENSOR_NHWC(32F, 0)};
			ccv_nnc_tensor_t* inputs[6];
			ccv_nnc_tensor_t* outputs[6];
			for (j = 0; j < 6; j++)
				inputs[j] = outputs[j] = &empty;
			REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, inputs, input_sizes[i], 0, 0, 0), CCV_NNC_EXEC_INVALID, "missing outputs must return invalid without dereferencing the output array");
			REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, inputs, input_sizes[i], outputs, 1, 0), CCV_NNC_EXEC_INVALID, "an empty output must not bypass output arity validation");
			REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, inputs, 1, outputs, output_sizes[i], 0), CCV_NNC_EXEC_INVALID, "an empty output must not bypass input arity validation");
			if (i == 2)
			{
				outputs[0] = 0;
				REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, inputs, input_sizes[i], outputs, output_sizes[i], 0), CCV_NNC_EXEC_INVALID, "streaming must reject a null first output before checking its shape");
			}
		}
}

TEST_CASE("empty decoder models retain their parameter slots and infer empty outputs")
{
	ccv_cnnp_model_t* const models[] = {
		ccv_cnnp_swiglu(32, 10, 0, "swiglu"),
		ccv_cnnp_segmented_swiglu(4, 32, 10, 0, 0, "segmented_swiglu"),
		ccv_cnnp_segmented_dense(4, 32, 1, 0, 0, 0, "segmented_dense"),
		ccv_cnnp_segmented_dense(4, 32, 0, 0, 0, 0, "segmented_dense_with_bias"),
		ccv_cnnp_scatter_add(0, 2, "scatter_add"),
		ccv_cnnp_scatter_add(8, 2, "scatter_add_with_bins"),
	};
	const int input_sizes[] = {1, 4, 3, 3, 2, 2};
	const int parameter_counts[] = {2, 2, 1, 2, 0, 0};
	const ccv_nnc_tensor_param_t inputs[] = {
		CPU_TENSOR_NHWC(32F, 0),
		CPU_TENSOR_NHWC(32S, 0),
		CPU_TENSOR_NHWC(32S, 0),
		CPU_TENSOR_NHWC(32F, 0),
	};
	int i;
	for (i = 0; i < sizeof(models) / sizeof(models[0]); i++)
	{
		ccv_cnnp_model_compile(models[i], inputs, input_sizes[i], CMD_NOOP(), CMD_NOOP());
		ccv_nnc_tensor_param_t output_params = {};
		ccv_cnnp_model_tensor_auto(models[i], &output_params, 1);
		REQUIRE_EQ(ccv_nnc_tensor_count(output_params), 0, "empty model must not infer an activation or index a missing feature dimension");
		REQUIRE_EQ(output_params.datatype, CCV_32F, "empty model preserves the activation datatype");
		REQUIRE_EQ(ccv_cnnp_model_parameter_count(models[i]), parameter_counts[i], "empty model retains the same parameter slots as an active model");
		ccv_cnnp_model_free(models[i]);
	}
}

#include "case_main.h"
