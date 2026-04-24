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

TEST_CASE("cnnp send to devices and all-to-all")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_COMM_ALL_TO_ALL_FORWARD, CCV_NNC_BACKEND_GPU_NCCL));
	const int device_count = ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU);
	GUARD_ELSE_RETURN(device_count > 1);
	const int chunk = 5;
	const ccv_cnnp_model_io_t x = ccv_cnnp_input();
	ccv_cnnp_model_io_t chunks = ccv_cnnp_model_apply(ccv_cnnp_chunk(device_count, 0, "chunk"), MODEL_IO_LIST(x));
	ccv_cnnp_model_io_t sent[device_count];
	int i, j, k;
	for (i = 0; i < device_count; i++)
	{
		ccv_cnnp_model_io_t chunk_i = ccv_cnnp_model_apply(ccv_cnnp_extract(i, 0), &chunks, 1);
		sent[i] = ccv_cnnp_model_apply(ccv_cnnp_send(i, 0), &chunk_i, 1);
	}
	ccv_cnnp_model_io_t exchanged = ccv_cnnp_model_apply(ccv_cnnp_all_to_all(device_count, 1, "all_to_all"), sent, device_count);
	ccv_cnnp_model_t* const model = ccv_cnnp_model_new(MODEL_IO_LIST(x), &exchanged, 1, 0, "send_all_to_all");
	ccv_nnc_tensor_param_t input_params = GPU_TENSOR_NHWC(000, 32F, device_count, device_count * chunk);
	ccv_cnnp_model_compile(model, TENSOR_PARAM_LIST(input_params), CMD_NOOP(), CMD_NOOP());
	ccv_nnc_tensor_param_t output_params[device_count];
	ccv_cnnp_model_tensor_auto(model, output_params, device_count);
	ccv_nnc_tensor_t* const h_input = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, device_count, device_count * chunk), 0);
	for (i = 0; i < device_count; i++)
		for (j = 0; j < device_count * chunk; j++)
			h_input->data.f32[i * device_count * chunk + j] = (float)(i * 1000 + j);
	ccv_nnc_tensor_t* const d_input = ccv_nnc_tensor_new(0, input_params, 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(h_input), TENSOR_LIST(d_input), 0);
	ccv_nnc_tensor_t* outputs[device_count];
	for (i = 0; i < device_count; i++)
	{
		REQUIRE_EQ(CCV_TENSOR_GET_DEVICE_ID(output_params[i].type), i, "output device should match all-to-all rank");
		outputs[i] = ccv_nnc_tensor_new(0, output_params[i], 0);
	}
	ccv_cnnp_model_evaluate(model, (ccv_cnnp_evaluate_param_t){}, TENSOR_LIST(d_input), outputs, device_count, 0, 0);
	for (j = 0; j < device_count; j++)
	{
		ccv_nnc_tensor_t* const h_output = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, device_count * chunk), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(outputs[j]), TENSOR_LIST(h_output), 0);
		for (i = 0; i < device_count; i++)
			for (k = 0; k < chunk; k++)
			{
				const float expected = h_input->data.f32[i * device_count * chunk + j * chunk + k];
				REQUIRE_EQ_WITH_TOLERANCE(h_output->data.f32[i * chunk + k], expected, 1e-5, "all-to-all output should match expected exchange");
			}
		ccv_nnc_tensor_free(h_output);
	}
	for (i = 0; i < device_count; i++)
		ccv_nnc_tensor_free(outputs[i]);
	ccv_nnc_tensor_free(d_input);
	ccv_nnc_tensor_free(h_input);
	ccv_cnnp_model_free(model);
}

TEST_CASE("cnnp replicated dense feeds all-to-all")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_COMM_ALL_TO_ALL_FORWARD, CCV_NNC_BACKEND_GPU_NCCL));
	const int device_count = ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU);
	GUARD_ELSE_RETURN(device_count > 1);
	const int rank_count = device_count;
	const int rows_per_rank = 2;
	const int input_dim = 4;
	const int output_chunk = 3;
	const int output_dim = rank_count * output_chunk;
	const ccv_cnnp_model_io_t x = ccv_cnnp_input();
	ccv_cnnp_model_io_t chunks = ccv_cnnp_model_apply(ccv_cnnp_chunk(rank_count, 0, "chunk"), MODEL_IO_LIST(x));
	ccv_cnnp_model_io_t sent[rank_count];
	int i, j, k, r;
	for (i = 0; i < rank_count; i++)
	{
		ccv_cnnp_model_io_t chunk_i = ccv_cnnp_model_apply(ccv_cnnp_extract(i, 0), &chunks, 1);
		sent[i] = ccv_cnnp_model_apply(ccv_cnnp_send(i, 0), &chunk_i, 1);
	}
	ccv_cnnp_model_t* const dense = ccv_cnnp_dense(output_dim, 1, 0, 1, "dense");
	ccv_cnnp_model_io_t dense_outputs = ccv_cnnp_model_apply(ccv_cnnp_replicated(dense, rank_count, 1, "replicated_dense"), sent, rank_count);
	ccv_cnnp_model_io_t exchanged = ccv_cnnp_model_apply(ccv_cnnp_all_to_all(rank_count, 1, "all_to_all"), MODEL_IO_LIST(dense_outputs));
	ccv_cnnp_model_t* const model = ccv_cnnp_model_new(MODEL_IO_LIST(x), &exchanged, 1, 0, "replicated_dense_all_to_all");
	ccv_nnc_tensor_param_t input_params = GPU_TENSOR_NHWC(000, 32F, rank_count * rows_per_rank, input_dim);
	ccv_cnnp_model_compile(model, TENSOR_PARAM_LIST(input_params), CMD_NOOP(), CMD_NOOP());
	ccv_nnc_tensor_t* const h_input = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rank_count * rows_per_rank, input_dim), 0);
	for (i = 0; i < rank_count * rows_per_rank; i++)
		for (j = 0; j < input_dim; j++)
			h_input->data.f32[i * input_dim + j] = (float)(i * 10 + j + 1);
	ccv_nnc_tensor_t* const h_weight = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_dim, input_dim), 0);
	for (i = 0; i < output_dim; i++)
		for (j = 0; j < input_dim; j++)
			h_weight->data.f32[i * input_dim + j] = (float)((i + 1) * 0.25 + (j + 1) * 0.125);
	ccv_cnnp_model_set_parameter(model, ccv_cnnp_model_parameters(dense, ALL_PARAMETERS, 0), h_weight);
	ccv_nnc_tensor_t* const d_input = ccv_nnc_tensor_new(0, input_params, 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(h_input), TENSOR_LIST(d_input), 0);
	ccv_nnc_tensor_param_t output_params[rank_count];
	ccv_cnnp_model_tensor_auto(model, output_params, rank_count);
	ccv_nnc_tensor_t* outputs[rank_count];
	for (i = 0; i < rank_count; i++)
	{
		REQUIRE_EQ(CCV_TENSOR_GET_DEVICE_ID(output_params[i].type), i, "output device should match all-to-all rank");
		outputs[i] = ccv_nnc_tensor_new(0, output_params[i], 0);
	}
	ccv_cnnp_model_evaluate(model, (ccv_cnnp_evaluate_param_t){}, TENSOR_LIST(d_input), outputs, rank_count, 0, 0);
	for (j = 0; j < rank_count; j++)
	{
		ccv_nnc_tensor_t* const h_output = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows_per_rank, output_dim), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(outputs[j]), TENSOR_LIST(h_output), 0);
		for (r = 0; r < rows_per_rank; r++)
			for (i = 0; i < rank_count; i++)
				for (k = 0; k < output_chunk; k++)
				{
					float expected = 0;
					const int input_row = i * rows_per_rank + r;
					const int output_channel = j * output_chunk + k;
					int l;
					for (l = 0; l < input_dim; l++)
						expected += h_input->data.f32[input_row * input_dim + l] * h_weight->data.f32[output_channel * input_dim + l];
					REQUIRE_EQ_WITH_TOLERANCE(h_output->data.f32[r * output_dim + i * output_chunk + k], expected, 1e-4, "replicated dense all-to-all output should match CPU reference");
				}
		ccv_nnc_tensor_free(h_output);
	}
	for (i = 0; i < rank_count; i++)
		ccv_nnc_tensor_free(outputs[i]);
	ccv_nnc_tensor_free(d_input);
	ccv_nnc_tensor_free(h_weight);
	ccv_nnc_tensor_free(h_input);
	ccv_cnnp_model_free(model);
}

TEST_CASE("cnnp replicated embedding matches CPU reference")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	const int device_count = ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU);
	GUARD_ELSE_RETURN(device_count > 1);
	const int rank_count = device_count;
	const int rows_per_rank = 3;
	const int vocab_size = 11;
	const int embed_size = 5;
	const ccv_cnnp_model_io_t x = ccv_cnnp_input();
	ccv_cnnp_model_io_t chunks = ccv_cnnp_model_apply(ccv_cnnp_chunk(rank_count, 0, "chunk"), MODEL_IO_LIST(x));
	ccv_cnnp_model_io_t sent[rank_count];
	int i, j, r;
	for (i = 0; i < rank_count; i++)
	{
		ccv_cnnp_model_io_t chunk_i = ccv_cnnp_model_apply(ccv_cnnp_extract(i, 0), &chunks, 1);
		sent[i] = ccv_cnnp_model_apply(ccv_cnnp_send(i, 0), &chunk_i, 1);
	}
	ccv_cnnp_model_t* const embedding = ccv_cnnp_embedding(CCV_32F, vocab_size, embed_size, 1, "embedding");
	ccv_cnnp_model_io_t embedding_outputs = ccv_cnnp_model_apply(ccv_cnnp_replicated(embedding, rank_count, 1, "replicated_embedding"), sent, rank_count);
	ccv_cnnp_model_t* const model = ccv_cnnp_model_new(MODEL_IO_LIST(x), MODEL_IO_LIST(embedding_outputs), 0, "replicated_embedding");
	ccv_nnc_tensor_param_t input_params = GPU_TENSOR_NHWC(000, 32S, rank_count * rows_per_rank);
	ccv_cnnp_model_compile(model, TENSOR_PARAM_LIST(input_params), CMD_NOOP(), CMD_NOOP());
	ccv_nnc_tensor_t* const h_input = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, rank_count * rows_per_rank), 0);
	for (i = 0; i < rank_count * rows_per_rank; i++)
		h_input->data.i32[i] = (i * 3 + 1) % vocab_size;
	ccv_nnc_tensor_t* const h_vocab = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, vocab_size, embed_size), 0);
	for (i = 0; i < vocab_size; i++)
		for (j = 0; j < embed_size; j++)
			h_vocab->data.f32[i * embed_size + j] = (float)(i * 0.5 + j * 0.25 + 1);
	ccv_cnnp_model_set_parameter(model, ccv_cnnp_model_parameters(embedding, ALL_PARAMETERS, 0), h_vocab);
	ccv_nnc_tensor_t* const d_input = ccv_nnc_tensor_new(0, input_params, 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(h_input), TENSOR_LIST(d_input), 0);
	ccv_nnc_tensor_param_t output_params[rank_count];
	ccv_cnnp_model_tensor_auto(model, output_params, rank_count);
	ccv_nnc_tensor_t* outputs[rank_count];
	for (i = 0; i < rank_count; i++)
	{
		REQUIRE_EQ(CCV_TENSOR_GET_DEVICE_ID(output_params[i].type), i, "output device should match replicated rank");
		outputs[i] = ccv_nnc_tensor_new(0, output_params[i], 0);
	}
	ccv_cnnp_model_evaluate(model, (ccv_cnnp_evaluate_param_t){}, TENSOR_LIST(d_input), outputs, rank_count, 0, 0);
	for (i = 0; i < rank_count; i++)
	{
		ccv_nnc_tensor_t* const h_output = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows_per_rank, embed_size), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(outputs[i]), TENSOR_LIST(h_output), 0);
		for (r = 0; r < rows_per_rank; r++)
		{
			const int token = h_input->data.i32[i * rows_per_rank + r];
			for (j = 0; j < embed_size; j++)
				REQUIRE_EQ_WITH_TOLERANCE(h_output->data.f32[r * embed_size + j], h_vocab->data.f32[token * embed_size + j], 1e-5, "replicated embedding output should match CPU reference");
		}
		ccv_nnc_tensor_free(h_output);
	}
	for (i = 0; i < rank_count; i++)
		ccv_nnc_tensor_free(outputs[i]);
	ccv_nnc_tensor_free(d_input);
	ccv_nnc_tensor_free(h_vocab);
	ccv_nnc_tensor_free(h_input);
	ccv_cnnp_model_free(model);
}

TEST_CASE("cnnp replicated layer norm matches CPU reference")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_LAYER_NORM_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN));
	const int device_count = ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU);
	GUARD_ELSE_RETURN(device_count > 1);
	const int rank_count = device_count;
	const int rows_per_rank = 2;
	const int cols = 6;
	const float epsilon = 1e-5;
	const int axis[] = {1};
	const ccv_cnnp_model_io_t x = ccv_cnnp_input();
	ccv_cnnp_model_io_t chunks = ccv_cnnp_model_apply(ccv_cnnp_chunk(rank_count, 0, "chunk"), MODEL_IO_LIST(x));
	ccv_cnnp_model_io_t sent[rank_count];
	int i, j, r;
	for (i = 0; i < rank_count; i++)
	{
		ccv_cnnp_model_io_t chunk_i = ccv_cnnp_model_apply(ccv_cnnp_extract(i, 0), &chunks, 1);
		sent[i] = ccv_cnnp_model_apply(ccv_cnnp_send(i, 0), &chunk_i, 1);
	}
	ccv_cnnp_model_t* const layer_norm = ccv_cnnp_layer_norm(epsilon, axis, 1, 1, 1, "layer_norm");
	ccv_cnnp_model_io_t normalized = ccv_cnnp_model_apply(ccv_cnnp_replicated(layer_norm, rank_count, 1, "replicated_layer_norm"), sent, rank_count);
	ccv_cnnp_model_t* const model = ccv_cnnp_model_new(MODEL_IO_LIST(x), MODEL_IO_LIST(normalized), 0, "replicated_layer_norm");
	ccv_nnc_tensor_param_t input_params = GPU_TENSOR_NHWC(000, 32F, rank_count * rows_per_rank, cols);
	ccv_cnnp_model_compile(model, TENSOR_PARAM_LIST(input_params), CMD_NOOP(), CMD_NOOP());
	ccv_nnc_tensor_t* const h_input = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rank_count * rows_per_rank, cols), 0);
	for (i = 0; i < rank_count * rows_per_rank; i++)
		for (j = 0; j < cols; j++)
			h_input->data.f32[i * cols + j] = (float)((i + 1) * 0.75 + (j + 1) * 0.5);
	ccv_nnc_tensor_t* const h_scale = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, cols), 0);
	ccv_nnc_tensor_t* const h_bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, cols), 0);
	for (j = 0; j < cols; j++)
	{
		h_scale->data.f32[j] = 0.5f + j * 0.125f;
		h_bias->data.f32[j] = -0.75f + j * 0.2f;
	}
	ccv_cnnp_model_set_parameter(model, ccv_cnnp_model_parameters(layer_norm, ALL_PARAMETERS, 0), h_scale);
	ccv_cnnp_model_set_parameter(model, ccv_cnnp_model_parameters(layer_norm, ALL_PARAMETERS, 1), h_bias);
	ccv_nnc_tensor_t* const d_input = ccv_nnc_tensor_new(0, input_params, 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(h_input), TENSOR_LIST(d_input), 0);
	ccv_nnc_tensor_param_t output_params[rank_count];
	ccv_cnnp_model_tensor_auto(model, output_params, rank_count);
	ccv_nnc_tensor_t* outputs[rank_count];
	for (i = 0; i < rank_count; i++)
	{
		REQUIRE_EQ(CCV_TENSOR_GET_DEVICE_ID(output_params[i].type), i, "output device should match replicated rank");
		outputs[i] = ccv_nnc_tensor_new(0, output_params[i], 0);
	}
	ccv_cnnp_model_evaluate(model, (ccv_cnnp_evaluate_param_t){}, TENSOR_LIST(d_input), outputs, rank_count, 0, 0);
	for (i = 0; i < rank_count; i++)
	{
		ccv_nnc_tensor_t* const h_output = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows_per_rank, cols), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(outputs[i]), TENSOR_LIST(h_output), 0);
		for (r = 0; r < rows_per_rank; r++)
		{
			const int row = i * rows_per_rank + r;
			float mean = 0;
			for (j = 0; j < cols; j++)
				mean += h_input->data.f32[row * cols + j];
			mean /= cols;
			float variance = 0;
			for (j = 0; j < cols; j++)
			{
				const float centered = h_input->data.f32[row * cols + j] - mean;
				variance += centered * centered;
			}
			variance /= cols;
			const float inv_std = 1.0 / sqrtf(variance + epsilon);
			for (j = 0; j < cols; j++)
			{
				const float centered = h_input->data.f32[row * cols + j] - mean;
				const float expected = centered * inv_std * h_scale->data.f32[j] + h_bias->data.f32[j];
				REQUIRE_EQ_WITH_TOLERANCE(h_output->data.f32[r * cols + j], expected, 1e-4, "replicated layer norm output should match CPU reference");
			}
		}
		ccv_nnc_tensor_free(h_output);
	}
	for (i = 0; i < rank_count; i++)
		ccv_nnc_tensor_free(outputs[i]);
	ccv_nnc_tensor_free(d_input);
	ccv_nnc_tensor_free(h_bias);
	ccv_nnc_tensor_free(h_scale);
	ccv_nnc_tensor_free(h_input);
	ccv_cnnp_model_free(model);
}

TEST_CASE("schedule symbolic graph to data parallel with broadcast and reduce")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN));
	GUARD_ELSE_RETURN(ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU) >= 2);
	ccv_nnc_tensor_t* updated[4];
	ccv_nnc_tensor_t* cpu_inputs[2];
	ccv_nnc_tensor_t* cpu_fits[2];
	ccv_nnc_tensor_t* w1_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8, 3, 5, 5), 0);
	ccv_nnc_tensor_t* w3_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8, 8, 5, 5), 0);
	{
		ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
		const ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 16, 3, 32, 32), 0);
		const ccv_nnc_tensor_symbol_t w1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 8, 3, 5, 5), 0);
		const ccv_nnc_tensor_symbol_t bias1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 8), 0);
		const ccv_nnc_tensor_symbol_t y1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 16, 8, 32, 32), 0);
		const ccv_nnc_graph_exec_symbol_t conv1 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CONVOLUTION_FORWARD(1, 8, 5, 5, 32), TENSOR_SYMBOL_LIST(x, w1, bias1), TENSOR_SYMBOL_LIST(y1), "conv1");
		ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, conv1, HINT((1, 1), (2, 2)));
		const ccv_nnc_tensor_symbol_t y2 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 16, 8, 16, 16), 0);
		const ccv_nnc_graph_exec_symbol_t avg2 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_AVERAGE_POOL_FORWARD(2, 2), TENSOR_SYMBOL_LIST(y1), TENSOR_SYMBOL_LIST(y2), "avg2");
		ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, avg2, HINT((2, 2)));
		const ccv_nnc_tensor_symbol_t w3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 8, 8, 5, 5), 0);
		const ccv_nnc_tensor_symbol_t bias3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 8), 0);
		const ccv_nnc_tensor_symbol_t y3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 16, 8, 8, 8), 0);
		const ccv_nnc_graph_exec_symbol_t conv3 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CONVOLUTION_FORWARD(1, 8, 5, 5, 32), TENSOR_SYMBOL_LIST(y2, w3, bias3), TENSOR_SYMBOL_LIST(y3), "conv3");
		ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, conv3, HINT((2, 2), (2, 2)));
		const ccv_nnc_tensor_symbol_t y4 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 16, 8, 1, 1), 0);
		ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_AVERAGE_POOL_FORWARD(8, 8), TENSOR_SYMBOL_LIST(y3), TENSOR_SYMBOL_LIST(y4), "avg4");
		const ccv_nnc_tensor_symbol_t y4a = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, y4, ccv_nnc_no_ofs, DIM_ALLOC(8, 1, 1, 1), GPU_TENSOR_NCHW(000, 32F, 16, 8), 0);
		const ccv_nnc_tensor_symbol_t label = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 16), "label");
		const ccv_nnc_tensor_symbol_t y5 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 16, 8), "y5");
		const ccv_nnc_tensor_symbol_t loss = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 16), "loss");
		ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SOFTMAX_CROSSENTROPY_FORWARD(), TENSOR_SYMBOL_LIST(y4a, label), TENSOR_SYMBOL_LIST(loss, y5), "softmax crossentropy");
		ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
		ccv_nnc_tensor_symbol_t updated_params[4];
		ccv_nnc_tensor_symbol_t gradients[4];
		const int saved_aux_size = ccv_nnc_minimizer_saved_aux_size(CMD_SGD_FORWARD(0, 0.001, 1, 0.99, 0.9, 0.9));
		ccv_nnc_tensor_symbol_map_t saved_aux[saved_aux_size * 4];
		ccv_nnc_graph_exec_symbol_t updated_execs[4];
		ccv_nnc_symbolic_graph_minimize(symbolic_graph, CMD_SGD_FORWARD(0, 0.001, 1, 0.99, 0.9, 0.9), TENSOR_SYMBOL_LIST(loss), TENSOR_SYMBOL_LIST(w1, bias1, w3, bias3), 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), gradients, updated_params, saved_aux, updated_execs);
		const ccv_nnc_tensor_symbol_t dloss = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, loss);
		ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SET_FORWARD(1), TENSOR_SYMBOL_LIST(), TENSOR_SYMBOL_LIST(dloss), "set 1");
		int i;
		for (i = 0; i < saved_aux_size * 4; i++)
			ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SET_FORWARD(0), TENSOR_SYMBOL_LIST(), TENSOR_SYMBOL_LIST(saved_aux[i].source), "set 0");
		ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
		ccv_nnc_symbolic_graph_data_parallel(symbolic_graph, 2, TENSOR_SYMBOL_LIST(w1, bias1, w3, bias3), 0, 0, 0, gradients, 4, 0, CCV_NNC_PARALLEL_REDUCE_OP_SUM, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), updated_execs, 4);
		ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
		SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
		ccv_nnc_graph_t* graph;
		ccv_nnc_tensor_arena_t* tensor_arena;
		ccv_nnc_graph_exec_arena_t* graph_exec_arena;
		ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params,
			0, 0,
			updated_params, 4,
			SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph),
			&graph, &tensor_arena, &graph_exec_arena);
		ccv_nnc_graph_set_default_static_schedule(graph, CCV_STREAM_CONTEXT_GPU, 0);
		GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
		cpu_inputs[0] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 16, 3, 32, 32), 0);
		cpu_inputs[1] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 16, 3, 32, 32), 0);
		dsfmt_t dsfmt;
		dsfmt_init_gen_rand(&dsfmt, 0);
		for (i = 0; i < 16 * 3 * 32 * 32; i++)
			cpu_inputs[0]->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		for (i = 0; i < 16 * 3 * 32 * 32; i++)
			cpu_inputs[1]->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		cpu_fits[0] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 16), 0);
		cpu_fits[1] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 16), 0);
		for (i = 0; i < 16; i++)
			cpu_fits[0]->data.f32[i] = cpu_fits[1]->data.f32[i] = (int)(dsfmt_genrand_open_close(&dsfmt) * 7.4); // Between 0 to 7.
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(cpu_inputs[0], cpu_inputs[1]), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, x), ccv_nnc_tensor_from_symbol(tensor_arena, ccv_nnc_tensor_symbol_copy(symbolic_graph, x, 1))), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(cpu_fits[0], cpu_fits[1]), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, label), ccv_nnc_tensor_from_symbol(tensor_arena, ccv_nnc_tensor_symbol_copy(symbolic_graph, label, 1))), 0);
		ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, bias1), ccv_nnc_tensor_from_symbol(tensor_arena, bias3)), 0);
		for (i = 0; i < 8 * 3 * 5 * 5; i++)
			w1_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		for (i = 0; i < 8 * 8 * 5 * 5; i++)
			w3_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(w1_tensor, w3_tensor), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, w1), ccv_nnc_tensor_from_symbol(tensor_arena, w3)), 0);
		ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, ccv_nnc_graph_default_stream(graph));
		ccv_nnc_stream_context_wait(ccv_nnc_graph_default_stream(graph));
		updated[0] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8, 3, 5, 5), 0);
		updated[1] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8), 0);
		updated[2] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8, 8, 5, 5), 0);
		updated[3] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, updated_params[0]), ccv_nnc_tensor_from_symbol(tensor_arena, updated_params[1]), ccv_nnc_tensor_from_symbol(tensor_arena, updated_params[2]), ccv_nnc_tensor_from_symbol(tensor_arena, updated_params[3])), updated, 4, 0);
		ccv_nnc_symbolic_graph_free(symbolic_graph);
		ccv_nnc_graph_free(graph);
		ccv_nnc_tensor_arena_free(tensor_arena);
		ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	}
	// Now, doing exactly the same, but with no parallel.
	{
		ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
		const ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 32, 3, 32, 32), 0);
		const ccv_nnc_tensor_symbol_t w1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 8, 3, 5, 5), 0);
		const ccv_nnc_tensor_symbol_t bias1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 8), 0);
		const ccv_nnc_tensor_symbol_t y1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 32, 8, 32, 32), 0);
		const ccv_nnc_graph_exec_symbol_t conv1 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CONVOLUTION_FORWARD(1, 8, 5, 5, 32), TENSOR_SYMBOL_LIST(x, w1, bias1), TENSOR_SYMBOL_LIST(y1), "conv1");
		ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, conv1, HINT((1, 1), (2, 2)));
		const ccv_nnc_tensor_symbol_t y2 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 32, 8, 16, 16), 0);
		const ccv_nnc_graph_exec_symbol_t avg2 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_AVERAGE_POOL_FORWARD(2, 2), TENSOR_SYMBOL_LIST(y1), TENSOR_SYMBOL_LIST(y2), "avg2");
		ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, avg2, HINT((2, 2)));
		const ccv_nnc_tensor_symbol_t w3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 8, 8, 5, 5), 0);
		const ccv_nnc_tensor_symbol_t bias3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 8), 0);
		const ccv_nnc_tensor_symbol_t y3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 32, 8, 8, 8), 0);
		const ccv_nnc_graph_exec_symbol_t conv3 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CONVOLUTION_FORWARD(1, 8, 5, 5, 32), TENSOR_SYMBOL_LIST(y2, w3, bias3), TENSOR_SYMBOL_LIST(y3), "conv3");
		ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, conv3, HINT((2, 2), (2, 2)));
		const ccv_nnc_tensor_symbol_t y4 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 32, 8, 1, 1), 0);
		ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_AVERAGE_POOL_FORWARD(8, 8), TENSOR_SYMBOL_LIST(y3), TENSOR_SYMBOL_LIST(y4), "avg4");
		const ccv_nnc_tensor_symbol_t label = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 32), "label");
		const ccv_nnc_tensor_symbol_t y5 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 32, 8), "y5");
		const ccv_nnc_tensor_symbol_t loss = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 32), "loss");
		ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SOFTMAX_CROSSENTROPY_FORWARD(), TENSOR_SYMBOL_LIST(y4, label), TENSOR_SYMBOL_LIST(loss, y5), "softmax crossentropy");
		ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
		ccv_nnc_tensor_symbol_t updated_params[4];
		ccv_nnc_tensor_symbol_t gradients[4];
		const int saved_aux_size = ccv_nnc_minimizer_saved_aux_size(CMD_SGD_FORWARD(0, 0.001, 1, 0.99, 0.9, 0.9));
		ccv_nnc_tensor_symbol_map_t saved_aux[saved_aux_size * 4];
		ccv_nnc_graph_exec_symbol_t updated_execs[4];
		ccv_nnc_symbolic_graph_minimize(symbolic_graph, CMD_SGD_FORWARD(0, 0.001, 1, 0.99, 0.9, 0.9), TENSOR_SYMBOL_LIST(loss), TENSOR_SYMBOL_LIST(w1, bias1, w3, bias3), 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), gradients, updated_params, saved_aux, updated_execs);
		const ccv_nnc_tensor_symbol_t dloss = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, loss);
		ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SET_FORWARD(1), TENSOR_SYMBOL_LIST(), TENSOR_SYMBOL_LIST(dloss), "set 1");
		int i;
		for (i = 0; i < saved_aux_size * 4; i++)
			ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SET_FORWARD(0), TENSOR_SYMBOL_LIST(), TENSOR_SYMBOL_LIST(saved_aux[i].source), "set 0");
		ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
		ccv_nnc_graph_t* graph;
		ccv_nnc_tensor_arena_t* tensor_arena;
		ccv_nnc_graph_exec_arena_t* graph_exec_arena;
		ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params,
			0, 0,
			updated_params, 4,
			SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph),
			&graph, &tensor_arena, &graph_exec_arena);
		ccv_nnc_tensor_t* cpu_input = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 32, 3, 32, 32), 0);
		memcpy(cpu_input->data.f32, cpu_inputs[0]->data.f32, sizeof(float) * 16 * 3 * 32 * 32);
		memcpy(cpu_input->data.f32 + 16 * 3 * 32 * 32, cpu_inputs[1]->data.f32, sizeof(float) * 16 * 3 * 32 * 32);
		ccv_nnc_tensor_t* cpu_fit = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 32), 0);
		memcpy(cpu_fit->data.f32, cpu_fits[0]->data.f32, sizeof(float) * 16);
		memcpy(cpu_fit->data.f32 + 16, cpu_fits[1]->data.f32, sizeof(float) * 16);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(cpu_input), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, x)), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(cpu_fit), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, label)), 0);
		ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, bias1), ccv_nnc_tensor_from_symbol(tensor_arena, bias3)), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(w1_tensor, w3_tensor), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, w1), ccv_nnc_tensor_from_symbol(tensor_arena, w3)), 0);
		ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
		ccv_nnc_tensor_t* np_updated[4];
		np_updated[0] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8, 3, 5, 5), 0);
		np_updated[1] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8), 0);
		np_updated[2] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8, 8, 5, 5), 0);
		np_updated[3] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, updated_params[0]), ccv_nnc_tensor_from_symbol(tensor_arena, updated_params[1]), ccv_nnc_tensor_from_symbol(tensor_arena, updated_params[2]), ccv_nnc_tensor_from_symbol(tensor_arena, updated_params[3])), np_updated, 4, 0);
		ccv_nnc_symbolic_graph_free(symbolic_graph);
		ccv_nnc_graph_free(graph);
		ccv_nnc_tensor_arena_free(tensor_arena);
		ccv_nnc_graph_exec_arena_free(graph_exec_arena);
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, np_updated[0]->data.f32, updated[0]->data.f32, 8 * 3 * 5 * 5, 1e-4, "updated params should be equal");
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, np_updated[1]->data.f32, updated[1]->data.f32, 8, 1e-5, "updated params should be equal");
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, np_updated[2]->data.f32, updated[2]->data.f32, 8 * 8 * 5 * 5, 1e-4, "updated params should be equal");
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, np_updated[3]->data.f32, updated[3]->data.f32, 8, 1e-4, "updated params should be equal");
		ccv_nnc_tensor_free(cpu_input);
		ccv_nnc_tensor_free(cpu_fit);
		ccv_nnc_tensor_free(np_updated[0]);
		ccv_nnc_tensor_free(np_updated[1]);
		ccv_nnc_tensor_free(np_updated[2]);
		ccv_nnc_tensor_free(np_updated[3]);
	}
	ccv_nnc_tensor_free(updated[0]);
	ccv_nnc_tensor_free(updated[1]);
	ccv_nnc_tensor_free(updated[2]);
	ccv_nnc_tensor_free(updated[3]);
	ccv_nnc_tensor_free(cpu_inputs[0]);
	ccv_nnc_tensor_free(cpu_inputs[1]);
	ccv_nnc_tensor_free(cpu_fits[0]);
	ccv_nnc_tensor_free(cpu_fits[1]);
	ccv_nnc_tensor_free(w1_tensor);
	ccv_nnc_tensor_free(w3_tensor);
}

TEST_CASE("schedule symbolic graph to data parallel with allreduce")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN));
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_COMM_ALLREDUCE_FORWARD, CCV_NNC_BACKEND_GPU_NCCL));
	GUARD_ELSE_RETURN(ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU) >= 2);
	ccv_nnc_tensor_t* updated[4];
	ccv_nnc_tensor_t* cpu_inputs[2];
	ccv_nnc_tensor_t* cpu_fits[2];
	ccv_nnc_tensor_t* w1_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8, 3, 5, 5), 0);
	ccv_nnc_tensor_t* w3_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8, 8, 5, 5), 0);
	{
		ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
		const ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 16, 3, 32, 32), 0);
		const ccv_nnc_tensor_symbol_t w1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 8, 3, 5, 5), 0);
		const ccv_nnc_tensor_symbol_t bias1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 8), 0);
		const ccv_nnc_tensor_symbol_t y1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 16, 8, 32, 32), 0);
		const ccv_nnc_graph_exec_symbol_t conv1 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CONVOLUTION_FORWARD(1, 8, 5, 5, 32), TENSOR_SYMBOL_LIST(x, w1, bias1), TENSOR_SYMBOL_LIST(y1), "conv1");
		ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, conv1, HINT((1, 1), (2, 2)));
		const ccv_nnc_tensor_symbol_t y2 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 16, 8, 16, 16), 0);
		const ccv_nnc_graph_exec_symbol_t avg2 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_AVERAGE_POOL_FORWARD(2, 2), TENSOR_SYMBOL_LIST(y1), TENSOR_SYMBOL_LIST(y2), "avg2");
		ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, avg2, HINT((2, 2)));
		const ccv_nnc_tensor_symbol_t w3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 8, 8, 5, 5), 0);
		const ccv_nnc_tensor_symbol_t bias3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 8), 0);
		const ccv_nnc_tensor_symbol_t y3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 16, 8, 8, 8), 0);
		const ccv_nnc_graph_exec_symbol_t conv3 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CONVOLUTION_FORWARD(1, 8, 5, 5, 32), TENSOR_SYMBOL_LIST(y2, w3, bias3), TENSOR_SYMBOL_LIST(y3), "conv3");
		ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, conv3, HINT((2, 2), (2, 2)));
		const ccv_nnc_tensor_symbol_t y4 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 16, 8, 1, 1), 0);
		ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_AVERAGE_POOL_FORWARD(8, 8), TENSOR_SYMBOL_LIST(y3), TENSOR_SYMBOL_LIST(y4), "avg4");
		const ccv_nnc_tensor_symbol_t y4a = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, y4, ccv_nnc_no_ofs, DIM_ALLOC(8, 1, 1, 1), GPU_TENSOR_NCHW(000, 32F, 16, 8), 0);
		const ccv_nnc_tensor_symbol_t label = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 16), "label");
		const ccv_nnc_tensor_symbol_t y5 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 16, 8), "y5");
		const ccv_nnc_tensor_symbol_t loss = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 16), "loss");
		ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SOFTMAX_CROSSENTROPY_FORWARD(), TENSOR_SYMBOL_LIST(y4a, label), TENSOR_SYMBOL_LIST(loss, y5), "softmax crossentropy");
		ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
		ccv_nnc_tensor_symbol_t updated_params[4];
		ccv_nnc_tensor_symbol_t gradients[4];
		const int saved_aux_size = ccv_nnc_minimizer_saved_aux_size(CMD_SGD_FORWARD(0, 0.001, 1, 0.99, 0.9, 0.9));
		ccv_nnc_tensor_symbol_map_t saved_aux[saved_aux_size * 4];
		ccv_nnc_graph_exec_symbol_t updated_execs[4];
		ccv_nnc_symbolic_graph_minimize(symbolic_graph, CMD_SGD_FORWARD(0, 0.001, 1, 0.99, 0.9, 0.9), TENSOR_SYMBOL_LIST(loss), TENSOR_SYMBOL_LIST(w1, bias1, w3, bias3), 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), gradients, updated_params, saved_aux, updated_execs);
		const ccv_nnc_tensor_symbol_t dloss = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, loss);
		ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SET_FORWARD(1), TENSOR_SYMBOL_LIST(), TENSOR_SYMBOL_LIST(dloss), "set 1");
		int i;
		for (i = 0; i < saved_aux_size * 4; i++)
			ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SET_FORWARD(0), TENSOR_SYMBOL_LIST(), TENSOR_SYMBOL_LIST(saved_aux[i].source), "set 0");
		ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
		ccv_nnc_symbolic_graph_data_parallel(symbolic_graph, 2, TENSOR_SYMBOL_LIST(w1, bias1, w3, bias3), gradients, 4, 0, 0, 0, 0, CCV_NNC_PARALLEL_REDUCE_OP_SUM, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), updated_execs, 4);
		ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
		SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
		ccv_nnc_graph_t* graph;
		ccv_nnc_tensor_arena_t* tensor_arena;
		ccv_nnc_graph_exec_arena_t* graph_exec_arena;
		ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params,
			0, 0,
			updated_params, 4,
			SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph),
			&graph, &tensor_arena, &graph_exec_arena);
		ccv_nnc_graph_set_default_static_schedule(graph, CCV_STREAM_CONTEXT_GPU, 0);
		GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
		cpu_inputs[0] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 16, 3, 32, 32), 0);
		cpu_inputs[1] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 16, 3, 32, 32), 0);
		dsfmt_t dsfmt;
		dsfmt_init_gen_rand(&dsfmt, 0);
		for (i = 0; i < 16 * 3 * 32 * 32; i++)
			cpu_inputs[0]->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		for (i = 0; i < 16 * 3 * 32 * 32; i++)
			cpu_inputs[1]->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		cpu_fits[0] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 16), 0);
		cpu_fits[1] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 16), 0);
		for (i = 0; i < 16; i++)
			cpu_fits[0]->data.f32[i] = cpu_fits[1]->data.f32[i] = (int)(dsfmt_genrand_open_close(&dsfmt) * 7.4); // Between 0 to 7.
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(cpu_inputs[0], cpu_inputs[1]), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, x), ccv_nnc_tensor_from_symbol(tensor_arena, ccv_nnc_tensor_symbol_copy(symbolic_graph, x, 1))), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(cpu_fits[0], cpu_fits[1]), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, label), ccv_nnc_tensor_from_symbol(tensor_arena, ccv_nnc_tensor_symbol_copy(symbolic_graph, label, 1))), 0);
		ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, bias1), ccv_nnc_tensor_from_symbol(tensor_arena, bias3)), 0);
		for (i = 0; i < 8 * 3 * 5 * 5; i++)
			w1_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		for (i = 0; i < 8 * 8 * 5 * 5; i++)
			w3_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(w1_tensor, w3_tensor), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, w1), ccv_nnc_tensor_from_symbol(tensor_arena, w3)), 0);
		ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, ccv_nnc_graph_default_stream(graph));
		ccv_nnc_stream_context_wait(ccv_nnc_graph_default_stream(graph));
		updated[0] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8, 3, 5, 5), 0);
		updated[1] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8), 0);
		updated[2] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8, 8, 5, 5), 0);
		updated[3] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, updated_params[0]), ccv_nnc_tensor_from_symbol(tensor_arena, updated_params[1]), ccv_nnc_tensor_from_symbol(tensor_arena, updated_params[2]), ccv_nnc_tensor_from_symbol(tensor_arena, updated_params[3])), updated, 4, 0);
		ccv_nnc_symbolic_graph_free(symbolic_graph);
		ccv_nnc_graph_free(graph);
		ccv_nnc_tensor_arena_free(tensor_arena);
		ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	}
	// Now, doing exactly the same, but with no parallel.
	{
		ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
		const ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 32, 3, 32, 32), 0);
		const ccv_nnc_tensor_symbol_t w1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 8, 3, 5, 5), 0);
		const ccv_nnc_tensor_symbol_t bias1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 8), 0);
		const ccv_nnc_tensor_symbol_t y1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 32, 8, 32, 32), 0);
		const ccv_nnc_graph_exec_symbol_t conv1 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CONVOLUTION_FORWARD(1, 8, 5, 5, 32), TENSOR_SYMBOL_LIST(x, w1, bias1), TENSOR_SYMBOL_LIST(y1), "conv1");
		ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, conv1, HINT((1, 1), (2, 2)));
		const ccv_nnc_tensor_symbol_t y2 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 32, 8, 16, 16), 0);
		const ccv_nnc_graph_exec_symbol_t avg2 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_AVERAGE_POOL_FORWARD(2, 2), TENSOR_SYMBOL_LIST(y1), TENSOR_SYMBOL_LIST(y2), "avg2");
		ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, avg2, HINT((2, 2)));
		const ccv_nnc_tensor_symbol_t w3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 8, 8, 5, 5), 0);
		const ccv_nnc_tensor_symbol_t bias3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 8), 0);
		const ccv_nnc_tensor_symbol_t y3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 32, 8, 8, 8), 0);
		const ccv_nnc_graph_exec_symbol_t conv3 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CONVOLUTION_FORWARD(1, 8, 5, 5, 32), TENSOR_SYMBOL_LIST(y2, w3, bias3), TENSOR_SYMBOL_LIST(y3), "conv3");
		ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, conv3, HINT((2, 2), (2, 2)));
		const ccv_nnc_tensor_symbol_t y4 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 32, 8, 1, 1), 0);
		ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_AVERAGE_POOL_FORWARD(8, 8), TENSOR_SYMBOL_LIST(y3), TENSOR_SYMBOL_LIST(y4), "avg4");
		const ccv_nnc_tensor_symbol_t label = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 32), "label");
		const ccv_nnc_tensor_symbol_t y5 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 32, 8), "y5");
		const ccv_nnc_tensor_symbol_t loss = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 32), "loss");
		ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SOFTMAX_CROSSENTROPY_FORWARD(), TENSOR_SYMBOL_LIST(y4, label), TENSOR_SYMBOL_LIST(loss, y5), "softmax crossentropy");
		ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
		ccv_nnc_tensor_symbol_t updated_params[4];
		ccv_nnc_tensor_symbol_t gradients[4];
		const int saved_aux_size = ccv_nnc_minimizer_saved_aux_size(CMD_SGD_FORWARD(0, 0.001, 1, 0.99, 0.9, 0.9));
		ccv_nnc_tensor_symbol_map_t saved_aux[saved_aux_size * 4];
		ccv_nnc_graph_exec_symbol_t updated_execs[4];
		ccv_nnc_symbolic_graph_minimize(symbolic_graph, CMD_SGD_FORWARD(0, 0.001, 1, 0.99, 0.9, 0.9), TENSOR_SYMBOL_LIST(loss), TENSOR_SYMBOL_LIST(w1, bias1, w3, bias3), 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), gradients, updated_params, saved_aux, updated_execs);
		const ccv_nnc_tensor_symbol_t dloss = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, loss);
		ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SET_FORWARD(1), TENSOR_SYMBOL_LIST(), TENSOR_SYMBOL_LIST(dloss), "set 1");
		int i;
		for (i = 0; i < saved_aux_size * 4; i++)
			ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SET_FORWARD(0), TENSOR_SYMBOL_LIST(), TENSOR_SYMBOL_LIST(saved_aux[i].source), "set 0");
		ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
		ccv_nnc_graph_t* graph;
		ccv_nnc_tensor_arena_t* tensor_arena;
		ccv_nnc_graph_exec_arena_t* graph_exec_arena;
		ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params,
			0, 0,
			updated_params, 4,
			SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph),
			&graph, &tensor_arena, &graph_exec_arena);
		ccv_nnc_tensor_t* cpu_input = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 32, 3, 32, 32), 0);
		memcpy(cpu_input->data.f32, cpu_inputs[0]->data.f32, sizeof(float) * 16 * 3 * 32 * 32);
		memcpy(cpu_input->data.f32 + 16 * 3 * 32 * 32, cpu_inputs[1]->data.f32, sizeof(float) * 16 * 3 * 32 * 32);
		ccv_nnc_tensor_t* cpu_fit = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 32), 0);
		memcpy(cpu_fit->data.f32, cpu_fits[0]->data.f32, sizeof(float) * 16);
		memcpy(cpu_fit->data.f32 + 16, cpu_fits[1]->data.f32, sizeof(float) * 16);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(cpu_input), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, x)), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(cpu_fit), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, label)), 0);
		ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, bias1), ccv_nnc_tensor_from_symbol(tensor_arena, bias3)), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(w1_tensor, w3_tensor), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, w1), ccv_nnc_tensor_from_symbol(tensor_arena, w3)), 0);
		ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
		ccv_nnc_tensor_t* np_updated[4];
		np_updated[0] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8, 3, 5, 5), 0);
		np_updated[1] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8), 0);
		np_updated[2] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8, 8, 5, 5), 0);
		np_updated[3] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, updated_params[0]), ccv_nnc_tensor_from_symbol(tensor_arena, updated_params[1]), ccv_nnc_tensor_from_symbol(tensor_arena, updated_params[2]), ccv_nnc_tensor_from_symbol(tensor_arena, updated_params[3])), np_updated, 4, 0);
		ccv_nnc_symbolic_graph_free(symbolic_graph);
		ccv_nnc_graph_free(graph);
		ccv_nnc_tensor_arena_free(tensor_arena);
		ccv_nnc_graph_exec_arena_free(graph_exec_arena);
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, np_updated[0]->data.f32, updated[0]->data.f32, 8 * 3 * 5 * 5, 1e-4, "updated params should be equal");
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, np_updated[1]->data.f32, updated[1]->data.f32, 8, 1e-5, "updated params should be equal");
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, np_updated[2]->data.f32, updated[2]->data.f32, 8 * 8 * 5 * 5, 1e-4, "updated params should be equal");
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, np_updated[3]->data.f32, updated[3]->data.f32, 8, 1e-4, "updated params should be equal");
		ccv_nnc_tensor_free(cpu_input);
		ccv_nnc_tensor_free(cpu_fit);
		ccv_nnc_tensor_free(np_updated[0]);
		ccv_nnc_tensor_free(np_updated[1]);
		ccv_nnc_tensor_free(np_updated[2]);
		ccv_nnc_tensor_free(np_updated[3]);
	}
	ccv_nnc_tensor_free(updated[0]);
	ccv_nnc_tensor_free(updated[1]);
	ccv_nnc_tensor_free(updated[2]);
	ccv_nnc_tensor_free(updated[3]);
	ccv_nnc_tensor_free(cpu_inputs[0]);
	ccv_nnc_tensor_free(cpu_inputs[1]);
	ccv_nnc_tensor_free(cpu_fits[0]);
	ccv_nnc_tensor_free(cpu_fits[1]);
	ccv_nnc_tensor_free(w1_tensor);
	ccv_nnc_tensor_free(w3_tensor);
}

#include "case_main.h"
