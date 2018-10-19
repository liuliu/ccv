#include <ctype.h>
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <3rdparty/dsfmt/dSFMT.h>
#include <sys/time.h>
#include <ctype.h>

static unsigned int get_current_time(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

static ccv_cnnp_model_t* _building_block_new(const int filters, const int strides, const int border, const int projection_shortcut)
{
	ccv_cnnp_model_io_t input = ccv_cnnp_input();
	ccv_cnnp_model_io_t shortcut = input;
	ccv_cnnp_model_t* const identity = ccv_cnnp_identity((ccv_cnnp_param_t){
		.norm = CCV_CNNP_BATCH_NORM,
		.activation = CCV_CNNP_ACTIVATION_RELU,
	});
	ccv_cnnp_model_io_t output = ccv_cnnp_model_apply(identity, MODEL_IO_LIST(input));
	if (projection_shortcut)
	{
		ccv_cnnp_model_t* const conv0 = ccv_cnnp_convolution(1, filters, DIM_ALLOC(1, 1), (ccv_cnnp_param_t){
			.no_bias = 1,
			.hint = HINT((strides, strides), (0, 0)),
		});
		shortcut = ccv_cnnp_model_apply(conv0, MODEL_IO_LIST(output));
	}
	ccv_cnnp_model_t* const conv1 = ccv_cnnp_convolution(1, filters, DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
		.norm = CCV_CNNP_BATCH_NORM,
		.activation = CCV_CNNP_ACTIVATION_RELU,
		.hint = HINT((strides, strides), (border, border)),
	});
	output = ccv_cnnp_model_apply(conv1, MODEL_IO_LIST(output));
	ccv_cnnp_model_t* const conv2 = ccv_cnnp_convolution(1, filters, DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
		.no_bias = 1,
		.hint = HINT((1, 1), (1, 1)),
	});
	output = ccv_cnnp_model_apply(conv2, MODEL_IO_LIST(output));
	ccv_cnnp_model_t* const add = ccv_cnnp_add();
	output = ccv_cnnp_model_apply(add, MODEL_IO_LIST(output, shortcut));
	return ccv_cnnp_model_new(MODEL_IO_LIST(input), MODEL_IO_LIST(output));
}

static ccv_cnnp_model_t* _block_layer_new(const int filters, const int strides, const int border, const int blocks)
{
	ccv_cnnp_model_io_t input = ccv_cnnp_input();
	ccv_cnnp_model_t* first_block = _building_block_new(filters, strides, border, 1);
	ccv_cnnp_model_io_t output = ccv_cnnp_model_apply(first_block, MODEL_IO_LIST(input));
	int i;
	for (i = 1; i < blocks; i++)
	{
		ccv_cnnp_model_t* block = _building_block_new(filters, 1, 1, 0);
		output = ccv_cnnp_model_apply(block, MODEL_IO_LIST(output));
	}
	return ccv_cnnp_model_new(MODEL_IO_LIST(input), MODEL_IO_LIST(output));
}

ccv_cnnp_model_t* _cifar_10_resnet56(void)
{
	const ccv_cnnp_model_io_t input = ccv_cnnp_input();
	ccv_cnnp_model_t* init_conv = ccv_cnnp_convolution(1, 16, DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
		.no_bias = 1,
		.hint = HINT((1, 1), (1, 1)),
	});
	ccv_cnnp_model_io_t output = ccv_cnnp_model_apply(init_conv, MODEL_IO_LIST(input));
	output = ccv_cnnp_model_apply(_block_layer_new(16, 1, 1, 9), MODEL_IO_LIST(output));
	output = ccv_cnnp_model_apply(_block_layer_new(32, 2, 1, 9), MODEL_IO_LIST(output));
	output = ccv_cnnp_model_apply(_block_layer_new(64, 2, 1, 9), MODEL_IO_LIST(output));
	ccv_cnnp_model_t* identity = ccv_cnnp_identity((ccv_cnnp_param_t){
		.norm = CCV_CNNP_BATCH_NORM,
		.activation = CCV_CNNP_ACTIVATION_RELU,
	});
	output = ccv_cnnp_model_apply(identity, MODEL_IO_LIST(output));
	output = ccv_cnnp_model_apply(ccv_cnnp_average_pool(DIM_ALLOC(0, 0), (ccv_cnnp_param_t){}), MODEL_IO_LIST(output));
	output = ccv_cnnp_model_apply(ccv_cnnp_flatten(), MODEL_IO_LIST(output));
	output = ccv_cnnp_model_apply(ccv_cnnp_dense(10, (ccv_cnnp_param_t){
		.activation = CCV_CNNP_ACTIVATION_SOFTMAX,
	}), MODEL_IO_LIST(output));
	return ccv_cnnp_model_new(MODEL_IO_LIST(input), MODEL_IO_LIST(output));
}

ccv_cnnp_model_t* _cifar_10_alexnet(void)
{
	return ccv_cnnp_sequential_new(MODEL_LIST(
		ccv_cnnp_convolution(1, 32, DIM_ALLOC(5, 5), (ccv_cnnp_param_t){
			.norm = CCV_CNNP_BATCH_NORM,
			.activation = CCV_CNNP_ACTIVATION_RELU,
			.hint = HINT((1, 1), (2, 2)),
		}),
		ccv_cnnp_max_pool(DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.hint = HINT((2, 2), (0, 0)),
		}),
		ccv_cnnp_convolution(1, 32, DIM_ALLOC(5, 5), (ccv_cnnp_param_t){
			.norm = CCV_CNNP_BATCH_NORM,
			.activation = CCV_CNNP_ACTIVATION_RELU,
			.hint = HINT((1, 1), (2, 2)),
		}),
		ccv_cnnp_average_pool(DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.hint = HINT((2, 2), (0, 0)),
		}),
		ccv_cnnp_convolution(1, 64, DIM_ALLOC(5, 5), (ccv_cnnp_param_t){
			.norm = CCV_CNNP_BATCH_NORM,
			.activation = CCV_CNNP_ACTIVATION_RELU,
			.hint = HINT((1, 1), (2, 2)),
		}),
		ccv_cnnp_average_pool(DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.hint = HINT((2, 2), (0, 0)),
		}),
		ccv_cnnp_flatten(),
		ccv_cnnp_dense(256, (ccv_cnnp_param_t){
			.norm = CCV_CNNP_BATCH_NORM,
			.activation = CCV_CNNP_ACTIVATION_RELU,
		}),
		ccv_cnnp_dense(10, (ccv_cnnp_param_t){
			.activation = CCV_CNNP_ACTIVATION_SOFTMAX,
		})
	));
}

static void train_cifar_10(ccv_array_t* const training_set, const int batch_size, const float mean[3], ccv_array_t* const test_set)
{
	ccv_cnnp_model_t* const cifar_10 = _cifar_10_resnet56();
	const ccv_nnc_tensor_param_t input = GPU_TENSOR_NCHW(000, batch_size, 3, 32, 32);
	float learn_rate = 0.001;
	ccv_cnnp_model_compile(cifar_10, &input, 1, CMD_SGD_FORWARD(learn_rate, 0.99, 0.9, 0.9), CMD_CATEGORICAL_CROSSENTROPY_FORWARD());
	FILE *w = fopen("cifar-10.dot", "w+");
	ccv_cnnp_model_dot(cifar_10, CCV_NNC_LONG_DOT_GRAPH, w);
	fclose(w);
	ccv_nnc_tensor_t* input_tensors[2];
	input_tensors[0] = ccv_nnc_tensor_new(0, input, 0);
	input_tensors[1] = ccv_nnc_tensor_new(0, input, 0);
	ccv_nnc_tensor_t* output_tensors[2];
	output_tensors[0] = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, batch_size, 10), 0);
	output_tensors[1] = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, batch_size, 10), 0);
	ccv_nnc_tensor_t* fit_tensors[2];
	fit_tensors[0] = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, batch_size, 1), 0);
	fit_tensors[1] = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, batch_size, 1), 0);
	ccv_nnc_tensor_t* cpu_inputs[2];
	cpu_inputs[0] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(batch_size, 3, 32, 32), 0);
	ccv_nnc_tensor_pin_memory(cpu_inputs[0]);
	cpu_inputs[1] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(batch_size, 3, 32, 32), 0);
	ccv_nnc_tensor_pin_memory(cpu_inputs[1]);
	ccv_nnc_tensor_t* cpu_fits[2];
	cpu_fits[0] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(batch_size, 1), 0);
	ccv_nnc_tensor_pin_memory(cpu_fits[0]);
	cpu_fits[1] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(batch_size, 1), 0);
	ccv_nnc_tensor_pin_memory(cpu_fits[1]);
	ccv_nnc_tensor_t* cpu_output = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(batch_size, 10), 0);
	ccv_nnc_tensor_pin_memory(cpu_output);
	int i, j, k;
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int c[batch_size];
	ccv_nnc_stream_context_t* stream_contexts[2];
	stream_contexts[0] = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	stream_contexts[1] = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	int p = 0, q = 1;
	const int epoch_end = (training_set->rnum + batch_size - 1) / batch_size;
	unsigned int current_time = get_current_time();
	for (i = 0; i < 100000; i++)
	{
		memset(cpu_inputs[p]->data.f32, 0, sizeof(float) * batch_size * 32 * 32 * 3);
		for (j = 0; j < batch_size; j++)
		{
			k = (int)(dsfmt_genrand_close_open(&dsfmt) * training_set->rnum);
			assert(k < training_set->rnum);
			ccv_categorized_t* const categorized = (ccv_categorized_t*)ccv_array_get(training_set, k);
			float* const ip = cpu_inputs[p]->data.f32 + j * 32 * 32 * 3;
			float* const cp = categorized->matrix->data.f32;
			int fi, fj, fk;
			const int flip = dsfmt_genrand_close_open(&dsfmt) >= 0.5;
			const int padx = (int)(dsfmt_genrand_close_open(&dsfmt) * 8 + 0.5) - 4;
			const int pady = (int)(dsfmt_genrand_close_open(&dsfmt) * 8 + 0.5) - 4;
			if (!flip)
			{
				for (fi = ccv_max(0, pady); fi < ccv_min(32 + pady, 32); fi++)
					for (fj = ccv_max(0, padx); fj < ccv_min(32 + padx, 32); fj++)
						for (fk = 0; fk < 3; fk++)
							ip[fi * 32 + fj + fk * 32 * 32] = cp[(fi - pady) * 32 * 3 + (fj - padx) * 3 + fk] - mean[fk];
			} else {
				for (fi = ccv_max(0, pady); fi < ccv_min(32 + pady, 32); fi++)
					for (fj = ccv_max(0, padx); fj < ccv_min(32 + padx, 32); fj++)
						for (fk = 0; fk < 3; fk++)
							ip[fi * 32 + (31 - fj) + fk * 32 * 32] = cp[(fi - pady) * 32 * 3 + (fj - padx) * 3 + fk] - mean[fk];
			}
			assert(categorized->c >= 0 && categorized->c < 10);
			cpu_fits[p]->data.f32[j] = categorized->c;
		}
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(cpu_inputs[p], cpu_fits[p]), TENSOR_LIST(input_tensors[p], fit_tensors[p]), stream_contexts[p]);
		ccv_nnc_stream_context_wait(stream_contexts[q]); // Need to wait the other context to finish, we use the same tensor_arena.
		ccv_cnnp_model_fit(cifar_10, TENSOR_LIST(input_tensors[p]), TENSOR_LIST(fit_tensors[p]), TENSOR_LIST(output_tensors[p]), stream_contexts[p]);
		if ((i + 1) % epoch_end == 0)
		{
			ccv_nnc_stream_context_wait(stream_contexts[p]);
			ccv_nnc_stream_context_wait(stream_contexts[q]);
			unsigned int elapsed_time = get_current_time() - current_time;
			int correct = 0;
			p = 0, q = 1;
			for (j = 0; j < test_set->rnum; j += batch_size)
			{
				for (k = 0; k < ccv_min(test_set->rnum - j, batch_size); k++)
				{
					ccv_categorized_t* const categorized = (ccv_categorized_t*)ccv_array_get(test_set, j + k);
					float* const ip = cpu_inputs[p]->data.f32 + k * 32 * 32 * 3;
					float* const cp = categorized->matrix->data.f32;
					int fi, fj, fk;
					for (fi = 0; fi < 32; fi++)
						for (fj = 0; fj < 32; fj++)
							for (fk = 0; fk < 3; fk++)
								ip[fi * 32 + fj + fk * 32 * 32] = cp[fi * 32 * 3 + fj * 3 + fk] - mean[fk];
					assert(categorized->c >= 0 && categorized->c < 10);
					c[k] = categorized->c;
				}
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(cpu_inputs[p], cpu_fits[p]), TENSOR_LIST(input_tensors[p], fit_tensors[p]), 0);
				ccv_cnnp_model_evaluate(cifar_10, TENSOR_LIST(input_tensors[p]), TENSOR_LIST(output_tensors[p]), 0);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(output_tensors[p]), TENSOR_LIST(cpu_output), 0);
				for (k = 0; k < ccv_min(test_set->rnum - j, batch_size); k++)
				{
					float max = -FLT_MAX;
					int t = -1;
					int fi;
					for (fi = 0; fi < 10; fi++)
						if (cpu_output->data.f32[k * 10 + fi] > max)
							max = cpu_output->data.f32[k * 10 + fi], t = fi;
					if (c[k] == t)
						++correct;
				}
			}
			PRINT(CCV_CLI_INFO, "Epoch %03d (%d), %.2f%% (%.3f seconds)\n", (i + 1) / epoch_end, epoch_end * batch_size, (float)correct / test_set->rnum * 100, (float)elapsed_time / 1000);
			current_time = get_current_time();
		}
		if ((i + 1) % 10000 == 0)
		{
			learn_rate *= 0.5;
			ccv_cnnp_model_set_minimizer(cifar_10, CMD_SGD_FORWARD(learn_rate, 0.99, 0.9, 0.9));
		}
		int t;
		CCV_SWAP(p, q, t);
	}
	ccv_cnnp_model_free(cifar_10);
	ccv_nnc_stream_context_free(stream_contexts[0]);
	ccv_nnc_tensor_free(input_tensors[0]);
	ccv_nnc_tensor_free(fit_tensors[0]);
	ccv_nnc_tensor_free(output_tensors[0]);
	ccv_nnc_stream_context_free(stream_contexts[1]);
	ccv_nnc_tensor_free(input_tensors[1]);
	ccv_nnc_tensor_free(fit_tensors[1]);
	ccv_nnc_tensor_free(output_tensors[1]);
	ccv_nnc_tensor_free(cpu_inputs[0]);
	ccv_nnc_tensor_free(cpu_inputs[1]);
	ccv_nnc_tensor_free(cpu_fits[0]);
	ccv_nnc_tensor_free(cpu_fits[1]);
	ccv_nnc_tensor_free(cpu_output);
}

int main(int argc, char** argv)
{
	ccv_nnc_init();
	assert(argc == 5);
	int num1 = atoi(argv[2]);
	int num2 = atoi(argv[4]);
	FILE* r1 = fopen(argv[1], "rb");
	FILE* r2 = fopen(argv[3], "rb");
	if (r1 && r2)
	{
		int i, j, k;
		unsigned char bytes[32 * 32 + 1];
		double mean[3] = {};
		ccv_array_t* categorizeds = ccv_array_new(sizeof(ccv_categorized_t), num1, 0);
		for (k = 0; k < num1; k++)
		{
			fread(bytes, 32 * 32 + 1, 1, r1);
			double per_mean[3] = {};
			int c = bytes[0];
			ccv_dense_matrix_t* a = ccv_dense_matrix_new(32, 32, CCV_32F | CCV_C3, 0, 0);
			for (i = 0; i < 32; i++)
				for (j = 0; j < 32; j++)
					per_mean[0] += (a->data.f32[(j + i * 32) * 3] = bytes[j + i * 32 + 1] * 2. / 255.);
			fread(bytes, 32 * 32, 1, r1);
			for (i = 0; i < 32; i++)
				for (j = 0; j < 32; j++)
					per_mean[1] += (a->data.f32[(j + i * 32) * 3 + 1] = bytes[j + i * 32] * 2. / 255.);
			fread(bytes, 32 * 32, 1, r1);
			for (i = 0; i < 32; i++)
				for (j = 0; j < 32; j++)
					per_mean[2] += (a->data.f32[(j + i * 32) * 3 + 2] = bytes[j + i * 32] * 2. / 255.);
			ccv_categorized_t categorized = ccv_categorized(c, a, 0);
			ccv_array_push(categorizeds, &categorized);
			mean[0] += per_mean[0] / (32 * 32);
			mean[1] += per_mean[1] / (32 * 32);
			mean[2] += per_mean[2] / (32 * 32);
		}
		ccv_array_t* tests = ccv_array_new(sizeof(ccv_categorized_t), num2, 0);
		for (k = 0; k < num2; k++)
		{
			fread(bytes, 32 * 32 + 1, 1, r2);
			int c = bytes[0];
			ccv_dense_matrix_t* a = ccv_dense_matrix_new(32, 32, CCV_32F | CCV_C3, 0, 0);
			for (i = 0; i < 32; i++)
				for (j = 0; j < 32; j++)
					a->data.f32[(j + i * 32) * 3] = bytes[j + i * 32 + 1] * 2. / 255.;
			fread(bytes, 32 * 32, 1, r2);
			for (i = 0; i < 32; i++)
				for (j = 0; j < 32; j++)
					a->data.f32[(j + i * 32) * 3 + 1] = bytes[j + i * 32] * 2. / 255.;
			fread(bytes, 32 * 32, 1, r2);
			for (i = 0; i < 32; i++)
				for (j = 0; j < 32; j++)
					a->data.f32[(j + i * 32) * 3 + 2] = bytes[j + i * 32] * 2. / 255.;
			ccv_categorized_t categorized = ccv_categorized(c, a, 0);
			ccv_array_push(tests, &categorized);
		}
		float meanf[3];
		meanf[0] = mean[0] / num1;
		meanf[1] = mean[1] / num1;
		meanf[2] = mean[2] / num1;
		train_cifar_10(categorizeds, 256, meanf, tests);
	}
	if (r1)
		fclose(r1);
	if (r2)
		fclose(r2);
	return 0;
}
