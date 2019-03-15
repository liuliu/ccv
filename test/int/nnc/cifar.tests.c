#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <3rdparty/dsfmt/dSFMT.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

static ccv_cnnp_model_t* _dawn_layer_new(const int filters, const int strides, const int residual)
{
	ccv_cnnp_model_io_t input = ccv_cnnp_input();
	ccv_cnnp_model_t* conv = ccv_cnnp_convolution(1, filters, DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
		.norm = CCV_CNNP_BATCH_NORM,
		.activation = CCV_CNNP_ACTIVATION_RELU,
		.hint = HINT((1, 1), (1, 1)),
	});
	ccv_cnnp_model_io_t output = ccv_cnnp_model_apply(conv, MODEL_IO_LIST(input));
	ccv_cnnp_model_t* pool = ccv_cnnp_max_pool(DIM_ALLOC(strides, strides), (ccv_cnnp_param_t){
		.hint = HINT((strides, strides), (0, 0)),
	});
	output = ccv_cnnp_model_apply(pool, MODEL_IO_LIST(output));
	if (residual)
	{
		ccv_cnnp_model_io_t shortcut = output;
		ccv_cnnp_model_t* res1 = ccv_cnnp_convolution(1, filters, DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.norm = CCV_CNNP_BATCH_NORM,
			.activation = CCV_CNNP_ACTIVATION_RELU,
			.hint = HINT((1, 1), (1, 1)),
		});
		output = ccv_cnnp_model_apply(res1, MODEL_IO_LIST(output));
		ccv_cnnp_model_t* res2 = ccv_cnnp_convolution(1, filters, DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.norm = CCV_CNNP_BATCH_NORM,
			.activation = CCV_CNNP_ACTIVATION_RELU,
			.hint = HINT((1, 1), (1, 1)),
		});
		output = ccv_cnnp_model_apply(res2, MODEL_IO_LIST(output));
		ccv_cnnp_model_t* const add = ccv_cnnp_add();
		output = ccv_cnnp_model_apply(add, MODEL_IO_LIST(output, shortcut));
	}
	return ccv_cnnp_model_new(MODEL_IO_LIST(input), MODEL_IO_LIST(output));
}

static ccv_cnnp_model_t* _cifar_10_dawn(void)
{
	ccv_cnnp_model_t* prep = ccv_cnnp_convolution(1, 64, DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
		.norm = CCV_CNNP_BATCH_NORM,
		.activation = CCV_CNNP_ACTIVATION_RELU,
		.hint = HINT((1, 1), (1, 1)),
	});
	ccv_cnnp_model_t* layer1 = _dawn_layer_new(128, 2, 1);
	ccv_cnnp_model_t* layer2 = _dawn_layer_new(256, 2, 0);
	ccv_cnnp_model_t* layer3 = _dawn_layer_new(512, 2, 1);
	return ccv_cnnp_sequential_new(MODEL_LIST(
		prep,
		layer1,
		layer2,
		layer3,
		ccv_cnnp_max_pool(DIM_ALLOC(0, 0), (ccv_cnnp_param_t){}),
		ccv_cnnp_flatten(),
		ccv_cnnp_dense(10, (ccv_cnnp_param_t){
			.activation = CCV_CNNP_ACTIVATION_SOFTMAX,
		})));
}

static int train_cifar_10(ccv_array_t* const training_set, const int batch_size, const float mean[3], ccv_array_t* const test_set)
{
	ccv_cnnp_model_t* const cifar_10 = _cifar_10_dawn();
	const int device_count = ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU);
	if (device_count < 1)
		return -1;
	const ccv_nnc_tensor_param_t input = GPU_TENSOR_NCHW(000, 32F, batch_size, 3, 32, 32);
	float learn_rate = 0.001;
	ccv_cnnp_model_compile(cifar_10, &input, 1, CMD_SGD_FORWARD(learn_rate, 0.99, 0.9, 0.9), CMD_CATEGORICAL_CROSSENTROPY_FORWARD());
	int i, j, k;
	ccv_nnc_tensor_t* cpu_outputs[device_count];
	for (i = 0; i < device_count; i++)
	{
		cpu_outputs[i] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, batch_size, 10), 0);
		ccv_nnc_tensor_pin_memory(cpu_outputs[i]);
	}
	ccv_cnnp_dataframe_t* const raw_train_data = ccv_cnnp_dataframe_from_array_new(training_set);
	const ccv_cnnp_random_jitter_t random_jitter = {
		.resize = {
			.min = 32,
			.max = 32,
		},
		.size = {
			.rows = 32,
			.cols = 32,
		},
		.symmetric = 1,
		.normalize = {
			.mean = {
				mean[0], mean[1], mean[2],
			},
		},
		.offset = {
			.x = 4,
			.y = 4,
		},
		.seed = 1,
	};
	const int images = ccv_cnnp_dataframe_extract_value(raw_train_data, 0, offsetof(ccv_categorized_t, matrix));
	const int jitter_images = ccv_cnnp_dataframe_image_random_jitter(raw_train_data, images, CCV_32F, random_jitter);
	const int one_hot = ccv_cnnp_dataframe_one_hot(raw_train_data, 0, offsetof(ccv_categorized_t, c), 10, 1, 0, CCV_32F, CCV_TENSOR_FORMAT_NCHW);
	ccv_cnnp_dataframe_t* const batch_train_data = ccv_cnnp_dataframe_batching_new(raw_train_data, COLUMN_ID_LIST(jitter_images, one_hot), batch_size, device_count, CCV_TENSOR_FORMAT_NCHW);
	ccv_cnnp_dataframe_t* const raw_test_data = ccv_cnnp_dataframe_from_array_new(test_set);
	const int test_images = ccv_cnnp_dataframe_extract_value(raw_test_data, 0, offsetof(ccv_categorized_t, matrix));
	ccv_cnnp_dataframe_t* const batch_test_data = ccv_cnnp_dataframe_batching_new(raw_test_data, COLUMN_ID_LIST(test_images), batch_size, device_count, CCV_TENSOR_FORMAT_NCHW);
	int train_device_columns[device_count * 2];
	int test_device_columns[device_count * 2];
	for (i = 0; i < device_count; i++)
	{
		int stream_type = CCV_STREAM_CONTEXT_GPU;
		CCV_STREAM_SET_DEVICE_ID(stream_type, i);
		train_device_columns[i] = ccv_cnnp_dataframe_copy_to_gpu(batch_train_data, 0, i * 2, 2, i);
		ccv_nnc_tensor_param_t params = GPU_TENSOR_NCHW(000, 32F, batch_size, 10);
		CCV_TENSOR_SET_DEVICE_ID(params.type, i);
		train_device_columns[device_count + i] = ccv_cnnp_dataframe_add_aux(batch_train_data, params);
		test_device_columns[i] = ccv_cnnp_dataframe_copy_to_gpu(batch_test_data, 0, i, 1, i);
		test_device_columns[device_count + i] = ccv_cnnp_dataframe_add_aux(batch_test_data, params);
	}
	ccv_cnnp_dataframe_iter_t* const test_iter = ccv_cnnp_dataframe_iter_new(batch_test_data, test_device_columns, device_count * 2);
	ccv_cnnp_dataframe_iter_t* const iter = ccv_cnnp_dataframe_iter_new(batch_train_data, train_device_columns, device_count * 2);
	ccv_nnc_stream_context_t* stream_contexts[2];
	stream_contexts[0] = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	stream_contexts[1] = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	int p = 0, q = 1;
	const int epoch_end = (training_set->rnum + batch_size * device_count - 1) / (batch_size * device_count);
	int correct = 0;
	int epoch = 0;
	ccv_cnnp_model_set_data_parallel(cifar_10, device_count);
	ccv_cnnp_dataframe_iter_prefetch(iter, 1, stream_contexts[p]);
	ccv_nnc_tensor_t** input_fits[device_count * 2];
	ccv_nnc_tensor_t* input_fit_inputs[device_count];
	ccv_nnc_tensor_t* input_fit_fits[device_count];
	ccv_nnc_tensor_t* outputs[device_count];
	for (i = 0; epoch < 30; i++)
	{
		// Piece-wise linear learning rate: https://www.myrtle.ai/2018/09/24/how_to_train_your_resnet_3/
		learn_rate = ((i + 1) < 10 * epoch_end ? 0.4 * (i + 1) / (10 * epoch_end) : 0.4 * (35 * epoch_end - (i + 1)) / ((35 - 10) * epoch_end)) / batch_size;
		learn_rate = ccv_max(learn_rate, 0.000001);
		ccv_cnnp_model_set_minimizer(cifar_10, CMD_SGD_FORWARD(learn_rate, 0.99, 0.9, 0.9));
		ccv_cnnp_dataframe_iter_next(iter, (void**)input_fits, device_count * 2, stream_contexts[p]);
		ccv_nnc_stream_context_wait(stream_contexts[q]); // Need to wait the other context to finish, we use the same tensor_arena.
		for (j = 0; j < device_count; j++)
		{
			input_fit_inputs[j] = input_fits[j][0];
			input_fit_fits[j] = input_fits[j][1];
			outputs[j] = (ccv_nnc_tensor_t*)input_fits[device_count + j];
		}
		ccv_cnnp_model_fit(cifar_10, input_fit_inputs, device_count, input_fit_fits, device_count, outputs, device_count, stream_contexts[p]);
		// Prefetch the next round.
		ccv_cnnp_dataframe_iter_prefetch(iter, 1, stream_contexts[q]);
		if ((i + 1) % epoch_end == 0)
		{
			++epoch;
			// Reshuffle and reset cursor.
			ccv_cnnp_dataframe_shuffle(raw_train_data);
			ccv_cnnp_dataframe_iter_set_cursor(iter, 0);
		}
		int t;
		CCV_SWAP(p, q, t);
	}
	ccv_cnnp_dataframe_iter_set_cursor(test_iter, 0);
	ccv_nnc_stream_context_wait(stream_contexts[p]);
	ccv_nnc_stream_context_wait(stream_contexts[q]);
	correct = 0;
	p = 0, q = 1;
	for (j = 0; j < test_set->rnum; j += batch_size * device_count)
	{
		ccv_cnnp_dataframe_iter_next(test_iter, (void**)input_fits, device_count * 2, 0);
		for (k = 0; k < device_count; k++)
		{
			input_fit_inputs[k] = input_fits[k][0];
			outputs[k] = (ccv_nnc_tensor_t*)input_fits[device_count + k];
		}
		ccv_cnnp_model_evaluate(cifar_10, 0, input_fit_inputs, device_count, outputs, device_count, 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, outputs, device_count, cpu_outputs, device_count, 0);
		for (k = 0; k < ccv_min(test_set->rnum - j, batch_size * device_count); k++)
		{
			ccv_categorized_t* const categorized = (ccv_categorized_t*)ccv_array_get(test_set, j + k);
			const int d = k / batch_size;
			const int b = k % batch_size;
			float max = -FLT_MAX;
			int t = -1;
			int fi;
			for (fi = 0; fi < 10; fi++)
				if (cpu_outputs[d]->data.f32[b * 10 + fi] > max)
					max = cpu_outputs[d]->data.f32[b * 10 + fi], t = fi;
			if (categorized->c == t)
				++correct;
		}
	}
	ccv_cnnp_dataframe_iter_free(iter);
	ccv_cnnp_dataframe_free(batch_train_data);
	ccv_cnnp_dataframe_free(raw_train_data);
	ccv_cnnp_dataframe_iter_free(test_iter);
	ccv_cnnp_dataframe_free(batch_test_data);
	ccv_cnnp_dataframe_free(raw_test_data);
	ccv_cnnp_model_free(cifar_10);
	ccv_nnc_stream_context_free(stream_contexts[0]);
	ccv_nnc_stream_context_free(stream_contexts[1]);
	for (i = 0; i < device_count; i++)
		ccv_nnc_tensor_free(cpu_outputs[i]);
	return correct;
}

TEST_CASE("cifar-10 with resnet20 to > 90% under 3 minutes")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) &&
			ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_BACKWARD, CCV_NNC_BACKEND_GPU_CUDNN));
	FILE* train = fopen("/fast/Data/cifar-10/cifar-10-batches-bin/data_batch.bin", "rb");
	FILE* test = fopen("/fast/Data/cifar-10/cifar-10-batches-bin/test_batch.bin", "rb");
	if (!train || !test)
	{
		if (train)
			fclose(train);
		if (test)
			fclose(test);
		GUARD_ELSE_RETURN(0);
	}
	int i, j, k;
	unsigned char bytes[32 * 32 + 1];
	double mean[3] = {};
	const int train_count = 50000;
	const int test_count = 10000;
	ccv_array_t* categorizeds = ccv_array_new(sizeof(ccv_categorized_t), train_count, 0);
	for (k = 0; k < train_count; k++)
	{
		fread(bytes, 32 * 32 + 1, 1, train);
		double per_mean[3] = {};
		int c = bytes[0];
		ccv_dense_matrix_t* a = ccv_dense_matrix_new(32, 32, CCV_32F | CCV_C3, 0, 0);
		for (i = 0; i < 32; i++)
			for (j = 0; j < 32; j++)
				per_mean[0] += (a->data.f32[(j + i * 32) * 3] = bytes[j + i * 32 + 1] * 2. / 255.);
		fread(bytes, 32 * 32, 1, train);
		for (i = 0; i < 32; i++)
			for (j = 0; j < 32; j++)
				per_mean[1] += (a->data.f32[(j + i * 32) * 3 + 1] = bytes[j + i * 32] * 2. / 255.);
		fread(bytes, 32 * 32, 1, train);
		for (i = 0; i < 32; i++)
			for (j = 0; j < 32; j++)
				per_mean[2] += (a->data.f32[(j + i * 32) * 3 + 2] = bytes[j + i * 32] * 2. / 255.);
		ccv_categorized_t categorized = ccv_categorized(c, a, 0);
		ccv_array_push(categorizeds, &categorized);
		mean[0] += per_mean[0] / (32 * 32);
		mean[1] += per_mean[1] / (32 * 32);
		mean[2] += per_mean[2] / (32 * 32);
	}
	float meanf[3];
	meanf[0] = mean[0] / train_count;
	meanf[1] = mean[1] / train_count;
	meanf[2] = mean[2] / train_count;
	ccv_array_t* tests = ccv_array_new(sizeof(ccv_categorized_t), test_count, 0);
	for (k = 0; k < test_count; k++)
	{
		fread(bytes, 32 * 32 + 1, 1, test);
		int c = bytes[0];
		ccv_dense_matrix_t* a = ccv_dense_matrix_new(32, 32, CCV_32F | CCV_C3, 0, 0);
		for (i = 0; i < 32; i++)
			for (j = 0; j < 32; j++)
				a->data.f32[(j + i * 32) * 3] = bytes[j + i * 32 + 1] * 2. / 255. - meanf[0];
		fread(bytes, 32 * 32, 1, test);
		for (i = 0; i < 32; i++)
			for (j = 0; j < 32; j++)
				a->data.f32[(j + i * 32) * 3 + 1] = bytes[j + i * 32] * 2. / 255. - meanf[1];
		fread(bytes, 32 * 32, 1, test);
		for (i = 0; i < 32; i++)
			for (j = 0; j < 32; j++)
				a->data.f32[(j + i * 32) * 3 + 2] = bytes[j + i * 32] * 2. / 255. - meanf[2];
		ccv_categorized_t categorized = ccv_categorized(c, a, 0);
		ccv_array_push(tests, &categorized);
	}
	int correct = train_cifar_10(categorizeds, 256, meanf, tests);
	fclose(train);
	fclose(test);
	REQUIRE(correct > 9000, "accuracy %.2f after 30 epoch should be higher than 90%%", (float)correct / 10000);
}

#include "case_main.h"
