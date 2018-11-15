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

ccv_cnnp_model_t* _cifar_10_resnet16(void)
{
	const ccv_cnnp_model_io_t input = ccv_cnnp_input();
	ccv_cnnp_model_t* init_conv = ccv_cnnp_convolution(1, 16, DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
		.no_bias = 1,
		.hint = HINT((1, 1), (1, 1)),
	});
	ccv_cnnp_model_io_t output = ccv_cnnp_model_apply(init_conv, MODEL_IO_LIST(input));
	output = ccv_cnnp_model_apply(_block_layer_new(16, 1, 1, 2), MODEL_IO_LIST(output));
	output = ccv_cnnp_model_apply(_block_layer_new(32, 2, 1, 2), MODEL_IO_LIST(output));
	output = ccv_cnnp_model_apply(_block_layer_new(64, 2, 1, 3), MODEL_IO_LIST(output));
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

typedef struct {
	ccv_nnc_tensor_t* input;
	ccv_nnc_tensor_t* fit;
} ccv_nnc_input_fit_t;

static void _input_fit_deinit(void* const self, void* const context)
{
	ccv_nnc_input_fit_t* const input_fit = (ccv_nnc_input_fit_t*)self;
	ccv_nnc_tensor_free(input_fit->input);
	ccv_nnc_tensor_free(input_fit->fit);
	ccfree(input_fit);
}

typedef struct {
	dsfmt_t dsfmt;
	float mean[3];
	int batch_size;
	int device_count;
} ccv_nnc_reduce_context_t;

static void _reduce_train_batch_new(void** const input_data, const int batch_size, void** const output_data, void* const context, ccv_nnc_stream_context_t* const stream_context)
{
	ccv_nnc_reduce_context_t* const reduce_context = (ccv_nnc_reduce_context_t*)context;
	const int total_size = reduce_context->batch_size * reduce_context->device_count;
	if (!output_data[0])
	{
		ccv_nnc_input_fit_t* const input_fit = (ccv_nnc_input_fit_t*)(output_data[0] = ccmalloc(sizeof(ccv_nnc_input_fit_t)));
		input_fit->input = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(total_size, 3, 32, 32), 0);
		ccv_nnc_tensor_pin_memory(input_fit->input);
		input_fit->fit = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(total_size, 1), 0);
		ccv_nnc_tensor_pin_memory(input_fit->fit);
	}
	ccv_nnc_input_fit_t* const input_fit = (ccv_nnc_input_fit_t*)output_data[0];
	memset(input_fit->input->data.f32, 0, sizeof(float) * total_size * 3 * 32 * 32);
	float* mean = reduce_context->mean;
	int i;
	for (i = 0; i < total_size; i++)
	{
		const int b = i % batch_size;
		ccv_categorized_t* const categorized = (ccv_categorized_t*)input_data[b];
		float* const ip = input_fit->input->data.f32 + i * 32 * 32 * 3;
		float* const cp = categorized->matrix->data.f32;
		int fi, fj, fk;
		const int flip = dsfmt_genrand_close_open(&reduce_context->dsfmt) >= 0.5;
		const int padx = (int)(dsfmt_genrand_close_open(&reduce_context->dsfmt) * 8 + 0.5) - 4;
		const int pady = (int)(dsfmt_genrand_close_open(&reduce_context->dsfmt) * 8 + 0.5) - 4;
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
		input_fit->fit->data.f32[i] = categorized->c;
	}
}

static void _reduce_test_batch_new(void** const input_data, const int batch_size, void** const output_data, void* const context, ccv_nnc_stream_context_t* const stream_context)
{
	ccv_nnc_reduce_context_t* const reduce_context = (ccv_nnc_reduce_context_t*)context;
	const int total_size = reduce_context->batch_size * reduce_context->device_count;
	if (!output_data[0])
	{
		ccv_nnc_input_fit_t* const input_fit = (ccv_nnc_input_fit_t*)(output_data[0] = ccmalloc(sizeof(ccv_nnc_input_fit_t)));
		input_fit->input = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(total_size, 3, 32, 32), 0);
		ccv_nnc_tensor_pin_memory(input_fit->input);
		input_fit->fit = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(total_size, 1), 0);
		ccv_nnc_tensor_pin_memory(input_fit->fit);
	}
	ccv_nnc_input_fit_t* const input_fit = (ccv_nnc_input_fit_t*)output_data[0];
	float* mean = reduce_context->mean;
	parallel_for(i, total_size) {
		const int b = i % batch_size;
		ccv_categorized_t* const categorized = (ccv_categorized_t*)input_data[b];
		float* const ip = input_fit->input->data.f32 + i * 32 * 32 * 3;
		float* const cp = categorized->matrix->data.f32;
		int fi, fj, fk;
		for (fi = 0; fi < 32; fi++)
			for (fj = 0; fj < 32; fj++)
				for (fk = 0; fk < 3; fk++)
					ip[fi * 32 + fj + fk * 32 * 32] = cp[fi * 32 * 3 + fj * 3 + fk] - mean[fk];
		assert(categorized->c >= 0 && categorized->c < 10);
		input_fit->fit->data.f32[i] = categorized->c;
	} parallel_endfor
}

typedef struct {
	int device_id;
	int batch_size;
} ccv_nnc_map_context_t;

static void _copy_to_gpu(void*** const column_data, const int column_size, const int batch_size, void** const data, void* const context, ccv_nnc_stream_context_t* const stream_context)
{
	ccv_nnc_map_context_t* const map_context = (ccv_nnc_map_context_t*)context;
	ccv_nnc_tensor_param_t input = GPU_TENSOR_NCHW(000, map_context->batch_size, 3, 32, 32);
	ccv_nnc_tensor_param_t fit = GPU_TENSOR_NCHW(000, map_context->batch_size, 1);
	CCV_TENSOR_SET_DEVICE_ID(input.type, map_context->device_id);
	CCV_TENSOR_SET_DEVICE_ID(fit.type, map_context->device_id);
	int i;
	for (i = 0; i < batch_size; i++)
	{
		if (!data[i])
		{
			ccv_nnc_input_fit_t* const input_fit = (ccv_nnc_input_fit_t*)(data[i] = ccmalloc(sizeof(ccv_nnc_input_fit_t)));
			input_fit->input = ccv_nnc_tensor_new(0, input, 0);
			input_fit->fit = ccv_nnc_tensor_new(0, fit, 0);
		}
		ccv_nnc_input_fit_t* const gpu_input_fit = data[i];
		ccv_nnc_input_fit_t* const input_fit = column_data[0][i];
		ccv_nnc_tensor_t input = ccv_nnc_tensor(input_fit->input->data.f32 + map_context->device_id * map_context->batch_size * 3 * 32 * 32, CPU_TENSOR_NCHW(map_context->batch_size, 3, 32, 32), 0);
		ccv_nnc_tensor_t fit = ccv_nnc_tensor(input_fit->fit->data.f32 + map_context->device_id * map_context->batch_size, CPU_TENSOR_NCHW(map_context->batch_size, 1), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(&input, &fit), TENSOR_LIST(gpu_input_fit->input, gpu_input_fit->fit), stream_context);
	}
}

static int train_cifar_10(ccv_array_t* const training_set, const int batch_size, const float mean[3], ccv_array_t* const test_set)
{
	ccv_cnnp_model_t* const cifar_10 = _cifar_10_resnet16();
	const int device_count = ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU);
	if (device_count < 1)
		return -1;
	const ccv_nnc_tensor_param_t input = GPU_TENSOR_NCHW(000, batch_size, 3, 32, 32);
	float learn_rate = 0.001;
	ccv_cnnp_model_compile(cifar_10, &input, 1, CMD_SGD_FORWARD(learn_rate, 0.99, 0.9, 0.9), CMD_CATEGORICAL_CROSSENTROPY_FORWARD());
	int i, j, k;
	ccv_nnc_tensor_t* cpu_outputs[device_count];
	for (i = 0; i < device_count; i++)
	{
		cpu_outputs[i] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(batch_size, 10), 0);
		ccv_nnc_tensor_pin_memory(cpu_outputs[i]);
	}
	ccv_cnnp_dataframe_t* const raw_train_data = ccv_cnnp_dataframe_from_array_new(training_set);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_reduce_context_t reduce_context = {
		.dsfmt = dsfmt,
		.mean = {
			mean[0], mean[1], mean[2]
		},
		.batch_size = batch_size,
		.device_count = device_count
	};
	ccv_cnnp_dataframe_shuffle(raw_train_data);
	ccv_cnnp_dataframe_t* const batch_train_data = ccv_cnnp_dataframe_reduce_new(raw_train_data, _reduce_train_batch_new, _input_fit_deinit, 0, batch_size * device_count, &reduce_context, 0);
	ccv_cnnp_dataframe_t* const raw_test_data = ccv_cnnp_dataframe_from_array_new(test_set);
	ccv_cnnp_dataframe_t* const batch_test_data = ccv_cnnp_dataframe_reduce_new(raw_test_data, _reduce_test_batch_new, _input_fit_deinit, 0, batch_size * device_count, &reduce_context, 0);
	int train_device_columns[device_count * 2];
	int test_device_columns[device_count * 2];
	ccv_nnc_map_context_t map_context[device_count];
	for (i = 0; i < device_count; i++)
	{
		map_context[i].device_id = i;
		map_context[i].batch_size = batch_size;
		int stream_type = CCV_STREAM_CONTEXT_GPU;
		CCV_STREAM_SET_DEVICE_ID(stream_type, i);
		train_device_columns[i] = ccv_cnnp_dataframe_map(batch_train_data, _copy_to_gpu, stream_type, _input_fit_deinit, COLUMN_ID_LIST(0), map_context + i, 0);
		ccv_nnc_tensor_param_t params = GPU_TENSOR_NCHW(000, batch_size, 10);
		CCV_TENSOR_SET_DEVICE_ID(params.type, i);
		train_device_columns[device_count + i] = ccv_cnnp_dataframe_add_aux_tensors(batch_train_data, params);
		test_device_columns[i] = ccv_cnnp_dataframe_map(batch_test_data, _copy_to_gpu, stream_type, _input_fit_deinit, COLUMN_ID_LIST(0), map_context + i, 0);
		test_device_columns[device_count + i] = ccv_cnnp_dataframe_add_aux_tensors(batch_test_data, params);
	}
	ccv_cnnp_dataframe_iter_t* const test_iter = ccv_cnnp_dataframe_iter_new(batch_test_data, test_device_columns, device_count * 2);
	ccv_cnnp_dataframe_iter_t* const iter = ccv_cnnp_dataframe_iter_new(batch_train_data, train_device_columns, device_count * 2);
	ccv_nnc_stream_context_t* stream_contexts[2];
	stream_contexts[0] = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	stream_contexts[1] = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	int p = 0, q = 1;
	const int epoch_end = (training_set->rnum + batch_size - 1) / (batch_size * device_count);
	int correct = 0;
	int epoch = 0;
	ccv_cnnp_model_set_data_parallel(cifar_10, device_count);
	ccv_cnnp_dataframe_iter_prefetch(iter, 1, stream_contexts[p]);
	ccv_nnc_input_fit_t* input_fits[device_count * 2];
	ccv_nnc_tensor_t* input_fit_inputs[device_count];
	ccv_nnc_tensor_t* input_fit_fits[device_count];
	ccv_nnc_tensor_t* outputs[device_count];
	for (i = 0; epoch < 30; i++)
	{
		ccv_cnnp_dataframe_iter_next(iter, (void**)input_fits, device_count * 2, stream_contexts[p]);
		ccv_nnc_stream_context_wait(stream_contexts[q]); // Need to wait the other context to finish, we use the same tensor_arena.
		for (j = 0; j < device_count; j++)
		{
			input_fit_inputs[j] = input_fits[j]->input;
			input_fit_fits[j] = input_fits[j]->fit;
			outputs[j] = (ccv_nnc_tensor_t*)input_fits[device_count + j];
		}
		ccv_cnnp_model_fit(cifar_10, input_fit_inputs, device_count, input_fit_fits, device_count, outputs, device_count, stream_contexts[p]);
		// Prefetch the next round.
		ccv_cnnp_dataframe_iter_prefetch(iter, 1, stream_contexts[q]);
		if ((i + 1) % epoch_end == 0)
		{
			++epoch;
			if (epoch % 5 == 0)
			{
				learn_rate *= 0.5;
				ccv_cnnp_model_set_minimizer(cifar_10, CMD_SGD_FORWARD(learn_rate, 0.99, 0.9, 0.9));
			}
			ccv_nnc_stream_context_wait(stream_contexts[p]);
			ccv_nnc_stream_context_wait(stream_contexts[q]);
			correct = 0;
			p = 0, q = 1;
			for (j = 0; j < test_set->rnum; j += batch_size * device_count)
			{
				ccv_cnnp_dataframe_iter_next(test_iter, (void**)input_fits, device_count * 2, 0);
				for (k = 0; k < device_count; k++)
				{
					input_fit_inputs[k] = input_fits[k]->input;
					outputs[k] = (ccv_nnc_tensor_t*)input_fits[device_count + k];
				}
				ccv_cnnp_model_evaluate(cifar_10, input_fit_inputs, device_count, outputs, device_count, 0);
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
			ccv_cnnp_dataframe_iter_set_cursor(test_iter, 0);
			// Reshuffle and reset cursor.
			ccv_cnnp_dataframe_shuffle(raw_train_data);
			ccv_cnnp_dataframe_iter_set_cursor(iter, 0);
		}
		int t;
		CCV_SWAP(p, q, t);
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

TEST_CASE("cifar-10 with resnet16 to > 85% under 3 minutes")
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
	ccv_array_t* tests = ccv_array_new(sizeof(ccv_categorized_t), test_count, 0);
	for (k = 0; k < test_count; k++)
	{
		fread(bytes, 32 * 32 + 1, 1, test);
		int c = bytes[0];
		ccv_dense_matrix_t* a = ccv_dense_matrix_new(32, 32, CCV_32F | CCV_C3, 0, 0);
		for (i = 0; i < 32; i++)
			for (j = 0; j < 32; j++)
				a->data.f32[(j + i * 32) * 3] = bytes[j + i * 32 + 1] * 2. / 255.;
		fread(bytes, 32 * 32, 1, test);
		for (i = 0; i < 32; i++)
			for (j = 0; j < 32; j++)
				a->data.f32[(j + i * 32) * 3 + 1] = bytes[j + i * 32] * 2. / 255.;
		fread(bytes, 32 * 32, 1, test);
		for (i = 0; i < 32; i++)
			for (j = 0; j < 32; j++)
				a->data.f32[(j + i * 32) * 3 + 2] = bytes[j + i * 32] * 2. / 255.;
		ccv_categorized_t categorized = ccv_categorized(c, a, 0);
		ccv_array_push(tests, &categorized);
	}
	float meanf[3];
	meanf[0] = mean[0] / train_count;
	meanf[1] = mean[1] / train_count;
	meanf[2] = mean[2] / train_count;
	int correct = train_cifar_10(categorizeds, 256, meanf, tests);
	fclose(train);
	fclose(test);
	REQUIRE(correct > 8500, "accuracy %.2f after 30 epoch should be higher than 85%%", (float)correct / 10000);
}

#include "case_main.h"
