#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <3rdparty/dsfmt/dSFMT.h>
#include <ctype.h>
#include <3rdparty/khash/khash.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

KHASH_MAP_INIT_STR(vocab_map, int)

static CCV_WARN_UNUSED(ccv_nnc_tensor_t*) _text_to_tensor_index(const char* const filename, const khash_t(vocab_map)* const vocab, const int vocab_size, const int max_length)
{
	const int end_flag = vocab_size - 2;
	const int pad_flag = vocab_size - 1;
	char* const word = (char*)ccmalloc(1024);
	ccv_nnc_tensor_t* const tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, max_length), 0);
	FILE* const file = fopen(filename, "r");
	int t = 0;
	while (fscanf(file, "%1023s", word) != EOF)
	{
		if (t >= max_length)
			break;
		int j;
		for(j = 0; word[j]; j++)
			word[j] = tolower(word[j]);
		char* saveptr;
		const char* token = strtok_r(word, ".,<>/~`@#$%^&*+\\\"", &saveptr);
		while (token)
		{
			if (t >= max_length)
				break;
			const khiter_t k = kh_get(vocab_map, vocab, token);
			if (k != kh_end(vocab))
				tensor->data.i32[t++] = kh_val(vocab, k);
			token = strtok_r(0, ".,<>/~`@#$%^&*+\\\"", &saveptr);
		}
	}
	fclose(file);
	if (t < max_length)
	{
		tensor->data.i32[t] = end_flag;
		for (++t; t < max_length; t++)
			tensor->data.i32[t] = pad_flag;
	}
	ccfree(word);
	return tensor;
}

typedef struct {
	ccv_nnc_tensor_t* tensor;
	ccv_nnc_tensor_t* mask;
	int c;
} ccv_nnc_text_t;

static ccv_array_t* _array_from_disk_new(const char* const list, const char* const base_dir, const khash_t(vocab_map)* const vocab, const int vocab_size, const int max_length)
{
	FILE *r = fopen(list, "r");
	assert(r && "list doesn't exists");
	const int pad_flag = vocab_size - 1;
	int dirlen = (base_dir != 0) ? strlen(base_dir) + 1 : 0;
	ccv_array_t* categorizeds = ccv_array_new(sizeof(ccv_nnc_text_t), 64, 0);
	int c;
	char* file = (char*)ccmalloc(1024);
	char* filename = (char*)ccmalloc(1024);
	while (fscanf(r, "%d %1023s", &c, file) != EOF)
	{
		if (base_dir != 0)
		{
			strncpy(filename, base_dir, 1024);
			filename[dirlen - 1] = '/';
		}
		strncpy(filename + dirlen, file, 1024 - dirlen);
		ccv_nnc_tensor_t* const tensor = _text_to_tensor_index(filename, vocab, vocab_size, max_length);
		int length = 0;
		int i;
		for (i = 0; !length && i < max_length; i++)
			if (tensor->data.i32[i] == pad_flag)
				length = i;
		ccv_nnc_tensor_t* const mask = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 1), 0);
		mask->data.i32[0] = length;
		ccv_nnc_text_t categorized = {
			.tensor = tensor,
			.mask = mask,
			.c = c
		};
		ccv_array_push(categorizeds, &categorized);
	}
	ccfree(filename);
	ccfree(file);
	fclose(r);
	return categorizeds;
}

static ccv_cnnp_model_t* _self_attention_new(const int k, const int h, const int b, const int t)
{
	const ccv_cnnp_model_io_t x = ccv_cnnp_input();
	const ccv_cnnp_model_io_t mask = ccv_cnnp_input();
	ccv_cnnp_model_io_t multiheads = ccv_cnnp_model_apply(ccv_cnnp_reshape(DIM_ALLOC(b * t, k), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(x));
	ccv_cnnp_model_t* const tokeys = ccv_cnnp_dense(k * h, (ccv_cnnp_param_t){
		.no_bias = 1,
	}, "tokeys");
	ccv_cnnp_model_t* const toqueries = ccv_cnnp_dense(k * h, (ccv_cnnp_param_t){
		.no_bias = 1,
	}, "toqueries");
	ccv_cnnp_model_t* const tovalues = ccv_cnnp_dense(k * h, (ccv_cnnp_param_t){
		.no_bias = 1,
	}, "tovalues");
	ccv_cnnp_model_io_t keys = ccv_cnnp_model_apply(tokeys, MODEL_IO_LIST(multiheads));
	ccv_cnnp_model_io_t queries = ccv_cnnp_model_apply(toqueries, MODEL_IO_LIST(multiheads));
	ccv_cnnp_model_io_t values = ccv_cnnp_model_apply(tovalues, MODEL_IO_LIST(multiheads));
	keys = ccv_cnnp_model_apply(ccv_cnnp_reshape(DIM_ALLOC(b, t, h, k), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(keys));
	queries = ccv_cnnp_model_apply(ccv_cnnp_reshape(DIM_ALLOC(b, t, h, k), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(queries));
	values = ccv_cnnp_model_apply(ccv_cnnp_reshape(DIM_ALLOC(b, t, h, k), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(values));
	keys = ccv_cnnp_model_apply(ccv_cnnp_transpose(1, 2, 0), MODEL_IO_LIST(keys));
	queries = ccv_cnnp_model_apply(ccv_cnnp_transpose(1, 2, 0), MODEL_IO_LIST(queries));
	values = ccv_cnnp_model_apply(ccv_cnnp_transpose(1, 2, 0), MODEL_IO_LIST(values));
	keys = ccv_cnnp_model_apply(ccv_cnnp_reshape(DIM_ALLOC(b * h, t, k), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(keys));
	queries = ccv_cnnp_model_apply(ccv_cnnp_reshape(DIM_ALLOC(b * h, t, k), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(queries));
	values = ccv_cnnp_model_apply(ccv_cnnp_reshape(DIM_ALLOC(b * h, t, k), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(values));
	ccv_cnnp_model_io_t dot = ccv_cnnp_model_apply(ccv_cnnp_matmul(NO_TRANSPOSE, TRANSPOSE(1, 2), 0), MODEL_IO_LIST(queries, keys));
	const float scale = 1. / sqrt(k);
	dot = ccv_cnnp_model_apply(ccv_cnnp_scalar_mul(scale, 0), MODEL_IO_LIST(dot));
	dot = ccv_cnnp_model_apply(ccv_cnnp_masked_fill(0, -1e9, 0), MODEL_IO_LIST(dot, mask));
	dot = ccv_cnnp_model_apply(ccv_cnnp_reshape(DIM_ALLOC(b * h * t, t), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(dot));
	dot = ccv_cnnp_model_apply(ccv_cnnp_softmax(0), MODEL_IO_LIST(dot));
	dot = ccv_cnnp_model_apply(ccv_cnnp_dropout(0.1, 0), MODEL_IO_LIST(dot));
	dot = ccv_cnnp_model_apply(ccv_cnnp_reshape(DIM_ALLOC(b * h, t, t), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(dot));
	ccv_cnnp_model_io_t out = ccv_cnnp_model_apply(ccv_cnnp_matmul(NO_TRANSPOSE, NO_TRANSPOSE, 0), MODEL_IO_LIST(dot, values));
	out = ccv_cnnp_model_apply(ccv_cnnp_reshape(DIM_ALLOC(b, h, t, k), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_transpose(1, 2, 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_reshape(DIM_ALLOC(b * t, h * k), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(out));
	ccv_cnnp_model_t* const unifyheads = ccv_cnnp_dense(k, (ccv_cnnp_param_t){}, "unifyheads");
	out = ccv_cnnp_model_apply(unifyheads, MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_reshape(DIM_ALLOC(b, t, k), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(out));
	return ccv_cnnp_model_new(MODEL_IO_LIST(x, mask), MODEL_IO_LIST(out), "self-attention");
}

static ccv_cnnp_model_t* _transformer_block_new(const int k, const int h, const int b, const int t, const int ff)
{
	ccv_cnnp_model_io_t const x = ccv_cnnp_input();
	ccv_cnnp_model_io_t const mask = ccv_cnnp_input();
	ccv_cnnp_model_t* const self_attention = _self_attention_new(k, h, b, t);
	ccv_cnnp_model_io_t out = ccv_cnnp_model_apply(self_attention, MODEL_IO_LIST(x, mask));
	out = ccv_cnnp_model_apply(ccv_cnnp_add(0), MODEL_IO_LIST(x, out));
	ccv_cnnp_model_io_t first = ccv_cnnp_model_apply(ccv_cnnp_layer_norm(1e-5, DIM_ALLOC(2), 1, 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_dropout(0.1, 0), MODEL_IO_LIST(first));
	out = ccv_cnnp_model_apply(ccv_cnnp_reshape(DIM_ALLOC(b * t, k), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_dense(ff, (ccv_cnnp_param_t){}, 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_relu(0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_dense(k, (ccv_cnnp_param_t){}, 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_reshape(DIM_ALLOC(b, t, k), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_add(0), MODEL_IO_LIST(first, out));
	out = ccv_cnnp_model_apply(ccv_cnnp_layer_norm(1e-5, DIM_ALLOC(2), 1, 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_dropout(0.1, 0), MODEL_IO_LIST(out));
	return ccv_cnnp_model_new(MODEL_IO_LIST(x, mask), MODEL_IO_LIST(out), "transformer");
}

static ccv_cnnp_model_t* _classifier_transformer_new(const int layers, const int k, const int h, const int b, const int t, const int ff)
{
	ccv_cnnp_model_io_t const x = ccv_cnnp_input();
	ccv_cnnp_model_io_t const mask = ccv_cnnp_input();
	ccv_cnnp_model_io_t out = x;
	int i;
	for (i = 0; i < layers; i++)
		out = ccv_cnnp_model_apply(_transformer_block_new(k, h, b, t, ff), MODEL_IO_LIST(out, mask));
	out = ccv_cnnp_model_apply(ccv_cnnp_transpose(1, 2, 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_reshape(DIM_ALLOC(b, k, t, 1), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_average_pool(DIM_ALLOC(0, 0), (ccv_cnnp_param_t){}, 0), MODEL_IO_LIST(out));
	// Last layer, get it to 2.
	out = ccv_cnnp_model_apply(ccv_cnnp_flatten(0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_dense(2, (ccv_cnnp_param_t){}, 0), MODEL_IO_LIST(out));
	return ccv_cnnp_model_new(MODEL_IO_LIST(x, mask), MODEL_IO_LIST(out), "classifier");
}

typedef struct {
	int layers;
	int k;
	int h;
	int b;
	int t;
	int ff;
} classifier_transformer_params_t;

static ccv_cnnp_model_t* _dynamic_classifier_transformer(const ccv_nnc_tensor_param_t* const inputs, const int input_size, void* const context)
{
	const classifier_transformer_params_t* const params = (classifier_transformer_params_t*)context;
	return _classifier_transformer_new(params->layers, params->k, params->h, params->b, params->t, params->ff);
}

static void _vocab_init(const char* const vocab_file, khash_t(vocab_map)** const vocab_ref, int* const vocab_size_ref)
{
	FILE* const vocab_ptr = fopen(vocab_file, "r");
	khash_t(vocab_map)* const vocab = kh_init(vocab_map);
	int i, ret;
	char* const word = (char*)ccmalloc(1024);
	for (i = 0; fscanf(vocab_ptr, "%1023s", word) != EOF; i++)
	{
		const khiter_t k = kh_put(vocab_map, vocab, strdup(word), &ret);
		kh_val(vocab, k) = i;
	}
	ccfree(word);
	fclose(vocab_ptr);
	*vocab_ref = vocab;
	*vocab_size_ref = i;
}

static void _vocab_destroy(khash_t(vocab_map)* const vocab)
{
	// Free keys.
	for (khiter_t k = kh_begin(vocab); k != kh_end(vocab); k++)
		if (kh_exist(vocab, k))
			free((void*)kh_key(vocab, k));
	kh_destroy(vocab_map, vocab);
}

static int train_imdb(const int epoch_limit, const int vocab_size, const int batch_size, const int max_length, const int embedding_size, ccv_cnnp_dataframe_t* const train_data, ccv_cnnp_dataframe_t* const test_data)
{
	const int tensor_idx = ccv_cnnp_dataframe_extract_value(train_data, 0, offsetof(ccv_nnc_text_t, tensor));
	const int one_hot_idx = ccv_cnnp_dataframe_one_hot(train_data, 0, offsetof(ccv_nnc_text_t, c), 2, 1, 0, CCV_32F, CCV_TENSOR_FORMAT_NCHW);
	const int mask_idx = ccv_cnnp_dataframe_extract_value(train_data, 0, offsetof(ccv_nnc_text_t, mask));
	const int device_count = ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU);
	ccv_cnnp_dataframe_t* const batched_data = ccv_cnnp_dataframe_batching_new(train_data, COLUMN_ID_LIST(tensor_idx, one_hot_idx, mask_idx), batch_size, device_count, CCV_TENSOR_FORMAT_NCHW);
	const int test_tensor_idx = ccv_cnnp_dataframe_extract_value(test_data, 0, offsetof(ccv_nnc_text_t, tensor));
	const int test_one_hot_idx = ccv_cnnp_dataframe_one_hot(test_data, 0, offsetof(ccv_nnc_text_t, c), 2, 1, 0, CCV_32F, CCV_TENSOR_FORMAT_NCHW);
	const int test_mask_idx = ccv_cnnp_dataframe_extract_value(test_data, 0, offsetof(ccv_nnc_text_t, mask));
	ccv_cnnp_dataframe_t* const test_batched_data = ccv_cnnp_dataframe_batching_new(test_data, COLUMN_ID_LIST(test_tensor_idx, test_one_hot_idx, test_mask_idx), batch_size, device_count, CCV_TENSOR_FORMAT_NCHW);
	int gpu_batched[device_count * 2];
	int test_gpu_batched[device_count * 2];
	int i, j;
	for (i = 0; i < device_count; i++)
	{
		const int seq_len_batched = ccv_cnnp_dataframe_extract_tuple(batched_data, 0, i * 3 + 2);
		const int mask_batched = ccv_cnnp_dataframe_mask_for_sequence(batched_data, seq_len_batched, 0, max_length);
		const int tupled_mask_batched = ccv_cnnp_dataframe_make_tuple(batched_data, COLUMN_ID_LIST(mask_batched));
		gpu_batched[i] = ccv_cnnp_dataframe_copy_to_gpu(batched_data, 0, i * 3, 2, i);
		gpu_batched[i + device_count] = ccv_cnnp_dataframe_copy_to_gpu(batched_data, tupled_mask_batched, 0, 1, i);
		const int test_seq_len_batched = ccv_cnnp_dataframe_extract_tuple(test_batched_data, 0, i * 3 + 2);
		const int test_mask_batched = ccv_cnnp_dataframe_mask_for_sequence(test_batched_data, test_seq_len_batched, 0, max_length);
		const int test_tupled_mask_batched = ccv_cnnp_dataframe_make_tuple(test_batched_data, COLUMN_ID_LIST(test_mask_batched));
		test_gpu_batched[i] = ccv_cnnp_dataframe_copy_to_gpu(test_batched_data, 0, i * 3, 2, i);
		test_gpu_batched[i + device_count] = ccv_cnnp_dataframe_copy_to_gpu(test_batched_data, test_tupled_mask_batched, 0, 1, i);
	}
	ccv_cnnp_dataframe_iter_t* const iter = ccv_cnnp_dataframe_iter_new(batched_data, gpu_batched, device_count * 2);
	ccv_nnc_dynamic_graph_t* const dynamic_graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_tensor_variable_t vocab_vec[device_count];
	ccv_nnc_tensor_variable_t seq_vec[device_count];
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_tensor_param_t vocab_params = GPU_TENSOR_NCHW(000, 32F, vocab_size, embedding_size);
		CCV_TENSOR_SET_DEVICE_ID(vocab_params.type, i);
		vocab_vec[i] = ccv_nnc_tensor_variable_new(dynamic_graph, vocab_params);
		ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_RANDOM_UNIFORM_FORWARD(-1, 1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(), TENSOR_VARIABLE_LIST(vocab_vec[i]), 0, 0);
		ccv_nnc_tensor_param_t seq_params = GPU_TENSOR_NCHW(000, 32F, max_length, embedding_size);
		CCV_TENSOR_SET_DEVICE_ID(seq_params.type, i);
		seq_vec[i] = ccv_nnc_tensor_variable_new(dynamic_graph, seq_params);
		ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_RANDOM_UNIFORM_FORWARD(-1, 1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(), TENSOR_VARIABLE_LIST(seq_vec[i]), 0, 0);
	}
	ccv_nnc_tensor_t* const seq_indices_cpu = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32S, batch_size * max_length), 0);
	for (i = 0; i < batch_size; i++)
		for (j = 0; j < max_length; j++)
			seq_indices_cpu->data.i32[i * max_length + j] = j;
	ccv_nnc_tensor_variable_t seq_indices[device_count];
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_tensor_param_t seq_params = GPU_TENSOR_NCHW(000, 32S, batch_size * max_length);
		CCV_TENSOR_SET_DEVICE_ID(seq_params.type, i);
		seq_indices[i] = ccv_nnc_tensor_constant_new(dynamic_graph, seq_params);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(seq_indices_cpu), TENSOR_LIST(ccv_nnc_tensor_from_variable(dynamic_graph, seq_indices[i], 0)), 0);
	}
	classifier_transformer_params_t classifier_transformer_params = {
		.layers = 2,
		.k = embedding_size,
		.h = 8,
		.b = batch_size,
		.t = max_length,
		.ff = embedding_size * 4,
	};
	ccv_cnnp_model_t* const transformer = ccv_cnnp_dynamic_new(_dynamic_classifier_transformer, &classifier_transformer_params, 0);
	ccv_cnnp_model_set_data_parallel(transformer, device_count);
	const int epoch_end = (ccv_cnnp_dataframe_row_count(train_data) + device_count * batch_size - 1) / (device_count * batch_size);
	ccv_cnnp_dataframe_shuffle(train_data);
	ccv_nnc_cmd_t adam = CMD_ADAM_FORWARD(1, 0.0001, 0.9, 0.98, 0, 1e-9);
	const int aux_size = ccv_nnc_minimizer_saved_aux_size(adam);
	ccv_nnc_tensor_variable_t saved_auxs[device_count * aux_size * 2];
	for (i = 0; i < device_count; i++)
	{
		for (j = 0; j < aux_size; j++)
		{
			ccv_nnc_tensor_param_t saved_aux_params = GPU_TENSOR_NCHW(000, 32F, vocab_size, embedding_size);
			CCV_TENSOR_SET_DEVICE_ID(saved_aux_params.type, i);
			saved_auxs[i * aux_size * 2 + j] = ccv_nnc_tensor_variable_new(dynamic_graph, saved_aux_params);
		}
		for (j = 0; j < aux_size; j++)
		{
			ccv_nnc_tensor_param_t saved_aux_params = GPU_TENSOR_NCHW(000, 32F, max_length, embedding_size);
			CCV_TENSOR_SET_DEVICE_ID(saved_aux_params.type, i);
			saved_auxs[i* aux_size * 2 + aux_size + j] = ccv_nnc_tensor_variable_new(dynamic_graph, saved_aux_params);
		}
	}
	ccv_nnc_tensor_t* const out_cpu = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, batch_size, 2), 0);
	ccv_nnc_tensor_t* const fit_cpu = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, batch_size, 2), 0);
	ccv_nnc_tensor_t** tensor[device_count * 2];
	int epoch = 0;
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	for (i = 0; epoch < epoch_limit; i++)
	{
		float learn_rate = 0.0001 * ccv_min(i / (10000. / batch_size), 1) * device_count;
		adam = CMD_ADAM_FORWARD(i + 1, learn_rate, 0.9, 0.98, 0, 1e-9);
		ccv_cnnp_dataframe_iter_next(iter, (void**)tensor, device_count, stream);
		ccv_nnc_tensor_t word_indices_tensor[device_count];
		ccv_nnc_tensor_t mask_tensor[device_count];
		ccv_nnc_tensor_variable_t word_indices[device_count];
		ccv_nnc_tensor_variable_t word_vec[device_count];
		ccv_nnc_tensor_variable_t pos_vec[device_count];
		ccv_nnc_tensor_variable_t select_vec[device_count];
		ccv_nnc_tensor_variable_t vec[device_count * 2];
		ccv_nnc_tensor_variable_t out[device_count];
		for (j = 0; j < device_count; j++)
		{
			ccv_nnc_tensor_param_t word_indices_params = GPU_TENSOR_NCHW(000, 32S, batch_size * max_length);
			CCV_TENSOR_SET_DEVICE_ID(word_indices_params.type, j);
			word_indices_tensor[j] = ccv_nnc_tensor(tensor[j][0]->data.f32, word_indices_params, 0);
			word_indices[j] = ccv_nnc_tensor_variable_new(dynamic_graph, word_indices_params);
			ccv_nnc_tensor_variable_set(dynamic_graph, word_indices[j], &word_indices_tensor[j]);
			ccv_nnc_tensor_param_t pre_vec_params = GPU_TENSOR_NCHW(000, 32F, batch_size * max_length, embedding_size);
			CCV_TENSOR_SET_DEVICE_ID(pre_vec_params.type, j);
			word_vec[j] = ccv_nnc_tensor_variable_new(dynamic_graph, pre_vec_params);
			pos_vec[j] = ccv_nnc_tensor_variable_new(dynamic_graph, pre_vec_params);
			select_vec[j] = ccv_nnc_tensor_variable_new(dynamic_graph);
			ccv_nnc_tensor_param_t vec_params = GPU_TENSOR_NCHW(000, 32F, batch_size, max_length, embedding_size);
			CCV_TENSOR_SET_DEVICE_ID(vec_params.type, j);
			vec[j * 2] = ccv_nnc_tensor_variable_alias_new(dynamic_graph, select_vec[j], ccv_nnc_no_ofs, DIM_ALLOC(), vec_params);
			out[j] = ccv_nnc_tensor_variable_new(dynamic_graph);
		}
		ccv_nnc_tensor_variable_t tvin[device_count * 2];
		for (j = 0; j < device_count; j++)
			tvin[j * 2] = vocab_vec[j], tvin[j * 2 + 1] = word_indices[j];
		ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, tvin, device_count * 2, word_vec, device_count, device_count, stream);
		for (j = 0; j < device_count; j++)
			tvin[j * 2] = seq_vec[j], tvin[j * 2 + 1] = seq_indices[j];
		ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, tvin, device_count * 2, pos_vec, device_count, device_count, stream);
		for (j = 0; j < device_count; j++)
			tvin[j * 2] = word_vec[j], tvin[j * 2 + 1] = pos_vec[j];
		ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_ADD_FORWARD(1, 1), ccv_nnc_no_hint, 0, tvin, device_count * 2, select_vec, device_count, device_count, stream);
		ccv_cnnp_dataframe_iter_peek(iter, (void**)(tensor + device_count), device_count, device_count, stream);
		for (j = 0; j < device_count; j++)
		{
			ccv_nnc_tensor_param_t mask_params = GPU_TENSOR_NCHW(000, 32S, batch_size, max_length, max_length);
			CCV_TENSOR_SET_DEVICE_ID(mask_params.type, j);
			mask_tensor[j] = ccv_nnc_tensor(tensor[j + device_count][0]->data.i32, mask_params, 0);
			vec[j * 2 + 1] = ccv_nnc_tensor_constant_new(dynamic_graph, mask_params);
			ccv_nnc_tensor_variable_set(dynamic_graph, vec[j * 2 + 1], &mask_tensor[j]);
		}
		ccv_nnc_dynamic_graph_evaluate(dynamic_graph, transformer, 0, vec, device_count * 2, out, device_count, 0, stream);
		ccv_nnc_tensor_variable_t softmax[device_count];
		ccv_nnc_tensor_variable_t fit[device_count];
		ccv_nnc_tensor_variable_t vocab_vec_grad[device_count];
		ccv_nnc_tensor_variable_t seq_vec_grad[device_count];
		for (j = 0; j < device_count; j++)
		{
			softmax[j] = ccv_nnc_tensor_variable_new(dynamic_graph);
			fit[j] = ccv_nnc_tensor_variable_new(dynamic_graph);
			ccv_nnc_tensor_variable_set(dynamic_graph, fit[j], tensor[j][1]);
			vocab_vec_grad[j] = ccv_nnc_tensor_variable_new(dynamic_graph);
			seq_vec_grad[j] = ccv_nnc_tensor_variable_new(dynamic_graph);
		}
		ccv_nnc_tensor_variable_t tvout[device_count * 2];
		for (j = 0; j < device_count; j++)
			tvin[j * 2] = out[j], tvin[j * 2 + 1] = fit[j], tvout[j * 2] = 0, tvout[j * 2 + 1] = softmax[j];
		ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_SOFTMAX_CROSSENTROPY_FORWARD(), ccv_nnc_no_hint, 0, tvin, device_count * 2, tvout, device_count * 2, device_count, stream);
		for (j = 0; j < device_count; j++)
			tvin[j * 2] = vocab_vec[j], tvin[j * 2 + 1] = seq_vec[j], tvout[j * 2] = vocab_vec_grad[j], tvout[j * 2 + 1] = seq_vec_grad[j];
		ccv_nnc_dynamic_graph_backward(dynamic_graph, softmax, device_count, 0, tvin, device_count * 2, tvout, device_count * 2, stream);
		for (j = 0; j < device_count; j++)
			tvin[j * 2] = vocab_vec_grad[j], tvin[j * 2 + 1] = seq_vec_grad[j], tvout[j * 2] = vocab_vec[j], tvout[j * 2 + 1] = seq_vec[j];
		ccv_nnc_dynamic_graph_apply_gradients(dynamic_graph, adam, tvin, device_count * 2, tvout, device_count * 2, saved_auxs, device_count, stream);
		for (j = 0; j < device_count; j++)
		{
			ccv_nnc_tensor_variable_free(dynamic_graph, vec[j * 2]);
			ccv_nnc_tensor_variable_free(dynamic_graph, vec[j * 2 + 1]);
			ccv_nnc_tensor_variable_free(dynamic_graph, select_vec[j]);
			ccv_nnc_tensor_variable_free(dynamic_graph, word_vec[j]);
			ccv_nnc_tensor_variable_free(dynamic_graph, word_indices[j]);
			ccv_nnc_tensor_variable_free(dynamic_graph, out[j]);
			ccv_nnc_tensor_variable_free(dynamic_graph, fit[j]);
			ccv_nnc_tensor_variable_free(dynamic_graph, pos_vec[j]);
			ccv_nnc_tensor_variable_free(dynamic_graph, softmax[j]);
			ccv_nnc_tensor_variable_free(dynamic_graph, vocab_vec_grad[j]);
			ccv_nnc_tensor_variable_free(dynamic_graph, seq_vec_grad[j]);
		}
		if ((i + 1) % epoch_end == 0)
		{
			++epoch;
			ccv_cnnp_dataframe_shuffle(train_data);
			ccv_cnnp_dataframe_iter_set_cursor(iter, 0);
		}
	}
	ccv_nnc_stream_context_free(stream);
	int correct = 0;
	ccv_cnnp_dataframe_iter_t* const test_iter = ccv_cnnp_dataframe_iter_new(test_batched_data, test_gpu_batched, device_count * 2);
	int k;
	ccv_nnc_dynamic_graph_set_no_grad(dynamic_graph, 1);
	const int row_count = ccv_cnnp_dataframe_row_count(test_data);
	for (k = 0; k < row_count; k += batch_size * device_count)
	{
		ccv_cnnp_dataframe_iter_next(test_iter, (void**)tensor, device_count, 0);
		ccv_nnc_tensor_t word_indices_tensor[device_count];
		ccv_nnc_tensor_t mask_tensor[device_count];
		ccv_nnc_tensor_variable_t word_indices[device_count];
		ccv_nnc_tensor_variable_t word_vec[device_count];
		ccv_nnc_tensor_variable_t pos_vec[device_count];
		ccv_nnc_tensor_variable_t select_vec[device_count];
		ccv_nnc_tensor_variable_t vec[device_count * 2];
		ccv_nnc_tensor_variable_t out[device_count];
		for (j = 0; j < device_count; j++)
		{
			ccv_nnc_tensor_param_t word_indices_params = GPU_TENSOR_NCHW(000, 32S, batch_size * max_length);
			CCV_TENSOR_SET_DEVICE_ID(word_indices_params.type, j);
			word_indices_tensor[j] = ccv_nnc_tensor(tensor[j][0]->data.f32, word_indices_params, 0);
			word_indices[j] = ccv_nnc_tensor_variable_new(dynamic_graph, word_indices_params);
			ccv_nnc_tensor_variable_set(dynamic_graph, word_indices[j], &word_indices_tensor[j]);
			ccv_nnc_tensor_param_t pre_vec_params = GPU_TENSOR_NCHW(000, 32F, batch_size * max_length, embedding_size);
			CCV_TENSOR_SET_DEVICE_ID(pre_vec_params.type, j);
			word_vec[j] = ccv_nnc_tensor_variable_new(dynamic_graph, pre_vec_params);
			pos_vec[j] = ccv_nnc_tensor_variable_new(dynamic_graph, pre_vec_params);
			select_vec[j] = ccv_nnc_tensor_variable_new(dynamic_graph);
			ccv_nnc_tensor_param_t vec_params = GPU_TENSOR_NCHW(000, 32F, batch_size, max_length, embedding_size);
			CCV_TENSOR_SET_DEVICE_ID(vec_params.type, j);
			vec[j * 2] = ccv_nnc_tensor_variable_alias_new(dynamic_graph, select_vec[j], ccv_nnc_no_ofs, DIM_ALLOC(), vec_params);
			out[j] = ccv_nnc_tensor_variable_new(dynamic_graph);
		}
		ccv_nnc_tensor_variable_t tvin[device_count * 2];
		for (j = 0; j < device_count; j++)
			tvin[j * 2] = vocab_vec[j], tvin[j * 2 + 1] = word_indices[j];
		ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, tvin, device_count * 2, word_vec, device_count, device_count, 0);
		for (j = 0; j < device_count; j++)
			tvin[j * 2] = seq_vec[j], tvin[j * 2 + 1] = seq_indices[j];
		ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, tvin, device_count * 2, pos_vec, device_count, device_count, 0);
		for (j = 0; j < device_count; j++)
			tvin[j * 2] = word_vec[j], tvin[j * 2 + 1] = pos_vec[j];
		ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_ADD_FORWARD(1, 1), ccv_nnc_no_hint, 0, tvin, device_count * 2, select_vec, device_count, device_count, 0);
		ccv_cnnp_dataframe_iter_peek(test_iter, (void**)(tensor + device_count), device_count, device_count, 0);
		for (j = 0; j < device_count; j++)
		{
			ccv_nnc_tensor_param_t mask_params = GPU_TENSOR_NCHW(000, 32S, batch_size, max_length, max_length);
			CCV_TENSOR_SET_DEVICE_ID(mask_params.type, j);
			mask_tensor[j] = ccv_nnc_tensor(tensor[j + device_count][0]->data.i32, mask_params, 0);
			vec[j * 2 + 1] = ccv_nnc_tensor_constant_new(dynamic_graph, mask_params);
			ccv_nnc_tensor_variable_set(dynamic_graph, vec[j * 2 + 1], &mask_tensor[j]);
		}
		ccv_nnc_dynamic_graph_evaluate(dynamic_graph, transformer, 1, vec, device_count * 2, out, device_count, 0, 0);
		int d;
		for (d = 0; d < device_count; d++)
		{
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ccv_nnc_tensor_from_variable(dynamic_graph, out[d], 0)), TENSOR_LIST(out_cpu), 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor[d][1]), TENSOR_LIST(fit_cpu), 0);
			for (j = 0; j < ccv_min(row_count - k - d * batch_size, batch_size); j++)
			{
				const int truth = (fit_cpu->data.f32[j * 2] < fit_cpu->data.f32[j * 2 + 1]);
				const int prediction = (out_cpu->data.f32[j * 2] < out_cpu->data.f32[j * 2 + 1]);
				if (truth == prediction)
					++correct;
			}
		}
		for (j = 0; j < device_count; j++)
		{
			ccv_nnc_tensor_variable_free(dynamic_graph, vec[j * 2]);
			ccv_nnc_tensor_variable_free(dynamic_graph, vec[j * 2 + 1]);
			ccv_nnc_tensor_variable_free(dynamic_graph, select_vec[j]);
			ccv_nnc_tensor_variable_free(dynamic_graph, word_vec[j]);
			ccv_nnc_tensor_variable_free(dynamic_graph, word_indices[j]);
			ccv_nnc_tensor_variable_free(dynamic_graph, out[j]);
			ccv_nnc_tensor_variable_free(dynamic_graph, pos_vec[j]);
		}
	}
	ccv_cnnp_dataframe_iter_free(test_iter);
	ccv_nnc_dynamic_graph_set_no_grad(dynamic_graph, 0);
	ccv_cnnp_model_free(transformer);
	ccv_cnnp_dataframe_iter_free(iter);
	ccv_cnnp_dataframe_free(batched_data);
	ccv_cnnp_dataframe_free(test_batched_data);
	ccv_nnc_dynamic_graph_free(dynamic_graph);
	ccv_nnc_tensor_free(out_cpu);
	ccv_nnc_tensor_free(fit_cpu);
	return correct;
}

TEST_CASE("train a transformer classifier on imdb reviews to 80% with mix of dynamic graph and cnnp model")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) &&
			ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS) &&
			ccv_nnc_cmd_ok(CCV_NNC_AVERAGE_POOL_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) &&
			ccv_nnc_cmd_ok(CCV_NNC_AVERAGE_POOL_BACKWARD, CCV_NNC_BACKEND_GPU_CUDNN));
	const char* const train_list = "/fast/Data/IMDB_Movie_Reviews/aclImdb/train.txt";
	const char* const test_list = "/fast/Data/IMDB_Movie_Reviews/aclImdb/test.txt";
	const char* const vocab_file = "/fast/Data/IMDB_Movie_Reviews/aclImdb/imdb.vocab";
	const char* const base_dir = "/fast/Data/IMDB_Movie_Reviews/aclImdb/";
	FILE* train_open = fopen(train_list, "rb");
	FILE* test_open = fopen(test_list, "rb");
	FILE* vocab_open = fopen(vocab_file, "rb");
	if (train_open)
		fclose(train_open);
	if (test_open)
		fclose(test_open);
	if (vocab_open)
		fclose(vocab_open);
	if (!train_open || !test_open || !vocab_open)
		{ GUARD_ELSE_RETURN(0); }
	khash_t(vocab_map)* vocab;
	int vocab_size;
	_vocab_init(vocab_file, &vocab, &vocab_size);
	const int max_length = 512;
	ccv_array_t* const train_set = _array_from_disk_new(train_list, base_dir, vocab, vocab_size, max_length);
	ccv_cnnp_dataframe_t* const train_data = ccv_cnnp_dataframe_from_array_new(train_set);
	ccv_array_t* const test_set = _array_from_disk_new(test_list, base_dir, vocab, vocab_size, max_length);
	ccv_cnnp_dataframe_t* const test_data = ccv_cnnp_dataframe_from_array_new(test_set);
	if (!ccv_is_coverage())
	{
		const int correct = train_imdb(10, vocab_size, 64, max_length, 128, train_data, test_data);
		REQUIRE((float)correct / test_set->rnum > 0.80, "%f should be larger than 80%%", (float)correct / test_set->rnum);
	} else
		train_imdb(1, vocab_size, 64, max_length, 128, train_data, test_data);
	ccv_cnnp_dataframe_free(train_data);
	ccv_cnnp_dataframe_free(test_data);
	int i;
	for (i = 0; i < train_set->rnum; i++)
	{
		ccv_nnc_tensor_free(((ccv_nnc_text_t*)ccv_array_get(train_set, i))->tensor);
		ccv_nnc_tensor_free(((ccv_nnc_text_t*)ccv_array_get(train_set, i))->mask);
	}
	ccv_array_free(train_set);
	for (i = 0; i < test_set->rnum; i++)
	{
		ccv_nnc_tensor_free(((ccv_nnc_text_t*)ccv_array_get(test_set, i))->tensor);
		ccv_nnc_tensor_free(((ccv_nnc_text_t*)ccv_array_get(test_set, i))->mask);
	}
	ccv_array_free(test_set);
	_vocab_destroy(vocab);
}

#include "case_main.h"
