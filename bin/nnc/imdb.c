#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <sys/time.h>
#include <ctype.h>
#include <getopt.h>
#include <stddef.h>
#include <3rdparty/khash/khash.h>

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

static ccv_array_t* _array_from_disk_new(const char* const list, const char* const base_dir, const khash_t(vocab_map)* const vocab, const int vocab_size, const int max_length)
{
	FILE *r = fopen(list, "r");
	assert(r && "list doesn't exists");
	int dirlen = (base_dir != 0) ? strlen(base_dir) + 1 : 0;
	ccv_array_t* categorizeds = ccv_array_new(sizeof(ccv_categorized_t), 64, 0);
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
		ccv_categorized_t categorized = ccv_categorized(c, (ccv_dense_matrix_t*)tensor, 0);
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
	const float scale = 1. / powf(k, 0.25);
	queries = ccv_cnnp_model_apply(ccv_cnnp_scalar_mul(scale, 0), MODEL_IO_LIST(queries));
	keys = ccv_cnnp_model_apply(ccv_cnnp_scalar_mul(scale, 0), MODEL_IO_LIST(keys));
	ccv_cnnp_model_io_t dot = ccv_cnnp_model_apply(ccv_cnnp_matmul(NO_TRANSPOSE, TRANSPOSE(1, 2), 0), MODEL_IO_LIST(queries, keys));
	dot = ccv_cnnp_model_apply(ccv_cnnp_reshape(DIM_ALLOC(b * h * t, t), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(dot));
	dot = ccv_cnnp_model_apply(ccv_cnnp_softmax(0), MODEL_IO_LIST(dot));
	dot = ccv_cnnp_model_apply(ccv_cnnp_reshape(DIM_ALLOC(b * h, t, t), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(dot));
	ccv_cnnp_model_io_t out = ccv_cnnp_model_apply(ccv_cnnp_matmul(NO_TRANSPOSE, NO_TRANSPOSE, 0), MODEL_IO_LIST(dot, values));
	out = ccv_cnnp_model_apply(ccv_cnnp_reshape(DIM_ALLOC(b, h, t, k), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_transpose(1, 2, 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_reshape(DIM_ALLOC(b * t, h * k), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(out));
	ccv_cnnp_model_t* const unifyheads = ccv_cnnp_dense(k, (ccv_cnnp_param_t){}, "unifyheads");
	out = ccv_cnnp_model_apply(unifyheads, MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_reshape(DIM_ALLOC(b, t, k), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(out));
	return ccv_cnnp_model_new(MODEL_IO_LIST(x), MODEL_IO_LIST(out), "self-attention");
}

static ccv_cnnp_model_t* _transformer_block_new(const int k, const int h, const int b, const int t, const int ff)
{
	ccv_cnnp_model_io_t const x = ccv_cnnp_input();
	ccv_cnnp_model_t* const self_attention = _self_attention_new(k, h, b, t);
	ccv_cnnp_model_io_t out = ccv_cnnp_model_apply(self_attention, MODEL_IO_LIST(x));
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
	return ccv_cnnp_model_new(MODEL_IO_LIST(x), MODEL_IO_LIST(out), "transformer");
}

static ccv_cnnp_model_t* _classifier_transformer_new(const int layers, const int k, const int h, const int b, const int t, const int ff)
{
	ccv_cnnp_model_io_t const x = ccv_cnnp_input();
	ccv_cnnp_model_io_t out = x;
	int i;
	for (i = 0; i < layers; i++)
		out = ccv_cnnp_model_apply(_transformer_block_new(k, h, b, t, ff), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_transpose(1, 2, 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_reshape(DIM_ALLOC(b, k, t, 1), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_average_pool(DIM_ALLOC(0, 0), (ccv_cnnp_param_t){}, 0), MODEL_IO_LIST(out));
	// Last layer, get it to 2.
	out = ccv_cnnp_model_apply(ccv_cnnp_flatten(0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_dense(2, (ccv_cnnp_param_t){}, 0), MODEL_IO_LIST(out));
	return ccv_cnnp_model_new(MODEL_IO_LIST(x), MODEL_IO_LIST(out), "classifier");
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

static void train_imdb(const int vocab_size, const int batch_size, const int max_length, const int embedding_size, ccv_cnnp_dataframe_t* const train_data, ccv_cnnp_dataframe_t* const test_data, ccv_array_t* const test_set)
{
	const int tensor_idx = ccv_cnnp_dataframe_extract_value(train_data, 0, offsetof(ccv_categorized_t, matrix));
	const int one_hot_idx = ccv_cnnp_dataframe_one_hot(train_data, 0, offsetof(ccv_categorized_t, c), 2, 1, 0, CCV_32F, CCV_TENSOR_FORMAT_NCHW);
	ccv_cnnp_dataframe_t* const batched_data = ccv_cnnp_dataframe_batching_new(train_data, COLUMN_ID_LIST(tensor_idx, one_hot_idx), batch_size, 1, CCV_TENSOR_FORMAT_NCHW);
	const int gpu_batched = ccv_cnnp_dataframe_copy_to_gpu(batched_data, 0, 0, 2, 0);
	const int test_tensor_idx = ccv_cnnp_dataframe_extract_value(test_data, 0, offsetof(ccv_categorized_t, matrix));
	const int test_one_hot_idx = ccv_cnnp_dataframe_one_hot(test_data, 0, offsetof(ccv_categorized_t, c), 2, 1, 0, CCV_32F, CCV_TENSOR_FORMAT_NCHW);
	ccv_cnnp_dataframe_t* const test_batched_data = ccv_cnnp_dataframe_batching_new(test_data, COLUMN_ID_LIST(test_tensor_idx, test_one_hot_idx), batch_size, 1, CCV_TENSOR_FORMAT_NCHW);
	const int test_gpu_batched = ccv_cnnp_dataframe_copy_to_gpu(test_batched_data, 0, 0, 2, 0);
	ccv_cnnp_dataframe_iter_t* const iter = ccv_cnnp_dataframe_iter_new(batched_data, COLUMN_ID_LIST(gpu_batched));
	ccv_nnc_dynamic_graph_t* const dynamic_graph = ccv_nnc_dynamic_graph_new();
	const ccv_nnc_tensor_variable_t vocab_vec = ccv_nnc_tensor_variable_new(dynamic_graph, GPU_TENSOR_NCHW(000, 32F, vocab_size, embedding_size));
	ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_RANDOM_UNIFORM_FORWARD(-1, 1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(), TENSOR_VARIABLE_LIST(vocab_vec), 0, 0);
	const ccv_nnc_tensor_variable_t seq_vec = ccv_nnc_tensor_variable_new(dynamic_graph, GPU_TENSOR_NCHW(000, 32F, max_length, embedding_size));
	ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_RANDOM_UNIFORM_FORWARD(-1, 1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(), TENSOR_VARIABLE_LIST(seq_vec), 0, 0);
	ccv_nnc_tensor_t* const seq_indices_cpu = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32S, batch_size * max_length), 0);
	int i, j;
	for (i = 0; i < batch_size; i++)
		for (j = 0; j < max_length; j++)
			seq_indices_cpu->data.i32[i * max_length + j] = j;
	const ccv_nnc_tensor_variable_t seq_indices = ccv_nnc_tensor_constant_new(dynamic_graph, GPU_TENSOR_NCHW(000, 32S, batch_size * max_length));
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(seq_indices_cpu), TENSOR_LIST(ccv_nnc_tensor_from_variable(dynamic_graph, seq_indices)), 0);
	ccv_cnnp_model_t* const transformer = _classifier_transformer_new(6, embedding_size, 8, batch_size, max_length, embedding_size * 4);
	const int epoch_end = (ccv_cnnp_dataframe_row_count(train_data) + batch_size - 1) / batch_size;
	ccv_cnnp_dataframe_shuffle(train_data);
	ccv_nnc_cmd_t adam = CMD_ADAM_FORWARD(1, 0.0001, 0.9, 0.98, 0, 1e-9);
	const int aux_size = ccv_nnc_minimizer_saved_aux_size(adam);
	ccv_nnc_tensor_variable_t saved_auxs[aux_size * 2];
	for (i = 0; i < aux_size; i++)
		saved_auxs[i] = ccv_nnc_tensor_variable_new(dynamic_graph, GPU_TENSOR_NCHW(000, 32F, vocab_size, embedding_size));
	for (i = 0; i < aux_size; i++)
		saved_auxs[aux_size + i] = ccv_nnc_tensor_variable_new(dynamic_graph, GPU_TENSOR_NCHW(000, 32F, max_length, embedding_size));
	ccv_nnc_tensor_t* const out_cpu = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, batch_size, 2), 0);
	ccv_nnc_tensor_t* const fit_cpu = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, batch_size, 2), 0);
	// CCV_CLI_SET_OUTPUT_LEVEL_AND_ABOVE(CCV_CLI_VERBOSE);
	ccv_nnc_tensor_t** tensor = 0;
	double overall_accuracy = 0;
	int epoch = 0;
	// ccv_cnnp_dataframe_iter_next(iter, (void**)&tensor, 1, 0);
	for (i = 0; i < 100000; i++)
	{
		float learn_rate = 0.0001 * ccv_min(i / (10000. / batch_size), 1);
		adam = CMD_ADAM_FORWARD(i + 1, learn_rate, 0.9, 0.98, 0, 1e-9);
		ccv_cnnp_dataframe_iter_next(iter, (void**)&tensor, 1, 0);
		ccv_nnc_tensor_t word_indices_tensor = ccv_nnc_tensor(tensor[0]->data.f32, GPU_TENSOR_NCHW(000, 32S, batch_size * max_length), 0);
		const ccv_nnc_tensor_variable_t word_indices = ccv_nnc_tensor_variable_new(dynamic_graph, GPU_TENSOR_NCHW(000, 32S, batch_size * max_length));
		ccv_nnc_tensor_variable_set(dynamic_graph, word_indices, &word_indices_tensor);
		const ccv_nnc_tensor_variable_t word_vec = ccv_nnc_tensor_variable_new(dynamic_graph, GPU_TENSOR_NCHW(000, 32F, batch_size * max_length, embedding_size));
		const ccv_nnc_tensor_variable_t pos_vec = ccv_nnc_tensor_variable_new(dynamic_graph, GPU_TENSOR_NCHW(000, 32F, batch_size * max_length, embedding_size));
		ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(vocab_vec, word_indices), TENSOR_VARIABLE_LIST(word_vec), 0, 0);
		ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(seq_vec, seq_indices), TENSOR_VARIABLE_LIST(pos_vec), 0, 0);
		const ccv_nnc_tensor_variable_t select_vec = ccv_nnc_tensor_variable_new(dynamic_graph);
		ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_ADD_FORWARD(1, 1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(word_vec, pos_vec), TENSOR_VARIABLE_LIST(select_vec), 0, 0);
		const ccv_nnc_tensor_variable_t vec = ccv_nnc_tensor_variable_alias_new(dynamic_graph, select_vec, ccv_nnc_no_ofs, DIM_ALLOC(), GPU_TENSOR_NCHW(000, 32F, batch_size, max_length, embedding_size));
		const ccv_nnc_tensor_variable_t out = ccv_nnc_tensor_variable_new(dynamic_graph);
		ccv_nnc_dynamic_graph_evaluate(dynamic_graph, transformer, TENSOR_VARIABLE_LIST(vec), TENSOR_VARIABLE_LIST(out), 0, 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ccv_nnc_tensor_from_variable(dynamic_graph, out)), TENSOR_LIST(out_cpu), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor[1]), TENSOR_LIST(fit_cpu), 0);
		int correct = 0;
		for (j = 0; j < batch_size; j++)
		{
			const int truth = (fit_cpu->data.f32[j * 2] < fit_cpu->data.f32[j * 2 + 1]);
			const int prediction = (out_cpu->data.f32[j * 2] < out_cpu->data.f32[j * 2 + 1]);
			if (truth == prediction)
				++correct;
		}
		const double accuracy = (double)correct / batch_size;
		overall_accuracy = overall_accuracy * 0.9 + accuracy * 0.1;
		ccv_nnc_tensor_variable_t const softmax = ccv_nnc_tensor_variable_new(dynamic_graph);
		const ccv_nnc_tensor_variable_t fit = ccv_nnc_tensor_variable_new(dynamic_graph);
		ccv_nnc_tensor_variable_set(dynamic_graph, fit, tensor[1]);
		ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_SOFTMAX_CROSSENTROPY_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(out, fit), TENSOR_VARIABLE_LIST(0, softmax), 0, 0);
		const ccv_nnc_tensor_variable_t vocab_vec_grad = ccv_nnc_tensor_variable_new(dynamic_graph);
		const ccv_nnc_tensor_variable_t seq_vec_grad = ccv_nnc_tensor_variable_new(dynamic_graph);
		ccv_nnc_dynamic_graph_backward(dynamic_graph, softmax, 0, TENSOR_VARIABLE_LIST(vocab_vec, seq_vec), TENSOR_VARIABLE_LIST(vocab_vec_grad, seq_vec_grad), 0);
		if (i == 0)
		{
			FILE* model = fopen("imdb-model.dot", "w+");
			FILE* compiled = fopen("imdb-model-compiled.dot", "w+");
			FILE* ws[] = {model, compiled};
			ccv_cnnp_model_dot(transformer, CCV_NNC_LONG_DOT_GRAPH, ws, 2);
			fclose(compiled);
			fclose(model);
			FILE* exec = fopen("imdb.dot", "w+");
			ccv_nnc_dynamic_graph_dot(dynamic_graph, CCV_NNC_LONG_DOT_GRAPH, exec);
			fclose(exec);
		}
		ccv_nnc_dynamic_graph_apply_gradients(dynamic_graph, adam, TENSOR_VARIABLE_LIST(vocab_vec_grad, seq_vec_grad), TENSOR_VARIABLE_LIST(vocab_vec, seq_vec), saved_auxs, 0);
		ccv_nnc_tensor_variable_free(dynamic_graph, vec);
		ccv_nnc_tensor_variable_free(dynamic_graph, select_vec);
		ccv_nnc_tensor_variable_free(dynamic_graph, word_vec);
		ccv_nnc_tensor_variable_free(dynamic_graph, word_indices);
		ccv_nnc_tensor_variable_free(dynamic_graph, out);
		ccv_nnc_tensor_variable_free(dynamic_graph, fit);
		ccv_nnc_tensor_variable_free(dynamic_graph, pos_vec);
		ccv_nnc_tensor_variable_free(dynamic_graph, softmax);
		ccv_nnc_tensor_variable_free(dynamic_graph, vocab_vec_grad);
		ccv_nnc_tensor_variable_free(dynamic_graph, seq_vec_grad);
		if ((i + 1) % 50 == 0)
			printf("epoch %d (%d/%d), training accuracy %lf\n", epoch, (i + 1) - epoch * epoch_end, epoch_end, overall_accuracy);
		if ((i + 1) % epoch_end == 0)
		{
			int correct = 0;
			ccv_cnnp_dataframe_iter_t* const test_iter = ccv_cnnp_dataframe_iter_new(test_batched_data, COLUMN_ID_LIST(test_gpu_batched));
			int k;
			ccv_cnnp_dataframe_shuffle(test_data);
			ccv_nnc_dynamic_graph_set_no_grad(dynamic_graph, 1);
			const int row_count = ccv_cnnp_dataframe_row_count(test_data);
			for (k = 0; k < row_count; k += batch_size)
			{
				ccv_cnnp_dataframe_iter_next(test_iter, (void**)&tensor, 1, 0);
				ccv_nnc_tensor_t word_indices_tensor = ccv_nnc_tensor(tensor[0]->data.f32, GPU_TENSOR_NCHW(000, 32S, batch_size * max_length), 0);
				const ccv_nnc_tensor_variable_t word_indices = ccv_nnc_tensor_variable_new(dynamic_graph, GPU_TENSOR_NCHW(000, 32S, batch_size * max_length));
				ccv_nnc_tensor_variable_set(dynamic_graph, word_indices, &word_indices_tensor);
				const ccv_nnc_tensor_variable_t word_vec = ccv_nnc_tensor_variable_new(dynamic_graph, GPU_TENSOR_NCHW(000, 32F, batch_size * max_length, embedding_size));
				const ccv_nnc_tensor_variable_t pos_vec = ccv_nnc_tensor_variable_new(dynamic_graph, GPU_TENSOR_NCHW(000, 32F, batch_size * max_length, embedding_size));
				ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(vocab_vec, word_indices), TENSOR_VARIABLE_LIST(word_vec), 0, 0);
				ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(seq_vec, seq_indices), TENSOR_VARIABLE_LIST(pos_vec), 0, 0);
				const ccv_nnc_tensor_variable_t select_vec = ccv_nnc_tensor_variable_new(dynamic_graph);
				ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_ADD_FORWARD(1, 1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(word_vec, pos_vec), TENSOR_VARIABLE_LIST(select_vec), 0, 0);
				const ccv_nnc_tensor_variable_t vec = ccv_nnc_tensor_variable_alias_new(dynamic_graph, select_vec, ccv_nnc_no_ofs, DIM_ALLOC(), GPU_TENSOR_NCHW(000, 32F, batch_size, max_length, embedding_size));
				const ccv_nnc_tensor_variable_t out = ccv_nnc_tensor_variable_new(dynamic_graph);
				ccv_nnc_dynamic_graph_evaluate(dynamic_graph, transformer, TENSOR_VARIABLE_LIST(vec), TENSOR_VARIABLE_LIST(out), 0, 0);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ccv_nnc_tensor_from_variable(dynamic_graph, out)), TENSOR_LIST(out_cpu), 0);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor[1]), TENSOR_LIST(fit_cpu), 0);
				for (j = 0; j < ccv_min(row_count - k, batch_size); j++)
				{
					const int truth = (fit_cpu->data.f32[j * 2] < fit_cpu->data.f32[j * 2 + 1]);
					const int prediction = (out_cpu->data.f32[j * 2] < out_cpu->data.f32[j * 2 + 1]);
					if (truth == prediction)
						++correct;
				}
				ccv_nnc_tensor_variable_free(dynamic_graph, vec);
				ccv_nnc_tensor_variable_free(dynamic_graph, select_vec);
				ccv_nnc_tensor_variable_free(dynamic_graph, word_vec);
				ccv_nnc_tensor_variable_free(dynamic_graph, word_indices);
				ccv_nnc_tensor_variable_free(dynamic_graph, out);
				ccv_nnc_tensor_variable_free(dynamic_graph, pos_vec);
			}
			ccv_cnnp_dataframe_iter_free(test_iter);
			ccv_nnc_dynamic_graph_set_no_grad(dynamic_graph, 0);
			printf("epoch %d done, training accuracy %lf, test accuracy %lf\n", epoch, overall_accuracy, (double)correct / row_count);
			++epoch;
			ccv_cnnp_dataframe_shuffle(train_data);
			ccv_cnnp_dataframe_iter_set_cursor(iter, 0);
		}
	}
	ccv_cnnp_model_free(transformer);
	ccv_cnnp_dataframe_iter_free(iter);
	ccv_cnnp_dataframe_free(batched_data);
	ccv_cnnp_dataframe_free(test_batched_data);
	ccv_nnc_dynamic_graph_free(dynamic_graph);
	ccv_nnc_tensor_free(out_cpu);
}

int main(int argc, char** argv)
{
	ccv_nnc_init();
	static struct option imdb_options[] = {
		/* help */
		{"help", 0, 0, 0},
		/* required parameters */
		{"train-list", 1, 0, 0},
		{"test-list", 1, 0, 0},
		{"vocab", 1, 0, 0},
		/* optional parameters */
		{"base-dir", 1, 0, 0},
		{0, 0, 0, 0}
	};
	int c;
	char* train_list = 0;
	char* test_list = 0;
	char* base_dir = 0;
	char* vocab_file = 0;
	while (getopt_long_only(argc, argv, "", imdb_options, &c) != -1)
	{
		switch (c)
		{
			case 0:
				exit(0);
			case 1:
				train_list = optarg;
				break;
			case 2:
				test_list = optarg;
				break;
			case 3:
				vocab_file = optarg;
				break;
			case 4:
				base_dir = optarg;
				break;
		}
	}
	khash_t(vocab_map)* vocab;
	int vocab_size;
	_vocab_init(vocab_file, &vocab, &vocab_size);
	const int max_length = 512;
	ccv_array_t* const train_set = _array_from_disk_new(train_list, base_dir, vocab, vocab_size, max_length);
	ccv_cnnp_dataframe_t* const train_data = ccv_cnnp_dataframe_from_array_new(train_set);
	ccv_array_t* const test_set = _array_from_disk_new(test_list, base_dir, vocab, vocab_size, max_length);
	ccv_cnnp_dataframe_t* const test_data = ccv_cnnp_dataframe_from_array_new(test_set);
	train_imdb(vocab_size, 64, max_length, 128, train_data, test_data, test_set);
	ccv_cnnp_dataframe_free(train_data);
	ccv_cnnp_dataframe_free(test_data);
	int i;
	for (i = 0; i < train_set->rnum; i++)
		ccv_nnc_tensor_free((ccv_nnc_tensor_t*)((ccv_categorized_t*)ccv_array_get(train_set, i))->matrix);
	ccv_array_free(train_set);
	for (i = 0; i < test_set->rnum; i++)
		ccv_nnc_tensor_free((ccv_nnc_tensor_t*)((ccv_categorized_t*)ccv_array_get(test_set, i))->matrix);
	ccv_array_free(test_set);
	_vocab_destroy(vocab);
	return 0;
}
