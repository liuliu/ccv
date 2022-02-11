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

static ccv_cnnp_model_t* _classifier_lstm_new(const int batch_size, const int batch_length, const int num_layers, const int hidden_size, const float dropout)
{
	ccv_cnnp_model_io_t const x = ccv_cnnp_input();
	ccv_cnnp_model_io_t const mask = ccv_cnnp_input();
	ccv_cnnp_model_io_t const index = ccv_cnnp_input();
	ccv_cnnp_model_io_t out = ccv_cnnp_model_apply(ccv_cnnp_lstm(1, hidden_size, 0, num_layers, 1, 1, 0, dropout, 0), MODEL_IO_LIST(x, mask));
	out = ccv_cnnp_model_apply(ccv_cnnp_reshape(DIM_ALLOC(batch_size * batch_length, 128), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_index_select(0), MODEL_IO_LIST(out, index));
	// Last layer, get it to 1.
	out = ccv_cnnp_model_apply(ccv_cnnp_flatten(0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_dense(1, 0, 0), MODEL_IO_LIST(out));
	return ccv_cnnp_model_new(MODEL_IO_LIST(x, mask, index), MODEL_IO_LIST(out), "classifier");
}

typedef struct {
	int num_layers;
	int hidden_size;
	float dropout;
} classifier_lstm_params_t;

static ccv_cnnp_model_t* _dynamic_classifier_lstm(const ccv_nnc_tensor_param_t* const inputs, const int input_size, void* const context)
{
	const classifier_lstm_params_t* const params = (classifier_lstm_params_t*)context;
	const int batch_size = inputs[0].dim[0];
	const int batch_length = inputs[0].dim[1];
	return _classifier_lstm_new(batch_size, batch_length, params->num_layers, params->hidden_size, params->dropout);
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

static void _ccv_cnnp_mask_to_index(void* const* const* const column_data, const int column_size, const int batch_size, void** const data, void* const context, ccv_nnc_stream_context_t* const stream_context)
{
	int i, j;
	for (i = 0; i < batch_size; i++)
	{
		ccv_nnc_tensor_t* const input = (ccv_nnc_tensor_t*)column_data[0][i];
		ccv_nnc_tensor_t* output = (ccv_nnc_tensor_t*)data[i];
		ccv_nnc_tensor_param_t params = input->info;
		output = output ? ccv_nnc_tensor_resize(output, params) : ccv_nnc_tensor_new(0, params, 0);
		int max_seq_length = 0;
		for (j = 0; j < params.dim[0]; j++)
			if (input->data.i32[j] > max_seq_length)
				max_seq_length = input->data.i32[j];
		for (j = 0; j < params.dim[0]; j++)
			output->data.i32[j] = ccv_max(max_seq_length * j + input->data.i32[j] - 1, 0);
		data[i] = output;
	}
}

static void _ccv_cnnp_tensor_deinit(void* const data, void* const context)
{
	ccv_nnc_tensor_free((ccv_nnc_tensor_t*)data);
}

static void train_imdb(const int vocab_size, const int batch_size, const int max_length, const int embedding_size, ccv_cnnp_dataframe_t* const train_data, ccv_cnnp_dataframe_t* const test_data, ccv_array_t* const test_set)
{
	const int tensor_idx = ccv_cnnp_dataframe_extract_value(train_data, 0, offsetof(ccv_nnc_text_t, tensor), 0);
	const int one_hot_idx = ccv_cnnp_dataframe_copy_scalar(train_data, 0, offsetof(ccv_nnc_text_t, c), CCV_32S, CCV_32F, CCV_TENSOR_FORMAT_NCHW, 0);
	const int mask_idx = ccv_cnnp_dataframe_extract_value(train_data, 0, offsetof(ccv_nnc_text_t, mask), 0);
	ccv_cnnp_dataframe_t* const batched_data = ccv_cnnp_dataframe_combine_new(train_data, COLUMN_ID_LIST(tensor_idx, one_hot_idx, mask_idx), batch_size, 1, CCV_TENSOR_FORMAT_NCHW);
	const int test_tensor_idx = ccv_cnnp_dataframe_extract_value(test_data, 0, offsetof(ccv_nnc_text_t, tensor), 0);
	const int test_one_hot_idx = ccv_cnnp_dataframe_copy_scalar(test_data, 0, offsetof(ccv_nnc_text_t, c), CCV_32S, CCV_32F, CCV_TENSOR_FORMAT_NCHW, 0);
	const int test_mask_idx = ccv_cnnp_dataframe_extract_value(test_data, 0, offsetof(ccv_nnc_text_t, mask), 0);
	ccv_cnnp_dataframe_t* const test_batched_data = ccv_cnnp_dataframe_combine_new(test_data, COLUMN_ID_LIST(test_tensor_idx, test_one_hot_idx, test_mask_idx), batch_size, 1, CCV_TENSOR_FORMAT_NCHW);
	int gpu_batched[4];
	int seq_len_batched[1];
	int index_batched[1];
	int data_batched[1];
	int test_gpu_batched[4];
	int test_seq_len_batched[1];
	int test_index_batched[1];
	int test_data_batched[1];
	int i, j;
	for (i = 0; i < 1; i++)
	{
		seq_len_batched[i] = ccv_cnnp_dataframe_extract_tuple(batched_data, 0, i * 3 + 2, 0);
		index_batched[i] = ccv_cnnp_dataframe_map(batched_data, _ccv_cnnp_mask_to_index, CCV_STREAM_CONTEXT_CPU, _ccv_cnnp_tensor_deinit, COLUMN_ID_LIST(seq_len_batched[i]), 0, 0, 0);
		index_batched[i] = ccv_cnnp_dataframe_make_tuple(batched_data, COLUMN_ID_LIST(index_batched[i]), 0);
		data_batched[i] = ccv_cnnp_dataframe_extract_tuple(batched_data, 0, i * 3, 0);
		test_seq_len_batched[i] = ccv_cnnp_dataframe_extract_tuple(test_batched_data, 0, i * 3 + 2, 0);
		test_index_batched[i] = ccv_cnnp_dataframe_map(test_batched_data, _ccv_cnnp_mask_to_index, CCV_STREAM_CONTEXT_CPU, _ccv_cnnp_tensor_deinit, COLUMN_ID_LIST(test_seq_len_batched[i]), 0, 0, 0);
		test_index_batched[i] = ccv_cnnp_dataframe_make_tuple(test_batched_data, COLUMN_ID_LIST(test_index_batched[i]), 0);
		test_data_batched[i] = ccv_cnnp_dataframe_extract_tuple(test_batched_data, 0, i * 3, 0);
	}
	const int trunc_data_batched = ccv_cnnp_dataframe_truncate(batched_data, data_batched, 1, seq_len_batched, 1, 0);
	const int test_trunc_data_batched = ccv_cnnp_dataframe_truncate(test_batched_data, test_data_batched, 1, test_seq_len_batched, 1, 0);
	for (i = 0; i < 1; i++)
	{
		gpu_batched[i * 4] = ccv_cnnp_dataframe_copy_to_gpu(batched_data, trunc_data_batched, i, 1, i, 0);
		gpu_batched[i * 4 + 1] = ccv_cnnp_dataframe_copy_to_gpu(batched_data, 0, i * 3 + 1, 1, i, 0);
		gpu_batched[i * 4 + 2] = ccv_cnnp_dataframe_copy_to_gpu(batched_data, 0, i * 3 + 2, 1, i, 0);
		gpu_batched[i * 4 + 3] = ccv_cnnp_dataframe_copy_to_gpu(batched_data, index_batched[i], 0, 1, i, 0);
		test_gpu_batched[i * 4] = ccv_cnnp_dataframe_copy_to_gpu(test_batched_data, test_trunc_data_batched, i, 1, i, 0);
		test_gpu_batched[i * 4 + 1] = ccv_cnnp_dataframe_copy_to_gpu(test_batched_data, 0, i * 3 + 1, 1, i, 0);
		test_gpu_batched[i * 4 + 2] = ccv_cnnp_dataframe_copy_to_gpu(test_batched_data, 0, i * 3 + 2, 1, i, 0);
		test_gpu_batched[i * 4 + 3] = ccv_cnnp_dataframe_copy_to_gpu(test_batched_data, test_index_batched[i], 0, 1, i, 0);
	}
	ccv_cnnp_dataframe_iter_t* const iter = ccv_cnnp_dataframe_iter_new(batched_data, gpu_batched, 4);
	ccv_nnc_dynamic_graph_t* const dynamic_graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_tensor_t* const vocab_vec_cpu = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, vocab_size, embedding_size), 0);
	ccv_nnc_cmd_exec(CMD_RANDOM_UNIFORM_FORWARD(-1, 1), ccv_nnc_no_hint, 0, 0, 0, TENSOR_LIST(vocab_vec_cpu), 0);
	ccv_nnc_tensor_variable_t vocab_vec[1];
	for (i = 0; i < 1; i++)
	{
		ccv_nnc_tensor_param_t vocab_params = GPU_TENSOR_NCHW(000, 32F, vocab_size, embedding_size);
		CCV_TENSOR_SET_DEVICE_ID(vocab_params.type, i);
		vocab_vec[i] = ccv_nnc_tensor_variable_new(dynamic_graph, vocab_params);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(vocab_vec_cpu), TENSOR_LIST(ccv_nnc_tensor_from_variable(dynamic_graph, vocab_vec[i])), 0);
	}
	ccv_nnc_tensor_free(vocab_vec_cpu);
	classifier_lstm_params_t classifier_lstm_params = {
		.num_layers = 2,
		.hidden_size = 128,
		.dropout = 0.2,
	};
	ccv_cnnp_model_t* const lstm = ccv_cnnp_dynamic_new(_dynamic_classifier_lstm, &classifier_lstm_params, 0);
	const int epoch_end = (ccv_cnnp_dataframe_row_count(train_data) + batch_size - 1) / batch_size;
	ccv_cnnp_dataframe_shuffle(train_data);
	ccv_nnc_cmd_t optim = CMD_LAMB_FORWARD(1, 0.001, 0.9, 0.999, 0, 1e-6);
	const int aux_size = ccv_nnc_minimizer_saved_aux_size(optim);
	ccv_nnc_tensor_variable_t saved_auxs[aux_size * 2];
	for (i = 0; i < 1; i++)
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
	ccv_nnc_tensor_t* const out_cpu = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, batch_size, 1), 0);
	ccv_nnc_tensor_t* const fit_cpu = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, batch_size, 1), 0);
	// CCV_CLI_SET_OUTPUT_LEVEL_AND_ABOVE(CCV_CLI_VERBOSE);
	ccv_nnc_tensor_t** tensor[4];
	double overall_accuracy = 0;
	int epoch = 0;
	for (i = 0; i < 1000000; i++)
	{
		float learn_rate = 0.001;
		optim = CMD_LAMB_FORWARD(i + 1, learn_rate, 0.9, 0.999, 0, 1e-6);
		int status = ccv_cnnp_dataframe_iter_next(iter, (void**)tensor, 4, 0);
		assert(status == 0);
		ccv_nnc_tensor_t word_indices_tensor[1];
		ccv_nnc_tensor_t mask_tensor[1];
		ccv_nnc_tensor_t index_tensor[1];
		ccv_nnc_tensor_variable_t word_indices[1];
		ccv_nnc_tensor_variable_t word_vec[1];
		ccv_nnc_tensor_variable_t vec[1 * 3];
		ccv_nnc_tensor_variable_t out[1];
		for (j = 0; j < 1; j++)
		{
			const int batch_length = tensor[j * 4][0]->info.dim[1];
			ccv_nnc_tensor_param_t word_indices_params = GPU_TENSOR_NCHW(000, 32S, batch_size * batch_length);
			CCV_TENSOR_SET_DEVICE_ID(word_indices_params.type, j);
			word_indices_tensor[j] = ccv_nnc_tensor(tensor[j * 4][0]->data.f32, word_indices_params, 0);
			word_indices[j] = ccv_nnc_tensor_variable_new(dynamic_graph, word_indices_params);
			ccv_nnc_tensor_variable_set(dynamic_graph, word_indices[j], &word_indices_tensor[j]);
			ccv_nnc_tensor_param_t pre_vec_params = GPU_TENSOR_NCHW(000, 32F, batch_size * batch_length, embedding_size);
			CCV_TENSOR_SET_DEVICE_ID(pre_vec_params.type, j);
			word_vec[j] = ccv_nnc_tensor_variable_new(dynamic_graph, pre_vec_params);
			ccv_nnc_tensor_param_t vec_params = GPU_TENSOR_NCHW(000, 32F, batch_size, batch_length, embedding_size);
			CCV_TENSOR_SET_DEVICE_ID(vec_params.type, j);
			vec[j * 3] = ccv_nnc_tensor_variable_alias_new(dynamic_graph, word_vec[j], ccv_nnc_no_ofs, DIM_ALLOC(), vec_params);
			ccv_nnc_tensor_param_t mask_params = GPU_TENSOR_NCHW(000, 32S, batch_size);
			assert(tensor[j * 4 + 2][0]->info.dim[0] == batch_size);
			CCV_TENSOR_SET_DEVICE_ID(mask_params.type, j);
			ccv_nnc_tensor_param_t index_params = GPU_TENSOR_NCHW(000, 32S, batch_size);
			assert(tensor[j * 4 + 3][0]->info.dim[0] == batch_size);
			CCV_TENSOR_SET_DEVICE_ID(index_params.type, j);
			mask_tensor[j] = ccv_nnc_tensor(tensor[j * 4 + 2][0]->data.i32, mask_params, 0);
			index_tensor[j] = ccv_nnc_tensor(tensor[j * 4 + 3][0]->data.i32, index_params, 0);
			vec[j * 3 + 1] = ccv_nnc_tensor_constant_new(dynamic_graph, mask_params);
			vec[j * 3 + 2] = ccv_nnc_tensor_constant_new(dynamic_graph, index_params);
			ccv_nnc_tensor_variable_set(dynamic_graph, vec[j * 3 + 1], &mask_tensor[j]);
			ccv_nnc_tensor_variable_set(dynamic_graph, vec[j * 3 + 2], &index_tensor[j]);
			out[j] = ccv_nnc_tensor_variable_new(dynamic_graph);
		}
		ccv_nnc_tensor_variable_t tvin[1 * 2];
		for (j = 0; j < 1; j++)
			tvin[j * 2] = vocab_vec[j], tvin[j * 2 + 1] = word_indices[j];
		ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, tvin, 2, word_vec, 1, 1, 0);
		ccv_nnc_dynamic_graph_evaluate(dynamic_graph, lstm, 0, vec, 3, out, 1, 0, 0);
		int correct = 0;
		for (j = 0; j < 1; j++)
		{
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ccv_nnc_tensor_from_variable(dynamic_graph, out[j])), TENSOR_LIST(out_cpu), 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor[j * 4 + 1][0]), TENSOR_LIST(fit_cpu), 0);
			int k;
			for (k = 0; k < batch_size; k++)
			{
				const int truth = (fit_cpu->data.f32[k] > 0.5);
				const int prediction = (out_cpu->data.f32[k] > 0);
				if (truth == prediction)
					++correct;
			}
		}
		const double accuracy = (double)correct / batch_size;
		overall_accuracy = overall_accuracy * 0.9 + accuracy * 0.1;
		ccv_nnc_tensor_variable_t sigmoid[1];
		ccv_nnc_tensor_variable_t fit[1];
		ccv_nnc_tensor_variable_t vocab_vec_grad[1];
		for (j = 0; j < 1; j++)
		{
			sigmoid[j] = ccv_nnc_tensor_variable_new(dynamic_graph);
			fit[j] = ccv_nnc_tensor_variable_new(dynamic_graph);
			ccv_nnc_tensor_variable_set(dynamic_graph, fit[j], tensor[j * 4 + 1][0]);
			vocab_vec_grad[j] = ccv_nnc_tensor_variable_new(dynamic_graph);
		}
		ccv_nnc_tensor_variable_t tvout[2];
		for (j = 0; j < 1; j++)
			tvin[j * 2] = out[j], tvin[j * 2 + 1] = fit[j], tvout[j * 2] = 0, tvout[j * 2 + 1] = sigmoid[j];
		ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_SIGMOID_BINARY_CROSSENTROPY_FORWARD(), ccv_nnc_no_hint, 0, tvin, 2, tvout, 2, 1, 0);
		for (j = 0; j < 1; j++)
			tvin[j] = vocab_vec[j], tvout[j] = vocab_vec_grad[j];
		ccv_nnc_dynamic_graph_backward(dynamic_graph, sigmoid, 1, 0, tvin, 1, tvout, 1, 0);
		ccv_cnnp_model_set_minimizer(lstm, optim, 0, 0, 0);
		for (j = 0; j < 1; j++)
			tvin[j] = vocab_vec_grad[j], tvout[j] = vocab_vec[j];
		ccv_nnc_dynamic_graph_apply_gradients(dynamic_graph, optim, tvin, 1, tvout, 1, saved_auxs, 1, 0);
		for (j = 0; j < 1; j++)
		{
			ccv_nnc_tensor_variable_free(dynamic_graph, vec[j * 3]);
			ccv_nnc_tensor_variable_free(dynamic_graph, vec[j * 3 + 1]);
			ccv_nnc_tensor_variable_free(dynamic_graph, vec[j * 3 + 2]);
			ccv_nnc_tensor_variable_free(dynamic_graph, word_vec[j]);
			ccv_nnc_tensor_variable_free(dynamic_graph, word_indices[j]);
			ccv_nnc_tensor_variable_free(dynamic_graph, out[j]);
			ccv_nnc_tensor_variable_free(dynamic_graph, fit[j]);
			ccv_nnc_tensor_variable_free(dynamic_graph, sigmoid[j]);
			ccv_nnc_tensor_variable_free(dynamic_graph, vocab_vec_grad[j]);
		}
		if ((i + 1) % 50 == 0)
			printf("epoch %d (%d/%d), training accuracy %lf\n", epoch, (i + 1) - epoch * epoch_end, epoch_end, overall_accuracy);
		if ((i + 1) % epoch_end == 0)
		{
			int correct = 0;
			ccv_cnnp_dataframe_iter_t* const test_iter = ccv_cnnp_dataframe_iter_new(test_batched_data, test_gpu_batched, 4);
			int k;
			ccv_cnnp_dataframe_shuffle(test_data);
			ccv_nnc_dynamic_graph_set_no_grad(dynamic_graph, 1);
			const int row_count = ccv_cnnp_dataframe_row_count(test_data);
			for (k = 0; k < row_count; k += batch_size)
			{
				ccv_cnnp_dataframe_iter_next(test_iter, (void**)tensor, 4, 0);
				ccv_nnc_tensor_t word_indices_tensor[1];
				ccv_nnc_tensor_t mask_tensor[1];
				ccv_nnc_tensor_t index_tensor[1];
				ccv_nnc_tensor_variable_t word_indices[1];
				ccv_nnc_tensor_variable_t word_vec[1];
				ccv_nnc_tensor_variable_t vec[3];
				ccv_nnc_tensor_variable_t out[1];
				for (j = 0; j < 1; j++)
				{
					const int batch_length = tensor[j * 4][0]->info.dim[1];
					ccv_nnc_tensor_param_t word_indices_params = GPU_TENSOR_NCHW(000, 32S, batch_size * batch_length);
					CCV_TENSOR_SET_DEVICE_ID(word_indices_params.type, j);
					word_indices_tensor[j] = ccv_nnc_tensor(tensor[j * 4][0]->data.f32, word_indices_params, 0);
					word_indices[j] = ccv_nnc_tensor_variable_new(dynamic_graph, word_indices_params);
					ccv_nnc_tensor_variable_set(dynamic_graph, word_indices[j], &word_indices_tensor[j]);
					ccv_nnc_tensor_param_t pre_vec_params = GPU_TENSOR_NCHW(000, 32F, batch_size * batch_length, embedding_size);
					CCV_TENSOR_SET_DEVICE_ID(pre_vec_params.type, j);
					word_vec[j] = ccv_nnc_tensor_variable_new(dynamic_graph, pre_vec_params);
					ccv_nnc_tensor_param_t vec_params = GPU_TENSOR_NCHW(000, 32F, batch_size, batch_length, embedding_size);
					CCV_TENSOR_SET_DEVICE_ID(vec_params.type, j);
					vec[j * 3] = ccv_nnc_tensor_variable_alias_new(dynamic_graph, word_vec[j], ccv_nnc_no_ofs, DIM_ALLOC(), vec_params);
					ccv_nnc_tensor_param_t mask_params = GPU_TENSOR_NCHW(000, 32S, batch_size);
					CCV_TENSOR_SET_DEVICE_ID(mask_params.type, j);
					assert(tensor[j * 4 + 2][0]->info.dim[0] == batch_size);
					ccv_nnc_tensor_param_t index_params = GPU_TENSOR_NCHW(000, 32S, batch_size);
					CCV_TENSOR_SET_DEVICE_ID(index_params.type, j);
					assert(tensor[j * 4 + 3][0]->info.dim[0] == batch_size);
					mask_tensor[j] = ccv_nnc_tensor(tensor[j * 4 + 2][0]->data.i32, mask_params, 0);
					index_tensor[j] = ccv_nnc_tensor(tensor[j * 4 + 3][0]->data.i32, index_params, 0);
					vec[j * 3 + 1] = ccv_nnc_tensor_constant_new(dynamic_graph, mask_params);
					ccv_nnc_tensor_variable_set(dynamic_graph, vec[j * 3 + 1], &mask_tensor[j]);
					vec[j * 3 + 2] = ccv_nnc_tensor_constant_new(dynamic_graph, index_params);
					ccv_nnc_tensor_variable_set(dynamic_graph, vec[j * 3 + 2], &index_tensor[j]);
					out[j] = ccv_nnc_tensor_variable_new(dynamic_graph);
				}
				ccv_nnc_tensor_variable_t tvin[2];
				for (j = 0; j < 1; j++)
					tvin[j * 2] = vocab_vec[j], tvin[j * 2 + 1] = word_indices[j];
				ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, tvin, 2, word_vec, 1, 1, 0);
				ccv_nnc_dynamic_graph_evaluate(dynamic_graph, lstm, 1, vec, 3, out, 1, 0, 0);
				int d;
				for (d = 0; d < 1; d++)
				{
					ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ccv_nnc_tensor_from_variable(dynamic_graph, out[d])), TENSOR_LIST(out_cpu), 0);
					ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor[d * 4 + 1][0]), TENSOR_LIST(fit_cpu), 0);
					for (j = 0; j < ccv_min(row_count - k - d * batch_size, batch_size); j++)
					{
						const int truth = (fit_cpu->data.f32[j] > 0.5);
						const int prediction = (out_cpu->data.f32[j] > 0);
						if (truth == prediction)
							++correct;
					}
				}
				for (j = 0; j < 1; j++)
				{
					ccv_nnc_tensor_variable_free(dynamic_graph, vec[j * 3]);
					ccv_nnc_tensor_variable_free(dynamic_graph, vec[j * 3 + 1]);
					ccv_nnc_tensor_variable_free(dynamic_graph, vec[j * 3 + 2]);
					ccv_nnc_tensor_variable_free(dynamic_graph, word_vec[j]);
					ccv_nnc_tensor_variable_free(dynamic_graph, word_indices[j]);
					ccv_nnc_tensor_variable_free(dynamic_graph, out[j]);
				}
			}
			ccv_cnnp_dataframe_iter_free(test_iter);
			ccv_nnc_dynamic_graph_set_no_grad(dynamic_graph, 0);
			printf("epoch %d done, training accuracy %lf, test accuracy %lf\n", epoch, overall_accuracy, (double)correct / row_count);
			++epoch;
			ccv_cnnp_dataframe_shuffle(train_data);
			ccv_cnnp_dataframe_iter_set_cursor(iter, 0);
		}
	}
	ccv_cnnp_model_free(lstm);
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
	return 0;
}
