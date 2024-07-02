#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <sys/time.h>
#include <ctype.h>
#include <getopt.h>
#include <stddef.h>
#include <3rdparty/khash/khash.h>
#include <3rdparty/sqlite3/sqlite3.h>

static unsigned int get_current_time(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

KHASH_MAP_INIT_STR(vocab_map, int)

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

static CCV_WARN_UNUSED(ccv_nnc_tensor_t*) _text_to_tensor_index(char* const line, const khash_t(vocab_map)* const vocab, const int vocab_size, const int max_length, const int has_beg_flag)
{
	const int unk_flag = vocab_size - 4;
	const int beg_flag = vocab_size - 3;
	const int end_flag = vocab_size - 2;
	const int pad_flag = vocab_size - 1;
	const size_t linelen = strlen(line);
	if (line[linelen - 1] == '\n') // Remove new line.
		line[linelen - 1] = 0;
	ccv_nnc_tensor_t* const tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, max_length), 0);
	char* saveptr;
	const char* token = strtok_r(line, " ", &saveptr);
	if (has_beg_flag)
		tensor->data.i32[0] = beg_flag;
	int t = has_beg_flag ? 1 : 0;
	while (token)
	{
		if (t >= max_length)
			break;
		const khiter_t k = kh_get(vocab_map, vocab, token);
		if (k != kh_end(vocab))
			tensor->data.i32[t++] = kh_val(vocab, k);
		else
			tensor->data.i32[t++] = unk_flag;
		token = strtok_r(0, " ", &saveptr);
	}
	if (t < max_length)
	{
		tensor->data.i32[t] = end_flag;
		for (++t; t < max_length; t++)
			tensor->data.i32[t] = pad_flag;
	}
	return tensor;
}

typedef struct {
	ccv_nnc_tensor_t* src;
	ccv_nnc_tensor_t* tgt;
	ccv_nnc_tensor_t* out;
	ccv_nnc_tensor_t* src_mask;
	ccv_nnc_tensor_t* tgt_mask;
} ccv_nnc_seq2seq_t;

static ccv_array_t* _array_from_disk_new(const char* src_file, const char* tgt_file, khash_t(vocab_map)* const src_vocab, const int src_vocab_size, khash_t(vocab_map)* const tgt_vocab, const int tgt_vocab_size, const int max_length)
{
	FILE *r_src = fopen(src_file, "r");
	FILE *r_tgt = fopen(tgt_file, "r");
	assert(r_src && "list doesn't exists");
	assert(r_tgt && "list doesn't exists");
	const int src_pad_flag = src_vocab_size - 1;
	const int tgt_pad_flag = tgt_vocab_size - 1;
	ccv_array_t* categorizeds = ccv_array_new(sizeof(ccv_nnc_seq2seq_t), 64, 0);
	char* src_line = (char*)ccmalloc(64 * 1024);
	size_t src_len = 64 * 1024;
	char* tgt_line = (char*)ccmalloc(64 * 1024);
	size_t tgt_len = 64 * 1024;
	while (getline(&src_line, &src_len, r_src) >= 0 && getline(&tgt_line, &tgt_len, r_tgt) >= 0) {
		ccv_nnc_tensor_t* const src = _text_to_tensor_index(src_line, src_vocab, src_vocab_size, max_length, 0);
		ccv_nnc_tensor_t* const tgt = _text_to_tensor_index(tgt_line, tgt_vocab, tgt_vocab_size, max_length, 1);
		ccv_nnc_tensor_t* const out = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, max_length), 0);
		memcpy(out->data.i32, tgt->data.i32 + 1, sizeof(int) * (max_length - 1));
		out->data.i32[max_length - 1] = tgt_pad_flag;
		int src_length = 0;
		int i;
		for (i = 0; !src_length && i < max_length; i++)
			if (src->data.i32[i] == src_pad_flag)
				src_length = i;
		ccv_nnc_tensor_t* const src_mask = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 1), 0);
		src_mask->data.i32[0] = src_length ? src_length : max_length;
		int tgt_length = 0;
		for (i = 0; !tgt_length && i < max_length; i++)
			if (tgt->data.i32[i] == tgt_pad_flag)
				tgt_length = i;
		ccv_nnc_tensor_t* const tgt_mask = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 1), 0);
		tgt_mask->data.i32[0] = tgt_length ? tgt_length : max_length;
		ccv_nnc_seq2seq_t seq2seq = {
			.src = src,
			.tgt = tgt,
			.out = out,
			.src_mask = src_mask,
			.tgt_mask = tgt_mask,
		};
		ccv_array_push(categorizeds, &seq2seq);
	}
	ccfree(src_line);
	ccfree(tgt_line);
	fclose(r_src);
	fclose(r_tgt);
	return categorizeds;
}

static ccv_cnnp_model_t* _multihead_attention_new(const int k, const int h, const int b, const int t, const float dropout, const int has_m)
{
	const ccv_cnnp_model_io_t x = ccv_cnnp_input();
	ccv_cnnp_model_t* const tokeys = ccv_cnnp_dense(k * h, 1, 0, 1, 0);
	ccv_cnnp_model_t* const toqueries = ccv_cnnp_dense(k * h, 1, 0, 1, 0);
	ccv_cnnp_model_t* const tovalues = ccv_cnnp_dense(k * h, 1, 0, 1, 0);
	ccv_cnnp_model_io_t queries = ccv_cnnp_model_apply(toqueries, MODEL_IO_LIST(x));
	ccv_cnnp_model_io_t m = has_m ? ccv_cnnp_input() : 0;
	ccv_cnnp_model_io_t mask = ccv_cnnp_input();
	ccv_cnnp_model_io_t keys = (m) ? ccv_cnnp_model_apply(tokeys, MODEL_IO_LIST(m)) : ccv_cnnp_model_apply(tokeys, MODEL_IO_LIST(x));
	ccv_cnnp_model_io_t values = (m) ? ccv_cnnp_model_apply(tovalues, MODEL_IO_LIST(m)) : ccv_cnnp_model_apply(tovalues, MODEL_IO_LIST(x));
	keys = ccv_cnnp_model_apply(ccv_cnnp_reshape(0, DIM_ALLOC(t, b, h, k), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(keys));
	queries = ccv_cnnp_model_apply(ccv_cnnp_reshape(0, DIM_ALLOC(t, b, h, k), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(queries));
	values = ccv_cnnp_model_apply(ccv_cnnp_reshape(0, DIM_ALLOC(t, b, h, k), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(values));
	keys = ccv_cnnp_model_apply(ccv_cnnp_transpose(0, 2, 0), MODEL_IO_LIST(keys));
	queries = ccv_cnnp_model_apply(ccv_cnnp_transpose(0, 2, 0), MODEL_IO_LIST(queries));
	values = ccv_cnnp_model_apply(ccv_cnnp_transpose(0, 2, 0), MODEL_IO_LIST(values));
	keys = ccv_cnnp_model_apply(ccv_cnnp_reshape(0, DIM_ALLOC(b * h, t, k), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(keys));
	queries = ccv_cnnp_model_apply(ccv_cnnp_reshape(0, DIM_ALLOC(b * h, t, k), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(queries));
	values = ccv_cnnp_model_apply(ccv_cnnp_reshape(0, DIM_ALLOC(b * h, t, k), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(values));
	ccv_cnnp_model_io_t dot = ccv_cnnp_model_apply(ccv_cnnp_matmul(NO_TRANSPOSE, TRANSPOSE(1, 2), 0, 0), MODEL_IO_LIST(queries, keys));
	const float scale = 1. / sqrt(k);
	dot = ccv_cnnp_model_apply(ccv_cnnp_scalar_mul(scale, 0), MODEL_IO_LIST(dot));
	dot = ccv_cnnp_model_apply(ccv_cnnp_masked_fill(0, -1e9, 0), MODEL_IO_LIST(dot, mask));
	dot = ccv_cnnp_model_apply(ccv_cnnp_reshape(0, DIM_ALLOC(b * h * t, t), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(dot));
	dot = ccv_cnnp_model_apply(ccv_cnnp_softmax(0), MODEL_IO_LIST(dot));
	if (dropout > 0)
		dot = ccv_cnnp_model_apply(ccv_cnnp_dropout(dropout, 0, 0), MODEL_IO_LIST(dot));
	dot = ccv_cnnp_model_apply(ccv_cnnp_reshape(0, DIM_ALLOC(b * h, t, t), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(dot));
	ccv_cnnp_model_io_t out = ccv_cnnp_model_apply(ccv_cnnp_matmul(NO_TRANSPOSE, NO_TRANSPOSE, 0, 0), MODEL_IO_LIST(dot, values));
	out = ccv_cnnp_model_apply(ccv_cnnp_reshape(0, DIM_ALLOC(h, b, t, k), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_transpose(0, 2, 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_reshape(0, DIM_ALLOC(b * t, h * k), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(out));
	ccv_cnnp_model_t* const unifyheads = ccv_cnnp_dense(k * h, 0, 0, 1, 0);
	out = ccv_cnnp_model_apply(unifyheads, MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_reshape(0, DIM_ALLOC(t, b, k * h), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(out));
	if (m)
		return ccv_cnnp_model_new(MODEL_IO_LIST(x, m, mask), MODEL_IO_LIST(out), 1, 0);
	else
		return ccv_cnnp_model_new(MODEL_IO_LIST(x, mask), MODEL_IO_LIST(out), 1, 0);
}

static ccv_cnnp_model_t* _encoder_block_new(const int k, const int h, const int b, const int t, const int ff, const float dropout)
{
	ccv_cnnp_model_io_t const x = ccv_cnnp_input();
	ccv_cnnp_model_io_t const mask = ccv_cnnp_input();
	// self-attention
	ccv_cnnp_model_t* const self_attention = _multihead_attention_new(k, h, b, t, dropout, 0);
	ccv_cnnp_model_io_t out = ccv_cnnp_model_apply(self_attention, MODEL_IO_LIST(x, mask));
	out = ccv_cnnp_model_apply(ccv_cnnp_layer_norm(1e-5, DIM_ALLOC(2), 1, 1, 1, 0), MODEL_IO_LIST(out));
	const ccv_cnnp_model_io_t first = out = ccv_cnnp_model_apply(ccv_cnnp_sum(0), MODEL_IO_LIST(x, out));
	if (dropout)
		out = ccv_cnnp_model_apply(ccv_cnnp_dropout(dropout, 0, 0), MODEL_IO_LIST(out));
	// feed-forward
	out = ccv_cnnp_model_apply(ccv_cnnp_reshape(0, DIM_ALLOC(b * t, k * h), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_dense(ff, 0, 0, 1, 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_relu(0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_dense(k * h, 0, 0, 1, 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_reshape(0, DIM_ALLOC(t, b, k * h), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_layer_norm(1e-5, DIM_ALLOC(2), 1, 1, 1, 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_sum(0), MODEL_IO_LIST(first, out));
	if (dropout)
		out = ccv_cnnp_model_apply(ccv_cnnp_dropout(dropout, 0, 0), MODEL_IO_LIST(out));
	return ccv_cnnp_model_new(MODEL_IO_LIST(x, mask), MODEL_IO_LIST(out), 1, 0);
}

static ccv_cnnp_model_t* _decoder_block_new(const int k, const int h, const int b, const int t, const int ff, const float dropout)
{
	ccv_cnnp_model_io_t const x = ccv_cnnp_input();
	ccv_cnnp_model_io_t const m = ccv_cnnp_input();
	ccv_cnnp_model_io_t const src_mask = ccv_cnnp_input();
	ccv_cnnp_model_io_t const tgt_mask = ccv_cnnp_input();
	// self-attention
	ccv_cnnp_model_t* const self_attention = _multihead_attention_new(k, h, b, t, dropout, 0);
	ccv_cnnp_model_io_t out = ccv_cnnp_model_apply(self_attention, MODEL_IO_LIST(x, tgt_mask));
	out = ccv_cnnp_model_apply(ccv_cnnp_layer_norm(1e-5, DIM_ALLOC(2), 1, 1, 1, 0), MODEL_IO_LIST(out));
	ccv_cnnp_model_io_t first = out = ccv_cnnp_model_apply(ccv_cnnp_sum(0), MODEL_IO_LIST(x, out));
	if (dropout)
		out = ccv_cnnp_model_apply(ccv_cnnp_dropout(dropout, 0, 0), MODEL_IO_LIST(out));
	// source-attention
	ccv_cnnp_model_t* const src_attention = _multihead_attention_new(k, h, b, t, dropout, 1);
	out = ccv_cnnp_model_apply(src_attention, MODEL_IO_LIST(out, m, src_mask));
	out = ccv_cnnp_model_apply(ccv_cnnp_layer_norm(1e-5, DIM_ALLOC(2), 1, 1, 1, 0), MODEL_IO_LIST(out));
	first = out = ccv_cnnp_model_apply(ccv_cnnp_sum(0), MODEL_IO_LIST(first, out));
	if (dropout)
		out = ccv_cnnp_model_apply(ccv_cnnp_dropout(dropout, 0, 0), MODEL_IO_LIST(out));
	// feed-forward
	out = ccv_cnnp_model_apply(ccv_cnnp_reshape(0, DIM_ALLOC(b * t, k * h), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_dense(ff, 0, 0, 1, 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_relu(0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_dense(k * h, 0, 0, 1, 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_reshape(0, DIM_ALLOC(t, b, k * h), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_layer_norm(1e-5, DIM_ALLOC(2), 1, 1, 1, 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_sum(0), MODEL_IO_LIST(first, out));
	if (dropout)
		out = ccv_cnnp_model_apply(ccv_cnnp_dropout(dropout, 0, 0), MODEL_IO_LIST(out));
	return ccv_cnnp_model_new(MODEL_IO_LIST(x, m, src_mask, tgt_mask), MODEL_IO_LIST(out), 1, 0);
}

ccv_cnnp_model_t* _encoder_decoder_new(const int tgt_vocab_size, const int layers, const int k, const int h, const int b, const int t, const int ff, const float dropout)
{
	int i;
	ccv_cnnp_model_io_t const src = ccv_cnnp_input();
	ccv_cnnp_model_io_t const tgt = ccv_cnnp_input();
	ccv_cnnp_model_io_t const src_mask = ccv_cnnp_input();
	ccv_cnnp_model_io_t const tgt_mask = ccv_cnnp_input();
	ccv_cnnp_model_io_t const first = ccv_cnnp_model_apply(ccv_cnnp_transpose(0, 1, 0), MODEL_IO_LIST(src));
	ccv_cnnp_model_io_t encoder_out = first;
	for (i = 0; i < layers; i++)
		encoder_out = ccv_cnnp_model_apply(_encoder_block_new(k, h, b, t, ff, dropout), MODEL_IO_LIST(encoder_out, src_mask));
	ccv_cnnp_model_io_t decoder_out = ccv_cnnp_model_apply(ccv_cnnp_transpose(0, 1, 0), MODEL_IO_LIST(tgt));
	for (i = 0; i < layers; i++)
		decoder_out = ccv_cnnp_model_apply(_decoder_block_new(k, h, b, t, ff, dropout), MODEL_IO_LIST(decoder_out, encoder_out, src_mask, tgt_mask));
	ccv_cnnp_model_io_t out = ccv_cnnp_model_apply(ccv_cnnp_transpose(0, 1, 0), MODEL_IO_LIST(decoder_out)); // t, b, d -> b, t, d
	out = ccv_cnnp_model_apply(ccv_cnnp_reshape(0, DIM_ALLOC(b * t, k * h), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(out));
	return ccv_cnnp_model_new(MODEL_IO_LIST(src, tgt, src_mask, tgt_mask), MODEL_IO_LIST(out), 1, 0);
}

typedef struct {
	int tgt_vocab_size;
	int layers;
	int h;
	int ff;
	float dropout;
} encoder_decoder_params_t;

static ccv_cnnp_model_t* _dynamic_encoder_decoder(const ccv_nnc_tensor_param_t* const inputs, const int input_size, void* const context)
{
	const encoder_decoder_params_t* const params = (encoder_decoder_params_t*)context;
	const int b = inputs[0].dim[0];
	const int t = inputs[0].dim[1];
	const int k = inputs[0].dim[2] / params->h;
	const int ff = params->ff * k;
	return _encoder_decoder_new(params->tgt_vocab_size, params->layers, k, params->h, b, t, ff, params->dropout);
}

static ccv_nnc_tensor_t* _tensor_tril_new(const int word_size)
{
	ccv_nnc_tensor_t* const tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32S, word_size, word_size), 0);
	parallel_for(i, word_size) {
		int j;
		for (j = 0; j <= i; j++)
			tensor->data.i32[i * word_size + j] = 1;
		for (j = i + 1; j < word_size; j++)
			tensor->data.i32[i * word_size + j] = 0;
	} parallel_endfor
	return tensor;
}

static void eval_wmt(const int max_length, const int embedding_size, const char* const tst_file, khash_t(vocab_map)* const src_vocab, const int src_vocab_size, khash_t(vocab_map)* const tgt_vocab, const int tgt_vocab_size)
{
	// Load the model
	encoder_decoder_params_t encoder_decoder_params = {
		.tgt_vocab_size = tgt_vocab_size,
		.layers = 6,
		.h = 8,
		.ff = 4 * 8,
		.dropout = 0.1,
	};
	ccv_cnnp_model_t* const wmt = ccv_cnnp_dynamic_new(_dynamic_encoder_decoder, &encoder_decoder_params, 0);
	ccv_nnc_cmd_t adam = CMD_ADAM_FORWARD(1, 0.0001, 0.9, 0.98, 0, 1e-9, 0);
	ccv_nnc_tensor_param_t inputs[4];
	inputs[0] = GPU_TENSOR_NCHW(000, 32F, 1, max_length, embedding_size);
	inputs[1] = GPU_TENSOR_NCHW(000, 32F, 1, max_length, embedding_size);
	inputs[2] = GPU_TENSOR_NCHW(000, 32S, 1, max_length, max_length);
	inputs[3] = GPU_TENSOR_NCHW(000, 32S, 1, max_length, max_length);
	ccv_cnnp_model_compile(wmt, inputs, 4, adam, CMD_NOOP());
	ccv_cnnp_model_read_from_file("wmt.checkpoint", 0, wmt);
	ccv_nnc_tensor_t* const seq_vec_ = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, max_length, embedding_size), 0);
	int i;
	for (i = 0; i < max_length; i++)
	{
		int j;
		const float div_term = exp(-2 * i * log(10000) / embedding_size);
		if (i % 2 == 0)
			for (j = 0; j < embedding_size; j++)
				seq_vec_->data.f32[i * embedding_size + j] = sin(j * div_term);
		else
			for (j = 0; j < embedding_size; j++)
				seq_vec_->data.f32[i * embedding_size + j] = cos(j * div_term);
	}
	ccv_nnc_dynamic_graph_t* const dynamic_graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_dynamic_graph_set_no_grad(dynamic_graph, 1);
	ccv_nnc_tensor_variable_t const seq_vec = ccv_nnc_tensor_constant_new(dynamic_graph, GPU_TENSOR_NCHW(000, 32F, max_length, embedding_size));
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(seq_vec_), TENSOR_LIST(ccv_nnc_tensor_from_variable(dynamic_graph, seq_vec)), 0);
	ccv_nnc_tensor_variable_t const src_vocab_vec = ccv_nnc_tensor_constant_new(dynamic_graph, GPU_TENSOR_NCHW(000, 32F, src_vocab_size, embedding_size));
	ccv_nnc_tensor_variable_t const tgt_vocab_vec = ccv_nnc_tensor_constant_new(dynamic_graph, GPU_TENSOR_NCHW(000, 32F, tgt_vocab_size, embedding_size));
	sqlite3* conn = 0;
	if (SQLITE_OK == sqlite3_open("wmt.checkpoint", &conn))
	{
		ccv_nnc_tensor_t* src_vocab_vec_ = ccv_nnc_tensor_from_variable(dynamic_graph, src_vocab_vec);
		ccv_nnc_tensor_read(conn, "src_vocab", 0, 0, 0, &src_vocab_vec_);
		assert(src_vocab_vec_ == ccv_nnc_tensor_from_variable(dynamic_graph, src_vocab_vec));
		ccv_nnc_tensor_t* tgt_vocab_vec_ = ccv_nnc_tensor_from_variable(dynamic_graph, tgt_vocab_vec);
		ccv_nnc_tensor_read(conn, "tgt_vocab", 0, 0, 0, &tgt_vocab_vec_);
		assert(tgt_vocab_vec_ == ccv_nnc_tensor_from_variable(dynamic_graph, tgt_vocab_vec));
		sqlite3_close(conn);
	}
	const float sqrt_d_model = sqrt(embedding_size);
	// Run model
	FILE* r_tst = fopen(tst_file, "r");
	assert(r_tst);
	char* tst_line = (char*)ccmalloc(64 * 1024);
	size_t tst_len = 64 * 1024;
	const int src_pad_flag = src_vocab_size - 1;
	const int tgt_unk_flag = tgt_vocab_size - 4;
	const int tgt_beg_flag = tgt_vocab_size - 3;
	const int tgt_end_flag = tgt_vocab_size - 2;
	const int tgt_pad_flag = tgt_vocab_size - 1;
	ccv_nnc_tensor_t* const src_mask_ = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32S, 1, max_length, max_length), 0);
	ccv_nnc_tensor_t* const tril_mask_ = _tensor_tril_new(max_length);
	ccv_nnc_tensor_t* const tgt_mask_ = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32S, 1, max_length, max_length), 0);
	ccv_nnc_tensor_t* const out_ = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, max_length, tgt_vocab_size), 0);
	ccv_nnc_tensor_t* const tgt_indices_ = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32S, max_length), 0);
	const char** tgt_vocab_idx = (const char**)ccmalloc(sizeof(char*) * tgt_vocab_size);
	khiter_t k;
	for (k = kh_begin(tgt_vocab); k != kh_end(tgt_vocab); ++k)
		if (kh_exist(tgt_vocab, k))
			tgt_vocab_idx[kh_val(tgt_vocab, k)] = kh_key(tgt_vocab, k);
	while (getline(&tst_line, &tst_len, r_tst) >= 0) {
		if (memcmp(tst_line, "---", 3) == 0) // Another doc.
		{
			printf("---\n");
			continue;
		}
		ccv_nnc_tensor_t* src_indices_ = _text_to_tensor_index(tst_line, src_vocab, src_vocab_size, max_length, 0);
		int length = 0;
		for (i = 0; !length && i < max_length; i++)
			if (src_indices_->data.i32[i] == src_pad_flag)
				length = i;
		if (length == 0)
			length = max_length;
		for (i = 0; i < max_length; i++)
			src_mask_->data.i32[i] = i < length ? 1 : 0;
		for (i = 1; i < length; i++)
			memcpy(src_mask_->data.i32 + i * max_length, src_mask_->data.i32, sizeof(int) * max_length);
		if (length < max_length)
			memset(src_mask_->data.i32 + length * max_length, 0, sizeof(int) * max_length * (max_length - length));
		ccv_nnc_tensor_variable_t const src_mask = ccv_nnc_tensor_constant_new(dynamic_graph, GPU_TENSOR_NCHW(000, 32S, 1, max_length, max_length));
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(src_mask_), TENSOR_LIST(ccv_nnc_tensor_from_variable(dynamic_graph, src_mask)), 0);
		ccv_nnc_tensor_variable_t const src_indices = ccv_nnc_tensor_constant_new(dynamic_graph, GPU_TENSOR_NCHW(000, 32S, max_length));
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(src_indices_), TENSOR_LIST(ccv_nnc_tensor_from_variable(dynamic_graph, src_indices)), 0);
		ccv_nnc_tensor_free(src_indices_);
		ccv_nnc_tensor_variable_t const src_vec = ccv_nnc_tensor_constant_new(dynamic_graph);
		ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(src_vocab_vec, src_indices), TENSOR_VARIABLE_LIST(src_vec), 0, 0);
		ccv_nnc_tensor_variable_t const src_combine_vec = ccv_nnc_tensor_constant_new(dynamic_graph);
		ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_ADD_FORWARD(sqrt_d_model, 1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(src_vec, seq_vec), TENSOR_VARIABLE_LIST(src_combine_vec), 0, 0);
		ccv_nnc_tensor_variable_free(dynamic_graph, src_vec);
		ccv_nnc_tensor_variable_free(dynamic_graph, src_indices);
		ccv_nnc_tensor_variable_t src_vec_alias = ccv_nnc_tensor_variable_alias_new(dynamic_graph, src_combine_vec, ccv_nnc_no_ofs, DIM_ALLOC(), GPU_TENSOR_NCHW(000, 32F, 1, max_length, embedding_size));
		// First try greedy decoding.
		for (i = 1; i < max_length; i++)
			tgt_indices_->data.i32[i] = tgt_pad_flag;
		tgt_indices_->data.i32[0] = tgt_beg_flag;
		for (i = 1; i < max_length; i++)
		{
			ccv_nnc_tensor_variable_t const tgt_indices = ccv_nnc_tensor_constant_new(dynamic_graph, GPU_TENSOR_NCHW(000, 32S, max_length));
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tgt_indices_), TENSOR_LIST(ccv_nnc_tensor_from_variable(dynamic_graph, tgt_indices)), 0);
			memcpy(tgt_mask_->data.i32, tril_mask_->data.i32, sizeof(int) * i * max_length);
			if (i < max_length)
				memset(tgt_mask_->data.i32 + i * max_length, 0, sizeof(int) * max_length * (max_length - i));
			ccv_nnc_tensor_variable_t const tgt_mask = ccv_nnc_tensor_constant_new(dynamic_graph, GPU_TENSOR_NCHW(000, 32S, 1, max_length, max_length));
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tgt_mask_), TENSOR_LIST(ccv_nnc_tensor_from_variable(dynamic_graph, tgt_mask)), 0);
			ccv_nnc_tensor_variable_t const tgt_vec = ccv_nnc_tensor_variable_new(dynamic_graph);
			ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(tgt_vocab_vec, tgt_indices), TENSOR_VARIABLE_LIST(tgt_vec), 0, 0);
			ccv_nnc_tensor_variable_free(dynamic_graph, tgt_indices);
			ccv_nnc_tensor_variable_t const tgt_combine_vec = ccv_nnc_tensor_variable_new(dynamic_graph);
			ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_ADD_FORWARD(sqrt_d_model, 1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(tgt_vec, seq_vec), TENSOR_VARIABLE_LIST(tgt_combine_vec), 0, 0);
			ccv_nnc_tensor_variable_t tgt_vec_alias = ccv_nnc_tensor_variable_alias_new(dynamic_graph, tgt_combine_vec, ccv_nnc_no_ofs, DIM_ALLOC(), GPU_TENSOR_NCHW(000, 32F, 1, max_length, embedding_size));
			ccv_nnc_tensor_variable_free(dynamic_graph, tgt_vec);
			ccv_nnc_tensor_variable_t const embed = ccv_nnc_tensor_variable_new(dynamic_graph);
			ccv_nnc_dynamic_graph_evaluate(dynamic_graph, wmt, 1, TENSOR_VARIABLE_LIST(src_vec_alias, tgt_vec_alias, src_mask, tgt_mask), TENSOR_VARIABLE_LIST(embed), 0, 0);
			ccv_nnc_tensor_variable_free(dynamic_graph, tgt_combine_vec);
			ccv_nnc_tensor_variable_free(dynamic_graph, tgt_vec_alias);
			ccv_nnc_tensor_variable_free(dynamic_graph, tgt_mask);
			ccv_nnc_tensor_variable_t const out = ccv_nnc_tensor_variable_new(dynamic_graph);
			ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(embed, tgt_vocab_vec), TENSOR_VARIABLE_LIST(out), 0, 0);
			ccv_nnc_tensor_variable_free(dynamic_graph, embed);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ccv_nnc_tensor_from_variable(dynamic_graph, out)), TENSOR_LIST(out_), 0);
			int j;
			int word_idx = 0;
			float* const word = out_->data.f32 + (i - 1) * tgt_vocab_size;
			float most_likely = word[0];
			for (j = 1; j < tgt_vocab_size; j++)
				if (word[j] > most_likely)
					word_idx = j, most_likely = word[j];
			tgt_indices_->data.i32[i] = word_idx;
			ccv_nnc_tensor_variable_free(dynamic_graph, out);
			if (word_idx == tgt_end_flag)
				break;
		}
		ccv_nnc_tensor_variable_free(dynamic_graph, src_vec_alias);
		ccv_nnc_tensor_variable_free(dynamic_graph, src_combine_vec);
		ccv_nnc_tensor_variable_free(dynamic_graph, src_mask);
		for (i = 1; tgt_indices_->data.i32[i] != tgt_pad_flag && i < max_length; i++)
			if (tgt_indices_->data.i32[i] != tgt_beg_flag && tgt_indices_->data.i32[i] != tgt_end_flag)
				printf("%s ", (tgt_indices_->data.i32[i] != tgt_unk_flag) ? tgt_vocab_idx[tgt_indices_->data.i32[i]] : "<unk>");
		printf("\n");
	}
	ccv_nnc_dynamic_graph_set_no_grad(dynamic_graph, 0);
	ccv_nnc_tensor_free(tril_mask_);
	ccv_nnc_tensor_free(tgt_mask_);
	ccv_nnc_tensor_free(src_mask_);
	ccv_nnc_tensor_free(tgt_indices_);
	ccv_nnc_tensor_free(out_);
	ccfree(tst_line);
	fclose(r_tst);
	ccv_cnnp_model_free(wmt);
	ccv_nnc_dynamic_graph_free(dynamic_graph);
	ccfree(tgt_vocab_idx);
}

static void train_wmt(const int epoch_limit, const int src_vocab_size, const int tgt_vocab_size, const int batch_size, const int max_length, const int embedding_size, ccv_cnnp_dataframe_t* const train_data)
{
	const int src_idx = ccv_cnnp_dataframe_extract_value(train_data, 0, offsetof(ccv_nnc_seq2seq_t, src), 0);
	const int tgt_idx = ccv_cnnp_dataframe_extract_value(train_data, 0, offsetof(ccv_nnc_seq2seq_t, tgt), 0);
	const int out_idx = ccv_cnnp_dataframe_extract_value(train_data, 0, offsetof(ccv_nnc_seq2seq_t, out), 0);
	const int src_mask_idx = ccv_cnnp_dataframe_extract_value(train_data, 0, offsetof(ccv_nnc_seq2seq_t, src_mask), 0);
	const int tgt_mask_idx = ccv_cnnp_dataframe_extract_value(train_data, 0, offsetof(ccv_nnc_seq2seq_t, tgt_mask), 0);
	const int device_count = ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU);
	ccv_cnnp_dataframe_t* const batched_data = ccv_cnnp_dataframe_combine_new(train_data, COLUMN_ID_LIST(src_idx, tgt_idx, out_idx, src_mask_idx, tgt_mask_idx), batch_size, device_count, CCV_TENSOR_FORMAT_NCHW);
	int mask_seq_len_batched[device_count * 2];
	int seq_len_batched[device_count * 3];
	int data_batched[device_count * 3];
	int i;
	for (i = 0; i < device_count; i++)
	{
		data_batched[i * 3] = ccv_cnnp_dataframe_extract_tuple(batched_data, 0, i * 5, 0);
		data_batched[i * 3 + 1] = ccv_cnnp_dataframe_extract_tuple(batched_data, 0, i * 5 + 1, 0);
		data_batched[i * 3 + 2] = ccv_cnnp_dataframe_extract_tuple(batched_data, 0, i * 5 + 2, 0);
		mask_seq_len_batched[i * 2] = seq_len_batched[i * 3] = ccv_cnnp_dataframe_extract_tuple(batched_data, 0, i * 5 + 3, 0);
		mask_seq_len_batched[i * 2 + 1] = seq_len_batched[i * 3 + 1] = ccv_cnnp_dataframe_extract_tuple(batched_data, 0, i * 5 + 4, 0);
		seq_len_batched[i * 3 + 2] = seq_len_batched[i * 3 + 1];
	}
	const int mask_batched = ccv_cnnp_dataframe_one_squared(batched_data, mask_seq_len_batched, device_count * 2, 1, max_length, 0);
	const int trunc_data_batched = ccv_cnnp_dataframe_truncate(batched_data, data_batched, device_count * 3, seq_len_batched, device_count * 3, 0);
	int gpu_batched[device_count * 5];
	for (i = 0; i < device_count; i++)
	{
		gpu_batched[i * 5] = ccv_cnnp_dataframe_copy_to_gpu(batched_data, trunc_data_batched, i * 3, 1, i, 0);
		gpu_batched[i * 5 + 1] = ccv_cnnp_dataframe_copy_to_gpu(batched_data, trunc_data_batched, i * 3 + 1, 1, i, 0);
		gpu_batched[i * 5 + 2] = ccv_cnnp_dataframe_copy_to_gpu(batched_data, trunc_data_batched, i * 3 + 2, 1, i, 0);
		gpu_batched[i * 5 + 3] = ccv_cnnp_dataframe_copy_to_gpu(batched_data, mask_batched, i * 2, 1, i, 0);
		gpu_batched[i * 5 + 4] = ccv_cnnp_dataframe_copy_to_gpu(batched_data, mask_batched, i * 2 + 1, 1, i, 0);
	}
	encoder_decoder_params_t encoder_decoder_params = {
		.tgt_vocab_size = tgt_vocab_size,
		.layers = 6,
		.h = 8,
		.ff = 4 * 8,
		.dropout = 0.1,
	};
	ccv_cnnp_model_t* const wmt = ccv_cnnp_dynamic_new(_dynamic_encoder_decoder, &encoder_decoder_params, 0);
	ccv_cnnp_model_set_data_parallel(wmt, device_count);
	const int epoch_end = (ccv_cnnp_dataframe_row_count(train_data) + device_count * batch_size - 1) / (device_count * batch_size);
	ccv_cnnp_dataframe_shuffle(train_data);
	ccv_nnc_cmd_t adam = CMD_ADAM_FORWARD(1, 0.0001, 0.9, 0.98, 0, 1e-9, 0);
	const int aux_size = ccv_nnc_minimizer_saved_aux_size(adam);
	ccv_nnc_dynamic_graph_t* const dynamic_graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_tensor_t* const seq_vec_ = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, max_length, embedding_size), 0);
	for (i = 0; i < max_length; i++)
	{
		int j;
		const float div_term = exp(-2 * i * log(10000) / embedding_size);
		if (i % 2 == 0)
			for (j = 0; j < embedding_size; j++)
				seq_vec_->data.f32[i * embedding_size + j] = sin(j * div_term);
		else
			for (j = 0; j < embedding_size; j++)
				seq_vec_->data.f32[i * embedding_size + j] = cos(j * div_term);
	}
	ccv_nnc_tensor_variable_t seq_vec[device_count];
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_tensor_param_t seq_params = GPU_TENSOR_NCHW(000, 32F, max_length, embedding_size);
		CCV_TENSOR_SET_DEVICE_ID(seq_params.type, i);
		seq_vec[i] = ccv_nnc_tensor_constant_new(dynamic_graph, seq_params);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(seq_vec_), TENSOR_LIST(ccv_nnc_tensor_from_variable(dynamic_graph, seq_vec[i])), 0);
	}
	ccv_nnc_tensor_free(seq_vec_);
	ccv_nnc_tensor_variable_t src_vocab_vec[device_count];
	ccv_nnc_tensor_variable_t tgt_vocab_vec[device_count];
	ccv_nnc_dynamic_graph_set_no_grad(dynamic_graph, 1);
	ccv_nnc_tensor_param_t src_vocab_params = GPU_TENSOR_NCHW(000, 32F, src_vocab_size, embedding_size);
	ccv_nnc_tensor_variable_t src_vocab_vec_ = ccv_nnc_tensor_constant_new(dynamic_graph, CPU_TENSOR_NCHW(32F, src_vocab_size, embedding_size));
	const float src_bound = sqrtf(6) / sqrtf(embedding_size + src_vocab_size); // Xavier init.
	ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_RANDOM_UNIFORM_FORWARD(-src_bound, src_bound), ccv_nnc_no_hint, 0, 0, 0, TENSOR_VARIABLE_LIST(src_vocab_vec_), 0, 0);
	ccv_nnc_tensor_param_t tgt_vocab_params = GPU_TENSOR_NCHW(000, 32F, tgt_vocab_size, embedding_size);
	const float tgt_bound = sqrtf(6) / sqrtf(embedding_size + tgt_vocab_size); // Xavier init.
	ccv_nnc_tensor_variable_t tgt_vocab_vec_ = ccv_nnc_tensor_constant_new(dynamic_graph, CPU_TENSOR_NCHW(32F, tgt_vocab_size, embedding_size));
	ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_RANDOM_UNIFORM_FORWARD(-tgt_bound, tgt_bound), ccv_nnc_no_hint, 0, 0, 0, TENSOR_VARIABLE_LIST(tgt_vocab_vec_), 0, 0);
	for (i = 0; i < device_count; i++)
	{
		CCV_TENSOR_SET_DEVICE_ID(src_vocab_params.type, i);
		src_vocab_vec[i] = ccv_nnc_tensor_variable_new(dynamic_graph, src_vocab_params);
		ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(src_vocab_vec_), TENSOR_VARIABLE_LIST(src_vocab_vec[i]), 0, 0);
		CCV_TENSOR_SET_DEVICE_ID(tgt_vocab_params.type, i);
		tgt_vocab_vec[i] = ccv_nnc_tensor_variable_new(dynamic_graph, tgt_vocab_params);
		ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(tgt_vocab_vec_), TENSOR_VARIABLE_LIST(tgt_vocab_vec[i]), 0, 0);
	}
	ccv_nnc_tensor_variable_free(dynamic_graph, src_vocab_vec_);
	ccv_nnc_tensor_variable_free(dynamic_graph, tgt_vocab_vec_);
	ccv_nnc_dynamic_graph_set_no_grad(dynamic_graph, 0);
	ccv_nnc_tensor_t* const seq_indices_ = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32S, batch_size * max_length), 0);
	for (i = 0; i < batch_size; i++)
	{
		int j;
		for (j = 0; j < max_length; j++)
			seq_indices_->data.i32[i * max_length + j] = j;
	}
	ccv_nnc_tensor_variable_t seq_indices[device_count];
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_tensor_param_t seq_params = GPU_TENSOR_NCHW(000, 32S, batch_size * max_length);
		CCV_TENSOR_SET_DEVICE_ID(seq_params.type, i);
		seq_indices[i] = ccv_nnc_tensor_constant_new(dynamic_graph, seq_params);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(seq_indices_), TENSOR_LIST(ccv_nnc_tensor_from_variable(dynamic_graph, seq_indices[i])), 0);
	}
	ccv_nnc_tensor_variable_t saved_auxs[device_count * aux_size * 2];
	for (i = 0; i < device_count; i++)
	{
		int j;
		for (j = 0; j < aux_size; j++)
		{
			ccv_nnc_tensor_param_t saved_aux_params = GPU_TENSOR_NCHW(000, 32F, src_vocab_size, embedding_size);
			CCV_TENSOR_SET_DEVICE_ID(saved_aux_params.type, i);
			saved_auxs[i * aux_size * 2 + j] = ccv_nnc_tensor_variable_new(dynamic_graph, saved_aux_params);
		}
		for (j = 0; j < aux_size; j++)
		{
			ccv_nnc_tensor_param_t saved_aux_params = GPU_TENSOR_NCHW(000, 32F, tgt_vocab_size, embedding_size);
			CCV_TENSOR_SET_DEVICE_ID(saved_aux_params.type, i);
			saved_auxs[i* aux_size * 2 + aux_size + j] = ccv_nnc_tensor_variable_new(dynamic_graph, saved_aux_params);
		}
	}
	ccv_nnc_tensor_t* const out_ = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, batch_size * max_length, tgt_vocab_size), 0);
	ccv_nnc_tensor_t* const tgt_ = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32S, batch_size, max_length), 0);
	// CCV_CLI_SET_OUTPUT_LEVEL_AND_ABOVE(CCV_CLI_VERBOSE);
	ccv_cnnp_dataframe_iter_t* const iter = ccv_cnnp_dataframe_iter_new(batched_data, gpu_batched, device_count * 5);
	double overall_accuracy = 0;
	int epoch = 0;
	const int tgt_pad_flag = tgt_vocab_size - 1;
	unsigned int current_time = get_current_time();
	int elapsed_token = 0;
	const float sqrt_d_model = sqrt(embedding_size);
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	const int warmup_steps = 4000;
	ccv_nnc_tensor_variable_t src_vocab_vec_grad[device_count];
	ccv_nnc_tensor_variable_t tgt_vocab_vec_grad[device_count];
	for (i = 0; i < device_count; i++)
	{
		src_vocab_vec_grad[i] = ccv_nnc_tensor_variable_new(dynamic_graph);
		tgt_vocab_vec_grad[i] = ccv_nnc_tensor_variable_new(dynamic_graph);
	}
	for (i = 0; epoch < epoch_limit; i++)
	{
		ccv_nnc_tensor_t** tensors[device_count * 5];
		ccv_cnnp_dataframe_iter_next(iter, (void**)tensors, device_count * 5, stream);
		const int word_size = tensors[0][0]->info.dim[1];
		ccv_nnc_tensor_t* const tril_mask_ = _tensor_tril_new(word_size);
		ccv_nnc_tensor_t src_word_indices_[device_count];
		ccv_nnc_tensor_t src_mask_[device_count];
		ccv_nnc_tensor_variable_t src_word_indices[device_count];
		ccv_nnc_tensor_variable_t src_word_vec[device_count];
		ccv_nnc_tensor_variable_t src_combine_vec[device_count];
		ccv_nnc_tensor_t tgt_word_indices_[device_count];
		ccv_nnc_tensor_t tgt_mask_[device_count];
		ccv_nnc_tensor_variable_t tgt_word_indices[device_count];
		ccv_nnc_tensor_t out_word_indices_[device_count];
		ccv_nnc_tensor_variable_t out_word_indices[device_count];
		ccv_nnc_tensor_variable_t tgt_word_vec[device_count];
		ccv_nnc_tensor_variable_t tgt_combine_vec[device_count];
		ccv_nnc_tensor_variable_t vec[device_count * 4];
		ccv_nnc_tensor_variable_t pos_vec[device_count];
		ccv_nnc_tensor_variable_t embed[device_count];
		ccv_nnc_tensor_variable_t out[device_count];
		ccv_nnc_tensor_variable_t seq_indices_t[device_count];
		int j;
		for (j = 0; j < batch_size; j++)
		{
			int k;
			for (k = 0; k < word_size; k++)
				seq_indices_->data.i32[j * word_size + k] = k;
		}
		for (j = 0; j < device_count; j++)
		{
			// src
			ccv_nnc_tensor_param_t word_indices_params = GPU_TENSOR_NCHW(000, 32S, batch_size * word_size);
			CCV_TENSOR_SET_DEVICE_ID(word_indices_params.type, j);
			src_word_indices_[j] = ccv_nnc_tensor(tensors[j * 5][0]->data.i32, word_indices_params, 0);
			src_word_indices[j] = ccv_nnc_tensor_variable_new(dynamic_graph, word_indices_params);
			ccv_nnc_tensor_variable_set(dynamic_graph, src_word_indices[j], &src_word_indices_[j]);
			ccv_nnc_tensor_param_t pre_vec_params = GPU_TENSOR_NCHW(000, 32F, batch_size * word_size, embedding_size);
			CCV_TENSOR_SET_DEVICE_ID(pre_vec_params.type, j);
			src_word_vec[j] = ccv_nnc_tensor_variable_new(dynamic_graph, pre_vec_params);
			pos_vec[j] = ccv_nnc_tensor_variable_new(dynamic_graph, pre_vec_params);
			src_combine_vec[j] = ccv_nnc_tensor_variable_new(dynamic_graph);
			ccv_nnc_tensor_param_t vec_params = GPU_TENSOR_NCHW(000, 32F, batch_size, word_size, embedding_size);
			CCV_TENSOR_SET_DEVICE_ID(vec_params.type, j);
			vec[j * 4] = ccv_nnc_tensor_variable_alias_new(dynamic_graph, src_combine_vec[j], ccv_nnc_no_ofs, DIM_ALLOC(), vec_params);
			ccv_nnc_tensor_param_t mask_params = GPU_TENSOR_NCHW(000, 32S, batch_size, word_size, word_size);
			CCV_TENSOR_SET_DEVICE_ID(mask_params.type, j);
			assert(tensors[j * 5 + 3][0]->info.dim[1] == word_size);
			assert(tensors[j * 5 + 3][0]->info.dim[2] == word_size);
			src_mask_[j] = ccv_nnc_tensor(tensors[j * 5 + 3][0]->data.i32, mask_params, 0);
			vec[j * 4 + 2] = ccv_nnc_tensor_constant_new(dynamic_graph, mask_params);
			ccv_nnc_tensor_variable_set(dynamic_graph, vec[j * 4 + 2], &src_mask_[j]);
			// tgt
			tgt_word_indices_[j] = ccv_nnc_tensor(tensors[j * 5 + 1][0]->data.i32, word_indices_params, 0);
			tgt_word_indices[j] = ccv_nnc_tensor_variable_new(dynamic_graph, word_indices_params);
			ccv_nnc_tensor_variable_set(dynamic_graph, tgt_word_indices[j], &tgt_word_indices_[j]);
			tgt_word_vec[j] = ccv_nnc_tensor_variable_new(dynamic_graph, pre_vec_params);
			tgt_combine_vec[j] = ccv_nnc_tensor_variable_new(dynamic_graph);
			vec[j * 4 + 1] = ccv_nnc_tensor_variable_alias_new(dynamic_graph, tgt_combine_vec[j], ccv_nnc_no_ofs, DIM_ALLOC(), vec_params);
			assert(tensors[j * 5 + 4][0]->info.dim[1] == word_size);
			assert(tensors[j * 5 + 4][0]->info.dim[2] == word_size);
			tgt_mask_[j] = ccv_nnc_tensor(tensors[j * 5 + 4][0]->data.i32, mask_params, 0);
			ccv_nnc_tensor_param_t tril_params = GPU_TENSOR_NCHW(000, 32S, word_size, word_size);
			CCV_TENSOR_SET_DEVICE_ID(tril_params.type, j);
			ccv_nnc_tensor_variable_t tril_mask = ccv_nnc_tensor_constant_new(dynamic_graph, tril_params);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tril_mask_), TENSOR_LIST(ccv_nnc_tensor_from_variable(dynamic_graph, tril_mask)), 0);
			ccv_nnc_tensor_variable_t tgt_mask = ccv_nnc_tensor_constant_new(dynamic_graph, mask_params);
			ccv_nnc_tensor_variable_set(dynamic_graph, tgt_mask, &tgt_mask_[j]);
			out_word_indices_[j] = ccv_nnc_tensor(tensors[j * 5 + 2][0]->data.i32, word_indices_params, 0);
			out_word_indices[j] = ccv_nnc_tensor_variable_new(dynamic_graph, word_indices_params);
			ccv_nnc_tensor_variable_set(dynamic_graph, out_word_indices[j], &out_word_indices_[j]);
			vec[j * 4 + 3] = ccv_nnc_tensor_constant_new(dynamic_graph, mask_params);
			ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_MASKED_FILL_FORWARD(0, 0), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(tgt_mask, tril_mask), TENSOR_VARIABLE_LIST(vec[j * 4 + 3]), 0, 0);
			ccv_nnc_tensor_variable_free(dynamic_graph, tril_mask);
			ccv_nnc_tensor_variable_free(dynamic_graph, tgt_mask);
			// others.
			out[j] = ccv_nnc_tensor_variable_new(dynamic_graph);
			embed[j] = ccv_nnc_tensor_variable_new(dynamic_graph);
			ccv_nnc_tensor_param_t seq_params = GPU_TENSOR_NCHW(000, 32S, batch_size * word_size);
			CCV_TENSOR_SET_DEVICE_ID(seq_params.type, j);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(seq_indices_), TENSOR_LIST(ccv_nnc_tensor_from_variable(dynamic_graph, seq_indices[j])), 0);
			seq_indices_t[j] = ccv_nnc_tensor_variable_alias_new(dynamic_graph, seq_indices[j], ccv_nnc_no_ofs, DIM_ALLOC(), seq_params);
		}
		ccv_nnc_tensor_free(tril_mask_);
		ccv_nnc_tensor_variable_t tvin[device_count * 2];
		// pos
		for (j = 0; j < device_count; j++)
			tvin[j * 2] = seq_vec[j], tvin[j * 2 + 1] = seq_indices_t[j];
		ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, tvin, device_count * 2, pos_vec, device_count, device_count, stream);
		// src.
		for (j = 0; j < device_count; j++)
			tvin[j * 2] = src_vocab_vec[j], tvin[j * 2 + 1] = src_word_indices[j];
		ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, tvin, device_count * 2, src_word_vec, device_count, device_count, stream);
		for (j = 0; j < device_count; j++)
			tvin[j * 2] = src_word_vec[j], tvin[j * 2 + 1] = pos_vec[j];
		ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_ADD_FORWARD(sqrt_d_model, 1), ccv_nnc_no_hint, 0, tvin, device_count * 2, src_combine_vec, device_count, device_count, stream);
		// tgt
		for (j = 0; j < device_count; j++)
			tvin[j * 2] = tgt_vocab_vec[j], tvin[j * 2 + 1] = tgt_word_indices[j];
		ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, tvin, device_count * 2, tgt_word_vec, device_count, device_count, stream);
		for (j = 0; j < device_count; j++)
			tvin[j * 2] = tgt_word_vec[j], tvin[j * 2 + 1] = pos_vec[j];
		ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_ADD_FORWARD(sqrt_d_model, 1), ccv_nnc_no_hint, 0, tvin, device_count * 2, tgt_combine_vec, device_count, device_count, stream);
		ccv_nnc_dynamic_graph_evaluate(dynamic_graph, wmt, 0, vec, device_count * 4, embed, device_count, 0, stream);
		for (j = 0; j < device_count; j++)
			tvin[j * 2] = embed[j], tvin[j * 2 + 1] = tgt_vocab_vec[j];
		ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, tvin, device_count * 2, out, device_count, device_count, stream);
		// Loss.
		ccv_nnc_tensor_variable_t softmax[device_count];
		for (j = 0; j < device_count; j++)
			softmax[j] = ccv_nnc_tensor_variable_new(dynamic_graph);
		ccv_nnc_tensor_variable_t tvout[device_count * 2];
		for (j = 0; j < device_count; j++)
			tvin[j * 2] = out[j], tvin[j * 2 + 1] = out_word_indices[j], tvout[j * 2] = 0, tvout[j * 2 + 1] = softmax[j];
		const float eta = 0.1;
		const float trim0 = eta / tgt_vocab_size;
		const float trim1 = 1.0 - eta + trim0;
		ccv_nnc_dynamic_graph_exec(dynamic_graph, CMD_SOFTMAX_CROSSENTROPY_FORWARD(trim0, trim1), ccv_nnc_no_hint, 0, tvin, device_count * 2, tvout, device_count * 2, device_count, stream);
		for (j = 0; j < device_count; j++)
			tvin[j * 2] = src_vocab_vec[j], tvin[j * 2 + 1] = tgt_vocab_vec[j], tvout[j * 2] = src_vocab_vec_grad[j], tvout[j * 2 + 1] = tgt_vocab_vec_grad[j];
		ccv_nnc_dynamic_graph_backward(dynamic_graph, softmax, device_count, 0, tvin, device_count * 2, tvout, device_count * 2, stream);
		for (j = 0; j < device_count; j++)
		{
			ccv_nnc_tensor_variable_free(dynamic_graph, vec[j * 4]);
			ccv_nnc_tensor_variable_free(dynamic_graph, vec[j * 4 + 1]);
			ccv_nnc_tensor_variable_free(dynamic_graph, vec[j * 4 + 2]);
			ccv_nnc_tensor_variable_free(dynamic_graph, vec[j * 4 + 3]);
			ccv_nnc_tensor_variable_free(dynamic_graph, tgt_word_indices[j]);
			ccv_nnc_tensor_variable_free(dynamic_graph, tgt_word_vec[j]);
			ccv_nnc_tensor_variable_free(dynamic_graph, tgt_combine_vec[j]);
			ccv_nnc_tensor_variable_free(dynamic_graph, out_word_indices[j]);
			ccv_nnc_tensor_variable_free(dynamic_graph, seq_indices_t[j]);
			ccv_nnc_tensor_variable_free(dynamic_graph, embed[j]);
			ccv_nnc_tensor_variable_free(dynamic_graph, pos_vec[j]);
			ccv_nnc_tensor_variable_free(dynamic_graph, softmax[j]);
			ccv_nnc_tensor_variable_free(dynamic_graph, src_combine_vec[j]);
			ccv_nnc_tensor_variable_free(dynamic_graph, src_word_vec[j]);
			ccv_nnc_tensor_variable_free(dynamic_graph, src_word_indices[j]);
		}
		elapsed_token += batch_size * word_size * device_count;
		if ((i + 1) % 50 == 0)
		{
			ccv_nnc_tensor_t out_t = ccv_nnc_tensor(out_->data.f32, CPU_TENSOR_NCHW(32F, batch_size * word_size, tgt_vocab_size), 0);
			ccv_nnc_tensor_t tgt_t = ccv_nnc_tensor(tgt_->data.i32, CPU_TENSOR_NCHW(32S, batch_size, word_size), 0);
			int correct = 0, overall = 0;
			for (j = 0; j < device_count; j++)
			{
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ccv_nnc_tensor_from_variable(dynamic_graph, out[j])), TENSOR_LIST(&out_t), 0);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(&out_word_indices_[j]), TENSOR_LIST(&tgt_t), 0);
				int x, y;
				for (y = 0; y < batch_size * word_size; y++)
				{
					float* const word = out_t.data.f32 + y * tgt_vocab_size;
					if (tgt_t.data.i32[y] == tgt_pad_flag) // Ignore padding.
						continue;
					int word_idx = 0;
					float most_likely = word[0];
					for (x = 1; x < tgt_vocab_size; x++)
						if (word[x] > most_likely)
							word_idx = x, most_likely = word[x];
					if (word_idx == tgt_t.data.i32[y])
						++correct;
					++overall;
				}
			}
			unsigned int elapsed_time = get_current_time() - current_time;
			float token_per_sec = (float)elapsed_token * 1000 / (float)elapsed_time;
			elapsed_token = 0;
			const double accuracy = (double)correct / overall;
			overall_accuracy = overall_accuracy * 0.9 + accuracy * 0.1;
			printf("epoch %d (%d/%d), batch accuracy %lf, overall accuracy %lf, tokens per sec %.2lf\n", epoch, (i + 1) - epoch * epoch_end, epoch_end, accuracy, overall_accuracy, token_per_sec);
			ccv_cnnp_model_write_to_file(wmt, "wmt.checkpoint", 0);
			sqlite3* conn = 0;
			if (SQLITE_OK == sqlite3_open("wmt.checkpoint", &conn))
			{
				ccv_nnc_tensor_write(ccv_nnc_tensor_from_variable(dynamic_graph, src_vocab_vec[0]), conn, "src_vocab", 0);
				ccv_nnc_tensor_write(ccv_nnc_tensor_from_variable(dynamic_graph, tgt_vocab_vec[0]), conn, "tgt_vocab", 0);
				sqlite3_close(conn);
			}
			current_time = get_current_time();
		}
		for (j = 0; j < device_count; j++)
			ccv_nnc_tensor_variable_free(dynamic_graph, out[j]);
		if ((i + 1) % epoch_end == 0)
		{
			++epoch;
			ccv_cnnp_dataframe_shuffle(train_data);
			ccv_cnnp_dataframe_iter_set_cursor(iter, 0);
		}
		const int big_step = 5;
		if ((i + 1) % big_step == 0)
		{
			float learn_rate = 1. / sqrt_d_model * ccv_min(1. / sqrtf((i + 1) / big_step), (float)((i + 1) / big_step) / (sqrtf(warmup_steps) * warmup_steps));
			adam = CMD_ADAM_FORWARD((i + 1) / big_step, learn_rate, 0.9, 0.98, 0, 1e-9, 0);
			ccv_cnnp_model_set_minimizer(wmt, adam, 0, 0, 0);
			for (j = 0; j < device_count; j++)
				tvin[j * 2] = src_vocab_vec_grad[j], tvin[j * 2 + 1] = tgt_vocab_vec_grad[j], tvout[j * 2] = src_vocab_vec[j], tvout[j * 2 + 1] = tgt_vocab_vec[j];
			ccv_nnc_dynamic_graph_apply_gradients(dynamic_graph, adam, tvin, device_count * 2, tvout, device_count * 2, saved_auxs, device_count, stream);
			for (j = 0; j < device_count; j++)
			{
				ccv_nnc_tensor_variable_free(dynamic_graph, src_vocab_vec_grad[j]);
				ccv_nnc_tensor_variable_free(dynamic_graph, tgt_vocab_vec_grad[j]);
				src_vocab_vec_grad[j] = ccv_nnc_tensor_variable_new(dynamic_graph);
				tgt_vocab_vec_grad[j] = ccv_nnc_tensor_variable_new(dynamic_graph);
			}
		}
	}
	ccv_nnc_stream_context_free(stream);
	ccv_nnc_tensor_free(seq_indices_);
	ccv_nnc_tensor_free(out_);
	ccv_nnc_tensor_free(tgt_);
	ccv_cnnp_model_free(wmt);
	ccv_cnnp_dataframe_iter_free(iter);
	ccv_cnnp_dataframe_free(batched_data);
	ccv_nnc_dynamic_graph_free(dynamic_graph);
}

int main(int argc, char** argv)
{
	ccv_nnc_init();
	static struct option imdb_options[] = {
		/* help */
		{"help", 0, 0, 0},
		/* required parameters */
		{"src-vocab", 1, 0, 0},
		{"tgt-vocab", 1, 0, 0},
		/* optional parameters */
		{"src", 1, 0, 0},
		{"tgt", 1, 0, 0},
		{"tst", 1, 0, 0},
		{0, 0, 0, 0}
	};
	int c;
	char* src_file = 0;
	char* tgt_file = 0;
	char* tst_file = 0;
	char* src_vocab_file = 0;
	char* tgt_vocab_file = 0;
	while (getopt_long_only(argc, argv, "", imdb_options, &c) != -1)
	{
		switch (c)
		{
			case 0:
				printf("Data can be downloaded from https://nlp.stanford.edu/projects/nmt/\n");
				exit(0);
			case 1:
				src_vocab_file = optarg;
				break;
			case 2:
				tgt_vocab_file = optarg;
				break;
			case 3:
				src_file = optarg;
				break;
			case 4:
				tgt_file = optarg;
				break;
			case 5:
				tst_file = optarg;
				break;
		}
	}
	khash_t(vocab_map)* src_vocab;
	int src_vocab_size;
	_vocab_init(src_vocab_file, &src_vocab, &src_vocab_size);
	khash_t(vocab_map)* tgt_vocab;
	int tgt_vocab_size;
	_vocab_init(tgt_vocab_file, &tgt_vocab, &tgt_vocab_size);
	const int max_length = 128; // Training data max length is 99.
	src_vocab_size = src_vocab_size + 4;
	tgt_vocab_size = tgt_vocab_size + 4;
	// vocab_size - 1 - padding
	// vocab_size - 2 - ending
	// vocab_size - 3 - start
	// vocab_size - 4 - unknown
	if (!tst_file)
	{
		assert(src_file);
		assert(tgt_file);
		ccv_array_t* const train_set = _array_from_disk_new(src_file, tgt_file, src_vocab, src_vocab_size, tgt_vocab, tgt_vocab_size, max_length);
		ccv_cnnp_dataframe_t* const train_data = ccv_cnnp_dataframe_from_array_new(train_set);
		printf("%d pairs, source vocabulary size %d, target vocabulary size %d\n", train_set->rnum, src_vocab_size, tgt_vocab_size);
		train_wmt(10, src_vocab_size, tgt_vocab_size, 16, max_length, 512, train_data);
		ccv_cnnp_dataframe_free(train_data);
	} else {
		eval_wmt(max_length, 512, tst_file, src_vocab, src_vocab_size, tgt_vocab, tgt_vocab_size);
	}
	_vocab_destroy(src_vocab);
	_vocab_destroy(tgt_vocab);
	return 0;
}
