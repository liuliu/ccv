#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <sys/time.h>
#include <ctype.h>
#include <getopt.h>
#include <stddef.h>

static ccv_array_t* _array_from_disk_new(const char* src, const char* dest, const int vocab_size, const int max_length)
{
	return 0;
}

static ccv_cnnp_model_t* _multihead_attention_new(const int b, const int t, const int h, const int k, const float dropout, const int has_m, const int has_mask)
{
	const ccv_cnnp_model_io_t x = ccv_cnnp_input();
	ccv_cnnp_model_t* const tokeys = ccv_cnnp_dense(k * h, (ccv_cnnp_param_t){
		.no_bias = 1,
	}, "tokeys");
	ccv_cnnp_model_t* const toqueries = ccv_cnnp_dense(k * h, (ccv_cnnp_param_t){
		.no_bias = 1,
	}, "toqueries");
	ccv_cnnp_model_t* const tovalues = ccv_cnnp_dense(k * h, (ccv_cnnp_param_t){
		.no_bias = 1,
	}, "tovalues");
	ccv_cnnp_model_io_t queries = ccv_cnnp_model_apply(toqueries, MODEL_IO_LIST(x));
	ccv_cnnp_model_io_t m = has_m ? ccv_cnnp_input() : 0;
	ccv_cnnp_model_io_t mask = has_mask ? ccv_cnnp_input() : 0;
	ccv_cnnp_model_io_t keys = (m) ? ccv_cnnp_model_apply(tokeys, MODEL_IO_LIST(m)) : ccv_cnnp_model_apply(tokeys, MODEL_IO_LIST(x));
	ccv_cnnp_model_io_t values = (m) ? ccv_cnnp_model_apply(tovalues, MODEL_IO_LIST(m)) : ccv_cnnp_model_apply(tovalues, MODEL_IO_LIST(x));
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
	dot = ccv_cnnp_model_apply(ccv_cnnp_reshape(DIM_ALLOC(b * h * t, t), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(dot));
	dot = ccv_cnnp_model_apply(ccv_cnnp_softmax(0), MODEL_IO_LIST(dot));
	dot = ccv_cnnp_model_apply(ccv_cnnp_reshape(DIM_ALLOC(b * h, t, t), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(dot));
	if (mask)
		dot = ccv_cnnp_model_apply(ccv_cnnp_matmul(NO_TRANSPOSE, NO_TRANSPOSE, 0), MODEL_IO_LIST(dot, mask));
	if (dropout > 0)
		dot = ccv_cnnp_model_apply(ccv_cnnp_dropout(dropout, 0), MODEL_IO_LIST(dot));
	ccv_cnnp_model_io_t out = ccv_cnnp_model_apply(ccv_cnnp_matmul(NO_TRANSPOSE, NO_TRANSPOSE, 0), MODEL_IO_LIST(dot, values));
	out = ccv_cnnp_model_apply(ccv_cnnp_reshape(DIM_ALLOC(b, h, t, k), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_transpose(1, 2, 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_reshape(DIM_ALLOC(b * t, h * k), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(out));
	ccv_cnnp_model_t* const unifyheads = ccv_cnnp_dense(k, (ccv_cnnp_param_t){}, "unifyheads");
	out = ccv_cnnp_model_apply(unifyheads, MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_reshape(DIM_ALLOC(b, t, k), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(out));
	if (m)
		return ccv_cnnp_model_new(MODEL_IO_LIST(x, m, mask), MODEL_IO_LIST(out), "multihead-attention");
	else
		return ccv_cnnp_model_new(MODEL_IO_LIST(x, mask), MODEL_IO_LIST(out), "multihead-attention");
}

ccv_cnnp_model_t* _encoder_block_new(const int k, const int h, const int b, const int t, const int ff, const float dropout)
{
	ccv_cnnp_model_io_t const x = ccv_cnnp_input();
	ccv_cnnp_model_io_t const mask = ccv_cnnp_input();
	// self-attention
	ccv_cnnp_model_t* const self_attention = _multihead_attention_new(k, h, b, t, dropout, 0, 1);
	ccv_cnnp_model_io_t out = ccv_cnnp_model_apply(self_attention, MODEL_IO_LIST(x, mask));
	out = ccv_cnnp_model_apply(ccv_cnnp_layer_norm(1e-5, DIM_ALLOC(2), 1, 0), MODEL_IO_LIST(out));
	const ccv_cnnp_model_io_t first = out = ccv_cnnp_model_apply(ccv_cnnp_add(0), MODEL_IO_LIST(x, out));
	if (dropout)
		out = ccv_cnnp_model_apply(ccv_cnnp_dropout(dropout, 0), MODEL_IO_LIST(out));
	// feed-forward
	out = ccv_cnnp_model_apply(ccv_cnnp_reshape(DIM_ALLOC(b * t, k), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_dense(ff, (ccv_cnnp_param_t){}, 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_relu(0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_dense(k, (ccv_cnnp_param_t){}, 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_reshape(DIM_ALLOC(b, t, k), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_layer_norm(1e-5, DIM_ALLOC(2), 1, 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_add(0), MODEL_IO_LIST(first, out));
	if (dropout)
		out = ccv_cnnp_model_apply(ccv_cnnp_dropout(dropout, 0), MODEL_IO_LIST(out));
	return ccv_cnnp_model_new(MODEL_IO_LIST(x, mask), MODEL_IO_LIST(out), "transformer");
}

ccv_cnnp_model_t* _decoder_block_new(const int k, const int h, const int b, const int t, const int ff, const float dropout)
{
	ccv_cnnp_model_io_t const x = ccv_cnnp_input();
	ccv_cnnp_model_io_t const m = ccv_cnnp_input();
	ccv_cnnp_model_io_t const src_mask = ccv_cnnp_input();
	ccv_cnnp_model_io_t const tgt_mask = ccv_cnnp_input();
	// self-attention
	ccv_cnnp_model_t* const self_attention = _multihead_attention_new(k, h, b, t, dropout, 0, 1);
	ccv_cnnp_model_io_t out = ccv_cnnp_model_apply(self_attention, MODEL_IO_LIST(x, tgt_mask));
	out = ccv_cnnp_model_apply(ccv_cnnp_layer_norm(1e-5, DIM_ALLOC(2), 1, 0), MODEL_IO_LIST(out));
	ccv_cnnp_model_io_t first = out = ccv_cnnp_model_apply(ccv_cnnp_add(0), MODEL_IO_LIST(x, out));
	if (dropout)
		out = ccv_cnnp_model_apply(ccv_cnnp_dropout(dropout, 0), MODEL_IO_LIST(out));
	// source-attention
	ccv_cnnp_model_t* const src_attention = _multihead_attention_new(k, h, b, t, dropout, 1, 1);
	out = ccv_cnnp_model_apply(src_attention, MODEL_IO_LIST(out, m, src_mask));
	out = ccv_cnnp_model_apply(ccv_cnnp_layer_norm(1e-5, DIM_ALLOC(2), 1, 0), MODEL_IO_LIST(out));
	first = out = ccv_cnnp_model_apply(ccv_cnnp_add(0), MODEL_IO_LIST(first, out));
	if (dropout)
		out = ccv_cnnp_model_apply(ccv_cnnp_dropout(dropout, 0), MODEL_IO_LIST(out));
	// feed-forward
	out = ccv_cnnp_model_apply(ccv_cnnp_reshape(DIM_ALLOC(b * t, k), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_dense(ff, (ccv_cnnp_param_t){}, 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_relu(0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_dense(k, (ccv_cnnp_param_t){}, 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_reshape(DIM_ALLOC(b, t, k), DIM_ALLOC(), DIM_ALLOC(), 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_layer_norm(1e-5, DIM_ALLOC(2), 1, 0), MODEL_IO_LIST(out));
	out = ccv_cnnp_model_apply(ccv_cnnp_add(0), MODEL_IO_LIST(first, out));
	if (dropout)
		out = ccv_cnnp_model_apply(ccv_cnnp_dropout(dropout, 0), MODEL_IO_LIST(out));
	return ccv_cnnp_model_new(MODEL_IO_LIST(x, m, src_mask, tgt_mask), MODEL_IO_LIST(out), "transformer");
}

static void train_wmt(const int vocab_size, const int batch_size, const int max_length, const int embedding_size, ccv_cnnp_dataframe_t* const train_data)
{
}

int main(int argc, char** argv)
{
	ccv_nnc_init();
	static struct option imdb_options[] = {
		/* help */
		{"help", 0, 0, 0},
		/* required parameters */
		{"src", 1, 0, 0},
		{"dest", 1, 0, 0},
		{"vocab-size", 1, 0, 0},
		{0, 0, 0, 0}
	};
	int c;
	char* src = 0;
	char* dest = 0;
	int vocab_size;
	while (getopt_long_only(argc, argv, "", imdb_options, &c) != -1)
	{
		switch (c)
		{
			case 0:
				exit(0);
			case 1:
				src = optarg;
				break;
			case 2:
				dest = optarg;
				break;
			case 3:
				vocab_size = atoi(optarg);
				break;
		}
	}
	const int max_length = 512;
	ccv_array_t* const train_set = _array_from_disk_new(src, dest, vocab_size, max_length);
	ccv_cnnp_dataframe_t* const train_data = ccv_cnnp_dataframe_from_array_new(train_set);
	train_wmt(vocab_size, 64, max_length, 128, train_data);
	ccv_cnnp_dataframe_free(train_data);
	return 0;
}
