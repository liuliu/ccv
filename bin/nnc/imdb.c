#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <sys/time.h>
#include <ctype.h>
#include <getopt.h>
#include <stddef.h>
#include <3rdparty/khash/khash.h>

static ccv_array_t* _array_from_disk_new(const char* const list, const char* const base_dir)
{
	FILE *r = fopen(list, "r");
	assert(r && "list doesn't exists");
	int dirlen = (base_dir != 0) ? strlen(base_dir) + 1 : 0;
	ccv_array_t* categorizeds = ccv_array_new(sizeof(ccv_categorized_t), 64, 0);
	int c;
	char* file = (char*)ccmalloc(1024);
	while (fscanf(r, "%d %1023s", &c, file) != EOF)
	{
		char* filename = (char*)ccmalloc(1024);
		if (base_dir != 0)
		{
			strncpy(filename, base_dir, 1024);
			filename[dirlen - 1] = '/';
		}
		strncpy(filename + dirlen, file, 1024 - dirlen);
		ccv_file_info_t file_info = {
			.filename = filename,
		};
		ccv_categorized_t categorized = ccv_categorized(c, 0, &file_info);
		ccv_array_push(categorizeds, &categorized);
	}
	ccfree(file);
	fclose(r);
	return categorizeds;
}

KHASH_MAP_INIT_STR(vocab_map, int)

typedef struct {
	khash_t(vocab_map)* vocab;
	int vocab_size;
	int max_length;
} ccv_cnnp_text_context_t;

static void _ccv_cnnp_load_text(void* const* const* const column_data, const int column_size, const int batch_size, void** const data, void* const context, ccv_nnc_stream_context_t* const stream_context)
{
	int i;
	const ccv_cnnp_text_context_t* const text_context = (ccv_cnnp_text_context_t*)context;
	const khash_t(vocab_map)* const vocab = text_context->vocab;
	const int vocab_size = text_context->vocab_size;
	const int end_flag = vocab_size - 2;
	const int pad_flag = vocab_size - 1;
	const int max_length = text_context->max_length;
	char* const word = (char*)ccmalloc(1024);
	for (i = 0; i < batch_size; i++)
	{
		ccv_nnc_tensor_t* const tensor = data[i] = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, max_length), 0);
		const ccv_categorized_t* const categorized = (const ccv_categorized_t*)column_data[0][i];
		const char* const filename = categorized->file.filename;
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
	}
	ccfree(word);
}

static void _ccv_cnnp_tensor_deinit(void* const data, void* const context)
{
	ccv_nnc_tensor_free(data);
}

static void train_imdb(const int batch_size, const int embedding_size, const int max_length, const char* const vocab_file, ccv_cnnp_dataframe_t* const train_data, ccv_cnnp_dataframe_t* const test_data, ccv_array_t* const test_set)
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
	ccv_cnnp_text_context_t context = {
		.vocab = vocab,
		.vocab_size = i + 2, // end and padding.
		.max_length = max_length,
	};
	const int load_text = ccv_cnnp_dataframe_map(train_data, _ccv_cnnp_load_text, 0, _ccv_cnnp_tensor_deinit, COLUMN_ID_LIST(0), &context, 0);
	ccv_cnnp_dataframe_iter_t* const iter = ccv_cnnp_dataframe_iter_new(train_data, COLUMN_ID_LIST(load_text));
	ccv_nnc_tensor_t* tensor;
	ccv_cnnp_dataframe_iter_next(iter, (void**)&tensor, 1, 0);
	for (i = 0; i < max_length; i++)
		printf("%d ", tensor->data.i32[i]);
	printf("\n");
	ccv_cnnp_dataframe_iter_free(iter);
	// Free keys.
	for (khiter_t k = kh_begin(vocab); k != kh_end(vocab); k++)
		if (kh_exist(vocab, k))
			free((void*)kh_key(vocab, k));
	kh_destroy(vocab_map, vocab);
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
	ccv_array_t* const train_set = _array_from_disk_new(train_list, base_dir);
	ccv_cnnp_dataframe_t* const train_data = ccv_cnnp_dataframe_from_array_new(train_set);
	ccv_array_t* const test_set = _array_from_disk_new(test_list, base_dir);
	ccv_cnnp_dataframe_t* const test_data = ccv_cnnp_dataframe_from_array_new(test_set);
	train_imdb(128, 128, 512, vocab_file, train_data, test_data, test_set);
	ccv_cnnp_dataframe_free(train_data);
	ccv_cnnp_dataframe_free(test_data);
	int i;
	for (i = 0; i < train_set->rnum; i++)
		ccfree(((ccv_categorized_t*)ccv_array_get(train_set, i))->file.filename);
	ccv_array_free(train_set);
	for (i = 0; i < test_set->rnum; i++)
		ccfree(((ccv_categorized_t*)ccv_array_get(test_set, i))->file.filename);
	ccv_array_free(test_set);
	return 0;
}
