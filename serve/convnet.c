#include "uri.h"
#include "ccv.h"
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>

static void uri_convnet_on_model_string(void* context, char* string);
static void uri_convnet_on_source_blob(void* context, ebb_buf data);

static const param_dispatch_t param_map[] = {
	{
		.property = "model",
		.type = PARAM_TYPE_STRING,
		.on_string = uri_convnet_on_model_string,
		.offset = 0,
	},
	{
		.property = "source",
		.type = PARAM_TYPE_BODY,
		.on_blob = uri_convnet_on_source_blob,
		.offset = 0,
	},
	{
		.property = "top",
		.type = PARAM_TYPE_INT,
		.offset = 0,
	},
};

typedef struct {
	ccv_convnet_t* convnet;
	ccv_array_t* words;
} convnet_and_words_t;

typedef struct {
	ebb_buf desc;
	convnet_and_words_t image_net[2];
} convnet_context_t;

typedef struct {
	param_parser_t param_parser;
	convnet_context_t* context;
	int top;
	convnet_and_words_t* convnet_and_words;
	ebb_buf source;
} convnet_param_parser_t;

static void uri_convnet_param_parser_init(convnet_param_parser_t* parser)
{
	param_parser_init(&parser->param_parser, param_map, sizeof(param_map) / sizeof(param_dispatch_t), &parser->top, parser);
	parser->top = 5;
	parser->convnet_and_words  = 0;
	parser->source.data = 0;
}

static void uri_convnet_on_model_string(void* context, char* string)
{
	convnet_param_parser_t* parser = (convnet_param_parser_t*)context;
	if (strcmp(string, "image-net-2012") == 0)
		parser->convnet_and_words = &parser->context->image_net[0];
	else if (strcmp(string, "image-net-2012-vgg-d") == 0)
		parser->convnet_and_words = &parser->context->image_net[1];
}

static void uri_convnet_on_source_blob(void* context, ebb_buf data)
{
	convnet_param_parser_t* parser = (convnet_param_parser_t*)context;
	parser->source = data;
}

static ccv_array_t* uri_convnet_words_read(char* filename)
{
	FILE* r = fopen(filename, "rt");
	if(r)
	{
		ccv_array_t* words = ccv_array_new(sizeof(char*), 32, 0);
		size_t len = 1024;
		char* word = (char*)malloc(len);
		ssize_t read;
		while((read = getline(&word, &len, r)) != -1)
		{
			while(read > 1 && isspace(word[read - 1]))
				read--;
			word[read] = 0;
			char* new_word = (char*)malloc(sizeof(char) * (read + 1));
			memcpy(new_word, word, sizeof(char) * (read + 1));
			ccv_array_push(words, &new_word);
		}
		free(word);
		return words;
	}
	return 0;
}

void* uri_convnet_classify_init(void)
{
	convnet_context_t* context = (convnet_context_t*)malloc(sizeof(convnet_context_t));
	context->image_net[0].convnet = ccv_convnet_read(0, "../samples/image-net-2012.sqlite3");
	assert(context->image_net[0].convnet);
	context->image_net[0].words = uri_convnet_words_read("../samples/image-net-2012.words");
	assert(context->image_net[0].words);
	context->image_net[1].words = uri_convnet_words_read("../samples/image-net-2012.words");
	assert(context->image_net[1].words);
	context->image_net[1].convnet = ccv_convnet_read(0, "../samples/image-net-2012-vgg-d.sqlite3");
	assert(param_parser_map_alphabet(param_map, sizeof(param_map) / sizeof(param_dispatch_t)) == 0);
	context->desc = param_parser_map_http_body(param_map, sizeof(param_map) / sizeof(param_dispatch_t),
		"[{"
			"\"word\":\"string\","
			"\"confidence\":\"number\""
		"}]");
	return context;
}

void uri_convnet_classify_destroy(void* context)
{
	convnet_context_t* convnet_context = (convnet_context_t*)context;
	int i, j;
	for (i = 0; i < 2; i++)
	{
		ccv_convnet_free(convnet_context->image_net[i].convnet);
		for (j = 0; j < convnet_context->image_net[i].words->rnum; j++)
		{
			char* word = (char*)ccv_array_get(convnet_context->image_net[i].words, j);
			free(word);
		}
		ccv_array_free(convnet_context->image_net[i].words);
	}
	free(convnet_context->desc.data);
	free(convnet_context);
}

void* uri_convnet_classify_parse(const void* context, void* parsed, int resource_id, const char* buf, size_t len, uri_parse_state_t state, int header_index)
{
	convnet_param_parser_t* parser;
	if (parsed)
		parser = (convnet_param_parser_t*)parsed;
	else {
		parser = (convnet_param_parser_t*)malloc(sizeof(convnet_param_parser_t));
		uri_convnet_param_parser_init(parser);
		parser->context = (convnet_context_t*)context;
	}
	switch (state)
	{
		case URI_QUERY_STRING:
		case URI_CONTENT_BODY:
		case URI_PARSE_TERMINATE:
		case URI_MULTIPART_HEADER_FIELD:
		case URI_MULTIPART_HEADER_VALUE:
		case URI_MULTIPART_DATA:
			param_parser_execute(&parser->param_parser, resource_id, buf, len, state, header_index);
			break;
	}
	return parser;
}

int uri_convnet_classify_intro(const void* context, const void* parsed, ebb_buf* buf)
{
	convnet_context_t* convnet_context = (convnet_context_t*)context;
	buf->data = convnet_context->desc.data;
	buf->len = convnet_context->desc.len;
	return 0;
}

int uri_convnet_classify(const void* context, const void* parsed, ebb_buf* buf)
{
	if (!parsed)
		return -1;
	convnet_param_parser_t* parser = (convnet_param_parser_t*)parsed;
	param_parser_terminate(&parser->param_parser);
	if (parser->source.data == 0)
	{
		free(parser);
		return -1;
	}
	if (parser->convnet_and_words == 0)
	{
		free(parser->source.data);
		free(parser);
		return -1;
	}
	if (parser->top <= 0 || parser->top > parser->convnet_and_words->words->rnum)
	{
		free(parser->source.data);
		free(parser);
		return -1;
	}
	ccv_convnet_t* convnet = parser->convnet_and_words->convnet;
	if (convnet == 0)
	{
		free(parser->source.data);
		free(parser);
		return -1;
	}
	ccv_dense_matrix_t* image = 0;
	ccv_read(parser->source.data, &image, CCV_IO_ANY_STREAM | CCV_IO_RGB_COLOR, parser->source.written);
	free(parser->source.data);
	if (image == 0)
	{
		free(parser);
		return -1;
	}
	ccv_dense_matrix_t* input = 0;
	ccv_convnet_input_formation(convnet->input, image, &input);
	ccv_matrix_free(image);
	ccv_array_t* rank = 0;
	ccv_convnet_classify(convnet, &input, 1, &rank, parser->top, 1);
	// print out
	buf->len = 192 + rank->rnum * 30 + 2;
	char* data = (char*)malloc(buf->len);
	data[0] = '[';
	buf->written = 1;
	int i;
	for (i = 0; i < rank->rnum; i++)
	{
		char cell[1024];
		ccv_classification_t* classification = (ccv_classification_t*)ccv_array_get(rank, i);
		char* word = *(char**)ccv_array_get(parser->convnet_and_words->words, classification->id);
		snprintf(cell, 1024, "{\"word\":\"%s\",\"confidence\":%f}", word, classification->confidence);
		size_t len = strnlen(cell, 1024);
		while (buf->written + len + 1 >= buf->len)
		{
			buf->len = (buf->len * 3 + 1) / 2;
			data = (char*)realloc(data, buf->len);
		}
		memcpy(data + buf->written, cell, len);
		buf->written += len + 1;
		data[buf->written - 1] = (i == rank->rnum - 1) ? ']' : ',';
	}
	// copy the http header
	char http_header[192];
	snprintf(http_header, 192, ebb_http_header, buf->written + 1);
	size_t len = strnlen(http_header, 192);
	if (buf->written + len + 1 >= buf->len)
	{
		buf->len = buf->written + len + 1;
		data = (char*)realloc(data, buf->len);
	}
	memmove(data + len, data, buf->written);
	memcpy(data, http_header, len);
	buf->written += len + 1;
	data[buf->written - 1] = '\n';
	buf->data = data;
	buf->len = buf->written;
	buf->on_release = uri_ebb_buf_free;
	ccv_array_free(rank);
	ccv_matrix_free(input);
	free(parser);
	return 0;
}
