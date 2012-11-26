#include "uri.h"
#include "ccv.h"
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>

static void uri_bbf_on_model_string(void* context, char* string);
static void uri_bbf_on_source_blob(void* context, ebb_buf data);

static const param_dispatch_t param_map[] = {
	{
		.property = "accurate",
		.type = PARAM_TYPE_BOOL,
		.offset = offsetof(ccv_bbf_param_t, accurate),
	},
	{
		.property = "interval",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_bbf_param_t, interval),
	},
	{
		.property = "min_neighbors",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_bbf_param_t, min_neighbors),
	},
	{
		.property = "model",
		.type = PARAM_TYPE_STRING,
		.on_string = uri_bbf_on_model_string,
		.offset = 0,
	},
	{
		.property = "size",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_bbf_param_t, min_neighbors),
	},
	{
		.property = "source",
		.type = PARAM_TYPE_BLOB,
		.on_blob = uri_bbf_on_source_blob,
		.offset = 0,
	},
};

typedef struct {
	ccv_bbf_classifier_cascade_t* face;
} bbf_context_t;

typedef struct {
	param_parser_t param_parser;
	bbf_context_t* context;
	ccv_bbf_param_t params;
	ccv_bbf_classifier_cascade_t* cascade;
	ebb_buf source;
} bbf_param_parser_t;

static void uri_bbf_param_parser_init(bbf_param_parser_t* parser)
{
	param_parser_init(&parser->param_parser, param_map, sizeof(param_map) / sizeof(param_dispatch_t), &parser->params, parser);
	parser->params = ccv_bbf_default_params;
	parser->cascade = 0;
	parser->source.data = 0;
}

static void uri_bbf_on_model_string(void* context, char* string)
{
	bbf_param_parser_t* parser = (bbf_param_parser_t*)context;
	if (strcmp(string, "face") == 0)
		parser->cascade = parser->context->face;
}

static void uri_bbf_on_source_blob(void* context, ebb_buf data)
{
	bbf_param_parser_t* parser = (bbf_param_parser_t*)context;
	parser->source = data;
}

void* uri_bbf_detect_objects_parse(const void* context, void* parsed, const char* buf, size_t len, uri_parse_state_t state, int header_index)
{
	bbf_param_parser_t* parser;
	if (parsed)
		parser = (bbf_param_parser_t*)parsed;
	else {
		parser = (bbf_param_parser_t*)malloc(sizeof(bbf_param_parser_t));
		uri_bbf_param_parser_init(parser);
		parser->context = (bbf_context_t*)context;
	}
	switch (state)
	{
		case URI_QUERY_STRING:
			break;
		case URI_CONTENT_BODY:
			break;
		case URI_PARSE_TERMINATE:
		case URI_MULTIPART_HEADER_FIELD:
		case URI_MULTIPART_HEADER_VALUE:
		case URI_MULTIPART_DATA:
			param_parser_execute(&parser->param_parser, buf, len, state, header_index);
			break;
	}
	return parser;
}

void* uri_bbf_detect_objects_init(void)
{
	bbf_context_t* context = (bbf_context_t*)malloc(sizeof(bbf_context_t));
	context->face = ccv_load_bbf_classifier_cascade("../samples/face");
	assert(context->face);
	assert(param_parser_map_alphabet(param_map, sizeof(param_map) / sizeof(param_dispatch_t)) == 0);
	return context;
}

void uri_bbf_detect_objects_destroy(void* context)
{
	bbf_context_t* bbf_context = (bbf_context_t*)context;
	ccv_bbf_classifier_cascade_free(bbf_context->face);
	free(bbf_context);
}

int uri_bbf_detect_objects_intro(const void* context, const void* parsed, ebb_buf* buf)
{
	/*
	const static char bbf_desc[] = 
		"HTTP/1.1 200 OK\r\nCache-Control: no-cache\r\nAccept: \r\nContent-Type: text/html\r\nContent-Length: 189\r\n\r\n"
		"<html><body><form enctype='multipart/form-data' method='post'><input name='size' value='24x24'><input name='model' value='face'><input type='file' name='source'><input type='submit'></form>\n";
	*/
	const static char bbf_desc[] =
		"HTTP/1.1 200 OK\r\nCache-Control: no-cache\r\nAccept: \r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: 162\r\n\r\n"
		"{"
			"\"request\":{"
				"\"model\":\"\","
				"\"size\":\"\","
				"\"interval\":\"\","
				"\"min_neighbors\":\"\","
				"\"accurate\":\"\","
				"\"source\":\"\""
			"},"
			"\"response\":[{"
				"\"x\":\"\","
				"\"y\":\"\","
				"\"width\":\"\","
				"\"height\":\"\","
				"\"confidence\":\"\""
			"}]"
		"}\n";
	buf->data = (void*)bbf_desc;
	buf->len = sizeof(bbf_desc);
	return 0;
}

int uri_bbf_detect_objects(const void* context, const void* parsed, ebb_buf* buf)
{
	bbf_param_parser_t* parser = (bbf_param_parser_t*)parsed;
	param_parser_terminate(&parser->param_parser);
	if (parser->source.data == 0)
	{
		free(parser);
		return -1;
	}
	if (parser->cascade == 0)
	{
		free(parser->source.data);
		free(parser);
		return -1;
	}
	ccv_dense_matrix_t* image = 0;
	ccv_read(parser->source.data, &image, CCV_IO_ANY_STREAM | CCV_IO_GRAY, parser->source.written);
	free(parser->source.data);
	if (image == 0)
	{
		free(parser);
		return -1;
	}
	ccv_array_t* seq = ccv_bbf_detect_objects(image, &parser->cascade, 1, parser->params);
	ccv_matrix_free(image);
	if (seq == 0)
	{
		free(parser);
		return -1;
	}
	if (seq->rnum > 0)
	{
		int i;
		buf->len = 192 + seq->rnum * 21 + 2;
		char* data = (char*)malloc(buf->len);
		data[0] = '[';
		buf->written = 1;
		for (i = 0; i < seq->rnum; i++)
		{
			char cell[128];
			ccv_comp_t* comp = (ccv_comp_t*)ccv_array_get(seq, i);
			snprintf(cell, 128, "{\"x\":%d,\"y\":%d,\"width\":%d,\"height\":%d,\"confidence\":%f}", comp->rect.x, comp->rect.y, comp->rect.width, comp->rect.height, comp->confidence);
			size_t len = strnlen(cell, 128);
			while (buf->written + len + 1 >= buf->len)
			{
				buf->len = (buf->len * 3 + 1) / 2;
				data = (char*)realloc(data, buf->len);
			}
			memcpy(data + buf->written, cell, len);
			buf->written += len + 1;
			data[buf->written - 1] = (i == seq->rnum - 1) ? ']' : ',';
		}
		char http_header[192];
		snprintf(http_header, 192, ebb_http_header, buf->written);
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
	} else {
		buf->data = (void*)ebb_http_empty_array;
		buf->len = sizeof(ebb_http_empty_array);
		buf->on_release = 0;
	}
	ccv_array_free(seq);
	free(parser);
	return 0;
}
