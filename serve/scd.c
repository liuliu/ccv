#include "uri.h"
#include "ccv.h"
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>

static void uri_scd_on_model_string(void* context, char* string);
static void uri_scd_on_source_blob(void* context, ebb_buf data);

typedef struct {
	ccv_scd_param_t params;
	int max_dimension;
} ccv_scd_uri_param_t;

static const param_dispatch_t param_map[] = {
	{
		.property = "interval",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_scd_uri_param_t, params) + offsetof(ccv_scd_param_t, interval),
	},
	{
		.property = "max_dimension",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_scd_uri_param_t, max_dimension),
	},
	{
		.property = "min_neighbors",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_scd_uri_param_t, params) + offsetof(ccv_scd_param_t, min_neighbors),
	},
	{
		.property = "model",
		.type = PARAM_TYPE_STRING,
		.on_string = uri_scd_on_model_string,
		.offset = 0,
	},
	{
		.property = "source",
		.type = PARAM_TYPE_BODY,
		.on_blob = uri_scd_on_source_blob,
		.offset = 0,
	},
};

typedef struct {
	ebb_buf desc;
	ccv_scd_classifier_cascade_t* face;
} scd_context_t;

typedef struct {
	param_parser_t param_parser;
	scd_context_t* context;
	ccv_scd_uri_param_t params;
	ccv_scd_classifier_cascade_t* cascade;
	ebb_buf source;
} scd_param_parser_t;

static void uri_scd_param_parser_init(scd_param_parser_t* parser)
{
	param_parser_init(&parser->param_parser, param_map, sizeof(param_map) / sizeof(param_dispatch_t), &parser->params, parser);
	parser->params.params = ccv_scd_default_params;
	parser->params.max_dimension = 0;
	parser->cascade = 0;
	parser->source.data = 0;
}

static void uri_scd_on_model_string(void* context, char* string)
{
	scd_param_parser_t* parser = (scd_param_parser_t*)context;
	if (strcmp(string, "face") == 0)
		parser->cascade = parser->context->face;
}

static void uri_scd_on_source_blob(void* context, ebb_buf data)
{
	scd_param_parser_t* parser = (scd_param_parser_t*)context;
	parser->source = data;
}

void* uri_scd_detect_objects_parse(const void* context, void* parsed, int resource_id, const char* buf, size_t len, uri_parse_state_t state, int header_index)
{
	scd_param_parser_t* parser;
	if (parsed)
		parser = (scd_param_parser_t*)parsed;
	else {
		parser = (scd_param_parser_t*)malloc(sizeof(scd_param_parser_t));
		uri_scd_param_parser_init(parser);
		parser->context = (scd_context_t*)context;
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

void* uri_scd_detect_objects_init(void)
{
	scd_context_t* context = (scd_context_t*)malloc(sizeof(scd_context_t));
	context->face = ccv_scd_classifier_cascade_read("../samples/face.sqlite3");
	assert(context->face);
	assert(param_parser_map_alphabet(param_map, sizeof(param_map) / sizeof(param_dispatch_t)) == 0);
	context->desc = param_parser_map_http_body(param_map, sizeof(param_map) / sizeof(param_dispatch_t),
		"[{"
			"\"x\":\"number\","
			"\"y\":\"number\","
			"\"width\":\"number\","
			"\"height\":\"number\","
			"\"confidence\":\"number\""
		"}]");
	return context;
}

void uri_scd_detect_objects_destroy(void* context)
{
	scd_context_t* scd_context = (scd_context_t*)context;
	ccv_scd_classifier_cascade_free(scd_context->face);
	free(scd_context->desc.data);
	free(scd_context);
}

int uri_scd_detect_objects_intro(const void* context, const void* parsed, ebb_buf* buf)
{
	scd_context_t* scd_context = (scd_context_t*)context;
	buf->data = scd_context->desc.data;
	buf->len = scd_context->desc.len;
	return 0;
}

int uri_scd_detect_objects(const void* context, const void* parsed, ebb_buf* buf)
{
	if (!parsed)
		return -1;
	scd_param_parser_t* parser = (scd_param_parser_t*)parsed;
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
	ccv_dense_matrix_t* resize = 0;
	if (parser->params.max_dimension > 0 && (image->rows > parser->params.max_dimension || image->cols > parser->params.max_dimension))
	{
		ccv_resample(image, &resize, 0, ccv_min(parser->params.max_dimension, (int)(image->rows * (float)parser->params.max_dimension / image->cols + 0.5)), ccv_min(parser->params.max_dimension, (int)(image->cols * (float)parser->params.max_dimension / image->rows + 0.5)), CCV_INTER_AREA);
		ccv_matrix_free(image);
	} else
		resize = image;
	ccv_array_t* seq = ccv_scd_detect_objects(resize, &parser->cascade, 1, parser->params.params);
	float width = resize->cols, height = resize->rows;
	ccv_matrix_free(resize);
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
			snprintf(cell, 128, "{\"x\":%f,\"y\":%f,\"width\":%f,\"height\":%f,\"confidence\":%f}", comp->rect.x / width, comp->rect.y / height, comp->rect.width / width, comp->rect.height / height, comp->classification.confidence);
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
	} else {
		buf->data = (void*)ebb_http_empty_array;
		buf->len = sizeof(ebb_http_empty_array);
		buf->on_release = 0;
	}
	ccv_array_free(seq);
	free(parser);
	return 0;
}
