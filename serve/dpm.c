#include "uri.h"
#include "ccv.h"
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>

static void uri_dpm_on_model_string(void* context, char* string);
static void uri_dpm_on_source_blob(void* context, ebb_buf data);

typedef struct {
	ccv_dpm_param_t params;
	int max_dimension;
} ccv_dpm_uri_param_t;

static const param_dispatch_t param_map[] = {
	{
		.property = "interval",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_dpm_uri_param_t, params) + offsetof(ccv_dpm_param_t, interval),
	},
	{
		.property = "max_dimension",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_dpm_uri_param_t, max_dimension),
	},
	{
		.property = "min_neighbors",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_dpm_uri_param_t, params) + offsetof(ccv_dpm_param_t, min_neighbors),
	},
	{
		.property = "model",
		.type = PARAM_TYPE_STRING,
		.on_string = uri_dpm_on_model_string,
		.offset = 0,
	},
	{
		.property = "source",
		.type = PARAM_TYPE_BODY,
		.on_blob = uri_dpm_on_source_blob,
		.offset = 0,
	},
	{
		.property = "threshold",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_dpm_uri_param_t, params) + offsetof(ccv_dpm_param_t, threshold),
	},
};

typedef struct {
	ebb_buf desc;
	ccv_dpm_mixture_model_t* pedestrian;
	ccv_dpm_mixture_model_t* car;
} dpm_context_t;

typedef struct {
	param_parser_t param_parser;
	dpm_context_t* context;
	ccv_dpm_uri_param_t params;
	ccv_dpm_mixture_model_t* mixture_model;
	ebb_buf source;
} dpm_param_parser_t;

static void uri_dpm_param_parser_init(dpm_param_parser_t* parser)
{
	param_parser_init(&parser->param_parser, param_map, sizeof(param_map) / sizeof(param_dispatch_t), &parser->params, parser);
	parser->params.params = ccv_dpm_default_params;
	parser->params.max_dimension = 0;
	parser->mixture_model = 0;
	parser->source.data = 0;
}

static void uri_dpm_on_model_string(void* context, char* string)
{
	dpm_param_parser_t* parser = (dpm_param_parser_t*)context;
	if (strcmp(string, "pedestrian") == 0)
		parser->mixture_model = parser->context->pedestrian;
	else if (strcmp(string, "car") == 0)
		parser->mixture_model = parser->context->car;
}

static void uri_dpm_on_source_blob(void* context, ebb_buf data)
{
	dpm_param_parser_t* parser = (dpm_param_parser_t*)context;
	parser->source = data;
}

void* uri_dpm_detect_objects_parse(const void* context, void* parsed, int resource_id, const char* buf, size_t len, uri_parse_state_t state, int header_index)
{
	dpm_param_parser_t* parser;
	if (parsed)
		parser = (dpm_param_parser_t*)parsed;
	else {
		parser = (dpm_param_parser_t*)malloc(sizeof(dpm_param_parser_t));
		uri_dpm_param_parser_init(parser);
		parser->context = (dpm_context_t*)context;
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

void* uri_dpm_detect_objects_init(void)
{
	dpm_context_t* context = (dpm_context_t*)malloc(sizeof(dpm_context_t));
	context->pedestrian = ccv_dpm_read_mixture_model("../samples/pedestrian.m");
	context->car = ccv_dpm_read_mixture_model("../samples/car.m");
	assert(context->pedestrian && context->car);
	assert(param_parser_map_alphabet(param_map, sizeof(param_map) / sizeof(param_dispatch_t)) == 0);
	context->desc = param_parser_map_http_body(param_map, sizeof(param_map) / sizeof(param_dispatch_t),
		"[{"
			"\"x\":\"number\","
			"\"y\":\"number\","
			"\"width\":\"number\","
			"\"height\":\"number\","
			"\"confidence\":\"number\","
			"\"parts\":[{"
				"\"x\":\"number\","
				"\"y\":\"number\","
				"\"width\":\"number\","
				"\"height\":\"number\","
				"\"confidence\":\"number\""
			"}]"
		"}]");
	return context;
}

void uri_dpm_detect_objects_destroy(void* context)
{
	dpm_context_t* dpm_context = (dpm_context_t*)context;
	ccv_dpm_mixture_model_free(dpm_context->pedestrian);
	ccv_dpm_mixture_model_free(dpm_context->car);
	free(dpm_context->desc.data);
	free(dpm_context);
}

int uri_dpm_detect_objects_intro(const void* context, const void* parsed, ebb_buf* buf)
{
	dpm_context_t* dpm_context = (dpm_context_t*)context;
	buf->data = dpm_context->desc.data;
	buf->len = dpm_context->desc.len;
	return 0;
}

int uri_dpm_detect_objects(const void* context, const void* parsed, ebb_buf* buf)
{
	if (!parsed)
		return -1;
	dpm_param_parser_t* parser = (dpm_param_parser_t*)parsed;
	param_parser_terminate(&parser->param_parser);
	if (parser->source.data == 0)
	{
		free(parser);
		return -1;
	}
	if (parser->mixture_model == 0)
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
	ccv_array_t* seq = ccv_dpm_detect_objects(resize, &parser->mixture_model, 1, parser->params.params);
	float width = resize->cols, height = resize->rows;
	ccv_matrix_free(resize);
	if (seq  == 0)
	{
		free(parser);
		return -1;
	}
	if (seq->rnum > 0)
	{
		int i, j;
		buf->len = 192 + seq->rnum * 131 + 2;
		char* data = (char*)malloc(buf->len);
		data[0] = '[';
		buf->written = 1;
		for (i = 0; i < seq->rnum; i++)
		{
			char cell[128];
			ccv_root_comp_t* comp = (ccv_root_comp_t*)ccv_array_get(seq, i);
			snprintf(cell, 128, "{\"x\":%f,\"y\":%f,\"width\":%f,\"height\":%f,\"confidence\":%f,\"parts\":[", comp->rect.x / width, comp->rect.y / height, comp->rect.width / width, comp->rect.height / height, comp->classification.confidence);
			size_t len = strnlen(cell, 128);
			while (buf->written + len >= buf->len)
			{
				buf->len = (buf->len * 3 + 1) / 2;
				data = (char*)realloc(data, buf->len);
			}
			memcpy(data + buf->written, cell, len);
			buf->written += len;
			for (j = 0; j < comp->pnum; j++)
			{
				snprintf(cell, 128, "{\"x\":%f,\"y\":%f,\"width\":%f,\"height\":%f,\"confidence\":%f}", comp->part[j].rect.x / width, comp->part[j].rect.y / height, comp->part[j].rect.width / width, comp->part[j].rect.height / height, comp->part[j].classification.confidence);
				len = strnlen(cell, 128);
				while (buf->written + len + 3 >= buf->len)
				{
					buf->len = (buf->len * 3 + 1) / 2;
					data = (char*)realloc(data, buf->len);
				}
				memcpy(data + buf->written, cell, len);
				buf->written += len + 1;
				data[buf->written - 1] = (j == comp->pnum - 1) ? ']' : ',';
			}
			buf->written += 2;
			data[buf->written - 2] = '}';
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
