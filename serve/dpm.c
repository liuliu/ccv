#include "uri.h"
#include "ccv.h"
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>

typedef struct {
	ccv_dpm_mixture_model_t* pedestrian;
	ccv_dpm_mixture_model_t* car;
} dpm_context_t;

typedef enum {
	s_dpm_start,
	s_dpm_skip,
	s_dpm_name_interval,
	s_dpm_name_min_neighbors,
	s_dpm_name_threshold,
	s_dpm_name_model,
	s_dpm_name_source,
} dpm_param_parse_state_t;

typedef struct {
	dpm_context_t* context;
	dpm_param_parse_state_t state;
	form_data_parser_t form_data_parser;
	ccv_dpm_param_t params;
	int cursor;
	char name[16];
	ccv_dpm_mixture_model_t* mixture_model;
	ebb_buf source;
	union {
		numeric_parser_t numeric_parser;
		string_parser_t string_parser;
	};
} dpm_param_parser_t;

static void on_form_data_name(void* context, const char* buf, size_t len)
{
	dpm_param_parser_t* parser = (dpm_param_parser_t*)context;
	if (len + parser->cursor > 15)
		return;
	memcpy(parser->name + parser->cursor, buf, len);
	parser->cursor += len;
}

static void uri_dpm_param_parser_init(dpm_param_parser_t* parser)
{
	form_data_parser_init(&parser->form_data_parser, parser);
	parser->form_data_parser.on_name = on_form_data_name;
	parser->params = ccv_dpm_default_params;
	parser->state = s_dpm_start;
	parser->cursor = 0;
	parser->source.data = 0;
	parser->source.len = 0;
	parser->source.written = 0;
	memset(parser->name, 0, sizeof(parser->name));
}

static void uri_dpm_param_parser_terminate(dpm_param_parser_t* parser)
{
	switch (parser->state)
	{
		case s_dpm_name_interval:
			parser->params.interval = (int)(parser->numeric_parser.result + 0.5);
			break;
		case s_dpm_name_min_neighbors:
			parser->params.min_neighbors = (int)(parser->numeric_parser.result + 0.5);
			break;
		case s_dpm_name_threshold:
			parser->params.threshold = parser->numeric_parser.result;
			break;
		case s_dpm_name_model:
			if (parser->string_parser.state == s_string_start)
			{
				if (strcmp(parser->string_parser.string, "pedestrian") == 0)
					parser->mixture_model = parser->context->pedestrian;
				else if (strcmp(parser->string_parser.string, "car") == 0)
					parser->mixture_model = parser->context->car;
			}
			break;
		default:
			break;
	}
	if (parser->state != s_dpm_start)
	{
		parser->state = s_dpm_start;
		memset(parser->name, 0, sizeof(parser->name));
		parser->cursor = 0;
	}
}

void* uri_dpm_detect_objects_parse(const void* context, void* parsed, const char* buf, size_t len, uri_parse_state_t state, int header_index)
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
			break;
		case URI_PARSE_TERMINATE:
			if (parser->state != s_dpm_start)
				uri_dpm_param_parser_terminate(parser); // collect result
			break;
		case URI_MULTIPART_HEADER_FIELD:
			if (parser->state != s_dpm_start)
				uri_dpm_param_parser_terminate(parser); // collect previous result
			form_data_parser_execute(&parser->form_data_parser, buf, len, header_index);
			break;
		case URI_MULTIPART_HEADER_VALUE:
			if (parser->state != s_dpm_start)
				uri_dpm_param_parser_terminate(parser); // collect previous result
			form_data_parser_execute(&parser->form_data_parser, buf, len, header_index);
			break;
		case URI_MULTIPART_DATA:
			if (parser->state == s_dpm_start)
			{
				// need to use name to get the correct state
				if (strcmp(parser->name, "interval") == 0)
				{
					parser->state = s_dpm_name_interval;
					numeric_parser_init(&parser->numeric_parser);
				} else if (strcmp(parser->name, "min_neighbors") == 0) {
					parser->state = s_dpm_name_min_neighbors;
					numeric_parser_init(&parser->numeric_parser);
				} else if (strcmp(parser->name, "threshold") == 0) {
					parser->state = s_dpm_name_threshold;
					numeric_parser_init(&parser->numeric_parser);
				} else if (strcmp(parser->name, "model") == 0) {
					parser->state = s_dpm_name_model;
					string_parser_init(&parser->string_parser);
				} else if (strcmp(parser->name, "source") == 0) {
					parser->state = s_dpm_name_source;
				} else
					parser->state = s_dpm_skip;
			}
			switch (parser->state)
			{
				default:
					break;
				case s_dpm_name_interval:
				case s_dpm_name_min_neighbors:
				case s_dpm_name_threshold:
					numeric_parser_execute(&parser->numeric_parser, buf, len);
					break;
				case s_dpm_name_model:
					string_parser_execute(&parser->string_parser, buf, len);
					if (parser->string_parser.state == s_string_overflow)
						parser->state = s_dpm_skip;
					break;
				case s_dpm_name_source:
					if (parser->source.len == 0)
					{
						parser->source.len = (len * 3 + 1) / 2;
						parser->source.data = (unsigned char*)malloc(parser->source.len);
					} else if (parser->source.written + len > parser->source.len) {
						parser->source.len = ((parser->source.len + len) * 3 + 1) / 2;
						parser->source.data = (unsigned char*)realloc(parser->source.data, parser->source.len);
					}
					memcpy(parser->source.data + parser->source.written, buf, len);
					parser->source.written += len;
					break;
			}
			break;
	}
	return parser;
}

void* uri_dpm_detect_objects_init(void)
{
	dpm_context_t* context = (dpm_context_t*)malloc(sizeof(dpm_context_t));
	context->pedestrian = ccv_load_dpm_mixture_model("../samples/pedestrian.m");
	context->car = ccv_load_dpm_mixture_model("../samples/car.m");
	return context;
}

int uri_dpm_detect_objects_intro(const void* context, const void* parsed, ebb_buf* buf)
{
	/*
	const static char dpm_desc[] = 
		"HTTP/1.1 200 OK\r\nCache-Control: no-cache\r\nAccept: \r\nContent-Type: text/html\r\nContent-Length: 163\r\n\r\n"
		"<html><body><form enctype='multipart/form-data' method='post'><input name='model' value='pedestrian'><input type='file' name='source'><input type='submit'></form>\n";
	*/
	const static char dpm_desc[] =
		"HTTP/1.1 200 OK\r\nCache-Control: no-cache\r\nAccept: \r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: 218\r\n\r\n"
		"{"
			"\"request\":{"
				"\"model\":\"\","
				"\"interval\":\"\","
				"\"min_neighbors\":\"\","
				"\"threshold\":\"\","
				"\"source\":\"\""
			"},"
			"\"response\":[{"
				"\"x\":\"\","
				"\"y\":\"\","
				"\"width\":\"\","
				"\"height\":\"\","
				"\"confidence\":\"\","
				"\"parts\":[{"
					"\"x\":\"\","
					"\"y\":\"\","
					"\"width\":\"\","
					"\"height\":\"\","
					"\"confidence\":\"\""
				"}]"
			"}]"
		"}\n";
	buf->data = (void*)dpm_desc;
	buf->len = sizeof(dpm_desc);
	return 0;
}

int uri_dpm_detect_objects(const void* context, const void* parsed, ebb_buf* buf)
{
	dpm_param_parser_t* parser = (dpm_param_parser_t*)parsed;
	if (parser->state != s_dpm_start)
		uri_dpm_param_parser_terminate(parser);
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
	ccv_array_t* seq = ccv_dpm_detect_objects(image, &parser->mixture_model, 1, parser->params);
	ccv_matrix_free(image);
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
			snprintf(cell, 128, "{\"x\":%d,\"y\":%d,\"width\":%d,\"height\":%d,\"confidence\":%f,\"parts\":[", comp->rect.x, comp->rect.y, comp->rect.width, comp->rect.height, comp->confidence);
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
				snprintf(cell, 128, "{\"x\":%d,\"y\":%d,\"width\":%d,\"height\":%d,\"confidence\":%f}", comp->part[j].rect.x, comp->part[j].rect.y, comp->part[j].rect.width, comp->part[j].rect.height, comp->part[j].confidence);
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
