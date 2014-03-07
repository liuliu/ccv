#include "uri.h"
#include "ccv.h"
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>

static void uri_sift_on_source_blob(void* context, ebb_buf data);

static const param_dispatch_t param_map[] = {
	{
		.property = "edge_threshold",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_sift_param_t, edge_threshold),
	},
	{
		.property = "nlevels",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_sift_param_t, nlevels),
	},
	{
		.property = "noctaves",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_sift_param_t, noctaves),
	},
	{
		.property = "norm_threshold",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_sift_param_t, norm_threshold),
	},
	{
		.property = "peak_threshold",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_sift_param_t, peak_threshold),
	},
	{
		.property = "source",
		.type = PARAM_TYPE_BODY,
		.on_blob = uri_sift_on_source_blob,
		.offset = 0,
	},
	{
		.property = "up2x",
		.type = PARAM_TYPE_BOOL,
		.offset = offsetof(ccv_sift_param_t, up2x),
	},
};

typedef struct {
	ebb_buf desc;
} sift_context_t;

typedef struct {
	param_parser_t param_parser;
	sift_context_t* context;
	ccv_sift_param_t params;
	ebb_buf source;
} sift_param_parser_t;

static void uri_sift_on_source_blob(void* context, ebb_buf data)
{
	sift_param_parser_t* parser = (sift_param_parser_t*)context;
	parser->source = data;
}

void* uri_sift_init(void)
{
	sift_context_t* context = (sift_context_t*)malloc(sizeof(sift_context_t));
	assert(param_parser_map_alphabet(param_map, sizeof(param_map) / sizeof(param_dispatch_t)) == 0);
	context->desc = param_parser_map_http_body(param_map, sizeof(param_map) / sizeof(param_dispatch_t),
		"[{"
			"\"x\":\"number\","
			"\"y\":\"number\","
			"\"octave\":\"integer\","
			"\"level\":\"integer\","
			"\"scale\":\"number\","
			"\"angle\":\"number\","
			"\"descriptor\":[\"number\"]"
		"}]");
	return context;
}

static void uri_sift_param_parser_init(sift_param_parser_t* parser)
{
	param_parser_init(&parser->param_parser, param_map, sizeof(param_map) / sizeof(param_dispatch_t), &parser->params, parser);
	parser->params = ccv_sift_default_params;
	parser->source.data = 0;
}

void uri_sift_destroy(void* context)
{
	sift_context_t* sift_context = (sift_context_t*)context;
	free(sift_context->desc.data);
	free(sift_context);
}

void* uri_sift_parse(const void* context, void* parsed, int resource_id, const char* buf, size_t len, uri_parse_state_t state, int header_index)
{
	sift_param_parser_t* parser;
	if (parsed)
		parser = (sift_param_parser_t*)parsed;
	else {
		parser = (sift_param_parser_t*)malloc(sizeof(sift_param_parser_t));
		uri_sift_param_parser_init(parser);
		parser->context = (sift_context_t*)context;
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

int uri_sift_intro(const void* context, const void* parsed, ebb_buf* buf)
{
	sift_context_t* sift_context = (sift_context_t*)context;
	buf->data = sift_context->desc.data;
	buf->len = sift_context->desc.len;
	return 0;
}

int uri_sift(const void* context, const void* parsed, ebb_buf* buf)
{
	if (!parsed)
		return -1;
	sift_param_parser_t* parser = (sift_param_parser_t*)parsed;
	param_parser_terminate(&parser->param_parser);
	if (parser->source.data == 0)
	{
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
	ccv_array_t* keypoints = 0;
	ccv_dense_matrix_t* desc = 0;
	ccv_sift(image, &keypoints, &desc, 0, parser->params);
	ccv_matrix_free(image);
	int i, j;
	if (keypoints->rnum > 0)
	{
		buf->len = 192 + keypoints->rnum * 21 + 2;
		char* data = (char*)malloc(buf->len);
		data[0] = '[';
		buf->written = 1;
		float* f32 = desc->data.f32;
		for (i = 0; i < keypoints->rnum; i++)
		{
			ccv_keypoint_t* keypoint = (ccv_keypoint_t*)ccv_array_get(keypoints, i);
			char cell[128];
			snprintf(cell, 128, "{\"x\":%f,\"y\":%f,\"octave\":%d,\"level\":%d,\"scale\":%f,\"angle\":%f,\"descriptor\":[", keypoint->x, keypoint->y, keypoint->octave, keypoint->level, keypoint->regular.scale, keypoint->regular.angle);
			size_t len = strnlen(cell, 128);
			while (buf->written + len >= buf->len)
			{
				buf->len = (buf->len * 3 + 1) / 2;
				data = (char*)realloc(data, buf->len);
			}
			memcpy(data + buf->written, cell, len);
			buf->written += len;
			for (j = 0; j < 128; j++)
			{
				snprintf(cell, 128, "%f", f32[j]);
				size_t len = strnlen(cell, 128);
				while (buf->written + len + 3 >= buf->len)
				{
					buf->len = (buf->len * 3 + 1) / 2;
					data = (char*)realloc(data, buf->len);
				}
				memcpy(data + buf->written, cell, len);
				buf->written += len + 1;
				data[buf->written - 1] = (j == 128 - 1) ? ']' : ',';
			}
			buf->written += 2;
			data[buf->written - 2] = '}';
			data[buf->written - 1] = (i == keypoints->rnum - 1) ? ']' : ',';
			f32 += 128;
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
	ccv_array_free(keypoints);
	ccv_matrix_free(desc);
	return 0;
}
