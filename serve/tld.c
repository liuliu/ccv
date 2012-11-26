#include "uri.h"
#include "ccv.h"
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>

static void uri_tld_on_previous_blob(void* context, ebb_buf data);
static void uri_tld_on_source_blob(void* context, ebb_buf data);

static const param_dispatch_t param_map[] = {
	{
		.property = "bad_patches",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_tld_param_t, bad_patches),
	},
	{
		.property = "exclude_overlap",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_tld_param_t, exclude_overlap),
	},
	{
		.property = "features",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_tld_param_t, features),
	},
	{
		.property = "include_overlap",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_tld_param_t, include_overlap),
	},
	{
		.property = "interval",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_tld_param_t, interval),
	},
	{
		.property = "level",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_tld_param_t, level),
	},
	{
		.property = "min_eigen",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_tld_param_t, min_eigen),
	},
	{
		.property = "min_forward_backward_error",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_tld_param_t, min_forward_backward_error),
	},
	{
		.property = "min_win",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_tld_param_t, min_win),
	},
	{
		.property = "new_deform",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_tld_param_t, new_deform),
	},
	{
		.property = "new_deform_angle",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_tld_param_t, new_deform_angle),
	},
	{
		.property = "new_deform_scale",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_tld_param_t, new_deform_scale),
	},
	{
		.property = "new_deform_shift",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_tld_param_t, new_deform_shift),
	},
	{
		.property = "nnc_beyond",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_tld_param_t, nnc_beyond),
	},
	{
		.property = "nnc_collect",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_tld_param_t, nnc_collect),
	},
	{
		.property = "nnc_same",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_tld_param_t, nnc_same),
	},
	{
		.property = "nnc_thres",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_tld_param_t, nnc_thres),
	},
	{
		.property = "nnc_verify",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_tld_param_t, nnc_verify),
	},
	{
		.property = "previous",
		.type = PARAM_TYPE_BLOB,
		.on_blob = uri_tld_on_previous_blob,
		.offset = 0,
	},
	{
		.property = "rotation",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_tld_param_t, rotation),
	},
	{
		.property = "shift",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_tld_param_t, shift),
	},
	{
		.property = "source",
		.type = PARAM_TYPE_BLOB,
		.on_blob = uri_tld_on_source_blob,
		.offset = 0,
	},
	{
		.property = "structs",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_tld_param_t, structs),
	},
	{
		.property = "top_n",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_tld_param_t, top_n),
	},
	{
		.property = "track_deform",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_tld_param_t, track_deform),
	},
	{
		.property = "track_deform_angle",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_tld_param_t, track_deform_angle),
	},
	{
		.property = "track_deform_scale",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_tld_param_t, track_deform_scale),
	},
	{
		.property = "track_deform_shift",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_tld_param_t, track_deform_shift),
	},
	{
		.property = "validate_set",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_tld_param_t, validate_set),
	},
	{
		.property = "win_size",
		.type = PARAM_TYPE_SIZE,
		.offset = offsetof(ccv_tld_param_t, win_size),
	},
};

typedef struct {
	ebb_buf desc;
	ccv_tld_t* tlds;
} tld_context_t;

typedef struct {
	param_parser_t param_parser;
	ccv_tld_param_t params;
	ebb_buf previous;
	ebb_buf source;
} tld_param_parser_t;

static void uri_tld_on_previous_blob(void* context, ebb_buf data)
{
	tld_param_parser_t* parser = (tld_param_parser_t*)context;
	parser->previous = data;
}

static void uri_tld_on_source_blob(void* context, ebb_buf data)
{
	tld_param_parser_t* parser = (tld_param_parser_t*)context;
	parser->source = data;
}

void* uri_tld_track_object_init(void)
{
	assert(param_parser_map_alphabet(param_map, sizeof(param_map) / sizeof(param_dispatch_t)) == 0);
	tld_context_t* context = (tld_context_t*)malloc(sizeof(tld_context_t));
	context->desc = param_parser_map_http_body(param_map, sizeof(param_map) / sizeof(param_dispatch_t),
		"{"
			"\"tld\":\"integer\","
			"\"source\":\"blob\","
			"\"x\":\"integer\","
			"\"y\":\"integer\","
			"\"width\":\"integer\","
			"\"height\":\"integer\","
			"\"confidence\":\"number\""
		"}");
	return context;
}

void uri_tld_track_object_destroy(void* context)
{
	tld_context_t* tld_context = (tld_context_t*)context;
	free(tld_context->desc.data);
	free(tld_context);
}

static void uri_tld_param_parser_init(tld_param_parser_t* parser)
{
	param_parser_init(&parser->param_parser, param_map, sizeof(param_map) / sizeof(param_dispatch_t), &parser->params, parser);
	parser->params = ccv_tld_default_params;
	parser->previous.data = 0;
	parser->source.data = 0;
}

void* uri_tld_track_object_parse(const void* context, void* parsed, const char* buf, size_t len, uri_parse_state_t state, int header_index)
{
	tld_param_parser_t* parser;
	if (parsed)
		parser = (tld_param_parser_t*)parsed;
	else {
		parser = (tld_param_parser_t*)malloc(sizeof(tld_param_parser_t));
		uri_tld_param_parser_init(parser);
	}
	switch (state)
	{
		case URI_QUERY_STRING:
		case URI_CONTENT_BODY:
		case URI_PARSE_TERMINATE:
		case URI_MULTIPART_HEADER_FIELD:
		case URI_MULTIPART_HEADER_VALUE:
		case URI_MULTIPART_DATA:
			param_parser_execute(&parser->param_parser, buf, len, state, header_index);
			break;
	}
	return parser;
}

int uri_tld_track_object_intro(const void* context, const void* parsed, ebb_buf* buf)
{
	tld_context_t* tld_context = (tld_context_t*)context;
	buf->data = tld_context->desc.data;
	buf->len = tld_context->desc.len;
	return 0;
}

int uri_tld_track_object(const void* context, const void* parsed, ebb_buf* buf)
{
	tld_param_parser_t* parser = (tld_param_parser_t*)parsed;
	if (parser->source.data == 0)
	{
		free(parser);
		return -1;
	}
	return -1;
}
