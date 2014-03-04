#include "uri.h"
#include "ccv.h"
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <dispatch/dispatch.h>

static void uri_tld_on_previous_blob(void* context, ebb_buf data);
static void uri_tld_on_source_blob(void* context, ebb_buf data);

typedef struct {
	int tld;
	ccv_tld_param_t params;
	ccv_rect_t box;
} ccv_tld_uri_param_t;

static const param_dispatch_t param_map[] = {
	{
		.property = "bad_patches",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_tld_uri_param_t, params) + offsetof(ccv_tld_param_t, bad_patches),
	},
	{
		.property = "exclude_overlap",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_tld_uri_param_t, params) + offsetof(ccv_tld_param_t, exclude_overlap),
	},
	{
		.property = "features",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_tld_uri_param_t, params) + offsetof(ccv_tld_param_t, features),
	},
	{
		.property = "height",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_tld_uri_param_t, box) + offsetof(ccv_rect_t, height),
	},
	{
		.property = "include_overlap",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_tld_uri_param_t, params) + offsetof(ccv_tld_param_t, include_overlap),
	},
	{
		.property = "interval",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_tld_uri_param_t, params) + offsetof(ccv_tld_param_t, interval),
	},
	{
		.property = "level",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_tld_uri_param_t, params) + offsetof(ccv_tld_param_t, level),
	},
	{
		.property = "min_eigen",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_tld_uri_param_t, params) + offsetof(ccv_tld_param_t, min_eigen),
	},
	{
		.property = "min_forward_backward_error",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_tld_uri_param_t, params) + offsetof(ccv_tld_param_t, min_forward_backward_error),
	},
	{
		.property = "min_win",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_tld_uri_param_t, params) + offsetof(ccv_tld_param_t, min_win),
	},
	{
		.property = "new_deform",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_tld_uri_param_t, params) + offsetof(ccv_tld_param_t, new_deform),
	},
	{
		.property = "new_deform_angle",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_tld_uri_param_t, params) + offsetof(ccv_tld_param_t, new_deform_angle),
	},
	{
		.property = "new_deform_scale",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_tld_uri_param_t, params) + offsetof(ccv_tld_param_t, new_deform_scale),
	},
	{
		.property = "new_deform_shift",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_tld_uri_param_t, params) + offsetof(ccv_tld_param_t, new_deform_shift),
	},
	{
		.property = "nnc_beyond",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_tld_uri_param_t, params) + offsetof(ccv_tld_param_t, nnc_beyond),
	},
	{
		.property = "nnc_collect",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_tld_uri_param_t, params) + offsetof(ccv_tld_param_t, nnc_collect),
	},
	{
		.property = "nnc_same",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_tld_uri_param_t, params) + offsetof(ccv_tld_param_t, nnc_same),
	},
	{
		.property = "nnc_thres",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_tld_uri_param_t, params) + offsetof(ccv_tld_param_t, nnc_thres),
	},
	{
		.property = "nnc_verify",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_tld_uri_param_t, params) + offsetof(ccv_tld_param_t, nnc_verify),
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
		.offset = offsetof(ccv_tld_uri_param_t, params) + offsetof(ccv_tld_param_t, rotation),
	},
	{
		.property = "shift",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_tld_uri_param_t, params) + offsetof(ccv_tld_param_t, shift),
	},
	{
		.property = "source",
		.type = PARAM_TYPE_BODY,
		.on_blob = uri_tld_on_source_blob,
		.offset = 0,
	},
	{
		.property = "structs",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_tld_uri_param_t, params) + offsetof(ccv_tld_param_t, structs),
	},
	{
		.property = "tld",
		.type = PARAM_TYPE_ID,
		.offset = offsetof(ccv_tld_uri_param_t, tld),
	},
	{
		.property = "top_n",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_tld_uri_param_t, params) + offsetof(ccv_tld_param_t, top_n),
	},
	{
		.property = "track_deform",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_tld_uri_param_t, params) + offsetof(ccv_tld_param_t, track_deform),
	},
	{
		.property = "track_deform_angle",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_tld_uri_param_t, params) + offsetof(ccv_tld_param_t, track_deform_angle),
	},
	{
		.property = "track_deform_scale",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_tld_uri_param_t, params) + offsetof(ccv_tld_param_t, track_deform_scale),
	},
	{
		.property = "track_deform_shift",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_tld_uri_param_t, params) + offsetof(ccv_tld_param_t, track_deform_shift),
	},
	{
		.property = "validate_set",
		.type = PARAM_TYPE_FLOAT,
		.offset = offsetof(ccv_tld_uri_param_t, params) + offsetof(ccv_tld_param_t, validate_set),
	},
	{
		.property = "width",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_tld_uri_param_t, box) + offsetof(ccv_rect_t, width),
	},
	{
		.property = "win_size",
		.type = PARAM_TYPE_SIZE,
		.offset = offsetof(ccv_tld_uri_param_t, params) + offsetof(ccv_tld_param_t, win_size),
	},
	{
		.property = "x",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_tld_uri_param_t, box) + offsetof(ccv_rect_t, x),
	},
	{
		.property = "y",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_tld_uri_param_t, box) + offsetof(ccv_rect_t, y),
	},
};

typedef struct {
	ebb_buf desc;
	ccv_array_t* tlds;
	dispatch_semaphore_t semaphore;
} tld_context_t;

typedef struct {
	dispatch_semaphore_t semaphore;
	ccv_tld_t* tld;
} ccv_thread_safe_tld_t;

typedef struct {
	param_parser_t param_parser;
	ccv_tld_uri_param_t uri_params;
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
	context->tlds = ccv_array_new(sizeof(ccv_tld_t*), 64, 0);
	context->semaphore = dispatch_semaphore_create(1);
	context->desc = param_parser_map_http_body(param_map, sizeof(param_map) / sizeof(param_dispatch_t),
		"{"
			"\"tld\":\"integer\","
			"\"box\":{"
				"\"x\":\"integer\","
				"\"y\":\"integer\","
				"\"width\":\"integer\","
				"\"height\":\"integer\","
				"\"confidence\":\"number\""
			"},"
			"\"info\":{"
				"\"perform_track\":\"boolean\","
				"\"perform_learn\":\"boolean\","
				"\"track_success\":\"boolean\","
				"\"ferns_detects\":\"integer\","
				"\"nnc_detects\":\"integer\","
				"\"clustered_detects\":\"integer\","
				"\"confident_matches\":\"integer\","
				"\"close_matches\":\"integer\""
		"}}");
	return context;
}

void uri_tld_track_object_destroy(void* context)
{
	tld_context_t* tld_context = (tld_context_t*)context;
	ccv_array_free(tld_context->tlds);
	dispatch_release(tld_context->semaphore);
	free(tld_context->desc.data);
	free(tld_context);
}

static void uri_tld_param_parser_init(tld_param_parser_t* parser)
{
	param_parser_init(&parser->param_parser, param_map, sizeof(param_map) / sizeof(param_dispatch_t), &parser->uri_params, parser);
	parser->uri_params.params = ccv_tld_default_params;
	parser->uri_params.box = ccv_rect(0, 0, 0, 0);
	parser->uri_params.tld = -1;
	parser->previous.data = 0;
	parser->source.data = 0;
}

void* uri_tld_track_object_parse(const void* context, void* parsed, int resource_id, const char* buf, size_t len, uri_parse_state_t state, int header_index)
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
			param_parser_execute(&parser->param_parser, resource_id, buf, len, state, header_index);
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
	if (!parsed)
		return -1;
	tld_context_t* tld_context = (tld_context_t*)context;
	tld_param_parser_t* parser = (tld_param_parser_t*)parsed;
	param_parser_terminate(&parser->param_parser);
	if (parser->source.data == 0)
	{
		if (parser->previous.data)
			free(parser->previous.data);
		free(parser);
		return -1;
	}
	ccv_dense_matrix_t* source = 0;
	if (((char*)parser->source.data)[0] == '@' && parser->source.len == 11)
	{
		// TODO: find source in cache
	} else
		ccv_read(parser->source.data, &source, CCV_IO_ANY_STREAM | CCV_IO_GRAY, parser->source.written);
	free(parser->source.data);
	if (source == 0)
	{
		if (parser->previous.data)
			free(parser->previous.data);
		free(parser);
		return -1;
	}
	int i;
	ccv_dense_matrix_t* previous = 0;
	if (parser->previous.data == 0 || parser->previous.written == 0)
	{
		if (parser->previous.data)
			free(parser->previous.data);
		// to initialize
		if (ccv_rect_is_zero(parser->uri_params.box))
		{
			ccv_matrix_free(source);
			free(parser);
			return -1;
		}
		ccv_thread_safe_tld_t thread_safe_tld = {
			.tld = 0
		};
		int tld_ident = -1;
		dispatch_semaphore_wait(tld_context->semaphore, DISPATCH_TIME_FOREVER);
		for (i = 0; i < tld_context->tlds->rnum; i++)
		{
			ccv_thread_safe_tld_t* tld = (ccv_thread_safe_tld_t*)ccv_array_get(tld_context->tlds, i);
			if (tld->tld == 0)
			{
				tld_ident = i;
				thread_safe_tld = *tld;
				break;
			}
		}
		if (thread_safe_tld.tld == 0)
		{
			thread_safe_tld.tld = (ccv_tld_t*)1;
			thread_safe_tld.semaphore = dispatch_semaphore_create(1);
			tld_ident = tld_context->tlds->rnum;
			ccv_array_push(tld_context->tlds, &thread_safe_tld);
		}
		dispatch_semaphore_signal(tld_context->semaphore);
		dispatch_semaphore_wait(thread_safe_tld.semaphore, DISPATCH_TIME_FOREVER);
		thread_safe_tld.tld = ccv_tld_new(source, parser->uri_params.box, parser->uri_params.params);
		dispatch_semaphore_signal(thread_safe_tld.semaphore);
		ccv_matrix_free(source);
		dispatch_semaphore_wait(tld_context->semaphore, DISPATCH_TIME_FOREVER);
		*(ccv_thread_safe_tld_t*)ccv_array_get(tld_context->tlds, tld_ident) = thread_safe_tld; // set the real tld pointer
		dispatch_semaphore_signal(tld_context->semaphore);
		parser->uri_params.tld = tld_ident;
		// print out box and tld
		char cell[128];
		snprintf(cell, 128, "{\"tld\":%d,\"box\":{\"x\":%d,\"y\":%d,\"width\":%d,\"height\":%d,\"confidence\":1}}\n", tld_ident, parser->uri_params.box.x, parser->uri_params.box.y, parser->uri_params.box.width, parser->uri_params.box.height);
		size_t len = strlen(cell);
		char* data = (char*)malloc(192 + len);;
		static const char ebb_http_tld_created[] = "HTTP/1.1 201 Created\r\nCache-Control: no-cache\r\nContent-Type: application/json; charset=utf-8\r\nLocation: /tld/track.object/%d\r\nContent-Length: %zd\r\n\r\n";
		snprintf(data, 192, ebb_http_tld_created, tld_ident, len);
		size_t data_len = strlen(data);
		memcpy(data + data_len, cell, len);
		buf->data = data;
		buf->len = data_len + len;
		buf->on_release = uri_ebb_buf_free;
	} else {
		if (parser->uri_params.tld < 0)
		{
			free(parser->previous.data);
			free(parser);
			return -1;
		}
		if (((char*)parser->previous.data)[0] == '@' && parser->previous.written == 11)
		{
			// TODO: find previous in cache
		} else
			ccv_read(parser->previous.data, &previous, CCV_IO_ANY_STREAM | CCV_IO_GRAY, parser->previous.written);
		free(parser->previous.data);
		if (previous == 0)
		{
			free(parser);
			return -1;
		}
		ccv_thread_safe_tld_t thread_safe_tld = {
			.tld = 0,
		};
		dispatch_semaphore_wait(tld_context->semaphore, DISPATCH_TIME_FOREVER);
		if (parser->uri_params.tld < tld_context->tlds->rnum)
			thread_safe_tld = *(ccv_thread_safe_tld_t*)ccv_array_get(tld_context->tlds, parser->uri_params.tld);
		dispatch_semaphore_signal(tld_context->semaphore);
		if (thread_safe_tld.tld == 0)
		{
			ccv_matrix_free(previous);
			ccv_matrix_free(source);
			free(parser);
			return -1;
		}
		dispatch_semaphore_wait(thread_safe_tld.semaphore, DISPATCH_TIME_FOREVER);
		if (thread_safe_tld.tld->frame_signature != previous->sig)
		{
			dispatch_semaphore_signal(thread_safe_tld.semaphore);
			ccv_matrix_free(previous);
			ccv_matrix_free(source);
			free(parser);
			return -1;
		}
		ccv_tld_info_t info;
		ccv_comp_t box = ccv_tld_track_object(thread_safe_tld.tld, previous, source, &info);
		dispatch_semaphore_signal(thread_safe_tld.semaphore);
		ccv_matrix_free(previous);
		ccv_matrix_free(source);
		char cell[320];
		snprintf(cell, 320,
			"{\"tld\":%d,"
			"\"box\":{"
				"\"x\":%d,"
				"\"y\":%d,"
				"\"width\":%d,"
				"\"height\":%d,"
				"\"confidence\":%f"
			"},"
			"\"info\":{"
				"\"perform_track\":%s,"
				"\"perform_learn\":%s,"
				"\"track_success\":%s,"
				"\"ferns_detects\":%d,"
				"\"nnc_detects\":%d,"
				"\"clustered_detects\":%d,"
				"\"confident_matches\":%d,"
				"\"close_matches\":%d"
			"}}\n",
			parser->uri_params.tld,
			box.rect.x, box.rect.y, box.rect.width, box.rect.height, box.classification.confidence,
			info.perform_track ? "true" : "false",
			info.perform_learn ? "true" : "false",
			info.track_success ? "true" : "false",
			info.ferns_detects,
			info.nnc_detects,
			info.clustered_detects,
			info.confident_matches,
			info.close_matches
		);
		size_t len = strlen(cell);
		char* data = (char*)malloc(192 + len);;
		snprintf(data, 192, ebb_http_header, len);
		size_t data_len = strlen(data);
		memcpy(data + data_len, cell, len);
		buf->data = data;
		buf->len = data_len + len;
		buf->on_release = uri_ebb_buf_free;
	}
	free(parser);
	return 0;
}

int uri_tld_track_object_free(const void* context, const void* parsed, ebb_buf* buf)
{
	if (!parsed)
		return -1;
	tld_context_t* tld_context = (tld_context_t*)context;
	tld_param_parser_t* parser = (tld_param_parser_t*)parsed;
	param_parser_terminate(&parser->param_parser);
	if (parser->source.data)
		free(parser->source.data);
	if (parser->previous.data)
		free(parser->previous.data);
	if (parser->uri_params.tld < 0)
	{
		free(parser);
		return -1;
	}
	ccv_thread_safe_tld_t thread_safe_tld = {
		.tld = 0,
	};
	dispatch_semaphore_wait(tld_context->semaphore, DISPATCH_TIME_FOREVER);
	if (parser->uri_params.tld < tld_context->tlds->rnum)
	{
		thread_safe_tld = *(ccv_thread_safe_tld_t*)ccv_array_get(tld_context->tlds, parser->uri_params.tld);
		ccv_thread_safe_tld_t dummy_tld = thread_safe_tld;
		dummy_tld.tld = 0;
		*(ccv_thread_safe_tld_t*)ccv_array_get(tld_context->tlds, parser->uri_params.tld) = dummy_tld;
	}
	dispatch_semaphore_signal(tld_context->semaphore);
	if (!thread_safe_tld.tld)
	{
		free(parser);
		return -1;
	} else {
		dispatch_semaphore_wait(thread_safe_tld.semaphore, DISPATCH_TIME_FOREVER);
		ccv_tld_free(thread_safe_tld.tld);
		dispatch_semaphore_signal(thread_safe_tld.semaphore);
		buf->data = (void*)ebb_http_ok_true;
		buf->len = sizeof(ebb_http_ok_true) - 1;
		free(parser);
	}
	return 0;
}
