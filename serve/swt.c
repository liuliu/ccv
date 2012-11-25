#include "uri.h"
#include "ccv.h"
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>

typedef enum {
	s_swt_start,
	s_swt_skip,
	s_swt_name_interval,
	s_swt_name_min_neighbors,
	s_swt_name_scale_invariant,
	/* canny parameters */
	s_swt_name_size,
	s_swt_name_low_thresh,
	s_swt_name_high_thresh,
	s_swt_name_max_height,
	s_swt_name_min_height,
	s_swt_name_min_area,
	s_swt_name_letter_occlude_thresh,
	s_swt_name_aspect_ratio,
	s_swt_name_std_ratio,
	/* grouping parameters */
	s_swt_name_thickness_ratio,
	s_swt_name_height_ratio,
	s_swt_name_intensity_thresh,
	s_swt_name_distance_ratio,
	s_swt_name_intersect_ratio,
	s_swt_name_elongate_ratio,
	s_swt_name_letter_thresh,
	/* break textline into words */
	s_swt_name_breakdown,
	s_swt_name_breakdown_ratio,
	s_swt_name_source,
} swt_param_parse_state_t;

typedef struct {
	swt_param_parse_state_t state;
	form_data_parser_t form_data_parser;
	ccv_swt_param_t params;
	int cursor;
	char name[16];
	ebb_buf source;
	union {
		numeric_parser_t numeric_parser;
		bool_parser_t bool_parser;
		blob_parser_t blob_parser;
	};
} swt_param_parser_t;

static void on_form_data_name(void* context, const char* buf, size_t len)
{
	swt_param_parser_t* parser = (swt_param_parser_t*)context;
	if (len + parser->cursor > 15)
		return;
	memcpy(parser->name + parser->cursor, buf, len);
	parser->cursor += len;
}

static void uri_swt_param_parser_init(swt_param_parser_t* parser)
{
	form_data_parser_init(&parser->form_data_parser, parser);
	parser->form_data_parser.on_name = on_form_data_name;
	parser->params = ccv_swt_default_params;
	parser->state = s_swt_start;
	parser->cursor = 0;
	parser->source.data = 0;
	memset(parser->name, 0, sizeof(parser->name));
}

static void uri_swt_param_parser_terminate(swt_param_parser_t* parser)
{
	switch (parser->state)
	{
		case s_swt_name_interval:
			parser->params.interval = (int)(parser->numeric_parser.result + 0.5);
			break;
		case s_swt_name_min_neighbors:
			parser->params.min_neighbors = (int)(parser->numeric_parser.result + 0.5);
			break;
		case s_swt_name_scale_invariant:
			parser->params.scale_invariant = parser->bool_parser.result;
			break;
		/* canny parameters */
		case s_swt_name_size:
			parser->params.size = (int)(parser->numeric_parser.result + 0.5);
			break;
		case s_swt_name_low_thresh:
			parser->params.low_thresh = (int)(parser->numeric_parser.result + 0.5);
			break;
		case s_swt_name_high_thresh:
			parser->params.high_thresh = (int)(parser->numeric_parser.result + 0.5);
			break;
		case s_swt_name_max_height:
			parser->params.max_height = (int)(parser->numeric_parser.result + 0.5);
			break;
		case s_swt_name_min_height:
			parser->params.min_height = (int)(parser->numeric_parser.result + 0.5);
			break;
		case s_swt_name_min_area:
			parser->params.min_area = (int)(parser->numeric_parser.result + 0.5);
			break;
		case s_swt_name_letter_occlude_thresh:
			parser->params.letter_occlude_thresh = (int)(parser->numeric_parser.result + 0.5);
			break;
		case s_swt_name_aspect_ratio:
			parser->params.aspect_ratio = parser->numeric_parser.result;
			break;
		case s_swt_name_std_ratio:
			parser->params.std_ratio = parser->numeric_parser.result;
			break;
		/* grouping parameters */
		case s_swt_name_thickness_ratio:
			parser->params.thickness_ratio = parser->numeric_parser.result;
			break;
		case s_swt_name_height_ratio:
			parser->params.height_ratio = parser->numeric_parser.result;
			break;
		case s_swt_name_intensity_thresh:
			parser->params.intensity_thresh = (int)(parser->numeric_parser.result + 0.5);
			break;
		case s_swt_name_distance_ratio:
			parser->params.distance_ratio = parser->numeric_parser.result;
			break;
		case s_swt_name_intersect_ratio:
			parser->params.intersect_ratio = parser->numeric_parser.result;
			break;
		case s_swt_name_elongate_ratio:
			parser->params.elongate_ratio = parser->numeric_parser.result;
			break;
		case s_swt_name_letter_thresh:
			parser->params.letter_thresh = (int)(parser->numeric_parser.result + 0.5);
			break;
		/* break textline into words */
		case s_swt_name_breakdown:
			parser->params.breakdown = parser->bool_parser.result;
			break;
		case s_swt_name_breakdown_ratio:
			parser->params.breakdown_ratio = parser->numeric_parser.result;
			break;
		case s_swt_name_source:
			parser->source = parser->blob_parser.data;
			break;
		default:
			break;
	}
	if (parser->state != s_swt_start)
	{
		parser->state = s_swt_start;
		memset(parser->name, 0, sizeof(parser->name));
		parser->cursor = 0;
	}
}

void* uri_swt_detect_words_parse(const void* context, void* parsed, const char* buf, size_t len, uri_parse_state_t state, int header_index)
{
	swt_param_parser_t* parser;
	if (parsed)
		parser = (swt_param_parser_t*)parsed;
	else {
		parser = (swt_param_parser_t*)malloc(sizeof(swt_param_parser_t));
		uri_swt_param_parser_init(parser);
	}
	switch (state)
	{
		case URI_QUERY_STRING:
			break;
		case URI_CONTENT_BODY:
			break;
		case URI_PARSE_TERMINATE:
			if (parser->state != s_swt_start)
				uri_swt_param_parser_terminate(parser); // collect result
			break;
		case URI_MULTIPART_HEADER_FIELD:
			if (parser->state != s_swt_start)
				uri_swt_param_parser_terminate(parser); // collect previous result
			form_data_parser_execute(&parser->form_data_parser, buf, len, header_index);
			break;
		case URI_MULTIPART_HEADER_VALUE:
			if (parser->state != s_swt_start)
				uri_swt_param_parser_terminate(parser); // collect previous result
			form_data_parser_execute(&parser->form_data_parser, buf, len, header_index);
			break;
		case URI_MULTIPART_DATA:
			if (parser->state == s_swt_start)
			{
				// need to use name to get the correct state
				if (strcmp(parser->name, "interval") == 0)
				{
					parser->state = s_swt_name_interval;
					numeric_parser_init(&parser->numeric_parser);
				} else if (strcmp(parser->name, "min_neighbors") == 0) {
					parser->state = s_swt_name_min_neighbors;
					numeric_parser_init(&parser->numeric_parser);
				} else if (strcmp(parser->name, "size") == 0) {
					parser->state = s_swt_name_size;
					numeric_parser_init(&parser->numeric_parser);
				} else if (strcmp(parser->name, "low_thresh") == 0) {
					parser->state = s_swt_name_low_thresh;
					numeric_parser_init(&parser->numeric_parser);
				} else if (strcmp(parser->name, "high_thresh") == 0) {
					parser->state = s_swt_name_high_thresh;
					numeric_parser_init(&parser->numeric_parser);
				} else if (strcmp(parser->name, "max_height") == 0) {
					parser->state = s_swt_name_max_height;
					numeric_parser_init(&parser->numeric_parser);
				} else if (strcmp(parser->name, "min_height") == 0) {
					parser->state = s_swt_name_min_height;
					numeric_parser_init(&parser->numeric_parser);
				} else if (strcmp(parser->name, "min_area") == 0) {
					parser->state = s_swt_name_min_area;
					numeric_parser_init(&parser->numeric_parser);
				} else if (strcmp(parser->name, "letter_occlude_thresh") == 0) {
					parser->state = s_swt_name_letter_occlude_thresh;
					numeric_parser_init(&parser->numeric_parser);
				} else if (strcmp(parser->name, "aspect_ratio") == 0) {
					parser->state = s_swt_name_aspect_ratio;
					numeric_parser_init(&parser->numeric_parser);
				} else if (strcmp(parser->name, "std_ratio") == 0) {
					parser->state = s_swt_name_std_ratio;
					numeric_parser_init(&parser->numeric_parser);
				} else if (strcmp(parser->name, "thickness_ratio") == 0) {
					parser->state = s_swt_name_thickness_ratio;
					numeric_parser_init(&parser->numeric_parser);
				} else if (strcmp(parser->name, "height_ratio") == 0) {
					parser->state = s_swt_name_height_ratio;
					numeric_parser_init(&parser->numeric_parser);
				} else if (strcmp(parser->name, "intensity_thresh") == 0) {
					parser->state = s_swt_name_intensity_thresh;
					numeric_parser_init(&parser->numeric_parser);
				} else if (strcmp(parser->name, "distance_ratio") == 0) {
					parser->state = s_swt_name_distance_ratio;
					numeric_parser_init(&parser->numeric_parser);
				} else if (strcmp(parser->name, "intersect_ratio") == 0) {
					parser->state = s_swt_name_intersect_ratio;
					numeric_parser_init(&parser->numeric_parser);
				} else if (strcmp(parser->name, "elongate_ratio") == 0) {
					parser->state = s_swt_name_elongate_ratio;
					numeric_parser_init(&parser->numeric_parser);
				} else if (strcmp(parser->name, "letter_thresh") == 0) {
					parser->state = s_swt_name_letter_thresh;
					numeric_parser_init(&parser->numeric_parser);
				} else if (strcmp(parser->name, "breakdown_ratio") == 0) {
					parser->state = s_swt_name_breakdown_ratio;
					numeric_parser_init(&parser->numeric_parser);
				} else if (strcmp(parser->name, "scale_invariant") == 0) {
					parser->state = s_swt_name_scale_invariant;
					bool_parser_init(&parser->bool_parser);
				} else if (strcmp(parser->name, "breakdown") == 0) {
					parser->state = s_swt_name_breakdown;
					bool_parser_init(&parser->bool_parser);
				} else if (strcmp(parser->name, "source") == 0) {
					parser->state = s_swt_name_source;
					blob_parser_init(&parser->blob_parser);
				} else
					parser->state = s_swt_skip;
			}
			switch (parser->state)
			{
				default:
					break;
				case s_swt_name_interval:
				case s_swt_name_min_neighbors:
				case s_swt_name_size:
				case s_swt_name_low_thresh:
				case s_swt_name_high_thresh:
				case s_swt_name_max_height:
				case s_swt_name_min_height:
				case s_swt_name_min_area:
				case s_swt_name_letter_occlude_thresh:
				case s_swt_name_aspect_ratio:
				case s_swt_name_std_ratio:
				case s_swt_name_thickness_ratio:
				case s_swt_name_height_ratio:
				case s_swt_name_intensity_thresh:
				case s_swt_name_distance_ratio:
				case s_swt_name_intersect_ratio:
				case s_swt_name_elongate_ratio:
				case s_swt_name_letter_thresh:
				case s_swt_name_breakdown_ratio:
					numeric_parser_execute(&parser->numeric_parser, buf, len);
					if (parser->numeric_parser.state == s_numeric_illegal)
						parser->state = s_swt_skip;
					break;
				case s_swt_name_scale_invariant:
				case s_swt_name_breakdown:
					bool_parser_execute(&parser->bool_parser, buf, len);
					if (parser->bool_parser.state == s_bool_illegal)
						parser->state = s_swt_skip;
					break;
				case s_swt_name_source:
					blob_parser_execute(&parser->blob_parser, buf, len);
					break;
			}
			break;
	}
	return parser;
}

int uri_swt_detect_words_intro(const void* context, const void* parsed, ebb_buf* buf)
{
	const static char swt_desc[] = 
		"HTTP/1.1 200 OK\r\nCache-Control: no-cache\r\nAccept: \r\nContent-Type: text/html\r\nContent-Length: 163\r\n\r\n"
		"<html><body><form enctype='multipart/form-data' method='post'><input name='model' value='pedestrian'><input type='file' name='source'><input type='submit'></form>\n";
	buf->data = (void*)swt_desc;
	buf->len = sizeof(swt_desc);
	return 0;
}

int uri_swt_detect_words(const void* context, const void* parsed, ebb_buf* buf)
{
	swt_param_parser_t* parser = (swt_param_parser_t*)parsed;
	if (parser->state != s_swt_start)
		uri_swt_param_parser_terminate(parser);
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
	ccv_array_t* seq = ccv_swt_detect_words(image, parser->params);
	ccv_matrix_free(image);
	if (seq  == 0)
	{
		free(parser);
		return -1;
	}
	if (seq->rnum > 0)
	{
		int i;
		buf->len = 192 + seq->rnum * 131 + 2;
		char* data = (char*)malloc(buf->len);
		data[0] = '[';
		buf->written = 1;
		for (i = 0; i < seq->rnum; i++)
		{
			char cell[96];
			ccv_rect_t* rect = (ccv_rect_t*)ccv_array_get(seq, i);
			snprintf(cell, 96, "{\"x\":%d,\"y\":%d,\"width\":%d,\"height\":%d}", rect->x, rect->y, rect->width, rect->height);
			size_t len = strnlen(cell, 96);
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
