#include "uri.h"
#include "ccv.h"
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#ifdef HAVE_TESSERACT
#include <tesseract/capi.h>
#endif

static void uri_swt_on_source_blob(void* context, ebb_buf data);

typedef struct {
	ccv_swt_param_t params;
	int max_dimension;
} ccv_swt_uri_param_t;

static const param_dispatch_t param_map[] = {
	{
		.property = "aspect_ratio",
		.type = PARAM_TYPE_DOUBLE,
		.offset = offsetof(ccv_swt_uri_param_t, params) + offsetof(ccv_swt_param_t, aspect_ratio),
	},
	{
		.property = "breakdown",
		.type = PARAM_TYPE_BOOL,
		.offset = offsetof(ccv_swt_uri_param_t, params) + offsetof(ccv_swt_param_t, breakdown),
	},
	{
		.property = "breakdown_ratio",
		.type = PARAM_TYPE_DOUBLE,
		.offset = offsetof(ccv_swt_uri_param_t, params) + offsetof(ccv_swt_param_t, breakdown_ratio),
	},
	{
		.property = "distance_ratio",
		.type = PARAM_TYPE_DOUBLE,
		.offset = offsetof(ccv_swt_uri_param_t, params) + offsetof(ccv_swt_param_t, distance_ratio),
	},
	{
		.property = "elongate_ratio",
		.type = PARAM_TYPE_DOUBLE,
		.offset = offsetof(ccv_swt_uri_param_t, params) + offsetof(ccv_swt_param_t, elongate_ratio),
	},
	{
		.property = "height_ratio",
		.type = PARAM_TYPE_DOUBLE,
		.offset = offsetof(ccv_swt_uri_param_t, params) + offsetof(ccv_swt_param_t, height_ratio),
	},
	{
		.property = "high_thresh",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_swt_uri_param_t, params) + offsetof(ccv_swt_param_t, high_thresh),
	},
	{
		.property = "intensity_thresh",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_swt_uri_param_t, params) + offsetof(ccv_swt_param_t, intensity_thresh),
	},
	{
		.property = "intersect_ratio",
		.type = PARAM_TYPE_DOUBLE,
		.offset = offsetof(ccv_swt_uri_param_t, params) + offsetof(ccv_swt_param_t, intersect_ratio),
	},
	{
		.property = "interval",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_swt_uri_param_t, params) + offsetof(ccv_swt_param_t, interval),
	},
	{
		.property = "letter_occlude_thresh",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_swt_uri_param_t, params) + offsetof(ccv_swt_param_t, letter_occlude_thresh),
	},
	{
		.property = "letter_thresh",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_swt_uri_param_t, params) + offsetof(ccv_swt_param_t, letter_thresh),
	},
	{
		.property = "low_thresh",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_swt_uri_param_t, params) + offsetof(ccv_swt_param_t, low_thresh),
	},
	{
		.property = "max_dimension",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_swt_uri_param_t, max_dimension),
	},
	{
		.property = "max_height",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_swt_uri_param_t, params) + offsetof(ccv_swt_param_t, max_height),
	},
	{
		.property = "min_area",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_swt_uri_param_t, params) + offsetof(ccv_swt_param_t, min_area),
	},
	{
		.property = "min_height",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_swt_uri_param_t, params) + offsetof(ccv_swt_param_t, min_height),
	},
	{
		.property = "min_neighbors",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_swt_uri_param_t, params) + offsetof(ccv_swt_param_t, min_neighbors),
	},
	{
		.property = "scale_invariant",
		.type = PARAM_TYPE_BOOL,
		.offset = offsetof(ccv_swt_uri_param_t, params) + offsetof(ccv_swt_param_t, scale_invariant),
	},
	{
		.property = "size",
		.type = PARAM_TYPE_INT,
		.offset = offsetof(ccv_swt_uri_param_t, params) + offsetof(ccv_swt_param_t, size),
	},
	{
		.property = "source",
		.type = PARAM_TYPE_BODY,
		.on_blob = uri_swt_on_source_blob,
		.offset = 0,
	},
	{
		.property = "std_ratio",
		.type = PARAM_TYPE_DOUBLE,
		.offset = offsetof(ccv_swt_uri_param_t, params) + offsetof(ccv_swt_param_t, std_ratio),
	},
	{
		.property = "thickness_ratio",
		.type = PARAM_TYPE_DOUBLE,
		.offset = offsetof(ccv_swt_uri_param_t, params) + offsetof(ccv_swt_param_t, thickness_ratio),
	},
};

typedef struct {
	ebb_buf desc;
#ifdef HAVE_TESSERACT
	TessBaseAPI* tesseract;
#endif
} swt_context_t;

typedef struct {
	param_parser_t param_parser;
	ccv_swt_uri_param_t params;
	ebb_buf source;
	swt_context_t* context;
} swt_param_parser_t;

void* uri_swt_detect_words_init(void)
{
	assert(param_parser_map_alphabet(param_map, sizeof(param_map) / sizeof(param_dispatch_t)) == 0);
	swt_context_t* context = (swt_context_t*)malloc(sizeof(swt_context_t));
#ifdef HAVE_TESSERACT
	context->tesseract = TessBaseAPICreate();
	if (TessBaseAPIInit3(context->tesseract, 0, "eng") != 0)
		context->tesseract = 0;
#endif
	context->desc = param_parser_map_http_body(param_map, sizeof(param_map) / sizeof(param_dispatch_t),
		"[{"
			"\"x\":\"number\","
			"\"y\":\"number\","
			"\"width\":\"number\","
			"\"height\":\"number\""
		"}]");
	return context;
}

void uri_swt_detect_words_destroy(void* context)
{
	swt_context_t* swt_context = (swt_context_t*)context;
#ifdef HAVE_TESSERACT
	TessBaseAPIDelete(swt_context->tesseract);
#endif
	free(swt_context->desc.data);
	free(swt_context);
}

static void uri_swt_param_parser_init(swt_param_parser_t* parser)
{
	param_parser_init(&parser->param_parser, param_map, sizeof(param_map) / sizeof(param_dispatch_t), &parser->params, parser);
	parser->params.params = ccv_swt_default_params;
	parser->params.max_dimension = 0;
	parser->source.data = 0;
}

static void uri_swt_on_source_blob(void* context, ebb_buf data)
{
	swt_param_parser_t* parser = (swt_param_parser_t*)context;
	parser->source = data;
}

void* uri_swt_detect_words_parse(const void* context, void* parsed, int resource_id, const char* buf, size_t len, uri_parse_state_t state, int header_index)
{
	swt_param_parser_t* parser;
	if (parsed)
		parser = (swt_param_parser_t*)parsed;
	else {
		parser = (swt_param_parser_t*)malloc(sizeof(swt_param_parser_t));
		parser->context = (swt_context_t*)context;
		uri_swt_param_parser_init(parser);
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

int uri_swt_detect_words_intro(const void* context, const void* parsed, ebb_buf* buf)
{
	swt_context_t* swt_context = (swt_context_t*)context;
	buf->data = swt_context->desc.data;
	buf->len = swt_context->desc.len;
	return 0;
}

int uri_swt_detect_words(const void* context, const void* parsed, ebb_buf* buf)
{
	if (!parsed)
		return -1;
	swt_param_parser_t* parser = (swt_param_parser_t*)parsed;
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
	ccv_dense_matrix_t* resize = 0;
	if (parser->params.max_dimension > 0 && (image->rows > parser->params.max_dimension || image->cols > parser->params.max_dimension))
	{
		ccv_resample(image, &resize, 0, ccv_min(parser->params.max_dimension, (int)(image->rows * (float)parser->params.max_dimension / image->cols + 0.5)), ccv_min(parser->params.max_dimension, (int)(image->cols * (float)parser->params.max_dimension / image->rows + 0.5)), CCV_INTER_AREA);
		ccv_matrix_free(image);
	} else
		resize = image;
	ccv_array_t* seq = ccv_swt_detect_words(resize, parser->params.params);
	float width = resize->cols, height = resize->rows;
	if (seq  == 0)
	{
		ccv_matrix_free(resize);
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
			char cell[1024];
			ccv_rect_t* rect = (ccv_rect_t*)ccv_array_get(seq, i);
#ifdef HAVE_TESSERACT
			if (parser->context->tesseract)
			{
				char empty[] = "";
				char* word = TessBaseAPIRect(parser->context->tesseract, resize->data.u8, 1, resize->step, rect->x, rect->y, rect->width, rect->height);
				if (!word)
					word = empty;
				int wordlen = strlen(word); // trust tesseract to return correct thing
				int j;
				for (j = 0; j < wordlen; j++)
					if (!((word[j] >= 'a' && word[j] <= 'z') ||
							(word[j] >= 'A' && word[j] <= 'Z') ||
							(word[j] >= '0' && word[j] <= '9') ||
							word[j] == ' ' ||
							word[j] == '-')) // replace unsupported char to whitespace
						word[j] = ' ';
				for (j = wordlen - 1; j >= 0 && word[j] == ' '; j--); // remove trailing whitespace
				word[j + 1] = 0, wordlen = j + 1;
				for (j = 0; j < wordlen && word[j] == ' '; j++); // remove leading whitespace
				wordlen -= j;
				memmove(word, word + j, wordlen + 1);
				if (wordlen > 512) // if the wordlen is greater than 512, trim it
					word[512] = 0;
				snprintf(cell, 1024, "{\"x\":%f,\"y\":%f,\"width\":%f,\"height\":%f,\"word\":\"%s\"}", rect->x / width, rect->y / height, rect->width / width, rect->height / height, word);
			} else {
#endif
			snprintf(cell, 1024, "{\"x\":%f,\"y\":%f,\"width\":%f,\"height\":%f}", rect->x / width, rect->y / height, rect->width / width, rect->height / height);
#ifdef HAVE_TESSERACT
			}
#endif
			size_t len = strnlen(cell, 1024);
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
	ccv_matrix_free(resize);
	ccv_array_free(seq);
	free(parser);
	return 0;
}
