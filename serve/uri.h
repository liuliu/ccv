#ifndef _GUARD_uri_h_
#define _GUARD_uri_h_

#include "ebb.h"
#include <stddef.h>

/* have to be static const char so that can use sizeof */
static const char ebb_http_404[] = "HTTP/1.0 404 Not Found\r\nCache-Control: no-cache\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: 6\r\n\r\nfalse\n";
static const char ebb_http_empty_object[] = "HTTP/1.0 201 Created\r\nCache-Control: no-cache\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: 3\r\n\r\n{}\n";
static const char ebb_http_empty_array[] = "HTTP/1.0 201 Created\r\nCache-Control: no-cache\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: 3\r\n\r\n[]\n";
static const char ebb_http_ok_true[] = "HTTP/1.0 200 OK\r\nCache-Control: no-cache\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: 5\r\n\r\ntrue\n";
/* we should never sizeof ebb_http_header */
extern const char ebb_http_header[];

void uri_ebb_buf_free(ebb_buf* buf);

typedef enum {
	s_form_data_start,
	s_form_data_header_field,
	s_form_data_header_value_start,
	s_form_data_header_value_name_start,
	s_form_data_header_value_name_done,
	s_form_data_done,
} form_data_state_t;

typedef struct {
	form_data_state_t state;
	char lookbehind;
	int cursor;
	int disposition_index;
	void* context;
	void (*on_name)(void*, const char*, size_t);
} form_data_parser_t;

void form_data_parser_init(form_data_parser_t* parser, void* context);
void form_data_parser_execute(form_data_parser_t* parser, const char* buf, size_t len, int header_index);

typedef enum {
	s_query_string_start,
	s_query_string_field_start,
	s_query_string_value_start,
	s_query_string_value_done,
} query_string_state_t;

typedef struct {
	query_string_state_t state;
	void* context;
	int header_index;
	void (*on_field)(void*, const char*, size_t, int);
	void (*on_value)(void*, const char*, size_t, int);
} query_string_parser_t;

void query_string_parser_init(query_string_parser_t* parser, void* context);
void query_string_parser_execute(query_string_parser_t* parser, const char* buf, size_t len);

typedef enum {
	s_numeric_start,
	s_numeric_before_decimal,
	s_numeric_after_decimal,
	s_numeric_illegal,
} numeric_state_t;

typedef struct {
	numeric_state_t state;
	double result;
	double division;
} numeric_parser_t;

void numeric_parser_init(numeric_parser_t* parser);
void numeric_parser_execute(numeric_parser_t* parser, const char* buf, size_t len);

typedef enum {
	s_bool_start,
	s_bool_1,
	s_bool_0,
	s_bool_true,
	s_bool_false,
	s_bool_illegal,
} bool_state_t;

typedef struct {
	bool_state_t state;
	int cursor;
	int result;
} bool_parser_t;

void bool_parser_init(bool_parser_t* parser);
void bool_parser_execute(bool_parser_t* parser, const char* buf, size_t len);

typedef enum {
	s_coord_start,
	s_coord_x_before_decimal,
	s_coord_x_after_decimal,
	s_coord_y_before_decimal,
	s_coord_y_after_decimal,
	s_coord_illegal,
} coord_state_t;

typedef struct {
	coord_state_t state;
	double x;
	double y;
	double division;
} coord_parser_t;

void coord_parser_init(coord_parser_t* parser);
void coord_parser_execute(coord_parser_t* parser, const char* buf, size_t len);

typedef enum {
	s_string_start,
	s_string_overflow,
} string_state_t;

typedef struct {
	string_state_t state;
	char string[256];
	int cursor;
} string_parser_t;

void string_parser_init(string_parser_t* parser);
void string_parser_execute(string_parser_t* parser, const char* buf, size_t len);

typedef struct {
	ebb_buf data;
} blob_parser_t;

void blob_parser_init(blob_parser_t* parser);
void blob_parser_execute(blob_parser_t* parser, const char* buf, size_t len);

typedef enum {
	URI_QUERY_STRING,
	URI_CONTENT_BODY,
	URI_MULTIPART_HEADER_FIELD,
	URI_MULTIPART_HEADER_VALUE,
	URI_MULTIPART_DATA,
	URI_PARSE_TERMINATE,
} uri_parse_state_t;

typedef enum {
	PARAM_TYPE_INT,
	PARAM_TYPE_FLOAT,
	PARAM_TYPE_DOUBLE,
	PARAM_TYPE_SIZE,
	PARAM_TYPE_POINT,
	PARAM_TYPE_STRING,
	PARAM_TYPE_BOOL,
	PARAM_TYPE_BLOB,
	PARAM_TYPE_BODY, // alias for BLOB, in cases that is not multi-part, this is get result from http body
	PARAM_TYPE_ID, // alias for INT, in cases that is a part of uri, it will triumph any parameter passed in later
} param_type_t;

typedef struct {
	char* property;
	param_type_t type;
	size_t offset;
	void (*on_string)(void*, char*);
	void (*on_blob)(void*, ebb_buf);
} param_dispatch_t;

typedef enum {
	s_param_start = -1,
	s_param_skip = -2,
	/* the rest of the states are numerated from 0 to upper size of param_map */
} param_parse_state_t;

typedef struct {
	param_parse_state_t state;
	param_parse_state_t body;
	param_parse_state_t resource;
	form_data_parser_t form_data_parser;
	query_string_parser_t query_string_parser;
	int header_index;
	int cursor;
	char name[32];
	const param_dispatch_t* param_map;
	size_t len;
	char* parsed;
	void* context;
	union {
		numeric_parser_t numeric_parser;
		bool_parser_t bool_parser;
		coord_parser_t coord_parser;
		string_parser_t string_parser;
		blob_parser_t blob_parser;
	};
} param_parser_t;

void param_parser_terminate(param_parser_t* parser);
int param_parser_map_alphabet(const param_dispatch_t* param_map, size_t len);
ebb_buf param_parser_map_http_body(const param_dispatch_t* param_map, size_t len, const char* response_format);
void param_parser_init(param_parser_t* parser, const param_dispatch_t* param_map, size_t len, void* parsed, void* context);
void param_parser_execute(param_parser_t* parser, int resource_id, const char* buf, size_t len, uri_parse_state_t state, int header_index);

typedef struct {
	char* uri;
	void* context;
	void* (*init)(void); // this runs on server start
	void* (*parse)(const void*, void*, int, const char*, size_t, uri_parse_state_t, int); // this runs on main thread
	int (*get)(const void*, const void*, ebb_buf*); // this runs off thread
	int (*post)(const void*, const void*, ebb_buf*); // this runs off thread
	int (*delete)(const void*, const void*, ebb_buf*); // this runs off thread
	void (*destroy)(void*); // this runs on server shutdown
} uri_dispatch_t;

uri_dispatch_t* find_uri_dispatch(const char* path);
void uri_init(void);
void uri_destroy(void);

void* uri_root_init(void);
void uri_root_destroy(void* context);
int uri_root_discovery(const void* context, const void* parsed, ebb_buf* buf);

void* uri_bbf_detect_objects_init(void);
void uri_bbf_detect_objects_destroy(void* context);
void* uri_bbf_detect_objects_parse(const void* context, void* parsed, int resource_id, const char* buf, size_t len, uri_parse_state_t state, int header_index);
int uri_bbf_detect_objects_intro(const void* context, const void* parsed, ebb_buf* buf);
int uri_bbf_detect_objects(const void* context, const void* parsed, ebb_buf* buf);

void* uri_dpm_detect_objects_init(void);
void uri_dpm_detect_objects_destroy(void* context);
void* uri_dpm_detect_objects_parse(const void* context, void* parsed, int resource_id, const char* buf, size_t len, uri_parse_state_t state, int header_index);
int uri_dpm_detect_objects_intro(const void* context, const void* parsed, ebb_buf* buf);
int uri_dpm_detect_objects(const void* context, const void* parsed, ebb_buf* buf);

void* uri_icf_detect_objects_init(void);
void uri_icf_detect_objects_destroy(void* context);
void* uri_icf_detect_objects_parse(const void* context, void* parsed, int resource_id, const char* buf, size_t len, uri_parse_state_t state, int header_index);
int uri_icf_detect_objects_intro(const void* context, const void* parsed, ebb_buf* buf);
int uri_icf_detect_objects(const void* context, const void* parsed, ebb_buf* buf);

void* uri_scd_detect_objects_init(void);
void uri_scd_detect_objects_destroy(void* context);
void* uri_scd_detect_objects_parse(const void* context, void* parsed, int resource_id, const char* buf, size_t len, uri_parse_state_t state, int header_index);
int uri_scd_detect_objects_intro(const void* context, const void* parsed, ebb_buf* buf);
int uri_scd_detect_objects(const void* context, const void* parsed, ebb_buf* buf);

void* uri_sift_init(void);
void uri_sift_destroy(void* context);
void* uri_sift_parse(const void* context, void* parsed, int resource_id, const char* buf, size_t len, uri_parse_state_t state, int header_index);
int uri_sift_intro(const void* context, const void* parsed, ebb_buf* buf);
int uri_sift(const void* context, const void* parsed, ebb_buf* buf);

void* uri_swt_detect_words_init(void);
void uri_swt_detect_words_destroy(void* context);
void* uri_swt_detect_words_parse(const void* context, void* parsed, int resource_id, const char* buf, size_t len, uri_parse_state_t state, int header_index);
int uri_swt_detect_words_intro(const void* context, const void* parsed, ebb_buf* buf);
int uri_swt_detect_words(const void* context, const void* parsed, ebb_buf* buf);

void* uri_tld_track_object_init(void);
void uri_tld_track_object_destroy(void* context);
void* uri_tld_track_object_parse(const void* context, void* parsed, int resource_id, const char* buf, size_t len, uri_parse_state_t state, int header_index);
int uri_tld_track_object_intro(const void* context, const void* parsed, ebb_buf* buf);
int uri_tld_track_object(const void* context, const void* parsed, ebb_buf* buf);
int uri_tld_track_object_free(const void* context, const void* parsed, ebb_buf* buf);

void* uri_convnet_classify_init(void);
void uri_convnet_classify_destroy(void* context);
void* uri_convnet_classify_parse(const void* context, void* parsed, int resource_id, const char* buf, size_t len, uri_parse_state_t state, int header_index);
int uri_convnet_classify_intro(const void* context, const void* parsed, ebb_buf* buf);
int uri_convnet_classify(const void* context, const void* parsed, ebb_buf* buf);

#endif
