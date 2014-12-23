#include "uri.h"
#include "ccv.h"
#include "ccv_internal.h"
#include <string.h>
#include <ctype.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

void form_data_parser_init(form_data_parser_t* parser, void* context)
{
	parser->state = s_form_data_start;
	parser->cursor = 0;
	parser->disposition_index = -1;
	parser->context = context;
	parser->on_name = 0;
}

void form_data_parser_execute(form_data_parser_t* parser, const char* buf, size_t len, int header_index)
{
	static const char content_disposition[] = "content-disposition";
	static const char form_data[] = "form-data;";
	static const char name[] = "name=\"";

	int i, cl;
	size_t name_len = 0;
	const char* name_mark = buf;
	for (i = 0; i < len; i++)
	{
		cl = tolower(buf[i]);
		switch (parser->state)
		{
			default:
				// illegal state, just reset and return
				parser->state = s_form_data_start;
				/* fall-through */
			case s_form_data_start:
				parser->state = s_form_data_header_field;
				parser->cursor = 0;
				/* fall-through */
			case s_form_data_header_field:
				// only care about Content-Disposition
				if (cl != content_disposition[parser->cursor])
				{
					parser->state = s_form_data_start; // reset the state
					return;
				}
				++parser->cursor;
				if (parser->cursor == sizeof(content_disposition) - 1)
				{
					parser->disposition_index = header_index;
					parser->state = s_form_data_header_value_start;
					parser->cursor = 0;
				}
				break;
			case s_form_data_header_value_start:
				if (cl == ' ' || cl == '\t') // ignore space or tab
					continue;
				if (cl != form_data[parser->cursor])
				{
					parser->state = s_form_data_start; // we don't accept other disposition other than form-data
					return;
				}
				++parser->cursor;
				if (parser->cursor == sizeof(form_data) - 1)
				{
					// verified form-data, now get the name parameter
					parser->state = s_form_data_header_value_name_start;
					parser->cursor = 0;
				}
				break;
			case s_form_data_header_value_name_start:
				if (cl == ' ' || cl == '\t') // ignore space or tab
					continue;
				if (cl != name[parser->cursor])
				{
					parser->state = s_form_data_start; // we only accepts name parameter for form-data here
					return;
				}
				++parser->cursor;
				if (parser->cursor == sizeof(name) - 1)
				{
					parser->state = s_form_data_header_value_name_done;
					parser->lookbehind = '\0';
					parser->cursor = 0;
					name_mark = buf + i + 1;
					name_len = 0;
				}
				break;
			case s_form_data_header_value_name_done:
				if (parser->lookbehind != '\\' && buf[i] == '"')
				{
					parser->state = s_form_data_done; // the end of quote, return
					if (name_len > 0 && parser->on_name)
						parser->on_name(parser->context, name_mark, name_len);
					return;
				}
				name_len = buf + i + 1 - name_mark;
				// the lookbehind is only used for escape \, and if it is \ already
				// we will skip the current one, otherwise, we don't
				parser->lookbehind = (parser->lookbehind != '\\') ? buf[i] : '\0';
				break;
		}
	}
	if (name_len > 0 && parser->state == s_form_data_header_value_name_done && parser->on_name)
		parser->on_name(parser->context, name_mark, name_len);
}

void query_string_parser_init(query_string_parser_t* parser, void* context)
{
	parser->state = s_query_string_start;
	parser->context = context;
	parser->header_index = -1;
	parser->on_field = 0;
	parser->on_value = 0;
}

void query_string_parser_execute(query_string_parser_t* parser, const char* buf, size_t len)
{
	int i;
	size_t field_len = 0;
	const char* field_mark = buf;
	size_t value_len = 0;
	const char* value_mark = buf;
	for (i = 0; i < len; i++)
		switch (parser->state)
		{
			case s_query_string_start:
				parser->state = s_query_string_field_start;
				++parser->header_index;
				field_mark = buf + i;
				field_len = 0;
				/* fall-through */
			case s_query_string_field_start:
				if (buf[i] != '&')
				{
					if (buf[i] == '=')
					{
						parser->state = s_query_string_value_start;
						// setup value_len
						value_mark = buf + i + 1;
						value_len = 0;
					}
					if (parser->state == s_query_string_field_start)
						field_len = buf + i + 1 - field_mark;
					break;
				} else
					// it is marked as the start state, then quickly turns to done state
					parser->state = s_query_string_value_start;
					/* fall-through */
			case s_query_string_value_start:
				if (field_len > 0 && parser->on_field)
				{
					parser->on_field(parser->context, field_mark, field_len, parser->header_index);
					field_len = 0;
				}
				if (buf[i] != '&')
				{
					value_len = buf + i + 1 - value_mark;
					break;
				} else
					parser->state = s_query_string_value_done;
					/* fall-through */
			case s_query_string_value_done:
				// reset field_len
				field_mark = buf + i + 1;
				field_len = 0;
				if (value_len > 0 && parser->on_value)
				{
					parser->on_value(parser->context, value_mark, value_len, parser->header_index);
					// reset value_len
					value_len = 0;
				}
				++parser->header_index;
				parser->state = s_query_string_field_start;
				break;
		}
	if (field_len > 0 && parser->state == s_query_string_field_start && parser->on_field)
		parser->on_field(parser->context, field_mark, field_len, parser->header_index);
	else if (value_len > 0 && parser->state == s_query_string_value_start && parser->on_value)
		parser->on_value(parser->context, value_mark, value_len, parser->header_index);
}

void numeric_parser_init(numeric_parser_t* parser)
{
	parser->state = s_numeric_start;
	parser->result = 0;
	parser->division = 0.1;
}

void numeric_parser_execute(numeric_parser_t* parser, const char* buf, size_t len)
{
	int i;
	for (i = 0; i < len; i++)
	{
		int digit = buf[i] - '0';
		if ((digit < 0 || digit >= 10) && buf[i] != '.')
			parser->state = s_numeric_illegal;
		switch (parser->state)
		{
			case s_numeric_start:
				parser->result = 0;
				parser->state = s_numeric_before_decimal;
				/* fall-through */
			case s_numeric_before_decimal:
				if (buf[i] != '.')
					parser->result = parser->result * 10 + digit;
				else
					parser->state = s_numeric_after_decimal;
				break;
			case s_numeric_after_decimal:
				if (buf[i] == '.') // we cannot bear another .
					parser->state = s_numeric_illegal;
				else {
					parser->result += digit * parser->division;
					parser->division *= 0.1;
				}
				break;
			case s_numeric_illegal:
				break;
		}
		if (parser->state == s_numeric_illegal)
			break;
	}
}

void bool_parser_init(bool_parser_t* parser)
{
	parser->state = s_bool_start;
	parser->cursor = 0;
}

void bool_parser_execute(bool_parser_t* parser, const char* buf, size_t len)
{
	int i;
	static const char bool_true[] = "true";
	static const char bool_false[] = "false";
	for (i = 0; i < len; i++)
	{
		if (parser->state == s_bool_illegal)
			break;
		int cl = tolower(buf[i]);
		switch (parser->state)
		{
			default:
				break;
			case s_bool_start:
				switch (cl)
				{
					default:
						parser->state = s_bool_illegal;
						break;
					case '0':
						parser->state = s_bool_0;
						parser->result = 0;
						break;
					case '1':
						parser->state = s_bool_1;
						parser->result = 1;
						break;
					case 'f':
						parser->state = s_bool_false;
						parser->result = 0;
						parser->cursor = 1;
						break;
					case 't':
						parser->state = s_bool_true;
						parser->result = 1;
						parser->cursor = 1;
						break;
				}
				break;
			case s_bool_1: // it should be already 1-len, so this should be illegal
			case s_bool_0:
				parser->state = s_bool_illegal;
				break;
			case s_bool_true:
				if (parser->cursor > sizeof(bool_true) - 1 || cl != bool_true[parser->cursor])
					parser->state = s_bool_illegal;
				else
					++parser->cursor;
				break;
			case s_bool_false:
				if (parser->cursor > sizeof(bool_false) - 1 || cl != bool_false[parser->cursor])
					parser->state = s_bool_illegal;
				else
					++parser->cursor;
				break;
		}
	}
}

void coord_parser_init(coord_parser_t* parser)
{
	parser->state = s_coord_start;
	parser->x = parser->y = 0;
	parser->division = 0.1;
}

void coord_parser_execute(coord_parser_t* parser, const char* buf, size_t len)
{
	int i;
	for (i = 0; i < len; i++)
	{
		int digit = buf[i] - '0';
		if ((digit < 0 || digit >= 10) && buf[i] != '.' && buf[i] != 'x' && buf[i] != 'X')
			parser->state = s_coord_illegal;
		switch (parser->state)
		{
			case s_coord_start:
				parser->x = parser->y = 0;
				parser->state = s_coord_x_before_decimal;
				/* fall-through */
			case s_coord_x_before_decimal:
				if (buf[i] != 'x' && buf[i] != 'X')
				{
					if (buf[i] != '.')
						parser->x = parser->x * 10 + digit;
					else
						parser->state = s_coord_x_after_decimal;
				} else
					parser->state = s_coord_y_before_decimal;
				break;
			case s_coord_x_after_decimal:
				if (buf[i] != 'x' && buf[i] != 'X')
				{
					if (buf[i] == '.')
					{
						parser->state = s_coord_illegal;
						break;
					}
					parser->x += digit * parser->division;
					parser->division *= 0.1;
				} else {
					parser->division = 0.1;
					parser->state = s_coord_y_before_decimal;
				}
				break;
			case s_coord_y_before_decimal:
				if (buf[i] == 'x' || buf[i] == 'X')
				{
					parser->state = s_coord_illegal;
					break;
				}
				if (buf[i] != '.')
					parser->y = parser->y * 10 + digit;
				else
					parser->state = s_coord_y_after_decimal;
				break;
			case s_coord_y_after_decimal:
				if (buf[i] == 'x' || buf[i] == 'X' || buf[i] == '.')
				{
					parser->state = s_coord_illegal;
					break;
				}
				parser->y += digit * parser->division;
				parser->division *= 0.1;
				break;
			case s_coord_illegal:
				break;
		}
		if (parser->state == s_coord_illegal)
			break;
	}
}

void string_parser_init(string_parser_t* parser)
{
	memset(parser->string, 0, sizeof(parser->string));
	parser->state = s_string_start;
	parser->cursor = 0;
}

void string_parser_execute(string_parser_t* parser, const char* buf, size_t len)
{
	if (parser->cursor + len > sizeof(parser->string) - 1)
	{
		PRINT(CCV_CLI_INFO, "string parameter overflow %zu\n", sizeof(parser->string));
		parser->state = s_string_overflow;
	}
	else if (parser->state == s_string_start) {
		memcpy(parser->string + parser->cursor, buf, len);
		parser->cursor += len;
	}
}

void blob_parser_init(blob_parser_t* parser)
{
	parser->data.len = 0;
	parser->data.written = 0;
	parser->data.data = 0;
}

void blob_parser_execute(blob_parser_t* parser, const char* buf, size_t len)
{
	if (parser->data.len == 0)
	{
		parser->data.len = (len * 3 + 1) / 2;
		parser->data.data = (unsigned char*)malloc(parser->data.len);
	} else if (parser->data.written + len > parser->data.len) {
		parser->data.len = ((parser->data.len + len) * 3 + 1) / 2;
		parser->data.data = (unsigned char*)realloc(parser->data.data, parser->data.len);
	}
	memcpy(parser->data.data + parser->data.written, buf, len);
	parser->data.written += len;
}

int param_parser_map_alphabet(const param_dispatch_t* param_map, size_t len)
{
	int i;
	for (i = 1; i < len; i++)
		if (strcmp(param_map[i - 1].property, param_map[i].property) >= 0)
			return -1;
	return 0;
}

static int find_param_dispatch_state(param_parser_t* parser, const char* name)
{
	const param_dispatch_t* low = parser->param_map;
	const param_dispatch_t* high = parser->param_map + parser->len - 1;
	while (low <= high)
	{
		const param_dispatch_t* middle = low + (high - low) / 2;
		int flag = strcmp(middle->property, name);
		if (flag == 0)
			return middle - parser->param_map;
		else if (flag < 0)
			low = middle + 1;
		else
			high = middle - 1;
	}
	return s_param_skip;
}

void param_parser_terminate(param_parser_t* parser)
{
	if (parser->state >= 0)
	{
		const param_dispatch_t* dispatch = parser->param_map + parser->state;
		switch (dispatch->type)
		{
			case PARAM_TYPE_INT:
				*(int*)(parser->parsed + dispatch->offset) = (int)(parser->numeric_parser.result + 0.5);
				break;
			case PARAM_TYPE_ID:
				if (*(int*)(parser->parsed + dispatch->offset) < 0) // original is illegal resource id
					*(int*)(parser->parsed + dispatch->offset) = (int)(parser->numeric_parser.result + 0.5);
				break;
			case PARAM_TYPE_FLOAT:
				*(float*)(parser->parsed + dispatch->offset) = (float)parser->numeric_parser.result;
				break;
			case PARAM_TYPE_DOUBLE:
				*(double*)(parser->parsed + dispatch->offset) = parser->numeric_parser.result;
				break;
			case PARAM_TYPE_BOOL:
				*(int*)(parser->parsed + dispatch->offset) = parser->bool_parser.result;
				break;
			case PARAM_TYPE_SIZE:
				*(ccv_size_t*)(parser->parsed + dispatch->offset) = ccv_size((int)(parser->coord_parser.x + 0.5), (int)(parser->coord_parser.y + 0.5));
				break;
			case PARAM_TYPE_POINT:
				*(ccv_point_t*)(parser->parsed + dispatch->offset) = ccv_point((int)(parser->coord_parser.x + 0.5), (int)(parser->coord_parser.y + 0.5));
				break;
			case PARAM_TYPE_STRING:
				if (dispatch->on_string)
					dispatch->on_string(parser->context, parser->string_parser.string);
				break;
			case PARAM_TYPE_BLOB:
			case PARAM_TYPE_BODY:
				if (dispatch->on_blob)
					dispatch->on_blob(parser->context, parser->blob_parser.data);
				break;
		}
	}
	if (parser->state != s_param_start)
	{
		parser->state = s_param_start;
		memset(parser->name, 0, sizeof(parser->name));
		parser->cursor = 0;
	}
}

static void param_type_parser_init(param_parser_t* parser)
{
	assert(parser->state >= 0);
	switch (parser->param_map[parser->state].type)
	{
		case PARAM_TYPE_INT:
		case PARAM_TYPE_ID:
		case PARAM_TYPE_FLOAT:
		case PARAM_TYPE_DOUBLE:
			numeric_parser_init(&parser->numeric_parser);
			break;
		case PARAM_TYPE_BOOL:
			bool_parser_init(&parser->bool_parser);
			break;
		case PARAM_TYPE_SIZE:
		case PARAM_TYPE_POINT:
			coord_parser_init(&parser->coord_parser);
			break;
		case PARAM_TYPE_STRING:
			string_parser_init(&parser->string_parser);
			break;
		case PARAM_TYPE_BLOB:
		case PARAM_TYPE_BODY:
			blob_parser_init(&parser->blob_parser);
			break;
	}
}

static void param_type_parser_execute(param_parser_t* parser, const char* buf, size_t len)
{
	assert(parser->state >= 0);
	switch (parser->param_map[parser->state].type)
	{
		case PARAM_TYPE_INT:
		case PARAM_TYPE_ID:
		case PARAM_TYPE_FLOAT:
		case PARAM_TYPE_DOUBLE:
			numeric_parser_execute(&parser->numeric_parser, buf, len);
			if (parser->numeric_parser.state == s_numeric_illegal)
				parser->state = s_param_skip;
			break;
		case PARAM_TYPE_BOOL:
			bool_parser_execute(&parser->bool_parser, buf, len);
			if (parser->bool_parser.state == s_bool_illegal)
				parser->state = s_param_skip;
			break;
		case PARAM_TYPE_SIZE:
		case PARAM_TYPE_POINT:
			coord_parser_execute(&parser->coord_parser, buf, len);
			if (parser->coord_parser.state == s_coord_illegal)
				parser->state = s_param_skip;
			break;
		case PARAM_TYPE_STRING:
			string_parser_execute(&parser->string_parser, buf, len);
			if (parser->string_parser.state == s_string_overflow)
				parser->state = s_param_skip;
			break;
		case PARAM_TYPE_BLOB:
		case PARAM_TYPE_BODY:
			blob_parser_execute(&parser->blob_parser, buf, len);
			break;
	}
}

static void on_form_data_name(void* context, const char* buf, size_t len)
{
	param_parser_t* parser = (param_parser_t*)context;
	if (len + parser->cursor > 31)
		return;
	memcpy(parser->name + parser->cursor, buf, len);
	parser->cursor += len;
}

static void on_query_string_field(void* context, const char* buf, size_t len, int header_index)
{
	param_parser_t* parser = (param_parser_t*)context;
	if (parser->header_index != header_index)
	{
		parser->header_index = header_index;
		// if header index doesn't match, reset the name copy
		parser->cursor = 0;
		memset(parser->name, 0, sizeof(parser->name));
		// terminate last query string
		param_parser_terminate(parser);
	}
	on_form_data_name(context, buf, len);
}

static void on_query_string_value(void* context, const char* buf, size_t len, int header_index)
{
	param_parser_t* parser = (param_parser_t*)context;
	if (parser->header_index == header_index)
	{
		if (parser->state == s_param_start)
		{
			parser->state = find_param_dispatch_state(parser, parser->name);
			if (parser->state >= 0)
				param_type_parser_init(parser);
		}
		if (parser->state >= 0)
			param_type_parser_execute(parser, buf, len);
	}
}

void param_parser_init(param_parser_t* parser, const param_dispatch_t* param_map, size_t len, void* parsed, void* context)
{
	form_data_parser_init(&parser->form_data_parser, parser);
	query_string_parser_init(&parser->query_string_parser, parser);
	parser->form_data_parser.on_name = on_form_data_name;
	parser->query_string_parser.on_field = on_query_string_field;
	parser->query_string_parser.on_value = on_query_string_value;
	parser->state = s_param_start;
	parser->param_map = param_map;
	parser->len = len;
	parser->parsed = (char*)parsed;
	parser->context = context;
	parser->cursor = 0;
	memset(parser->name, 0, sizeof(parser->name));
	// find out the special bodies that we cared about (PARAM_TYPE_BODY and PARAM_TYPE_ID)
	parser->body = parser->resource = s_param_start;
	int i;
	for (i = 0; i < len; i++)
		switch (param_map[i].type)
		{
			default:
				break;
			case PARAM_TYPE_ID:
				assert(parser->resource == s_param_start);
				*(int*)(parser->parsed + param_map[i].offset) = -1; // set id == -1 first.
				parser->resource = i;
				break;
			case PARAM_TYPE_BODY:
				assert(parser->body == s_param_start);
				parser->body = i;
				break;
		}
}

void param_parser_execute(param_parser_t* parser, int resource_id, const char* buf, size_t len, uri_parse_state_t state, int header_index)
{
	switch (state)
	{
		default:
			break;
		case URI_QUERY_STRING:
			query_string_parser_execute(&parser->query_string_parser, buf, len);
			break;
		case URI_CONTENT_BODY:
			if (parser->body == s_param_skip)
				break;
			if (parser->state != s_param_start && parser->state != parser->body)
				param_parser_terminate(parser);
			if (parser->state == s_param_start)
			{
				parser->state = parser->body;
				param_type_parser_init(parser);
			}
			param_type_parser_execute(parser, buf, len);
			break;
		case URI_PARSE_TERMINATE:
			if (parser->state != s_param_start)
				param_parser_terminate(parser); // collect result
			break;
		case URI_MULTIPART_HEADER_FIELD:
		case URI_MULTIPART_HEADER_VALUE:
			if (parser->state != s_param_start)
				param_parser_terminate(parser);
			assert(header_index >= 0);
			form_data_parser_execute(&parser->form_data_parser, buf, len, header_index);
			break;
		case URI_MULTIPART_DATA:
			if (parser->state == s_param_start)
			{
				parser->state = find_param_dispatch_state(parser, parser->name);
				if (parser->state >= 0)
					param_type_parser_init(parser);
			}
			if (parser->state >= 0)
				param_type_parser_execute(parser, buf, len);
			break;
	}
	if (resource_id >= 0 && parser->resource != s_param_start)
		*(int*)(parser->parsed + parser->param_map[parser->resource].offset) = resource_id;
}

ebb_buf param_parser_map_http_body(const param_dispatch_t* param_map, size_t len, const char* response_format)
{
	ebb_buf body;
	int i;
	static const char int_type[] = "integer";
	static const char bool_type[] = "boolean";
	static const char number_type[] = "number";
	static const char size_type[] = "size";
	static const char point_type[] = "point";
	static const char string_type[] = "string";
	static const char blob_type[] = "blob";
	size_t body_len = 12;
	for (i = 0; i < len; i++)
	{
		body_len += strlen(param_map[i].property) + 6;
		switch (param_map[i].type)
		{
			case PARAM_TYPE_INT:
			case PARAM_TYPE_ID:
				body_len += sizeof(int_type) - 1;
				break;
			case PARAM_TYPE_FLOAT:
			case PARAM_TYPE_DOUBLE:
				body_len += sizeof(number_type) - 1;
				break;
			case PARAM_TYPE_BOOL:
				body_len += sizeof(bool_type) - 1;
				break;
			case PARAM_TYPE_POINT:
				body_len += sizeof(point_type) - 1;
				break;
			case PARAM_TYPE_SIZE:
				body_len += sizeof(size_type) - 1;
				break;
			case PARAM_TYPE_STRING:
				body_len += sizeof(string_type) - 1;
				break;
			case PARAM_TYPE_BLOB:
			case PARAM_TYPE_BODY:
				body_len += sizeof(blob_type) - 1;
				break;
		}
	}
	if (response_format)
		body_len += 12 + strlen(response_format);
	body_len += 1;
	char* data = (char*)malloc(192 /* the head start for http header */ + body_len);
	snprintf(data, 192, ebb_http_header, body_len);
	body.written = strlen(data);
	memcpy(data + body.written, "{\"request\":{", 12);
	body.written += 12 + 1;
	for (i = 0; i < len; i++)
	{
		data[body.written - 1] = '"';
		size_t property_len = strlen(param_map[i].property);
		memcpy(data + body.written, param_map[i].property, property_len);
		body.written += property_len + 3;
		data[body.written - 3] = '"';
		data[body.written - 2] = ':';
		data[body.written - 1] = '"';
		switch (param_map[i].type)
		{
			case PARAM_TYPE_INT:
			case PARAM_TYPE_ID:
				memcpy(data + body.written, int_type, sizeof(int_type) - 1);
				body.written += sizeof(int_type) + 2;
				break;
			case PARAM_TYPE_FLOAT:
			case PARAM_TYPE_DOUBLE:
				memcpy(data + body.written, number_type, sizeof(number_type) - 1);
				body.written += sizeof(number_type) + 2;
				break;
			case PARAM_TYPE_BOOL:
				memcpy(data + body.written, bool_type, sizeof(bool_type) - 1);
				body.written += sizeof(bool_type) + 2;
				break;
			case PARAM_TYPE_POINT:
				memcpy(data + body.written, point_type, sizeof(point_type) - 1);
				body.written += sizeof(point_type) + 2;
				break;
			case PARAM_TYPE_SIZE:
				memcpy(data + body.written, size_type, sizeof(size_type) - 1);
				body.written += sizeof(size_type) + 2;
				break;
			case PARAM_TYPE_STRING:
				memcpy(data + body.written, string_type, sizeof(string_type) - 1);
				body.written += sizeof(string_type) + 2;
				break;
			case PARAM_TYPE_BLOB:
			case PARAM_TYPE_BODY:
				memcpy(data + body.written, blob_type, sizeof(blob_type) - 1);
				body.written += sizeof(blob_type) + 2;
				break;
		}
		data[body.written - 3] = '"';
		data[body.written - 2] = (i == len - 1) ? '}' : ',';
	}
	if (response_format)
	{
		memcpy(data + body.written - 1, ",\"response\":", 12);
		body.written += 11;
		size_t response_len = strlen(response_format);
		memcpy(data + body.written, response_format, response_len);
		body.written += response_len + 1;
	}
	data[body.written - 1] = '}';
	data[body.written] = '\n';
	body.len = body.written + 1;
	body.data = data;
	body.on_release = 0;
	return body;
}
