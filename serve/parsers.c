#include "uri.h"
#include <string.h>
#include <ctype.h>
#include <stdio.h>

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
					parser->state = s_form_data_end; // the end of quote, return
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

void numeric_parser_init(numeric_parser_t* parser)
{
	parser->state = s_numeric_start;
	parser->result = 0;
	parser->division = 10;
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
				parser->result = digit;
				parser->state = s_numeric_before_decimal;
				break;
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
					parser->result += digit / parser->division;
					parser->division *= 10;
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
}

void coord_parser_execute(coord_parser_t* parser, const char* buf, size_t len)
{
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
		parser->state = s_string_overflow;
	else if (parser->state == s_string_start) {
		memcpy(parser->string + parser->cursor, buf, len);
		parser->cursor += len;
	}
}
