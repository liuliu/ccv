
#line 1 "ebb_request_parser.rl"
/* HTTP/1.1 Parser
 * Copyright 2008 ryah dahl, ry at tiny clouds punkt org
 *
 * Based on Zed Shaw's parser for Mongrel.
 * Copyright (c) 2005 Zed A. Shaw
 *
 * This software may be distributed under the "MIT" license included in the
 * README
 */
#include "ebb_request_parser.h"

#include <stdio.h>
#include <ctype.h>
#include <assert.h>

#define TRUE 1
#define FALSE 0
#define MIN(a,b) (a < b ? a : b)

#define REMAINING (pe - p)
#define CURRENT (parser->current_request)
#define CONTENT_LENGTH (parser->current_request->content_length)

#define LEN(FROM) (p - parser->FROM##_mark)
#define CALLBACK(FOR)                               \
  if(parser->FOR##_mark && CURRENT->on_##FOR) {     \
    CURRENT->on_##FOR( CURRENT                      \
                , parser->FOR##_mark                \
                , p - parser->FOR##_mark            \
                );                                  \
  }
#define HEADER_CALLBACK(FOR)                        \
  if(parser->FOR##_mark && CURRENT->on_##FOR) {     \
    CURRENT->on_##FOR( CURRENT                      \
                , parser->FOR##_mark                \
                , p - parser->FOR##_mark            \
                , CURRENT->number_of_headers        \
                );                                  \
  }

#define EMIT_HEADER_CB(FOR, ptr, len)               \
  if (CURRENT->on_##FOR) {                          \
    CURRENT->on_##FOR(CURRENT, ptr, len,            \
      CURRENT->number_of_multipart_headers);        \
  }
#define EMIT_DATA_CB(FOR, ptr, len)                 \
  if (CURRENT->on_##FOR) {                          \
    CURRENT->on_##FOR(CURRENT, ptr, len);           \
  }

#define END_REQUEST                       \
    if(CURRENT->on_complete)              \
      CURRENT->on_complete(CURRENT);      \
    CURRENT = NULL;


#line 381 "ebb_request_parser.rl"



#line 64 "ebb_request_parser.c"
static const int ebb_request_parser_start = 1;
static const int ebb_request_parser_first_final = 251;
static const int ebb_request_parser_error = 0;

static const int ebb_request_parser_en_ChunkedBody = 233;
static const int ebb_request_parser_en_ChunkedBody_chunk_chunk_end = 243;
static const int ebb_request_parser_en_main = 1;


#line 384 "ebb_request_parser.rl"

#define COPYSTACK(dest, src)  for(i = 0; i < EBB_RAGEL_STACK_SIZE; i++) { dest[i] = src[i]; }

enum multipart_state {
  s_uninitialized = 1,
  s_start,
  s_start_boundary,
  s_header_field_start,
  s_header_field,
  s_headers_almost_done,
  s_header_value_start,
  s_header_value,
  s_header_value_almost_done,
  s_part_data_start,
  s_part_data,
  s_part_data_almost_boundary,
  s_part_data_boundary,
  s_part_data_almost_end,
  s_part_data_end,
  s_part_data_final_hyphen,
  s_end
};

void ebb_request_parser_init(ebb_request_parser *parser) 
{
  int i;

  int cs = 0;
  int top = 0;
  int stack[EBB_RAGEL_STACK_SIZE];
  
#line 106 "ebb_request_parser.c"
	{
	cs = ebb_request_parser_start;
	top = 0;
	}

#line 415 "ebb_request_parser.rl"
  parser->cs = cs;
  parser->top = top;
  COPYSTACK(parser->stack, stack);

  parser->chunk_size = 0;
  parser->eating = 0;
  
  parser->current_request = NULL;

  parser->header_field_mark = parser->header_value_mark   = 
  parser->query_string_mark = parser->path_mark           = 
  parser->uri_mark          = parser->fragment_mark       = NULL;

  parser->multipart_state = s_uninitialized;
  parser->new_request = NULL;
}

#define LF 10
#define CR 13

size_t multipart_parser_execute(ebb_request_parser* parser, const char *buf, size_t len)
{
  size_t i = 0;
  size_t mark = 0;
  char c, cl;
  int is_last = 0;

  while(!is_last) {
    c = buf[i];
    is_last = (i == (len - 1));
    switch (parser->multipart_state) {
      case s_start:
		CURRENT->number_of_multipart_headers = 0;
        parser->multipart_index = 0;
        parser->multipart_state = s_start_boundary;

      /* fallthrough */
      case s_start_boundary:
	    // every time needs to take into account the first two '-'
        if (parser->multipart_index == CURRENT->multipart_boundary_len + 2) {
          if (c != CR) {
            return i;
          }
          parser->multipart_index++;
          break;
        } else if (parser->multipart_index == (CURRENT->multipart_boundary_len + 3)) {
          if (c != LF) {
            return i;
          }
		  CURRENT->number_of_multipart_headers = 0;
          parser->multipart_index = 0;
          parser->multipart_state = s_header_field_start;
          break;
        }
        if (c != CURRENT->multipart_boundary[parser->multipart_index]) {
          return i;
        }
        parser->multipart_index++;
        break;

      case s_header_field_start:
        mark = i;
        parser->multipart_state = s_header_field;

      /* fallthrough */
      case s_header_field:
        if (c == CR) {
          parser->multipart_state = s_headers_almost_done;
          break;
        }

        if (c == '-') {
          break;
        }

        if (c == ':') {
          EMIT_HEADER_CB(multipart_header_field, buf + mark, i - mark);
          parser->multipart_state = s_header_value_start;
          break;
        }

        cl = tolower(c);
        if (cl < 'a' || cl > 'z') {
          return i;
        }
        if (is_last)
          EMIT_HEADER_CB(multipart_header_field, buf + mark, (i - mark) + 1);
        break;

      case s_headers_almost_done:
        if (c != LF) {
          return i;
        }

        parser->multipart_state = s_part_data_start;
        break;

      case s_header_value_start:
        if (c == ' ') {
          break;
        }

        mark = i;
        parser->multipart_state = s_header_value;

      /* fallthrough */
      case s_header_value:
        if (c == CR) {
          EMIT_HEADER_CB(multipart_header_value, buf + mark, i - mark);
          parser->multipart_state = s_header_value_almost_done;
        }
        if (is_last)
          EMIT_HEADER_CB(multipart_header_value, buf + mark, (i - mark) + 1);
        break;

      case s_header_value_almost_done:
        if (c != LF) {
          return i;
        }
		CURRENT->number_of_multipart_headers++;
        parser->multipart_state = s_header_field_start;
        break;

      case s_part_data_start:
        if (CURRENT->on_multipart_headers_complete)
		  CURRENT->on_multipart_headers_complete(CURRENT);
        mark = i;
        parser->multipart_state = s_part_data;

      /* fallthrough */
      case s_part_data:
        if (c == CR) {
          EMIT_DATA_CB(part_data, buf + mark, i - mark);
          mark = i;
          parser->multipart_state = s_part_data_almost_boundary;
          parser->multipart_lookbehind[0] = CR;
          break;
        }
        if (is_last)
          EMIT_DATA_CB(part_data, buf + mark, (i - mark) + 1);
        break;

      case s_part_data_almost_boundary:
        if (c == LF) {
          parser->multipart_state = s_part_data_boundary;
          parser->multipart_lookbehind[1] = LF;
		  CURRENT->number_of_multipart_headers = 0;
          parser->multipart_index = 0;
          break;
        }
        EMIT_DATA_CB(part_data, parser->multipart_lookbehind, 1);
        parser->multipart_state = s_part_data;
        mark = i --;
        break;

      case s_part_data_boundary:
        if (CURRENT->multipart_boundary[parser->multipart_index] != c) {
          EMIT_DATA_CB(part_data, parser->multipart_lookbehind, 2 + parser->multipart_index);
          parser->multipart_state = s_part_data;
          mark = i --;
          break;
        }
        parser->multipart_lookbehind[2 + parser->multipart_index] = c;
        if ((++ parser->multipart_index) == CURRENT->multipart_boundary_len + 2) {
          if (CURRENT->on_part_data_complete)
		    CURRENT->on_part_data_complete(CURRENT);
          parser->multipart_state = s_part_data_almost_end;
        }
        break;

      case s_part_data_almost_end:
        if (c == '-') {
          parser->multipart_state = s_part_data_final_hyphen;
          break;
        }
        if (c == CR) {
          parser->multipart_state = s_part_data_end;
          break;
        }
        return i;
   
      case s_part_data_final_hyphen:
        if (c == '-') {
          parser->multipart_state = s_end;
          break;
        }
        return i;

      case s_part_data_end:
        if (c == LF) {
          parser->multipart_state = s_header_field_start;
          break;
        }
        return i;

      case s_end:
        break;

      default:
        return 0;
    }
    ++i;
  }

  return len;
}

/** exec **/
size_t ebb_request_parser_execute(ebb_request_parser *parser, const char *buffer, size_t len)
{
  const char *p, *pe;
  int i, cs = parser->cs;

  int top = parser->top;
  int stack[EBB_RAGEL_STACK_SIZE];
  COPYSTACK(stack, parser->stack);

  assert(parser->new_request && "undefined callback");

  p = buffer;
  pe = buffer+len;

  if(0 < parser->chunk_size && parser->eating) {
    /*
     *
     * eat chunked body
     * 
     */
    //printf("eat chunk body (before parse)\n");
    size_t eat = MIN(len, parser->chunk_size);
    if(eat == parser->chunk_size) {
      parser->eating = FALSE;
    }
	if (CURRENT->multipart_boundary_len > 0)
	  multipart_parser_execute(parser, p, eat);
    CURRENT->on_body(CURRENT, p, eat);
    p += eat;
    parser->chunk_size -= eat;
    //printf("eat: %d\n", eat);
  } else if( parser->current_request && CURRENT->eating_body ) {
    /*
     *
     * eat normal body
     * 
     */
    //printf("eat normal body (before parse)\n");
    size_t eat = MIN(len, CURRENT->content_length - CURRENT->body_read);
	if (CURRENT->multipart_boundary_len > 0)
	  multipart_parser_execute(parser, p, eat);
    CURRENT->on_body(CURRENT, p, eat);
    p += eat;
    CURRENT->body_read += eat;

    if(CURRENT->body_read == CURRENT->content_length) {
      END_REQUEST;
    }
  }

  if(parser->header_field_mark)   parser->header_field_mark   = buffer;
  if(parser->header_value_mark)   parser->header_value_mark   = buffer;
  if(parser->fragment_mark)       parser->fragment_mark       = buffer;
  if(parser->query_string_mark)   parser->query_string_mark   = buffer;
  if(parser->path_mark)           parser->path_mark           = buffer;
  if(parser->uri_mark)            parser->uri_mark            = buffer;

  
#line 379 "ebb_request_parser.c"
	{
	if ( p == pe )
		goto _test_eof;
	goto _resume;

_again:
	switch ( cs ) {
		case 1: goto st1;
		case 0: goto st0;
		case 2: goto st2;
		case 3: goto st3;
		case 4: goto st4;
		case 5: goto st5;
		case 6: goto st6;
		case 7: goto st7;
		case 8: goto st8;
		case 9: goto st9;
		case 10: goto st10;
		case 11: goto st11;
		case 12: goto st12;
		case 13: goto st13;
		case 14: goto st14;
		case 15: goto st15;
		case 16: goto st16;
		case 17: goto st17;
		case 18: goto st18;
		case 19: goto st19;
		case 251: goto st251;
		case 20: goto st20;
		case 21: goto st21;
		case 22: goto st22;
		case 23: goto st23;
		case 24: goto st24;
		case 25: goto st25;
		case 26: goto st26;
		case 27: goto st27;
		case 28: goto st28;
		case 29: goto st29;
		case 30: goto st30;
		case 31: goto st31;
		case 32: goto st32;
		case 33: goto st33;
		case 34: goto st34;
		case 35: goto st35;
		case 36: goto st36;
		case 37: goto st37;
		case 38: goto st38;
		case 39: goto st39;
		case 40: goto st40;
		case 41: goto st41;
		case 42: goto st42;
		case 43: goto st43;
		case 44: goto st44;
		case 45: goto st45;
		case 46: goto st46;
		case 47: goto st47;
		case 48: goto st48;
		case 49: goto st49;
		case 50: goto st50;
		case 51: goto st51;
		case 52: goto st52;
		case 53: goto st53;
		case 54: goto st54;
		case 55: goto st55;
		case 56: goto st56;
		case 57: goto st57;
		case 58: goto st58;
		case 59: goto st59;
		case 60: goto st60;
		case 61: goto st61;
		case 62: goto st62;
		case 63: goto st63;
		case 64: goto st64;
		case 65: goto st65;
		case 66: goto st66;
		case 67: goto st67;
		case 68: goto st68;
		case 69: goto st69;
		case 70: goto st70;
		case 71: goto st71;
		case 72: goto st72;
		case 73: goto st73;
		case 74: goto st74;
		case 75: goto st75;
		case 76: goto st76;
		case 77: goto st77;
		case 78: goto st78;
		case 79: goto st79;
		case 80: goto st80;
		case 81: goto st81;
		case 82: goto st82;
		case 83: goto st83;
		case 84: goto st84;
		case 85: goto st85;
		case 86: goto st86;
		case 87: goto st87;
		case 88: goto st88;
		case 89: goto st89;
		case 90: goto st90;
		case 91: goto st91;
		case 92: goto st92;
		case 93: goto st93;
		case 94: goto st94;
		case 95: goto st95;
		case 96: goto st96;
		case 97: goto st97;
		case 98: goto st98;
		case 99: goto st99;
		case 100: goto st100;
		case 101: goto st101;
		case 102: goto st102;
		case 103: goto st103;
		case 104: goto st104;
		case 105: goto st105;
		case 106: goto st106;
		case 107: goto st107;
		case 108: goto st108;
		case 109: goto st109;
		case 110: goto st110;
		case 111: goto st111;
		case 112: goto st112;
		case 113: goto st113;
		case 114: goto st114;
		case 115: goto st115;
		case 116: goto st116;
		case 117: goto st117;
		case 118: goto st118;
		case 119: goto st119;
		case 120: goto st120;
		case 121: goto st121;
		case 122: goto st122;
		case 123: goto st123;
		case 124: goto st124;
		case 125: goto st125;
		case 126: goto st126;
		case 127: goto st127;
		case 128: goto st128;
		case 129: goto st129;
		case 130: goto st130;
		case 131: goto st131;
		case 132: goto st132;
		case 133: goto st133;
		case 134: goto st134;
		case 135: goto st135;
		case 136: goto st136;
		case 137: goto st137;
		case 138: goto st138;
		case 139: goto st139;
		case 140: goto st140;
		case 141: goto st141;
		case 142: goto st142;
		case 143: goto st143;
		case 144: goto st144;
		case 145: goto st145;
		case 146: goto st146;
		case 147: goto st147;
		case 148: goto st148;
		case 149: goto st149;
		case 150: goto st150;
		case 151: goto st151;
		case 152: goto st152;
		case 153: goto st153;
		case 154: goto st154;
		case 155: goto st155;
		case 156: goto st156;
		case 157: goto st157;
		case 158: goto st158;
		case 159: goto st159;
		case 160: goto st160;
		case 161: goto st161;
		case 162: goto st162;
		case 163: goto st163;
		case 164: goto st164;
		case 165: goto st165;
		case 166: goto st166;
		case 167: goto st167;
		case 168: goto st168;
		case 169: goto st169;
		case 170: goto st170;
		case 171: goto st171;
		case 172: goto st172;
		case 173: goto st173;
		case 174: goto st174;
		case 175: goto st175;
		case 176: goto st176;
		case 177: goto st177;
		case 178: goto st178;
		case 179: goto st179;
		case 180: goto st180;
		case 181: goto st181;
		case 182: goto st182;
		case 183: goto st183;
		case 184: goto st184;
		case 185: goto st185;
		case 186: goto st186;
		case 187: goto st187;
		case 188: goto st188;
		case 189: goto st189;
		case 190: goto st190;
		case 191: goto st191;
		case 192: goto st192;
		case 193: goto st193;
		case 194: goto st194;
		case 195: goto st195;
		case 196: goto st196;
		case 197: goto st197;
		case 198: goto st198;
		case 199: goto st199;
		case 200: goto st200;
		case 201: goto st201;
		case 202: goto st202;
		case 203: goto st203;
		case 204: goto st204;
		case 205: goto st205;
		case 206: goto st206;
		case 207: goto st207;
		case 208: goto st208;
		case 209: goto st209;
		case 210: goto st210;
		case 211: goto st211;
		case 212: goto st212;
		case 213: goto st213;
		case 214: goto st214;
		case 215: goto st215;
		case 216: goto st216;
		case 217: goto st217;
		case 218: goto st218;
		case 219: goto st219;
		case 220: goto st220;
		case 221: goto st221;
		case 222: goto st222;
		case 223: goto st223;
		case 224: goto st224;
		case 225: goto st225;
		case 226: goto st226;
		case 227: goto st227;
		case 228: goto st228;
		case 229: goto st229;
		case 230: goto st230;
		case 231: goto st231;
		case 232: goto st232;
		case 233: goto st233;
		case 234: goto st234;
		case 235: goto st235;
		case 236: goto st236;
		case 237: goto st237;
		case 252: goto st252;
		case 238: goto st238;
		case 239: goto st239;
		case 240: goto st240;
		case 241: goto st241;
		case 242: goto st242;
		case 243: goto st243;
		case 244: goto st244;
		case 245: goto st245;
		case 246: goto st246;
		case 247: goto st247;
		case 248: goto st248;
		case 249: goto st249;
		case 250: goto st250;
	default: break;
	}

	if ( ++p == pe )
		goto _test_eof;
_resume:
	switch ( cs )
	{
st1:
	if ( ++p == pe )
		goto _test_eof1;
case 1:
	switch( (*p) ) {
		case 67: goto tr0;
		case 68: goto tr2;
		case 71: goto tr3;
		case 72: goto tr4;
		case 76: goto tr5;
		case 77: goto tr6;
		case 79: goto tr7;
		case 80: goto tr8;
		case 84: goto tr9;
		case 85: goto tr10;
	}
	goto st0;
st0:
cs = 0;
	goto _out;
tr0:
#line 205 "ebb_request_parser.rl"
	{
    assert(CURRENT == NULL);
    CURRENT = parser->new_request(parser->data);
  }
	goto st2;
st2:
	if ( ++p == pe )
		goto _test_eof2;
case 2:
#line 679 "ebb_request_parser.c"
	if ( (*p) == 79 )
		goto st3;
	goto st0;
st3:
	if ( ++p == pe )
		goto _test_eof3;
case 3:
	if ( (*p) == 80 )
		goto st4;
	goto st0;
st4:
	if ( ++p == pe )
		goto _test_eof4;
case 4:
	if ( (*p) == 89 )
		goto st5;
	goto st0;
st5:
	if ( ++p == pe )
		goto _test_eof5;
case 5:
	if ( (*p) == 32 )
		goto tr14;
	goto st0;
tr14:
#line 66 "ebb_request_parser.rl"
	{ CURRENT->method = EBB_COPY;      }
	goto st6;
tr42:
#line 67 "ebb_request_parser.rl"
	{ CURRENT->method = EBB_DELETE;    }
	goto st6;
tr45:
#line 68 "ebb_request_parser.rl"
	{ CURRENT->method = EBB_GET;       }
	goto st6;
tr49:
#line 69 "ebb_request_parser.rl"
	{ CURRENT->method = EBB_HEAD;      }
	goto st6;
tr53:
#line 70 "ebb_request_parser.rl"
	{ CURRENT->method = EBB_LOCK;      }
	goto st6;
tr59:
#line 71 "ebb_request_parser.rl"
	{ CURRENT->method = EBB_MKCOL;     }
	goto st6;
tr62:
#line 72 "ebb_request_parser.rl"
	{ CURRENT->method = EBB_MOVE;      }
	goto st6;
tr69:
#line 73 "ebb_request_parser.rl"
	{ CURRENT->method = EBB_OPTIONS;   }
	goto st6;
tr75:
#line 74 "ebb_request_parser.rl"
	{ CURRENT->method = EBB_POST;      }
	goto st6;
tr83:
#line 75 "ebb_request_parser.rl"
	{ CURRENT->method = EBB_PROPFIND;  }
	goto st6;
tr88:
#line 76 "ebb_request_parser.rl"
	{ CURRENT->method = EBB_PROPPATCH; }
	goto st6;
tr90:
#line 77 "ebb_request_parser.rl"
	{ CURRENT->method = EBB_PUT;       }
	goto st6;
tr95:
#line 78 "ebb_request_parser.rl"
	{ CURRENT->method = EBB_TRACE;     }
	goto st6;
tr101:
#line 79 "ebb_request_parser.rl"
	{ CURRENT->method = EBB_UNLOCK;    }
	goto st6;
st6:
	if ( ++p == pe )
		goto _test_eof6;
case 6:
#line 764 "ebb_request_parser.c"
	switch( (*p) ) {
		case 42: goto tr15;
		case 43: goto tr16;
		case 47: goto tr17;
		case 58: goto tr18;
	}
	if ( (*p) < 65 ) {
		if ( 45 <= (*p) && (*p) <= 57 )
			goto tr16;
	} else if ( (*p) > 90 ) {
		if ( 97 <= (*p) && (*p) <= 122 )
			goto tr16;
	} else
		goto tr16;
	goto st0;
tr15:
#line 64 "ebb_request_parser.rl"
	{ parser->uri_mark            = p; }
	goto st7;
st7:
	if ( ++p == pe )
		goto _test_eof7;
case 7:
#line 788 "ebb_request_parser.c"
	switch( (*p) ) {
		case 32: goto tr19;
		case 35: goto tr20;
	}
	goto st0;
tr19:
#line 93 "ebb_request_parser.rl"
	{ 
    //printf("request uri\n");
    CALLBACK(uri);
    parser->uri_mark = NULL;
  }
	goto st8;
tr260:
#line 61 "ebb_request_parser.rl"
	{ parser->fragment_mark       = p; }
#line 99 "ebb_request_parser.rl"
	{ 
    //printf("fragment\n");
    CALLBACK(fragment);
    parser->fragment_mark = NULL;
  }
	goto st8;
tr263:
#line 99 "ebb_request_parser.rl"
	{ 
    //printf("fragment\n");
    CALLBACK(fragment);
    parser->fragment_mark = NULL;
  }
	goto st8;
tr271:
#line 111 "ebb_request_parser.rl"
	{
    //printf("request path\n");
    CALLBACK(path);
    parser->path_mark = NULL;
  }
#line 93 "ebb_request_parser.rl"
	{ 
    //printf("request uri\n");
    CALLBACK(uri);
    parser->uri_mark = NULL;
  }
	goto st8;
tr277:
#line 62 "ebb_request_parser.rl"
	{ parser->query_string_mark   = p; }
#line 105 "ebb_request_parser.rl"
	{ 
    //printf("query  string\n");
    CALLBACK(query_string);
    parser->query_string_mark = NULL;
  }
#line 93 "ebb_request_parser.rl"
	{ 
    //printf("request uri\n");
    CALLBACK(uri);
    parser->uri_mark = NULL;
  }
	goto st8;
tr281:
#line 105 "ebb_request_parser.rl"
	{ 
    //printf("query  string\n");
    CALLBACK(query_string);
    parser->query_string_mark = NULL;
  }
#line 93 "ebb_request_parser.rl"
	{ 
    //printf("request uri\n");
    CALLBACK(uri);
    parser->uri_mark = NULL;
  }
	goto st8;
st8:
	if ( ++p == pe )
		goto _test_eof8;
case 8:
#line 868 "ebb_request_parser.c"
	if ( (*p) == 72 )
		goto st9;
	goto st0;
st9:
	if ( ++p == pe )
		goto _test_eof9;
case 9:
	if ( (*p) == 84 )
		goto st10;
	goto st0;
st10:
	if ( ++p == pe )
		goto _test_eof10;
case 10:
	if ( (*p) == 84 )
		goto st11;
	goto st0;
st11:
	if ( ++p == pe )
		goto _test_eof11;
case 11:
	if ( (*p) == 80 )
		goto st12;
	goto st0;
st12:
	if ( ++p == pe )
		goto _test_eof12;
case 12:
	if ( (*p) == 47 )
		goto st13;
	goto st0;
st13:
	if ( ++p == pe )
		goto _test_eof13;
case 13:
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr26;
	goto st0;
tr26:
#line 147 "ebb_request_parser.rl"
	{
    CURRENT->version_major *= 10;
    CURRENT->version_major += *p - '0';
  }
	goto st14;
st14:
	if ( ++p == pe )
		goto _test_eof14;
case 14:
#line 918 "ebb_request_parser.c"
	if ( (*p) == 46 )
		goto st15;
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr26;
	goto st0;
st15:
	if ( ++p == pe )
		goto _test_eof15;
case 15:
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr28;
	goto st0;
tr28:
#line 152 "ebb_request_parser.rl"
	{
    CURRENT->version_minor *= 10;
    CURRENT->version_minor += *p - '0';
  }
	goto st16;
st16:
	if ( ++p == pe )
		goto _test_eof16;
case 16:
#line 942 "ebb_request_parser.c"
	if ( (*p) == 13 )
		goto st17;
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr28;
	goto st0;
st17:
	if ( ++p == pe )
		goto _test_eof17;
case 17:
	if ( (*p) == 10 )
		goto st18;
	goto st0;
st18:
	if ( ++p == pe )
		goto _test_eof18;
case 18:
	switch( (*p) ) {
		case 13: goto st19;
		case 33: goto tr32;
		case 67: goto tr33;
		case 69: goto tr34;
		case 84: goto tr35;
		case 99: goto tr33;
		case 101: goto tr34;
		case 116: goto tr35;
		case 124: goto tr32;
		case 126: goto tr32;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto tr32;
		} else if ( (*p) >= 35 )
			goto tr32;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto tr32;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto tr32;
		} else
			goto tr32;
	} else
		goto tr32;
	goto st0;
tr110:
#line 157 "ebb_request_parser.rl"
	{
    CURRENT->number_of_headers++;
  }
	goto st19;
st19:
	if ( ++p == pe )
		goto _test_eof19;
case 19:
#line 999 "ebb_request_parser.c"
	if ( (*p) == 10 )
		goto tr36;
	goto st0;
tr36:
#line 161 "ebb_request_parser.rl"
	{
    if(CURRENT->on_headers_complete)
      CURRENT->on_headers_complete(CURRENT);
  }
#line 210 "ebb_request_parser.rl"
	{
    if(CURRENT->transfer_encoding == EBB_CHUNKED) {
      {stack[top++] = 251; goto st233;}
    } else {
      /*
       * EAT BODY
       * this is very ugly. sorry.
       *
       */
      if( CURRENT->content_length == 0) {

        END_REQUEST;

      } else if( CURRENT->content_length < REMAINING ) {
        /* 
         * 
         * FINISH EATING THE BODY. there is still more 
         * on the buffer - so we just let it continue
         * parsing after we're done
         *
         */
        p += 1;
	    if( CURRENT->multipart_boundary_len > 0 )
	      multipart_parser_execute(parser, p, CURRENT->content_length);
        if( CURRENT->on_body )
          CURRENT->on_body(CURRENT, p, CURRENT->content_length); 

        p += CURRENT->content_length;
        CURRENT->body_read = CURRENT->content_length;

        assert(0 <= REMAINING);

        END_REQUEST;

        p--;

      } else {
        /* 
         * The body is larger than the buffer
         * EAT REST OF BUFFER
         * there is still more to read though. this will  
         * be handled on the next invokion of ebb_request_parser_execute
         * right before we enter the state machine. 
         *
         */
        p += 1;
        size_t eat = REMAINING;

	    if( CURRENT->multipart_boundary_len > 0 && eat > 0 )
	      multipart_parser_execute(parser, p, eat);
        if( CURRENT->on_body && eat > 0)
          CURRENT->on_body(CURRENT, p, eat); 

        p += eat;
        CURRENT->body_read += eat;
        CURRENT->eating_body = TRUE;
        //printf("eating body!\n");

        assert(CURRENT->body_read < CURRENT->content_length);
        assert(REMAINING == 0);
        
        p--; {p++; cs = 251; goto _out;}  
      }
    }
  }
	goto st251;
st251:
	if ( ++p == pe )
		goto _test_eof251;
case 251:
#line 1080 "ebb_request_parser.c"
	switch( (*p) ) {
		case 67: goto tr0;
		case 68: goto tr2;
		case 71: goto tr3;
		case 72: goto tr4;
		case 76: goto tr5;
		case 77: goto tr6;
		case 79: goto tr7;
		case 80: goto tr8;
		case 84: goto tr9;
		case 85: goto tr10;
	}
	goto st0;
tr2:
#line 205 "ebb_request_parser.rl"
	{
    assert(CURRENT == NULL);
    CURRENT = parser->new_request(parser->data);
  }
	goto st20;
st20:
	if ( ++p == pe )
		goto _test_eof20;
case 20:
#line 1105 "ebb_request_parser.c"
	if ( (*p) == 69 )
		goto st21;
	goto st0;
st21:
	if ( ++p == pe )
		goto _test_eof21;
case 21:
	if ( (*p) == 76 )
		goto st22;
	goto st0;
st22:
	if ( ++p == pe )
		goto _test_eof22;
case 22:
	if ( (*p) == 69 )
		goto st23;
	goto st0;
st23:
	if ( ++p == pe )
		goto _test_eof23;
case 23:
	if ( (*p) == 84 )
		goto st24;
	goto st0;
st24:
	if ( ++p == pe )
		goto _test_eof24;
case 24:
	if ( (*p) == 69 )
		goto st25;
	goto st0;
st25:
	if ( ++p == pe )
		goto _test_eof25;
case 25:
	if ( (*p) == 32 )
		goto tr42;
	goto st0;
tr3:
#line 205 "ebb_request_parser.rl"
	{
    assert(CURRENT == NULL);
    CURRENT = parser->new_request(parser->data);
  }
	goto st26;
st26:
	if ( ++p == pe )
		goto _test_eof26;
case 26:
#line 1155 "ebb_request_parser.c"
	if ( (*p) == 69 )
		goto st27;
	goto st0;
st27:
	if ( ++p == pe )
		goto _test_eof27;
case 27:
	if ( (*p) == 84 )
		goto st28;
	goto st0;
st28:
	if ( ++p == pe )
		goto _test_eof28;
case 28:
	if ( (*p) == 32 )
		goto tr45;
	goto st0;
tr4:
#line 205 "ebb_request_parser.rl"
	{
    assert(CURRENT == NULL);
    CURRENT = parser->new_request(parser->data);
  }
	goto st29;
st29:
	if ( ++p == pe )
		goto _test_eof29;
case 29:
#line 1184 "ebb_request_parser.c"
	if ( (*p) == 69 )
		goto st30;
	goto st0;
st30:
	if ( ++p == pe )
		goto _test_eof30;
case 30:
	if ( (*p) == 65 )
		goto st31;
	goto st0;
st31:
	if ( ++p == pe )
		goto _test_eof31;
case 31:
	if ( (*p) == 68 )
		goto st32;
	goto st0;
st32:
	if ( ++p == pe )
		goto _test_eof32;
case 32:
	if ( (*p) == 32 )
		goto tr49;
	goto st0;
tr5:
#line 205 "ebb_request_parser.rl"
	{
    assert(CURRENT == NULL);
    CURRENT = parser->new_request(parser->data);
  }
	goto st33;
st33:
	if ( ++p == pe )
		goto _test_eof33;
case 33:
#line 1220 "ebb_request_parser.c"
	if ( (*p) == 79 )
		goto st34;
	goto st0;
st34:
	if ( ++p == pe )
		goto _test_eof34;
case 34:
	if ( (*p) == 67 )
		goto st35;
	goto st0;
st35:
	if ( ++p == pe )
		goto _test_eof35;
case 35:
	if ( (*p) == 75 )
		goto st36;
	goto st0;
st36:
	if ( ++p == pe )
		goto _test_eof36;
case 36:
	if ( (*p) == 32 )
		goto tr53;
	goto st0;
tr6:
#line 205 "ebb_request_parser.rl"
	{
    assert(CURRENT == NULL);
    CURRENT = parser->new_request(parser->data);
  }
	goto st37;
st37:
	if ( ++p == pe )
		goto _test_eof37;
case 37:
#line 1256 "ebb_request_parser.c"
	switch( (*p) ) {
		case 75: goto st38;
		case 79: goto st42;
	}
	goto st0;
st38:
	if ( ++p == pe )
		goto _test_eof38;
case 38:
	if ( (*p) == 67 )
		goto st39;
	goto st0;
st39:
	if ( ++p == pe )
		goto _test_eof39;
case 39:
	if ( (*p) == 79 )
		goto st40;
	goto st0;
st40:
	if ( ++p == pe )
		goto _test_eof40;
case 40:
	if ( (*p) == 76 )
		goto st41;
	goto st0;
st41:
	if ( ++p == pe )
		goto _test_eof41;
case 41:
	if ( (*p) == 32 )
		goto tr59;
	goto st0;
st42:
	if ( ++p == pe )
		goto _test_eof42;
case 42:
	if ( (*p) == 86 )
		goto st43;
	goto st0;
st43:
	if ( ++p == pe )
		goto _test_eof43;
case 43:
	if ( (*p) == 69 )
		goto st44;
	goto st0;
st44:
	if ( ++p == pe )
		goto _test_eof44;
case 44:
	if ( (*p) == 32 )
		goto tr62;
	goto st0;
tr7:
#line 205 "ebb_request_parser.rl"
	{
    assert(CURRENT == NULL);
    CURRENT = parser->new_request(parser->data);
  }
	goto st45;
st45:
	if ( ++p == pe )
		goto _test_eof45;
case 45:
#line 1322 "ebb_request_parser.c"
	if ( (*p) == 80 )
		goto st46;
	goto st0;
st46:
	if ( ++p == pe )
		goto _test_eof46;
case 46:
	if ( (*p) == 84 )
		goto st47;
	goto st0;
st47:
	if ( ++p == pe )
		goto _test_eof47;
case 47:
	if ( (*p) == 73 )
		goto st48;
	goto st0;
st48:
	if ( ++p == pe )
		goto _test_eof48;
case 48:
	if ( (*p) == 79 )
		goto st49;
	goto st0;
st49:
	if ( ++p == pe )
		goto _test_eof49;
case 49:
	if ( (*p) == 78 )
		goto st50;
	goto st0;
st50:
	if ( ++p == pe )
		goto _test_eof50;
case 50:
	if ( (*p) == 83 )
		goto st51;
	goto st0;
st51:
	if ( ++p == pe )
		goto _test_eof51;
case 51:
	if ( (*p) == 32 )
		goto tr69;
	goto st0;
tr8:
#line 205 "ebb_request_parser.rl"
	{
    assert(CURRENT == NULL);
    CURRENT = parser->new_request(parser->data);
  }
	goto st52;
st52:
	if ( ++p == pe )
		goto _test_eof52;
case 52:
#line 1379 "ebb_request_parser.c"
	switch( (*p) ) {
		case 79: goto st53;
		case 82: goto st56;
		case 85: goto st68;
	}
	goto st0;
st53:
	if ( ++p == pe )
		goto _test_eof53;
case 53:
	if ( (*p) == 83 )
		goto st54;
	goto st0;
st54:
	if ( ++p == pe )
		goto _test_eof54;
case 54:
	if ( (*p) == 84 )
		goto st55;
	goto st0;
st55:
	if ( ++p == pe )
		goto _test_eof55;
case 55:
	if ( (*p) == 32 )
		goto tr75;
	goto st0;
st56:
	if ( ++p == pe )
		goto _test_eof56;
case 56:
	if ( (*p) == 79 )
		goto st57;
	goto st0;
st57:
	if ( ++p == pe )
		goto _test_eof57;
case 57:
	if ( (*p) == 80 )
		goto st58;
	goto st0;
st58:
	if ( ++p == pe )
		goto _test_eof58;
case 58:
	switch( (*p) ) {
		case 70: goto st59;
		case 80: goto st63;
	}
	goto st0;
st59:
	if ( ++p == pe )
		goto _test_eof59;
case 59:
	if ( (*p) == 73 )
		goto st60;
	goto st0;
st60:
	if ( ++p == pe )
		goto _test_eof60;
case 60:
	if ( (*p) == 78 )
		goto st61;
	goto st0;
st61:
	if ( ++p == pe )
		goto _test_eof61;
case 61:
	if ( (*p) == 68 )
		goto st62;
	goto st0;
st62:
	if ( ++p == pe )
		goto _test_eof62;
case 62:
	if ( (*p) == 32 )
		goto tr83;
	goto st0;
st63:
	if ( ++p == pe )
		goto _test_eof63;
case 63:
	if ( (*p) == 65 )
		goto st64;
	goto st0;
st64:
	if ( ++p == pe )
		goto _test_eof64;
case 64:
	if ( (*p) == 84 )
		goto st65;
	goto st0;
st65:
	if ( ++p == pe )
		goto _test_eof65;
case 65:
	if ( (*p) == 67 )
		goto st66;
	goto st0;
st66:
	if ( ++p == pe )
		goto _test_eof66;
case 66:
	if ( (*p) == 72 )
		goto st67;
	goto st0;
st67:
	if ( ++p == pe )
		goto _test_eof67;
case 67:
	if ( (*p) == 32 )
		goto tr88;
	goto st0;
st68:
	if ( ++p == pe )
		goto _test_eof68;
case 68:
	if ( (*p) == 84 )
		goto st69;
	goto st0;
st69:
	if ( ++p == pe )
		goto _test_eof69;
case 69:
	if ( (*p) == 32 )
		goto tr90;
	goto st0;
tr9:
#line 205 "ebb_request_parser.rl"
	{
    assert(CURRENT == NULL);
    CURRENT = parser->new_request(parser->data);
  }
	goto st70;
st70:
	if ( ++p == pe )
		goto _test_eof70;
case 70:
#line 1518 "ebb_request_parser.c"
	if ( (*p) == 82 )
		goto st71;
	goto st0;
st71:
	if ( ++p == pe )
		goto _test_eof71;
case 71:
	if ( (*p) == 65 )
		goto st72;
	goto st0;
st72:
	if ( ++p == pe )
		goto _test_eof72;
case 72:
	if ( (*p) == 67 )
		goto st73;
	goto st0;
st73:
	if ( ++p == pe )
		goto _test_eof73;
case 73:
	if ( (*p) == 69 )
		goto st74;
	goto st0;
st74:
	if ( ++p == pe )
		goto _test_eof74;
case 74:
	if ( (*p) == 32 )
		goto tr95;
	goto st0;
tr10:
#line 205 "ebb_request_parser.rl"
	{
    assert(CURRENT == NULL);
    CURRENT = parser->new_request(parser->data);
  }
	goto st75;
st75:
	if ( ++p == pe )
		goto _test_eof75;
case 75:
#line 1561 "ebb_request_parser.c"
	if ( (*p) == 78 )
		goto st76;
	goto st0;
st76:
	if ( ++p == pe )
		goto _test_eof76;
case 76:
	if ( (*p) == 76 )
		goto st77;
	goto st0;
st77:
	if ( ++p == pe )
		goto _test_eof77;
case 77:
	if ( (*p) == 79 )
		goto st78;
	goto st0;
st78:
	if ( ++p == pe )
		goto _test_eof78;
case 78:
	if ( (*p) == 67 )
		goto st79;
	goto st0;
st79:
	if ( ++p == pe )
		goto _test_eof79;
case 79:
	if ( (*p) == 75 )
		goto st80;
	goto st0;
st80:
	if ( ++p == pe )
		goto _test_eof80;
case 80:
	if ( (*p) == 32 )
		goto tr101;
	goto st0;
tr32:
#line 59 "ebb_request_parser.rl"
	{ parser->header_field_mark   = p; }
	goto st81;
tr111:
#line 157 "ebb_request_parser.rl"
	{
    CURRENT->number_of_headers++;
  }
#line 59 "ebb_request_parser.rl"
	{ parser->header_field_mark   = p; }
	goto st81;
st81:
	if ( ++p == pe )
		goto _test_eof81;
case 81:
#line 1616 "ebb_request_parser.c"
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
tr103:
#line 81 "ebb_request_parser.rl"
	{ 
    //printf("write_field!\n");
    HEADER_CALLBACK(header_field);
    parser->header_field_mark = NULL;
  }
	goto st82;
st82:
	if ( ++p == pe )
		goto _test_eof82;
case 82:
#line 1653 "ebb_request_parser.c"
	switch( (*p) ) {
		case 13: goto tr105;
		case 32: goto st82;
	}
	goto tr104;
tr104:
#line 60 "ebb_request_parser.rl"
	{ parser->header_value_mark   = p; }
	goto st83;
st83:
	if ( ++p == pe )
		goto _test_eof83;
case 83:
#line 1667 "ebb_request_parser.c"
	if ( (*p) == 13 )
		goto tr108;
	goto st83;
tr105:
#line 60 "ebb_request_parser.rl"
	{ parser->header_value_mark   = p; }
#line 87 "ebb_request_parser.rl"
	{
    //printf("write_value!\n");
    HEADER_CALLBACK(header_value);
    parser->header_value_mark = NULL;
  }
	goto st84;
tr108:
#line 87 "ebb_request_parser.rl"
	{
    //printf("write_value!\n");
    HEADER_CALLBACK(header_value);
    parser->header_value_mark = NULL;
  }
	goto st84;
tr133:
#line 127 "ebb_request_parser.rl"
	{ CURRENT->keep_alive = FALSE; }
#line 87 "ebb_request_parser.rl"
	{
    //printf("write_value!\n");
    HEADER_CALLBACK(header_value);
    parser->header_value_mark = NULL;
  }
	goto st84;
tr143:
#line 126 "ebb_request_parser.rl"
	{ CURRENT->keep_alive = TRUE; }
#line 87 "ebb_request_parser.rl"
	{
    //printf("write_value!\n");
    HEADER_CALLBACK(header_value);
    parser->header_value_mark = NULL;
  }
	goto st84;
tr221:
#line 138 "ebb_request_parser.rl"
	{
    CURRENT->expect_continue = TRUE;
  }
#line 87 "ebb_request_parser.rl"
	{
    //printf("write_value!\n");
    HEADER_CALLBACK(header_value);
    parser->header_value_mark = NULL;
  }
	goto st84;
tr231:
#line 142 "ebb_request_parser.rl"
	{
    //printf("trailer\n");
    /* not implemenetd yet. (do requests even have trailing headers?) */
  }
#line 60 "ebb_request_parser.rl"
	{ parser->header_value_mark   = p; }
#line 87 "ebb_request_parser.rl"
	{
    //printf("write_value!\n");
    HEADER_CALLBACK(header_value);
    parser->header_value_mark = NULL;
  }
	goto st84;
tr234:
#line 142 "ebb_request_parser.rl"
	{
    //printf("trailer\n");
    /* not implemenetd yet. (do requests even have trailing headers?) */
  }
#line 87 "ebb_request_parser.rl"
	{
    //printf("write_value!\n");
    HEADER_CALLBACK(header_value);
    parser->header_value_mark = NULL;
  }
	goto st84;
tr258:
#line 123 "ebb_request_parser.rl"
	{ CURRENT->transfer_encoding = EBB_IDENTITY; }
#line 87 "ebb_request_parser.rl"
	{
    //printf("write_value!\n");
    HEADER_CALLBACK(header_value);
    parser->header_value_mark = NULL;
  }
	goto st84;
st84:
	if ( ++p == pe )
		goto _test_eof84;
case 84:
#line 1763 "ebb_request_parser.c"
	if ( (*p) == 10 )
		goto st85;
	goto st0;
st85:
	if ( ++p == pe )
		goto _test_eof85;
case 85:
	switch( (*p) ) {
		case 13: goto tr110;
		case 33: goto tr111;
		case 67: goto tr112;
		case 69: goto tr113;
		case 84: goto tr114;
		case 99: goto tr112;
		case 101: goto tr113;
		case 116: goto tr114;
		case 124: goto tr111;
		case 126: goto tr111;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto tr111;
		} else if ( (*p) >= 35 )
			goto tr111;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto tr111;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto tr111;
		} else
			goto tr111;
	} else
		goto tr111;
	goto st0;
tr33:
#line 59 "ebb_request_parser.rl"
	{ parser->header_field_mark   = p; }
	goto st86;
tr112:
#line 157 "ebb_request_parser.rl"
	{
    CURRENT->number_of_headers++;
  }
#line 59 "ebb_request_parser.rl"
	{ parser->header_field_mark   = p; }
	goto st86;
st86:
	if ( ++p == pe )
		goto _test_eof86;
case 86:
#line 1817 "ebb_request_parser.c"
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 79: goto st87;
		case 111: goto st87;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st87:
	if ( ++p == pe )
		goto _test_eof87;
case 87:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 78: goto st88;
		case 110: goto st88;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st88:
	if ( ++p == pe )
		goto _test_eof88;
case 88:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 78: goto st89;
		case 84: goto st112;
		case 110: goto st89;
		case 116: goto st112;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st89:
	if ( ++p == pe )
		goto _test_eof89;
case 89:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 69: goto st90;
		case 101: goto st90;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st90:
	if ( ++p == pe )
		goto _test_eof90;
case 90:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 67: goto st91;
		case 99: goto st91;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st91:
	if ( ++p == pe )
		goto _test_eof91;
case 91:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 84: goto st92;
		case 116: goto st92;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st92:
	if ( ++p == pe )
		goto _test_eof92;
case 92:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 73: goto st93;
		case 105: goto st93;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st93:
	if ( ++p == pe )
		goto _test_eof93;
case 93:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 79: goto st94;
		case 111: goto st94;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st94:
	if ( ++p == pe )
		goto _test_eof94;
case 94:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 78: goto st95;
		case 110: goto st95;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st95:
	if ( ++p == pe )
		goto _test_eof95;
case 95:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr125;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
tr125:
#line 81 "ebb_request_parser.rl"
	{ 
    //printf("write_field!\n");
    HEADER_CALLBACK(header_field);
    parser->header_field_mark = NULL;
  }
	goto st96;
st96:
	if ( ++p == pe )
		goto _test_eof96;
case 96:
#line 2126 "ebb_request_parser.c"
	switch( (*p) ) {
		case 13: goto tr105;
		case 32: goto st96;
		case 67: goto tr127;
		case 75: goto tr128;
		case 99: goto tr127;
		case 107: goto tr128;
	}
	goto tr104;
tr127:
#line 60 "ebb_request_parser.rl"
	{ parser->header_value_mark   = p; }
	goto st97;
st97:
	if ( ++p == pe )
		goto _test_eof97;
case 97:
#line 2144 "ebb_request_parser.c"
	switch( (*p) ) {
		case 13: goto tr108;
		case 76: goto st98;
		case 108: goto st98;
	}
	goto st83;
st98:
	if ( ++p == pe )
		goto _test_eof98;
case 98:
	switch( (*p) ) {
		case 13: goto tr108;
		case 79: goto st99;
		case 111: goto st99;
	}
	goto st83;
st99:
	if ( ++p == pe )
		goto _test_eof99;
case 99:
	switch( (*p) ) {
		case 13: goto tr108;
		case 83: goto st100;
		case 115: goto st100;
	}
	goto st83;
st100:
	if ( ++p == pe )
		goto _test_eof100;
case 100:
	switch( (*p) ) {
		case 13: goto tr108;
		case 69: goto st101;
		case 101: goto st101;
	}
	goto st83;
st101:
	if ( ++p == pe )
		goto _test_eof101;
case 101:
	if ( (*p) == 13 )
		goto tr133;
	goto st83;
tr128:
#line 60 "ebb_request_parser.rl"
	{ parser->header_value_mark   = p; }
	goto st102;
st102:
	if ( ++p == pe )
		goto _test_eof102;
case 102:
#line 2196 "ebb_request_parser.c"
	switch( (*p) ) {
		case 13: goto tr108;
		case 69: goto st103;
		case 101: goto st103;
	}
	goto st83;
st103:
	if ( ++p == pe )
		goto _test_eof103;
case 103:
	switch( (*p) ) {
		case 13: goto tr108;
		case 69: goto st104;
		case 101: goto st104;
	}
	goto st83;
st104:
	if ( ++p == pe )
		goto _test_eof104;
case 104:
	switch( (*p) ) {
		case 13: goto tr108;
		case 80: goto st105;
		case 112: goto st105;
	}
	goto st83;
st105:
	if ( ++p == pe )
		goto _test_eof105;
case 105:
	switch( (*p) ) {
		case 13: goto tr108;
		case 45: goto st106;
	}
	goto st83;
st106:
	if ( ++p == pe )
		goto _test_eof106;
case 106:
	switch( (*p) ) {
		case 13: goto tr108;
		case 65: goto st107;
		case 97: goto st107;
	}
	goto st83;
st107:
	if ( ++p == pe )
		goto _test_eof107;
case 107:
	switch( (*p) ) {
		case 13: goto tr108;
		case 76: goto st108;
		case 108: goto st108;
	}
	goto st83;
st108:
	if ( ++p == pe )
		goto _test_eof108;
case 108:
	switch( (*p) ) {
		case 13: goto tr108;
		case 73: goto st109;
		case 105: goto st109;
	}
	goto st83;
st109:
	if ( ++p == pe )
		goto _test_eof109;
case 109:
	switch( (*p) ) {
		case 13: goto tr108;
		case 86: goto st110;
		case 118: goto st110;
	}
	goto st83;
st110:
	if ( ++p == pe )
		goto _test_eof110;
case 110:
	switch( (*p) ) {
		case 13: goto tr108;
		case 69: goto st111;
		case 101: goto st111;
	}
	goto st83;
st111:
	if ( ++p == pe )
		goto _test_eof111;
case 111:
	if ( (*p) == 13 )
		goto tr143;
	goto st83;
st112:
	if ( ++p == pe )
		goto _test_eof112;
case 112:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 69: goto st113;
		case 101: goto st113;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st113:
	if ( ++p == pe )
		goto _test_eof113;
case 113:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 78: goto st114;
		case 110: goto st114;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st114:
	if ( ++p == pe )
		goto _test_eof114;
case 114:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 84: goto st115;
		case 116: goto st115;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st115:
	if ( ++p == pe )
		goto _test_eof115;
case 115:
	switch( (*p) ) {
		case 33: goto st81;
		case 45: goto st116;
		case 46: goto st81;
		case 58: goto tr103;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 48 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 57 ) {
		if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else if ( (*p) >= 65 )
			goto st81;
	} else
		goto st81;
	goto st0;
st116:
	if ( ++p == pe )
		goto _test_eof116;
case 116:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 76: goto st117;
		case 84: goto st125;
		case 108: goto st117;
		case 116: goto st125;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st117:
	if ( ++p == pe )
		goto _test_eof117;
case 117:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 69: goto st118;
		case 101: goto st118;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st118:
	if ( ++p == pe )
		goto _test_eof118;
case 118:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 78: goto st119;
		case 110: goto st119;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st119:
	if ( ++p == pe )
		goto _test_eof119;
case 119:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 71: goto st120;
		case 103: goto st120;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st120:
	if ( ++p == pe )
		goto _test_eof120;
case 120:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 84: goto st121;
		case 116: goto st121;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st121:
	if ( ++p == pe )
		goto _test_eof121;
case 121:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 72: goto st122;
		case 104: goto st122;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st122:
	if ( ++p == pe )
		goto _test_eof122;
case 122:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr155;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
tr155:
#line 81 "ebb_request_parser.rl"
	{ 
    //printf("write_field!\n");
    HEADER_CALLBACK(header_field);
    parser->header_field_mark = NULL;
  }
	goto st123;
st123:
	if ( ++p == pe )
		goto _test_eof123;
case 123:
#line 2628 "ebb_request_parser.c"
	switch( (*p) ) {
		case 13: goto tr105;
		case 32: goto st123;
	}
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr157;
	goto tr104;
tr157:
#line 117 "ebb_request_parser.rl"
	{
    //printf("content_length!\n");
    CURRENT->content_length *= 10;
    CURRENT->content_length += *p - '0';
  }
#line 60 "ebb_request_parser.rl"
	{ parser->header_value_mark   = p; }
	goto st124;
tr158:
#line 117 "ebb_request_parser.rl"
	{
    //printf("content_length!\n");
    CURRENT->content_length *= 10;
    CURRENT->content_length += *p - '0';
  }
	goto st124;
st124:
	if ( ++p == pe )
		goto _test_eof124;
case 124:
#line 2658 "ebb_request_parser.c"
	if ( (*p) == 13 )
		goto tr108;
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr158;
	goto st83;
st125:
	if ( ++p == pe )
		goto _test_eof125;
case 125:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 89: goto st126;
		case 121: goto st126;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st126:
	if ( ++p == pe )
		goto _test_eof126;
case 126:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 80: goto st127;
		case 112: goto st127;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st127:
	if ( ++p == pe )
		goto _test_eof127;
case 127:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 69: goto st128;
		case 101: goto st128;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st128:
	if ( ++p == pe )
		goto _test_eof128;
case 128:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr162;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
tr162:
#line 81 "ebb_request_parser.rl"
	{ 
    //printf("write_field!\n");
    HEADER_CALLBACK(header_field);
    parser->header_field_mark = NULL;
  }
	goto st129;
st129:
	if ( ++p == pe )
		goto _test_eof129;
case 129:
#line 2794 "ebb_request_parser.c"
	switch( (*p) ) {
		case 13: goto tr105;
		case 32: goto st129;
		case 109: goto tr164;
	}
	goto tr104;
tr164:
#line 60 "ebb_request_parser.rl"
	{ parser->header_value_mark   = p; }
	goto st130;
st130:
	if ( ++p == pe )
		goto _test_eof130;
case 130:
#line 2809 "ebb_request_parser.c"
	switch( (*p) ) {
		case 13: goto tr108;
		case 117: goto st131;
	}
	goto st83;
st131:
	if ( ++p == pe )
		goto _test_eof131;
case 131:
	switch( (*p) ) {
		case 13: goto tr108;
		case 108: goto st132;
	}
	goto st83;
st132:
	if ( ++p == pe )
		goto _test_eof132;
case 132:
	switch( (*p) ) {
		case 13: goto tr108;
		case 116: goto st133;
	}
	goto st83;
st133:
	if ( ++p == pe )
		goto _test_eof133;
case 133:
	switch( (*p) ) {
		case 13: goto tr108;
		case 105: goto st134;
	}
	goto st83;
st134:
	if ( ++p == pe )
		goto _test_eof134;
case 134:
	switch( (*p) ) {
		case 13: goto tr108;
		case 112: goto st135;
	}
	goto st83;
st135:
	if ( ++p == pe )
		goto _test_eof135;
case 135:
	switch( (*p) ) {
		case 13: goto tr108;
		case 97: goto st136;
	}
	goto st83;
st136:
	if ( ++p == pe )
		goto _test_eof136;
case 136:
	switch( (*p) ) {
		case 13: goto tr108;
		case 114: goto st137;
	}
	goto st83;
st137:
	if ( ++p == pe )
		goto _test_eof137;
case 137:
	switch( (*p) ) {
		case 13: goto tr108;
		case 116: goto st138;
	}
	goto st83;
st138:
	if ( ++p == pe )
		goto _test_eof138;
case 138:
	switch( (*p) ) {
		case 13: goto tr108;
		case 47: goto st139;
	}
	goto st83;
st139:
	if ( ++p == pe )
		goto _test_eof139;
case 139:
	switch( (*p) ) {
		case 13: goto tr108;
		case 102: goto st140;
	}
	goto st83;
st140:
	if ( ++p == pe )
		goto _test_eof140;
case 140:
	switch( (*p) ) {
		case 13: goto tr108;
		case 111: goto st141;
	}
	goto st83;
st141:
	if ( ++p == pe )
		goto _test_eof141;
case 141:
	switch( (*p) ) {
		case 13: goto tr108;
		case 114: goto st142;
	}
	goto st83;
st142:
	if ( ++p == pe )
		goto _test_eof142;
case 142:
	switch( (*p) ) {
		case 13: goto tr108;
		case 109: goto st143;
	}
	goto st83;
st143:
	if ( ++p == pe )
		goto _test_eof143;
case 143:
	switch( (*p) ) {
		case 13: goto tr108;
		case 45: goto st144;
	}
	goto st83;
st144:
	if ( ++p == pe )
		goto _test_eof144;
case 144:
	switch( (*p) ) {
		case 13: goto tr108;
		case 100: goto st145;
	}
	goto st83;
st145:
	if ( ++p == pe )
		goto _test_eof145;
case 145:
	switch( (*p) ) {
		case 13: goto tr108;
		case 97: goto st146;
	}
	goto st83;
st146:
	if ( ++p == pe )
		goto _test_eof146;
case 146:
	switch( (*p) ) {
		case 13: goto tr108;
		case 116: goto st147;
	}
	goto st83;
st147:
	if ( ++p == pe )
		goto _test_eof147;
case 147:
	switch( (*p) ) {
		case 13: goto tr108;
		case 97: goto st148;
	}
	goto st83;
st148:
	if ( ++p == pe )
		goto _test_eof148;
case 148:
	switch( (*p) ) {
		case 13: goto tr108;
		case 98: goto st149;
	}
	goto st148;
st149:
	if ( ++p == pe )
		goto _test_eof149;
case 149:
	switch( (*p) ) {
		case 13: goto tr108;
		case 98: goto st149;
		case 111: goto st150;
	}
	goto st148;
st150:
	if ( ++p == pe )
		goto _test_eof150;
case 150:
	switch( (*p) ) {
		case 13: goto tr108;
		case 98: goto st149;
		case 117: goto st151;
	}
	goto st148;
st151:
	if ( ++p == pe )
		goto _test_eof151;
case 151:
	switch( (*p) ) {
		case 13: goto tr108;
		case 98: goto st149;
		case 110: goto st152;
	}
	goto st148;
st152:
	if ( ++p == pe )
		goto _test_eof152;
case 152:
	switch( (*p) ) {
		case 13: goto tr108;
		case 98: goto st149;
		case 100: goto st153;
	}
	goto st148;
st153:
	if ( ++p == pe )
		goto _test_eof153;
case 153:
	switch( (*p) ) {
		case 13: goto tr108;
		case 97: goto st154;
		case 98: goto st149;
	}
	goto st148;
st154:
	if ( ++p == pe )
		goto _test_eof154;
case 154:
	switch( (*p) ) {
		case 13: goto tr108;
		case 98: goto st149;
		case 114: goto st155;
	}
	goto st148;
st155:
	if ( ++p == pe )
		goto _test_eof155;
case 155:
	switch( (*p) ) {
		case 13: goto tr108;
		case 98: goto st149;
		case 121: goto st156;
	}
	goto st148;
st156:
	if ( ++p == pe )
		goto _test_eof156;
case 156:
	switch( (*p) ) {
		case 13: goto tr108;
		case 61: goto st157;
		case 98: goto st149;
	}
	goto st148;
st157:
	if ( ++p == pe )
		goto _test_eof157;
case 157:
	switch( (*p) ) {
		case 13: goto tr108;
		case 34: goto st158;
		case 98: goto tr194;
		case 124: goto tr192;
		case 126: goto tr192;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto tr192;
		} else if ( (*p) >= 33 )
			goto tr192;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto tr192;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto tr192;
		} else
			goto tr192;
	} else
		goto tr192;
	goto st148;
tr192:
#line 129 "ebb_request_parser.rl"
	{
    if(CURRENT->multipart_boundary_len == EBB_MAX_MULTIPART_BOUNDARY_LEN) {
      cs = -1;
      {p++; cs = 158; goto _out;}
    }
    CURRENT->multipart_boundary[1 + (++CURRENT->multipart_boundary_len)] = *p;
    parser->multipart_state = s_start;
  }
	goto st158;
st158:
	if ( ++p == pe )
		goto _test_eof158;
case 158:
#line 3101 "ebb_request_parser.c"
	switch( (*p) ) {
		case 13: goto tr108;
		case 33: goto tr192;
		case 98: goto tr194;
		case 124: goto tr192;
		case 126: goto tr192;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto tr192;
		} else if ( (*p) >= 35 )
			goto tr192;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto tr192;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto tr192;
		} else
			goto tr192;
	} else
		goto tr192;
	goto st148;
tr194:
#line 129 "ebb_request_parser.rl"
	{
    if(CURRENT->multipart_boundary_len == EBB_MAX_MULTIPART_BOUNDARY_LEN) {
      cs = -1;
      {p++; cs = 159; goto _out;}
    }
    CURRENT->multipart_boundary[1 + (++CURRENT->multipart_boundary_len)] = *p;
    parser->multipart_state = s_start;
  }
	goto st159;
st159:
	if ( ++p == pe )
		goto _test_eof159;
case 159:
#line 3142 "ebb_request_parser.c"
	switch( (*p) ) {
		case 13: goto tr108;
		case 33: goto tr192;
		case 98: goto tr194;
		case 111: goto tr195;
		case 124: goto tr192;
		case 126: goto tr192;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto tr192;
		} else if ( (*p) >= 35 )
			goto tr192;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto tr192;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto tr192;
		} else
			goto tr192;
	} else
		goto tr192;
	goto st148;
tr195:
#line 129 "ebb_request_parser.rl"
	{
    if(CURRENT->multipart_boundary_len == EBB_MAX_MULTIPART_BOUNDARY_LEN) {
      cs = -1;
      {p++; cs = 160; goto _out;}
    }
    CURRENT->multipart_boundary[1 + (++CURRENT->multipart_boundary_len)] = *p;
    parser->multipart_state = s_start;
  }
	goto st160;
st160:
	if ( ++p == pe )
		goto _test_eof160;
case 160:
#line 3184 "ebb_request_parser.c"
	switch( (*p) ) {
		case 13: goto tr108;
		case 33: goto tr192;
		case 98: goto tr194;
		case 117: goto tr196;
		case 124: goto tr192;
		case 126: goto tr192;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto tr192;
		} else if ( (*p) >= 35 )
			goto tr192;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto tr192;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto tr192;
		} else
			goto tr192;
	} else
		goto tr192;
	goto st148;
tr196:
#line 129 "ebb_request_parser.rl"
	{
    if(CURRENT->multipart_boundary_len == EBB_MAX_MULTIPART_BOUNDARY_LEN) {
      cs = -1;
      {p++; cs = 161; goto _out;}
    }
    CURRENT->multipart_boundary[1 + (++CURRENT->multipart_boundary_len)] = *p;
    parser->multipart_state = s_start;
  }
	goto st161;
st161:
	if ( ++p == pe )
		goto _test_eof161;
case 161:
#line 3226 "ebb_request_parser.c"
	switch( (*p) ) {
		case 13: goto tr108;
		case 33: goto tr192;
		case 98: goto tr194;
		case 110: goto tr197;
		case 124: goto tr192;
		case 126: goto tr192;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto tr192;
		} else if ( (*p) >= 35 )
			goto tr192;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto tr192;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto tr192;
		} else
			goto tr192;
	} else
		goto tr192;
	goto st148;
tr197:
#line 129 "ebb_request_parser.rl"
	{
    if(CURRENT->multipart_boundary_len == EBB_MAX_MULTIPART_BOUNDARY_LEN) {
      cs = -1;
      {p++; cs = 162; goto _out;}
    }
    CURRENT->multipart_boundary[1 + (++CURRENT->multipart_boundary_len)] = *p;
    parser->multipart_state = s_start;
  }
	goto st162;
st162:
	if ( ++p == pe )
		goto _test_eof162;
case 162:
#line 3268 "ebb_request_parser.c"
	switch( (*p) ) {
		case 13: goto tr108;
		case 33: goto tr192;
		case 98: goto tr194;
		case 100: goto tr198;
		case 124: goto tr192;
		case 126: goto tr192;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto tr192;
		} else if ( (*p) >= 35 )
			goto tr192;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto tr192;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto tr192;
		} else
			goto tr192;
	} else
		goto tr192;
	goto st148;
tr198:
#line 129 "ebb_request_parser.rl"
	{
    if(CURRENT->multipart_boundary_len == EBB_MAX_MULTIPART_BOUNDARY_LEN) {
      cs = -1;
      {p++; cs = 163; goto _out;}
    }
    CURRENT->multipart_boundary[1 + (++CURRENT->multipart_boundary_len)] = *p;
    parser->multipart_state = s_start;
  }
	goto st163;
st163:
	if ( ++p == pe )
		goto _test_eof163;
case 163:
#line 3310 "ebb_request_parser.c"
	switch( (*p) ) {
		case 13: goto tr108;
		case 33: goto tr192;
		case 97: goto tr199;
		case 98: goto tr194;
		case 124: goto tr192;
		case 126: goto tr192;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto tr192;
		} else if ( (*p) >= 35 )
			goto tr192;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto tr192;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto tr192;
		} else
			goto tr192;
	} else
		goto tr192;
	goto st148;
tr199:
#line 129 "ebb_request_parser.rl"
	{
    if(CURRENT->multipart_boundary_len == EBB_MAX_MULTIPART_BOUNDARY_LEN) {
      cs = -1;
      {p++; cs = 164; goto _out;}
    }
    CURRENT->multipart_boundary[1 + (++CURRENT->multipart_boundary_len)] = *p;
    parser->multipart_state = s_start;
  }
	goto st164;
st164:
	if ( ++p == pe )
		goto _test_eof164;
case 164:
#line 3352 "ebb_request_parser.c"
	switch( (*p) ) {
		case 13: goto tr108;
		case 33: goto tr192;
		case 98: goto tr194;
		case 114: goto tr200;
		case 124: goto tr192;
		case 126: goto tr192;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto tr192;
		} else if ( (*p) >= 35 )
			goto tr192;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto tr192;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto tr192;
		} else
			goto tr192;
	} else
		goto tr192;
	goto st148;
tr200:
#line 129 "ebb_request_parser.rl"
	{
    if(CURRENT->multipart_boundary_len == EBB_MAX_MULTIPART_BOUNDARY_LEN) {
      cs = -1;
      {p++; cs = 165; goto _out;}
    }
    CURRENT->multipart_boundary[1 + (++CURRENT->multipart_boundary_len)] = *p;
    parser->multipart_state = s_start;
  }
	goto st165;
st165:
	if ( ++p == pe )
		goto _test_eof165;
case 165:
#line 3394 "ebb_request_parser.c"
	switch( (*p) ) {
		case 13: goto tr108;
		case 33: goto tr192;
		case 98: goto tr194;
		case 121: goto tr201;
		case 124: goto tr192;
		case 126: goto tr192;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto tr192;
		} else if ( (*p) >= 35 )
			goto tr192;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto tr192;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto tr192;
		} else
			goto tr192;
	} else
		goto tr192;
	goto st148;
tr201:
#line 129 "ebb_request_parser.rl"
	{
    if(CURRENT->multipart_boundary_len == EBB_MAX_MULTIPART_BOUNDARY_LEN) {
      cs = -1;
      {p++; cs = 166; goto _out;}
    }
    CURRENT->multipart_boundary[1 + (++CURRENT->multipart_boundary_len)] = *p;
    parser->multipart_state = s_start;
  }
	goto st166;
st166:
	if ( ++p == pe )
		goto _test_eof166;
case 166:
#line 3436 "ebb_request_parser.c"
	switch( (*p) ) {
		case 13: goto tr108;
		case 33: goto tr192;
		case 61: goto st157;
		case 98: goto tr194;
		case 124: goto tr192;
		case 126: goto tr192;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto tr192;
		} else if ( (*p) >= 35 )
			goto tr192;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto tr192;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto tr192;
		} else
			goto tr192;
	} else
		goto tr192;
	goto st148;
tr34:
#line 59 "ebb_request_parser.rl"
	{ parser->header_field_mark   = p; }
	goto st167;
tr113:
#line 157 "ebb_request_parser.rl"
	{
    CURRENT->number_of_headers++;
  }
#line 59 "ebb_request_parser.rl"
	{ parser->header_field_mark   = p; }
	goto st167;
st167:
	if ( ++p == pe )
		goto _test_eof167;
case 167:
#line 3479 "ebb_request_parser.c"
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 88: goto st168;
		case 120: goto st168;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st168:
	if ( ++p == pe )
		goto _test_eof168;
case 168:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 80: goto st169;
		case 112: goto st169;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st169:
	if ( ++p == pe )
		goto _test_eof169;
case 169:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 69: goto st170;
		case 101: goto st170;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st170:
	if ( ++p == pe )
		goto _test_eof170;
case 170:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 67: goto st171;
		case 99: goto st171;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st171:
	if ( ++p == pe )
		goto _test_eof171;
case 171:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 84: goto st172;
		case 116: goto st172;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st172:
	if ( ++p == pe )
		goto _test_eof172;
case 172:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr207;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
tr207:
#line 81 "ebb_request_parser.rl"
	{ 
    //printf("write_field!\n");
    HEADER_CALLBACK(header_field);
    parser->header_field_mark = NULL;
  }
	goto st173;
st173:
	if ( ++p == pe )
		goto _test_eof173;
case 173:
#line 3666 "ebb_request_parser.c"
	switch( (*p) ) {
		case 13: goto tr105;
		case 32: goto st173;
		case 49: goto tr209;
	}
	goto tr104;
tr209:
#line 60 "ebb_request_parser.rl"
	{ parser->header_value_mark   = p; }
	goto st174;
st174:
	if ( ++p == pe )
		goto _test_eof174;
case 174:
#line 3681 "ebb_request_parser.c"
	switch( (*p) ) {
		case 13: goto tr108;
		case 48: goto st175;
	}
	goto st83;
st175:
	if ( ++p == pe )
		goto _test_eof175;
case 175:
	switch( (*p) ) {
		case 13: goto tr108;
		case 48: goto st176;
	}
	goto st83;
st176:
	if ( ++p == pe )
		goto _test_eof176;
case 176:
	switch( (*p) ) {
		case 13: goto tr108;
		case 45: goto st177;
	}
	goto st83;
st177:
	if ( ++p == pe )
		goto _test_eof177;
case 177:
	switch( (*p) ) {
		case 13: goto tr108;
		case 67: goto st178;
		case 99: goto st178;
	}
	goto st83;
st178:
	if ( ++p == pe )
		goto _test_eof178;
case 178:
	switch( (*p) ) {
		case 13: goto tr108;
		case 79: goto st179;
		case 111: goto st179;
	}
	goto st83;
st179:
	if ( ++p == pe )
		goto _test_eof179;
case 179:
	switch( (*p) ) {
		case 13: goto tr108;
		case 78: goto st180;
		case 110: goto st180;
	}
	goto st83;
st180:
	if ( ++p == pe )
		goto _test_eof180;
case 180:
	switch( (*p) ) {
		case 13: goto tr108;
		case 84: goto st181;
		case 116: goto st181;
	}
	goto st83;
st181:
	if ( ++p == pe )
		goto _test_eof181;
case 181:
	switch( (*p) ) {
		case 13: goto tr108;
		case 73: goto st182;
		case 105: goto st182;
	}
	goto st83;
st182:
	if ( ++p == pe )
		goto _test_eof182;
case 182:
	switch( (*p) ) {
		case 13: goto tr108;
		case 78: goto st183;
		case 110: goto st183;
	}
	goto st83;
st183:
	if ( ++p == pe )
		goto _test_eof183;
case 183:
	switch( (*p) ) {
		case 13: goto tr108;
		case 85: goto st184;
		case 117: goto st184;
	}
	goto st83;
st184:
	if ( ++p == pe )
		goto _test_eof184;
case 184:
	switch( (*p) ) {
		case 13: goto tr108;
		case 69: goto st185;
		case 101: goto st185;
	}
	goto st83;
st185:
	if ( ++p == pe )
		goto _test_eof185;
case 185:
	if ( (*p) == 13 )
		goto tr221;
	goto st83;
tr35:
#line 59 "ebb_request_parser.rl"
	{ parser->header_field_mark   = p; }
	goto st186;
tr114:
#line 157 "ebb_request_parser.rl"
	{
    CURRENT->number_of_headers++;
  }
#line 59 "ebb_request_parser.rl"
	{ parser->header_field_mark   = p; }
	goto st186;
st186:
	if ( ++p == pe )
		goto _test_eof186;
case 186:
#line 3808 "ebb_request_parser.c"
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 82: goto st187;
		case 114: goto st187;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st187:
	if ( ++p == pe )
		goto _test_eof187;
case 187:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 65: goto st188;
		case 97: goto st188;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 66 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st188:
	if ( ++p == pe )
		goto _test_eof188;
case 188:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 73: goto st189;
		case 78: goto st195;
		case 105: goto st189;
		case 110: goto st195;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st189:
	if ( ++p == pe )
		goto _test_eof189;
case 189:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 76: goto st190;
		case 108: goto st190;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st190:
	if ( ++p == pe )
		goto _test_eof190;
case 190:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 69: goto st191;
		case 101: goto st191;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st191:
	if ( ++p == pe )
		goto _test_eof191;
case 191:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 82: goto st192;
		case 114: goto st192;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st192:
	if ( ++p == pe )
		goto _test_eof192;
case 192:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr229;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
tr229:
#line 81 "ebb_request_parser.rl"
	{ 
    //printf("write_field!\n");
    HEADER_CALLBACK(header_field);
    parser->header_field_mark = NULL;
  }
	goto st193;
st193:
	if ( ++p == pe )
		goto _test_eof193;
case 193:
#line 4027 "ebb_request_parser.c"
	switch( (*p) ) {
		case 13: goto tr231;
		case 32: goto st193;
	}
	goto tr230;
tr230:
#line 60 "ebb_request_parser.rl"
	{ parser->header_value_mark   = p; }
	goto st194;
st194:
	if ( ++p == pe )
		goto _test_eof194;
case 194:
#line 4041 "ebb_request_parser.c"
	if ( (*p) == 13 )
		goto tr234;
	goto st194;
st195:
	if ( ++p == pe )
		goto _test_eof195;
case 195:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 83: goto st196;
		case 115: goto st196;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st196:
	if ( ++p == pe )
		goto _test_eof196;
case 196:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 70: goto st197;
		case 102: goto st197;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st197:
	if ( ++p == pe )
		goto _test_eof197;
case 197:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 69: goto st198;
		case 101: goto st198;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st198:
	if ( ++p == pe )
		goto _test_eof198;
case 198:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 82: goto st199;
		case 114: goto st199;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st199:
	if ( ++p == pe )
		goto _test_eof199;
case 199:
	switch( (*p) ) {
		case 33: goto st81;
		case 45: goto st200;
		case 46: goto st81;
		case 58: goto tr103;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 48 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 57 ) {
		if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else if ( (*p) >= 65 )
			goto st81;
	} else
		goto st81;
	goto st0;
st200:
	if ( ++p == pe )
		goto _test_eof200;
case 200:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 69: goto st201;
		case 101: goto st201;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st201:
	if ( ++p == pe )
		goto _test_eof201;
case 201:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 78: goto st202;
		case 110: goto st202;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st202:
	if ( ++p == pe )
		goto _test_eof202;
case 202:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 67: goto st203;
		case 99: goto st203;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st203:
	if ( ++p == pe )
		goto _test_eof203;
case 203:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 79: goto st204;
		case 111: goto st204;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st204:
	if ( ++p == pe )
		goto _test_eof204;
case 204:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 68: goto st205;
		case 100: goto st205;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st205:
	if ( ++p == pe )
		goto _test_eof205;
case 205:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 73: goto st206;
		case 105: goto st206;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st206:
	if ( ++p == pe )
		goto _test_eof206;
case 206:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 78: goto st207;
		case 110: goto st207;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st207:
	if ( ++p == pe )
		goto _test_eof207;
case 207:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr103;
		case 71: goto st208;
		case 103: goto st208;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
st208:
	if ( ++p == pe )
		goto _test_eof208;
case 208:
	switch( (*p) ) {
		case 33: goto st81;
		case 58: goto tr248;
		case 124: goto st81;
		case 126: goto st81;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st81;
		} else if ( (*p) >= 35 )
			goto st81;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st81;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st81;
		} else
			goto st81;
	} else
		goto st81;
	goto st0;
tr248:
#line 124 "ebb_request_parser.rl"
	{ CURRENT->transfer_encoding = EBB_CHUNKED; }
#line 81 "ebb_request_parser.rl"
	{ 
    //printf("write_field!\n");
    HEADER_CALLBACK(header_field);
    parser->header_field_mark = NULL;
  }
	goto st209;
st209:
	if ( ++p == pe )
		goto _test_eof209;
case 209:
#line 4474 "ebb_request_parser.c"
	switch( (*p) ) {
		case 13: goto tr105;
		case 32: goto st209;
		case 105: goto tr250;
	}
	goto tr104;
tr250:
#line 60 "ebb_request_parser.rl"
	{ parser->header_value_mark   = p; }
	goto st210;
st210:
	if ( ++p == pe )
		goto _test_eof210;
case 210:
#line 4489 "ebb_request_parser.c"
	switch( (*p) ) {
		case 13: goto tr108;
		case 100: goto st211;
	}
	goto st83;
st211:
	if ( ++p == pe )
		goto _test_eof211;
case 211:
	switch( (*p) ) {
		case 13: goto tr108;
		case 101: goto st212;
	}
	goto st83;
st212:
	if ( ++p == pe )
		goto _test_eof212;
case 212:
	switch( (*p) ) {
		case 13: goto tr108;
		case 110: goto st213;
	}
	goto st83;
st213:
	if ( ++p == pe )
		goto _test_eof213;
case 213:
	switch( (*p) ) {
		case 13: goto tr108;
		case 116: goto st214;
	}
	goto st83;
st214:
	if ( ++p == pe )
		goto _test_eof214;
case 214:
	switch( (*p) ) {
		case 13: goto tr108;
		case 105: goto st215;
	}
	goto st83;
st215:
	if ( ++p == pe )
		goto _test_eof215;
case 215:
	switch( (*p) ) {
		case 13: goto tr108;
		case 116: goto st216;
	}
	goto st83;
st216:
	if ( ++p == pe )
		goto _test_eof216;
case 216:
	switch( (*p) ) {
		case 13: goto tr108;
		case 121: goto st217;
	}
	goto st83;
st217:
	if ( ++p == pe )
		goto _test_eof217;
case 217:
	if ( (*p) == 13 )
		goto tr258;
	goto st83;
tr20:
#line 93 "ebb_request_parser.rl"
	{ 
    //printf("request uri\n");
    CALLBACK(uri);
    parser->uri_mark = NULL;
  }
	goto st218;
tr272:
#line 111 "ebb_request_parser.rl"
	{
    //printf("request path\n");
    CALLBACK(path);
    parser->path_mark = NULL;
  }
#line 93 "ebb_request_parser.rl"
	{ 
    //printf("request uri\n");
    CALLBACK(uri);
    parser->uri_mark = NULL;
  }
	goto st218;
tr278:
#line 62 "ebb_request_parser.rl"
	{ parser->query_string_mark   = p; }
#line 105 "ebb_request_parser.rl"
	{ 
    //printf("query  string\n");
    CALLBACK(query_string);
    parser->query_string_mark = NULL;
  }
#line 93 "ebb_request_parser.rl"
	{ 
    //printf("request uri\n");
    CALLBACK(uri);
    parser->uri_mark = NULL;
  }
	goto st218;
tr282:
#line 105 "ebb_request_parser.rl"
	{ 
    //printf("query  string\n");
    CALLBACK(query_string);
    parser->query_string_mark = NULL;
  }
#line 93 "ebb_request_parser.rl"
	{ 
    //printf("request uri\n");
    CALLBACK(uri);
    parser->uri_mark = NULL;
  }
	goto st218;
st218:
	if ( ++p == pe )
		goto _test_eof218;
case 218:
#line 4612 "ebb_request_parser.c"
	switch( (*p) ) {
		case 32: goto tr260;
		case 37: goto tr261;
		case 60: goto st0;
		case 62: goto st0;
		case 127: goto st0;
	}
	if ( (*p) > 31 ) {
		if ( 34 <= (*p) && (*p) <= 35 )
			goto st0;
	} else if ( (*p) >= 0 )
		goto st0;
	goto tr259;
tr259:
#line 61 "ebb_request_parser.rl"
	{ parser->fragment_mark       = p; }
	goto st219;
st219:
	if ( ++p == pe )
		goto _test_eof219;
case 219:
#line 4634 "ebb_request_parser.c"
	switch( (*p) ) {
		case 32: goto tr263;
		case 37: goto st220;
		case 60: goto st0;
		case 62: goto st0;
		case 127: goto st0;
	}
	if ( (*p) > 31 ) {
		if ( 34 <= (*p) && (*p) <= 35 )
			goto st0;
	} else if ( (*p) >= 0 )
		goto st0;
	goto st219;
tr261:
#line 61 "ebb_request_parser.rl"
	{ parser->fragment_mark       = p; }
	goto st220;
st220:
	if ( ++p == pe )
		goto _test_eof220;
case 220:
#line 4656 "ebb_request_parser.c"
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto st221;
	} else if ( (*p) > 70 ) {
		if ( 97 <= (*p) && (*p) <= 102 )
			goto st221;
	} else
		goto st221;
	goto st0;
st221:
	if ( ++p == pe )
		goto _test_eof221;
case 221:
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto st219;
	} else if ( (*p) > 70 ) {
		if ( 97 <= (*p) && (*p) <= 102 )
			goto st219;
	} else
		goto st219;
	goto st0;
tr16:
#line 64 "ebb_request_parser.rl"
	{ parser->uri_mark            = p; }
	goto st222;
st222:
	if ( ++p == pe )
		goto _test_eof222;
case 222:
#line 4687 "ebb_request_parser.c"
	switch( (*p) ) {
		case 43: goto st222;
		case 58: goto st223;
	}
	if ( (*p) < 48 ) {
		if ( 45 <= (*p) && (*p) <= 46 )
			goto st222;
	} else if ( (*p) > 57 ) {
		if ( (*p) > 90 ) {
			if ( 97 <= (*p) && (*p) <= 122 )
				goto st222;
		} else if ( (*p) >= 65 )
			goto st222;
	} else
		goto st222;
	goto st0;
tr18:
#line 64 "ebb_request_parser.rl"
	{ parser->uri_mark            = p; }
	goto st223;
st223:
	if ( ++p == pe )
		goto _test_eof223;
case 223:
#line 4712 "ebb_request_parser.c"
	switch( (*p) ) {
		case 32: goto tr19;
		case 34: goto st0;
		case 35: goto tr20;
		case 37: goto st224;
		case 60: goto st0;
		case 62: goto st0;
		case 127: goto st0;
	}
	if ( 0 <= (*p) && (*p) <= 31 )
		goto st0;
	goto st223;
st224:
	if ( ++p == pe )
		goto _test_eof224;
case 224:
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto st225;
	} else if ( (*p) > 70 ) {
		if ( 97 <= (*p) && (*p) <= 102 )
			goto st225;
	} else
		goto st225;
	goto st0;
st225:
	if ( ++p == pe )
		goto _test_eof225;
case 225:
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto st223;
	} else if ( (*p) > 70 ) {
		if ( 97 <= (*p) && (*p) <= 102 )
			goto st223;
	} else
		goto st223;
	goto st0;
tr17:
#line 64 "ebb_request_parser.rl"
	{ parser->uri_mark            = p; }
#line 63 "ebb_request_parser.rl"
	{ parser->path_mark           = p; }
	goto st226;
st226:
	if ( ++p == pe )
		goto _test_eof226;
case 226:
#line 4761 "ebb_request_parser.c"
	switch( (*p) ) {
		case 32: goto tr271;
		case 34: goto st0;
		case 35: goto tr272;
		case 37: goto st227;
		case 60: goto st0;
		case 62: goto st0;
		case 63: goto tr274;
		case 127: goto st0;
	}
	if ( 0 <= (*p) && (*p) <= 31 )
		goto st0;
	goto st226;
st227:
	if ( ++p == pe )
		goto _test_eof227;
case 227:
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto st228;
	} else if ( (*p) > 70 ) {
		if ( 97 <= (*p) && (*p) <= 102 )
			goto st228;
	} else
		goto st228;
	goto st0;
st228:
	if ( ++p == pe )
		goto _test_eof228;
case 228:
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto st226;
	} else if ( (*p) > 70 ) {
		if ( 97 <= (*p) && (*p) <= 102 )
			goto st226;
	} else
		goto st226;
	goto st0;
tr274:
#line 111 "ebb_request_parser.rl"
	{
    //printf("request path\n");
    CALLBACK(path);
    parser->path_mark = NULL;
  }
	goto st229;
st229:
	if ( ++p == pe )
		goto _test_eof229;
case 229:
#line 4813 "ebb_request_parser.c"
	switch( (*p) ) {
		case 32: goto tr277;
		case 34: goto st0;
		case 35: goto tr278;
		case 37: goto tr279;
		case 60: goto st0;
		case 62: goto st0;
		case 127: goto st0;
	}
	if ( 0 <= (*p) && (*p) <= 31 )
		goto st0;
	goto tr276;
tr276:
#line 62 "ebb_request_parser.rl"
	{ parser->query_string_mark   = p; }
	goto st230;
st230:
	if ( ++p == pe )
		goto _test_eof230;
case 230:
#line 4834 "ebb_request_parser.c"
	switch( (*p) ) {
		case 32: goto tr281;
		case 34: goto st0;
		case 35: goto tr282;
		case 37: goto st231;
		case 60: goto st0;
		case 62: goto st0;
		case 127: goto st0;
	}
	if ( 0 <= (*p) && (*p) <= 31 )
		goto st0;
	goto st230;
tr279:
#line 62 "ebb_request_parser.rl"
	{ parser->query_string_mark   = p; }
	goto st231;
st231:
	if ( ++p == pe )
		goto _test_eof231;
case 231:
#line 4855 "ebb_request_parser.c"
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto st232;
	} else if ( (*p) > 70 ) {
		if ( 97 <= (*p) && (*p) <= 102 )
			goto st232;
	} else
		goto st232;
	goto st0;
st232:
	if ( ++p == pe )
		goto _test_eof232;
case 232:
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto st230;
	} else if ( (*p) > 70 ) {
		if ( 97 <= (*p) && (*p) <= 102 )
			goto st230;
	} else
		goto st230;
	goto st0;
st233:
	if ( ++p == pe )
		goto _test_eof233;
case 233:
	if ( (*p) == 48 )
		goto tr285;
	if ( (*p) < 65 ) {
		if ( 49 <= (*p) && (*p) <= 57 )
			goto tr286;
	} else if ( (*p) > 70 ) {
		if ( 97 <= (*p) && (*p) <= 102 )
			goto tr286;
	} else
		goto tr286;
	goto st0;
tr285:
#line 166 "ebb_request_parser.rl"
	{
    //printf("add to chunk size\n");
    parser->chunk_size *= 16;
    /* XXX: this can be optimized slightly  */
    if( 'A' <= *p && *p <= 'F') 
      parser->chunk_size += *p - 'A' + 10;
    else if( 'a' <= *p && *p <= 'f') 
      parser->chunk_size += *p - 'a' + 10;
    else if( '0' <= *p && *p <= '9') 
      parser->chunk_size += *p - '0';
    else  
      assert(0 && "bad hex char");
  }
	goto st234;
st234:
	if ( ++p == pe )
		goto _test_eof234;
case 234:
#line 4913 "ebb_request_parser.c"
	switch( (*p) ) {
		case 13: goto st235;
		case 48: goto tr285;
		case 59: goto st248;
	}
	if ( (*p) < 65 ) {
		if ( 49 <= (*p) && (*p) <= 57 )
			goto tr286;
	} else if ( (*p) > 70 ) {
		if ( 97 <= (*p) && (*p) <= 102 )
			goto tr286;
	} else
		goto tr286;
	goto st0;
st235:
	if ( ++p == pe )
		goto _test_eof235;
case 235:
	if ( (*p) == 10 )
		goto st236;
	goto st0;
st236:
	if ( ++p == pe )
		goto _test_eof236;
case 236:
	switch( (*p) ) {
		case 13: goto st237;
		case 33: goto st238;
		case 124: goto st238;
		case 126: goto st238;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st238;
		} else if ( (*p) >= 35 )
			goto st238;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st238;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st238;
		} else
			goto st238;
	} else
		goto st238;
	goto st0;
st237:
	if ( ++p == pe )
		goto _test_eof237;
case 237:
	if ( (*p) == 10 )
		goto tr292;
	goto st0;
tr292:
#line 199 "ebb_request_parser.rl"
	{
    //printf("end chunked body\n");
    END_REQUEST;
    {cs = stack[--top];goto _again;} // goto Request; 
  }
	goto st252;
st252:
	if ( ++p == pe )
		goto _test_eof252;
case 252:
#line 4982 "ebb_request_parser.c"
	goto st0;
st238:
	if ( ++p == pe )
		goto _test_eof238;
case 238:
	switch( (*p) ) {
		case 33: goto st238;
		case 58: goto st239;
		case 124: goto st238;
		case 126: goto st238;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st238;
		} else if ( (*p) >= 35 )
			goto st238;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st238;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st238;
		} else
			goto st238;
	} else
		goto st238;
	goto st0;
st239:
	if ( ++p == pe )
		goto _test_eof239;
case 239:
	if ( (*p) == 13 )
		goto st235;
	goto st239;
tr286:
#line 166 "ebb_request_parser.rl"
	{
    //printf("add to chunk size\n");
    parser->chunk_size *= 16;
    /* XXX: this can be optimized slightly  */
    if( 'A' <= *p && *p <= 'F') 
      parser->chunk_size += *p - 'A' + 10;
    else if( 'a' <= *p && *p <= 'f') 
      parser->chunk_size += *p - 'a' + 10;
    else if( '0' <= *p && *p <= '9') 
      parser->chunk_size += *p - '0';
    else  
      assert(0 && "bad hex char");
  }
	goto st240;
st240:
	if ( ++p == pe )
		goto _test_eof240;
case 240:
#line 5039 "ebb_request_parser.c"
	switch( (*p) ) {
		case 13: goto st241;
		case 59: goto st245;
	}
	if ( (*p) < 65 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr286;
	} else if ( (*p) > 70 ) {
		if ( 97 <= (*p) && (*p) <= 102 )
			goto tr286;
	} else
		goto tr286;
	goto st0;
st241:
	if ( ++p == pe )
		goto _test_eof241;
case 241:
	if ( (*p) == 10 )
		goto st242;
	goto st0;
st242:
	if ( ++p == pe )
		goto _test_eof242;
case 242:
	goto tr297;
tr297:
#line 180 "ebb_request_parser.rl"
	{
    //printf("skip chunk data\n");
    //printf("chunk_size: %d\n", parser->chunk_size);
    if(parser->chunk_size > REMAINING) {
      parser->eating = TRUE;
      CURRENT->on_body(CURRENT, p, REMAINING);
      parser->chunk_size -= REMAINING;
      p--; 
      {p++; cs = 243; goto _out;}
    } else {
      CURRENT->on_body(CURRENT, p, parser->chunk_size);
      p += parser->chunk_size;
      parser->chunk_size = 0;
      parser->eating = FALSE;
      p--; 
      {goto st243;} 
    }
  }
	goto st243;
st243:
	if ( ++p == pe )
		goto _test_eof243;
case 243:
#line 5090 "ebb_request_parser.c"
	if ( (*p) == 13 )
		goto st244;
	goto st0;
st244:
	if ( ++p == pe )
		goto _test_eof244;
case 244:
	if ( (*p) == 10 )
		goto st233;
	goto st0;
st245:
	if ( ++p == pe )
		goto _test_eof245;
case 245:
	switch( (*p) ) {
		case 13: goto st241;
		case 32: goto st245;
		case 33: goto st246;
		case 59: goto st245;
		case 61: goto st247;
		case 124: goto st246;
		case 126: goto st246;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st246;
		} else if ( (*p) >= 35 )
			goto st246;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st246;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st246;
		} else
			goto st246;
	} else
		goto st246;
	goto st0;
st246:
	if ( ++p == pe )
		goto _test_eof246;
case 246:
	switch( (*p) ) {
		case 13: goto st241;
		case 33: goto st246;
		case 59: goto st245;
		case 61: goto st247;
		case 124: goto st246;
		case 126: goto st246;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st246;
		} else if ( (*p) >= 35 )
			goto st246;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st246;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st246;
		} else
			goto st246;
	} else
		goto st246;
	goto st0;
st247:
	if ( ++p == pe )
		goto _test_eof247;
case 247:
	switch( (*p) ) {
		case 13: goto st241;
		case 33: goto st247;
		case 59: goto st245;
		case 124: goto st247;
		case 126: goto st247;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st247;
		} else if ( (*p) >= 35 )
			goto st247;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st247;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st247;
		} else
			goto st247;
	} else
		goto st247;
	goto st0;
st248:
	if ( ++p == pe )
		goto _test_eof248;
case 248:
	switch( (*p) ) {
		case 13: goto st235;
		case 32: goto st248;
		case 33: goto st249;
		case 59: goto st248;
		case 61: goto st250;
		case 124: goto st249;
		case 126: goto st249;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st249;
		} else if ( (*p) >= 35 )
			goto st249;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st249;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st249;
		} else
			goto st249;
	} else
		goto st249;
	goto st0;
st249:
	if ( ++p == pe )
		goto _test_eof249;
case 249:
	switch( (*p) ) {
		case 13: goto st235;
		case 33: goto st249;
		case 59: goto st248;
		case 61: goto st250;
		case 124: goto st249;
		case 126: goto st249;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st249;
		} else if ( (*p) >= 35 )
			goto st249;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st249;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st249;
		} else
			goto st249;
	} else
		goto st249;
	goto st0;
st250:
	if ( ++p == pe )
		goto _test_eof250;
case 250:
	switch( (*p) ) {
		case 13: goto st235;
		case 33: goto st250;
		case 59: goto st248;
		case 124: goto st250;
		case 126: goto st250;
	}
	if ( (*p) < 45 ) {
		if ( (*p) > 39 ) {
			if ( 42 <= (*p) && (*p) <= 43 )
				goto st250;
		} else if ( (*p) >= 35 )
			goto st250;
	} else if ( (*p) > 46 ) {
		if ( (*p) < 65 ) {
			if ( 48 <= (*p) && (*p) <= 57 )
				goto st250;
		} else if ( (*p) > 90 ) {
			if ( 94 <= (*p) && (*p) <= 122 )
				goto st250;
		} else
			goto st250;
	} else
		goto st250;
	goto st0;
	}
	_test_eof1: cs = 1; goto _test_eof; 
	_test_eof2: cs = 2; goto _test_eof; 
	_test_eof3: cs = 3; goto _test_eof; 
	_test_eof4: cs = 4; goto _test_eof; 
	_test_eof5: cs = 5; goto _test_eof; 
	_test_eof6: cs = 6; goto _test_eof; 
	_test_eof7: cs = 7; goto _test_eof; 
	_test_eof8: cs = 8; goto _test_eof; 
	_test_eof9: cs = 9; goto _test_eof; 
	_test_eof10: cs = 10; goto _test_eof; 
	_test_eof11: cs = 11; goto _test_eof; 
	_test_eof12: cs = 12; goto _test_eof; 
	_test_eof13: cs = 13; goto _test_eof; 
	_test_eof14: cs = 14; goto _test_eof; 
	_test_eof15: cs = 15; goto _test_eof; 
	_test_eof16: cs = 16; goto _test_eof; 
	_test_eof17: cs = 17; goto _test_eof; 
	_test_eof18: cs = 18; goto _test_eof; 
	_test_eof19: cs = 19; goto _test_eof; 
	_test_eof251: cs = 251; goto _test_eof; 
	_test_eof20: cs = 20; goto _test_eof; 
	_test_eof21: cs = 21; goto _test_eof; 
	_test_eof22: cs = 22; goto _test_eof; 
	_test_eof23: cs = 23; goto _test_eof; 
	_test_eof24: cs = 24; goto _test_eof; 
	_test_eof25: cs = 25; goto _test_eof; 
	_test_eof26: cs = 26; goto _test_eof; 
	_test_eof27: cs = 27; goto _test_eof; 
	_test_eof28: cs = 28; goto _test_eof; 
	_test_eof29: cs = 29; goto _test_eof; 
	_test_eof30: cs = 30; goto _test_eof; 
	_test_eof31: cs = 31; goto _test_eof; 
	_test_eof32: cs = 32; goto _test_eof; 
	_test_eof33: cs = 33; goto _test_eof; 
	_test_eof34: cs = 34; goto _test_eof; 
	_test_eof35: cs = 35; goto _test_eof; 
	_test_eof36: cs = 36; goto _test_eof; 
	_test_eof37: cs = 37; goto _test_eof; 
	_test_eof38: cs = 38; goto _test_eof; 
	_test_eof39: cs = 39; goto _test_eof; 
	_test_eof40: cs = 40; goto _test_eof; 
	_test_eof41: cs = 41; goto _test_eof; 
	_test_eof42: cs = 42; goto _test_eof; 
	_test_eof43: cs = 43; goto _test_eof; 
	_test_eof44: cs = 44; goto _test_eof; 
	_test_eof45: cs = 45; goto _test_eof; 
	_test_eof46: cs = 46; goto _test_eof; 
	_test_eof47: cs = 47; goto _test_eof; 
	_test_eof48: cs = 48; goto _test_eof; 
	_test_eof49: cs = 49; goto _test_eof; 
	_test_eof50: cs = 50; goto _test_eof; 
	_test_eof51: cs = 51; goto _test_eof; 
	_test_eof52: cs = 52; goto _test_eof; 
	_test_eof53: cs = 53; goto _test_eof; 
	_test_eof54: cs = 54; goto _test_eof; 
	_test_eof55: cs = 55; goto _test_eof; 
	_test_eof56: cs = 56; goto _test_eof; 
	_test_eof57: cs = 57; goto _test_eof; 
	_test_eof58: cs = 58; goto _test_eof; 
	_test_eof59: cs = 59; goto _test_eof; 
	_test_eof60: cs = 60; goto _test_eof; 
	_test_eof61: cs = 61; goto _test_eof; 
	_test_eof62: cs = 62; goto _test_eof; 
	_test_eof63: cs = 63; goto _test_eof; 
	_test_eof64: cs = 64; goto _test_eof; 
	_test_eof65: cs = 65; goto _test_eof; 
	_test_eof66: cs = 66; goto _test_eof; 
	_test_eof67: cs = 67; goto _test_eof; 
	_test_eof68: cs = 68; goto _test_eof; 
	_test_eof69: cs = 69; goto _test_eof; 
	_test_eof70: cs = 70; goto _test_eof; 
	_test_eof71: cs = 71; goto _test_eof; 
	_test_eof72: cs = 72; goto _test_eof; 
	_test_eof73: cs = 73; goto _test_eof; 
	_test_eof74: cs = 74; goto _test_eof; 
	_test_eof75: cs = 75; goto _test_eof; 
	_test_eof76: cs = 76; goto _test_eof; 
	_test_eof77: cs = 77; goto _test_eof; 
	_test_eof78: cs = 78; goto _test_eof; 
	_test_eof79: cs = 79; goto _test_eof; 
	_test_eof80: cs = 80; goto _test_eof; 
	_test_eof81: cs = 81; goto _test_eof; 
	_test_eof82: cs = 82; goto _test_eof; 
	_test_eof83: cs = 83; goto _test_eof; 
	_test_eof84: cs = 84; goto _test_eof; 
	_test_eof85: cs = 85; goto _test_eof; 
	_test_eof86: cs = 86; goto _test_eof; 
	_test_eof87: cs = 87; goto _test_eof; 
	_test_eof88: cs = 88; goto _test_eof; 
	_test_eof89: cs = 89; goto _test_eof; 
	_test_eof90: cs = 90; goto _test_eof; 
	_test_eof91: cs = 91; goto _test_eof; 
	_test_eof92: cs = 92; goto _test_eof; 
	_test_eof93: cs = 93; goto _test_eof; 
	_test_eof94: cs = 94; goto _test_eof; 
	_test_eof95: cs = 95; goto _test_eof; 
	_test_eof96: cs = 96; goto _test_eof; 
	_test_eof97: cs = 97; goto _test_eof; 
	_test_eof98: cs = 98; goto _test_eof; 
	_test_eof99: cs = 99; goto _test_eof; 
	_test_eof100: cs = 100; goto _test_eof; 
	_test_eof101: cs = 101; goto _test_eof; 
	_test_eof102: cs = 102; goto _test_eof; 
	_test_eof103: cs = 103; goto _test_eof; 
	_test_eof104: cs = 104; goto _test_eof; 
	_test_eof105: cs = 105; goto _test_eof; 
	_test_eof106: cs = 106; goto _test_eof; 
	_test_eof107: cs = 107; goto _test_eof; 
	_test_eof108: cs = 108; goto _test_eof; 
	_test_eof109: cs = 109; goto _test_eof; 
	_test_eof110: cs = 110; goto _test_eof; 
	_test_eof111: cs = 111; goto _test_eof; 
	_test_eof112: cs = 112; goto _test_eof; 
	_test_eof113: cs = 113; goto _test_eof; 
	_test_eof114: cs = 114; goto _test_eof; 
	_test_eof115: cs = 115; goto _test_eof; 
	_test_eof116: cs = 116; goto _test_eof; 
	_test_eof117: cs = 117; goto _test_eof; 
	_test_eof118: cs = 118; goto _test_eof; 
	_test_eof119: cs = 119; goto _test_eof; 
	_test_eof120: cs = 120; goto _test_eof; 
	_test_eof121: cs = 121; goto _test_eof; 
	_test_eof122: cs = 122; goto _test_eof; 
	_test_eof123: cs = 123; goto _test_eof; 
	_test_eof124: cs = 124; goto _test_eof; 
	_test_eof125: cs = 125; goto _test_eof; 
	_test_eof126: cs = 126; goto _test_eof; 
	_test_eof127: cs = 127; goto _test_eof; 
	_test_eof128: cs = 128; goto _test_eof; 
	_test_eof129: cs = 129; goto _test_eof; 
	_test_eof130: cs = 130; goto _test_eof; 
	_test_eof131: cs = 131; goto _test_eof; 
	_test_eof132: cs = 132; goto _test_eof; 
	_test_eof133: cs = 133; goto _test_eof; 
	_test_eof134: cs = 134; goto _test_eof; 
	_test_eof135: cs = 135; goto _test_eof; 
	_test_eof136: cs = 136; goto _test_eof; 
	_test_eof137: cs = 137; goto _test_eof; 
	_test_eof138: cs = 138; goto _test_eof; 
	_test_eof139: cs = 139; goto _test_eof; 
	_test_eof140: cs = 140; goto _test_eof; 
	_test_eof141: cs = 141; goto _test_eof; 
	_test_eof142: cs = 142; goto _test_eof; 
	_test_eof143: cs = 143; goto _test_eof; 
	_test_eof144: cs = 144; goto _test_eof; 
	_test_eof145: cs = 145; goto _test_eof; 
	_test_eof146: cs = 146; goto _test_eof; 
	_test_eof147: cs = 147; goto _test_eof; 
	_test_eof148: cs = 148; goto _test_eof; 
	_test_eof149: cs = 149; goto _test_eof; 
	_test_eof150: cs = 150; goto _test_eof; 
	_test_eof151: cs = 151; goto _test_eof; 
	_test_eof152: cs = 152; goto _test_eof; 
	_test_eof153: cs = 153; goto _test_eof; 
	_test_eof154: cs = 154; goto _test_eof; 
	_test_eof155: cs = 155; goto _test_eof; 
	_test_eof156: cs = 156; goto _test_eof; 
	_test_eof157: cs = 157; goto _test_eof; 
	_test_eof158: cs = 158; goto _test_eof; 
	_test_eof159: cs = 159; goto _test_eof; 
	_test_eof160: cs = 160; goto _test_eof; 
	_test_eof161: cs = 161; goto _test_eof; 
	_test_eof162: cs = 162; goto _test_eof; 
	_test_eof163: cs = 163; goto _test_eof; 
	_test_eof164: cs = 164; goto _test_eof; 
	_test_eof165: cs = 165; goto _test_eof; 
	_test_eof166: cs = 166; goto _test_eof; 
	_test_eof167: cs = 167; goto _test_eof; 
	_test_eof168: cs = 168; goto _test_eof; 
	_test_eof169: cs = 169; goto _test_eof; 
	_test_eof170: cs = 170; goto _test_eof; 
	_test_eof171: cs = 171; goto _test_eof; 
	_test_eof172: cs = 172; goto _test_eof; 
	_test_eof173: cs = 173; goto _test_eof; 
	_test_eof174: cs = 174; goto _test_eof; 
	_test_eof175: cs = 175; goto _test_eof; 
	_test_eof176: cs = 176; goto _test_eof; 
	_test_eof177: cs = 177; goto _test_eof; 
	_test_eof178: cs = 178; goto _test_eof; 
	_test_eof179: cs = 179; goto _test_eof; 
	_test_eof180: cs = 180; goto _test_eof; 
	_test_eof181: cs = 181; goto _test_eof; 
	_test_eof182: cs = 182; goto _test_eof; 
	_test_eof183: cs = 183; goto _test_eof; 
	_test_eof184: cs = 184; goto _test_eof; 
	_test_eof185: cs = 185; goto _test_eof; 
	_test_eof186: cs = 186; goto _test_eof; 
	_test_eof187: cs = 187; goto _test_eof; 
	_test_eof188: cs = 188; goto _test_eof; 
	_test_eof189: cs = 189; goto _test_eof; 
	_test_eof190: cs = 190; goto _test_eof; 
	_test_eof191: cs = 191; goto _test_eof; 
	_test_eof192: cs = 192; goto _test_eof; 
	_test_eof193: cs = 193; goto _test_eof; 
	_test_eof194: cs = 194; goto _test_eof; 
	_test_eof195: cs = 195; goto _test_eof; 
	_test_eof196: cs = 196; goto _test_eof; 
	_test_eof197: cs = 197; goto _test_eof; 
	_test_eof198: cs = 198; goto _test_eof; 
	_test_eof199: cs = 199; goto _test_eof; 
	_test_eof200: cs = 200; goto _test_eof; 
	_test_eof201: cs = 201; goto _test_eof; 
	_test_eof202: cs = 202; goto _test_eof; 
	_test_eof203: cs = 203; goto _test_eof; 
	_test_eof204: cs = 204; goto _test_eof; 
	_test_eof205: cs = 205; goto _test_eof; 
	_test_eof206: cs = 206; goto _test_eof; 
	_test_eof207: cs = 207; goto _test_eof; 
	_test_eof208: cs = 208; goto _test_eof; 
	_test_eof209: cs = 209; goto _test_eof; 
	_test_eof210: cs = 210; goto _test_eof; 
	_test_eof211: cs = 211; goto _test_eof; 
	_test_eof212: cs = 212; goto _test_eof; 
	_test_eof213: cs = 213; goto _test_eof; 
	_test_eof214: cs = 214; goto _test_eof; 
	_test_eof215: cs = 215; goto _test_eof; 
	_test_eof216: cs = 216; goto _test_eof; 
	_test_eof217: cs = 217; goto _test_eof; 
	_test_eof218: cs = 218; goto _test_eof; 
	_test_eof219: cs = 219; goto _test_eof; 
	_test_eof220: cs = 220; goto _test_eof; 
	_test_eof221: cs = 221; goto _test_eof; 
	_test_eof222: cs = 222; goto _test_eof; 
	_test_eof223: cs = 223; goto _test_eof; 
	_test_eof224: cs = 224; goto _test_eof; 
	_test_eof225: cs = 225; goto _test_eof; 
	_test_eof226: cs = 226; goto _test_eof; 
	_test_eof227: cs = 227; goto _test_eof; 
	_test_eof228: cs = 228; goto _test_eof; 
	_test_eof229: cs = 229; goto _test_eof; 
	_test_eof230: cs = 230; goto _test_eof; 
	_test_eof231: cs = 231; goto _test_eof; 
	_test_eof232: cs = 232; goto _test_eof; 
	_test_eof233: cs = 233; goto _test_eof; 
	_test_eof234: cs = 234; goto _test_eof; 
	_test_eof235: cs = 235; goto _test_eof; 
	_test_eof236: cs = 236; goto _test_eof; 
	_test_eof237: cs = 237; goto _test_eof; 
	_test_eof252: cs = 252; goto _test_eof; 
	_test_eof238: cs = 238; goto _test_eof; 
	_test_eof239: cs = 239; goto _test_eof; 
	_test_eof240: cs = 240; goto _test_eof; 
	_test_eof241: cs = 241; goto _test_eof; 
	_test_eof242: cs = 242; goto _test_eof; 
	_test_eof243: cs = 243; goto _test_eof; 
	_test_eof244: cs = 244; goto _test_eof; 
	_test_eof245: cs = 245; goto _test_eof; 
	_test_eof246: cs = 246; goto _test_eof; 
	_test_eof247: cs = 247; goto _test_eof; 
	_test_eof248: cs = 248; goto _test_eof; 
	_test_eof249: cs = 249; goto _test_eof; 
	_test_eof250: cs = 250; goto _test_eof; 

	_test_eof: {}
	_out: {}
	}

#line 681 "ebb_request_parser.rl"

  parser->cs = cs;
  parser->top = top;
  COPYSTACK(parser->stack, stack);

  HEADER_CALLBACK(header_field);
  HEADER_CALLBACK(header_value);
  CALLBACK(fragment);
  CALLBACK(query_string);
  CALLBACK(path);
  CALLBACK(uri);

  assert(p <= pe && "buffer overflow after parsing execute");

  return(p - buffer);
}

int ebb_request_parser_has_error(ebb_request_parser *parser) 
{
  return parser->cs == ebb_request_parser_error;
}

int ebb_request_parser_is_finished(ebb_request_parser *parser) 
{
  return parser->cs == ebb_request_parser_first_final;
}

void ebb_request_init(ebb_request *request)
{
  request->expect_continue = FALSE;
  request->eating_body = 0;
  request->body_read = 0;
  request->content_length = 0;
  request->version_major = 0;
  request->version_minor = 0;
  request->number_of_headers = 0;
  request->transfer_encoding = EBB_IDENTITY;
  request->number_of_multipart_headers = 0;
  request->multipart_boundary_len = 0;
  request->multipart_boundary[0] = request->multipart_boundary[1] = '-';
  request->keep_alive = -1;

  request->on_complete = NULL;
  request->on_headers_complete = NULL;
  request->on_body = NULL;
  request->on_multipart_headers_complete = NULL;
  request->on_multipart_header_field = NULL;
  request->on_multipart_header_value = NULL;
  request->on_part_data_complete = NULL;
  request->on_part_data = NULL;
  request->on_header_field = NULL;
  request->on_header_value = NULL;
  request->on_uri = NULL;
  request->on_fragment = NULL;
  request->on_path = NULL;
  request->on_query_string = NULL;
}

int ebb_request_should_keep_alive(ebb_request *request)
{
  if(request->keep_alive == -1)
    if(request->version_major == 1)
      return (request->version_minor != 0);
    else if(request->version_major == 0)
      return FALSE;
    else
      return TRUE;
  else
    return request->keep_alive;
}
