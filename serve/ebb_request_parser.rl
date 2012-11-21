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

%%{
  machine ebb_request_parser;

  action mark_header_field   { parser->header_field_mark   = p; }
  action mark_header_value   { parser->header_value_mark   = p; }
  action mark_fragment       { parser->fragment_mark       = p; }
  action mark_query_string   { parser->query_string_mark   = p; }
  action mark_request_path   { parser->path_mark           = p; }
  action mark_request_uri    { parser->uri_mark            = p; }

  action method_copy         { CURRENT->method = EBB_COPY;      }
  action method_delete       { CURRENT->method = EBB_DELETE;    }
  action method_get          { CURRENT->method = EBB_GET;       }
  action method_head         { CURRENT->method = EBB_HEAD;      }
  action method_lock         { CURRENT->method = EBB_LOCK;      }
  action method_mkcol        { CURRENT->method = EBB_MKCOL;     }
  action method_move         { CURRENT->method = EBB_MOVE;      }
  action method_options      { CURRENT->method = EBB_OPTIONS;   }
  action method_post         { CURRENT->method = EBB_POST;      }
  action method_propfind     { CURRENT->method = EBB_PROPFIND;  }
  action method_proppatch    { CURRENT->method = EBB_PROPPATCH; }
  action method_put          { CURRENT->method = EBB_PUT;       }
  action method_trace        { CURRENT->method = EBB_TRACE;     }
  action method_unlock       { CURRENT->method = EBB_UNLOCK;    }

  action write_field { 
    //printf("write_field!\n");
    HEADER_CALLBACK(header_field);
    parser->header_field_mark = NULL;
  }

  action write_value {
    //printf("write_value!\n");
    HEADER_CALLBACK(header_value);
    parser->header_value_mark = NULL;
  }

  action request_uri { 
    //printf("request uri\n");
    CALLBACK(uri);
    parser->uri_mark = NULL;
  }

  action fragment { 
    //printf("fragment\n");
    CALLBACK(fragment);
    parser->fragment_mark = NULL;
  }

  action query_string { 
    //printf("query  string\n");
    CALLBACK(query_string);
    parser->query_string_mark = NULL;
  }

  action request_path {
    //printf("request path\n");
    CALLBACK(path);
    parser->path_mark = NULL;
  }

  action content_length {
    //printf("content_length!\n");
    CURRENT->content_length *= 10;
    CURRENT->content_length += *p - '0';
  }

  action use_identity_encoding { CURRENT->transfer_encoding = EBB_IDENTITY; }
  action use_chunked_encoding { CURRENT->transfer_encoding = EBB_CHUNKED; }

  action set_keep_alive { CURRENT->keep_alive = TRUE; }
  action set_not_keep_alive { CURRENT->keep_alive = FALSE; }

  action multipart_boundary {
    if(CURRENT->multipart_boundary_len == EBB_MAX_MULTIPART_BOUNDARY_LEN) {
      cs = -1;
      fbreak;
    }
    CURRENT->multipart_boundary[1 + (++CURRENT->multipart_boundary_len)] = *p;
    parser->multipart_state = s_start;
  } 

  action expect_continue {
    CURRENT->expect_continue = TRUE;
  }

  action trailer {
    //printf("trailer\n");
    /* not implemenetd yet. (do requests even have trailing headers?) */
  }

  action version_major {
    CURRENT->version_major *= 10;
    CURRENT->version_major += *p - '0';
  }

  action version_minor {
    CURRENT->version_minor *= 10;
    CURRENT->version_minor += *p - '0';
  }

  action end_header_line {
    CURRENT->number_of_headers++;
  }

  action end_headers {
    if(CURRENT->on_headers_complete)
      CURRENT->on_headers_complete(CURRENT);
  }

  action add_to_chunk_size {
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

  action skip_chunk_data {
    //printf("skip chunk data\n");
    //printf("chunk_size: %d\n", parser->chunk_size);
    if(parser->chunk_size > REMAINING) {
      parser->eating = TRUE;
      CURRENT->on_body(CURRENT, p, REMAINING);
      parser->chunk_size -= REMAINING;
      fhold; 
      fbreak;
    } else {
      CURRENT->on_body(CURRENT, p, parser->chunk_size);
      p += parser->chunk_size;
      parser->chunk_size = 0;
      parser->eating = FALSE;
      fhold; 
      fgoto chunk_end; 
    }
  }

  action end_chunked_body {
    //printf("end chunked body\n");
    END_REQUEST;
    fret; // goto Request; 
  }

  action start_req {
    assert(CURRENT == NULL);
    CURRENT = parser->new_request(parser->data);
  }

  action body_logic {
    if(CURRENT->transfer_encoding == EBB_CHUNKED) {
      fcall ChunkedBody;
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

        fhold;

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
        
        fhold; fbreak;  
      }
    }
  }

#
##
###
#### HTTP/1.1 STATE MACHINE
###
##   RequestHeaders and character types are from
#    Zed Shaw's beautiful Mongrel parser.

  CRLF = "\r\n";

# character types
  CTL = (cntrl | 127);
  safe = ("$" | "-" | "_" | ".");
  extra = ("!" | "*" | "'" | "(" | ")" | ",");
  reserved = (";" | "/" | "?" | ":" | "@" | "&" | "=" | "+");
  unsafe = (CTL | " " | "\"" | "#" | "%" | "<" | ">");
  national = any -- (alpha | digit | reserved | extra | safe | unsafe);
  unreserved = (alpha | digit | safe | extra | national);
  escape = ("%" xdigit xdigit);
  uchar = (unreserved | escape);
  pchar = (uchar | ":" | "@" | "&" | "=" | "+");
  tspecials = ("(" | ")" | "<" | ">" | "@" | "," | ";" | ":" | "\\" | "\"" | "/" | "[" | "]" | "?" | "=" | "{" | "}" | " " | "\t");

# elements
  token = (ascii -- (CTL | tspecials));
  quote = "\"";
#  qdtext = token -- "\""; 
#  quoted_pair = "\" ascii;
#  quoted_string = "\"" (qdtext | quoted_pair )* "\"";

#  headers

  Method = ( "COPY"      %method_copy
           | "DELETE"    %method_delete
           | "GET"       %method_get
           | "HEAD"      %method_head
           | "LOCK"      %method_lock
           | "MKCOL"     %method_mkcol
           | "MOVE"      %method_move
           | "OPTIONS"   %method_options
           | "POST"      %method_post
           | "PROPFIND"  %method_propfind
           | "PROPPATCH" %method_proppatch
           | "PUT"       %method_put
           | "TRACE"     %method_trace
           | "UNLOCK"    %method_unlock
           ); # Not allowing extension methods

  HTTP_Version = "HTTP/" digit+ $version_major "." digit+ $version_minor;

  scheme = ( alpha | digit | "+" | "-" | "." )* ;
  absolute_uri = (scheme ":" (uchar | reserved )*);
  path = ( pchar+ ( "/" pchar* )* ) ;
  query = ( uchar | reserved )* >mark_query_string %query_string ;
  param = ( pchar | "/" )* ;
  params = ( param ( ";" param )* ) ;
  rel_path = ( path? (";" params)? ) ;
  absolute_path = ( "/"+ rel_path ) >mark_request_path %request_path ("?" query)?;
  Request_URI = ( "*" | absolute_uri | absolute_path ) >mark_request_uri %request_uri;
  Fragment = ( uchar | reserved )* >mark_fragment %fragment;

  field_name = ( token -- ":" )+;
  Field_Name = field_name >mark_header_field %write_field;

  field_value = ((any - " ") any*)?;
  Field_Value = field_value >mark_header_value %write_value;

  hsep = ":" " "*;
  header = (field_name hsep field_value) :> CRLF;
  Header = ( ("Content-Length"i hsep digit+ $content_length)
           | ("Connection"i hsep 
               ( "Keep-Alive"i %set_keep_alive
               | "close"i %set_not_keep_alive
               )
             )
           | ("Content-Type"i hsep 
              "multipart/form-data" any* 
              "boundary=" ( (quote token+ $multipart_boundary quote) | (token+ $multipart_boundary) ) # boundary can be quoted or not quoted
             )
           | ("Transfer-Encoding"i %use_chunked_encoding hsep "identity" %use_identity_encoding)
           | ("Expect"i hsep "100-continue"i %expect_continue)
           | ("Trailer"i hsep field_value %trailer)
           | (Field_Name hsep Field_Value)
           ) :> CRLF;

  Request_Line = ( Method " " Request_URI ("#" Fragment)? " " HTTP_Version CRLF ) ;
  RequestHeader = Request_Line (Header %end_header_line)* :> CRLF @end_headers;

# chunked message
  trailing_headers = header*;
  #chunk_ext_val   = token | quoted_string;
  chunk_ext_val = token*;
  chunk_ext_name = token*;
  chunk_extension = ( ";" " "* chunk_ext_name ("=" chunk_ext_val)? )*;
  last_chunk = "0"+ chunk_extension CRLF;
  chunk_size = (xdigit* [1-9a-fA-F] xdigit*) $add_to_chunk_size;
  chunk_end  = CRLF;
  chunk_body = any >skip_chunk_data;
  chunk_begin = chunk_size chunk_extension CRLF;
  chunk = chunk_begin chunk_body chunk_end;
  ChunkedBody := chunk* last_chunk trailing_headers CRLF @end_chunked_body;

  Request = RequestHeader >start_req @body_logic;

  main := Request+; # sequence of requests (for keep-alive)
}%%

%% write data;

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
  %% write init;
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

  %% write exec;

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
