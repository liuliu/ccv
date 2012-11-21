/* HTTP/1.1 Parser
 * Copyright 2008 ryah dahl, ry at tiny clouds punkt org
 *
 * Based on Zed Shaw's parser for Mongrel.
 * Copyright (c) 2005 Zed A. Shaw
 *
 * This software may be distributed under the "MIT" license included in the
 * README
 */
#ifndef ebb_request_parser_h
#define ebb_request_parser_h

#include <sys/types.h> 

typedef struct ebb_request ebb_request;
typedef struct ebb_request_parser  ebb_request_parser;
typedef void (*ebb_header_cb)(ebb_request*, const char *at, size_t length, int header_index);
typedef void (*ebb_element_cb)(ebb_request*, const char *at, size_t length);

#define EBB_RAGEL_STACK_SIZE 10
#define EBB_MAX_MULTIPART_BOUNDARY_LEN 72 // RFC 2046, page 19

struct ebb_request {
  enum { EBB_COPY
       , EBB_DELETE
       , EBB_GET
       , EBB_HEAD
       , EBB_LOCK
       , EBB_MKCOL
       , EBB_MOVE
       , EBB_OPTIONS
       , EBB_POST
       , EBB_PROPFIND
       , EBB_PROPPATCH
       , EBB_PUT
       , EBB_TRACE
       , EBB_UNLOCK
       } method;
  
  enum { EBB_IDENTITY
       , EBB_CHUNKED
       } transfer_encoding;          /* ro */

  size_t content_length;             /* ro - 0 if unknown */
  size_t body_read;                  /* ro */
  int eating_body;                   /* ro */
  int expect_continue;               /* ro */
  unsigned int version_major;        /* ro */
  unsigned int version_minor;        /* ro */
  int number_of_headers;             /* ro */
  int number_of_multipart_headers;   /* ro */
  int keep_alive;                    /* private - use ebb_request_should_keep_alive */

  char multipart_boundary[EBB_MAX_MULTIPART_BOUNDARY_LEN]; /* ro */
  unsigned int multipart_boundary_len; /* ro */

  /* Public  - ordered list of callbacks */
  ebb_element_cb on_path;
  ebb_element_cb on_query_string;
  ebb_element_cb on_uri;
  ebb_element_cb on_fragment;
  ebb_header_cb  on_header_field;
  ebb_header_cb  on_header_value;
  void (*on_headers_complete)(ebb_request *);
  /* multipart data only */
  ebb_header_cb  on_multipart_header_field;
  ebb_header_cb  on_multipart_header_value;
  ebb_element_cb on_part_data;
  void (*on_multipart_headers_complete)(ebb_request *);
  void (*on_part_data_complete)(ebb_request *);
  ebb_element_cb on_body;
  void (*on_complete)(ebb_request *);
  void *data;
};

struct ebb_request_parser {
  int cs;                           /* private */
  int stack[EBB_RAGEL_STACK_SIZE];  /* private */
  int top;                          /* private */
  size_t chunk_size;                /* private */
  unsigned eating:1;                /* private */
  ebb_request *current_request;     /* ro */
  const char *header_field_mark; 
  const char *header_value_mark; 
  const char *query_string_mark; 
  const char *path_mark; 
  const char *uri_mark; 
  const char *fragment_mark; 

  /* multipart data only */
  size_t multipart_index;
  unsigned char multipart_state;
  char multipart_lookbehind[EBB_MAX_MULTIPART_BOUNDARY_LEN + 2];

  /* Public */
  ebb_request* (*new_request)(void*);
  void *data;
};

void ebb_request_parser_init(ebb_request_parser *parser);
size_t ebb_request_parser_execute(ebb_request_parser *parser, const char *data, size_t len);
int ebb_request_parser_has_error(ebb_request_parser *parser);
int ebb_request_parser_is_finished(ebb_request_parser *parser);
void ebb_request_init(ebb_request *);
int ebb_request_should_keep_alive(ebb_request *request);
#define ebb_request_has_body(request) \
  (request->transfer_encoding == EBB_CHUNKED || request->content_length > 0 )

#endif
