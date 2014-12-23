#include <ccv.h>
#include <ev.h>
#include <dispatch/dispatch.h>
#include "ebb.h"
#include "uri.h"
#include "async.h"

typedef struct {
	ebb_request* request;
} ebb_connection_extras;

typedef struct {
	ebb_connection* connection;
	int resource;
	uri_dispatch_t* dispatcher;
	void* context;
	ebb_buf response;
	char uri[256];
	int cursor;
} ebb_request_extras;

static void on_request_path(ebb_request* request, const char* at, size_t length)
{
	ebb_request_extras* request_extras = (ebb_request_extras*)request->data;
	if (length + request_extras->cursor < 256)
		memcpy(request_extras->uri + request_extras->cursor, at, length);
	request_extras->cursor += length;
}

static void on_request_headers_complete(ebb_request* request)
{
	// resolve uri
	ebb_request_extras* request_extras = (ebb_request_extras*)request->data;
	if (request_extras->cursor > 0)
	{
		char* uri = request_extras->uri;
		size_t len = strnlen(request_extras->uri, 256);
		int i;
		int resource = 0, multiple = 1;
		for (i = len - 1; i >= 0; i--)
			if (uri[i] >= '0' && uri[i] <= '9')
			{
				resource += (uri[i] - '0') * multiple;
				multiple *= 10;
			} else if (i == len - 1 || uri[i] != '/') {
				// if we don't pass the first check, or it is not the end, reset i
				i = len;
				resource = -1;
				break;
			} else
				break;
		uri[i] = '\0';
		request_extras->dispatcher = find_uri_dispatch(uri);
		request_extras->resource = resource;
		request_extras->context = 0;
		if (resource >= 0 && request_extras->dispatcher && request_extras->dispatcher->parse)
			request_extras->context = request_extras->dispatcher->parse(request_extras->dispatcher->context, request_extras->context, request_extras->resource, uri, len, URI_PARSE_TERMINATE, 0); // this kicks off resource id
		request_extras->cursor = 0; // done work, reset cursor
	}
}

static void on_request_query_string(ebb_request* request, const char* at, size_t length)
{
	on_request_headers_complete(request); // resolve uri first
	ebb_request_extras* request_extras = (ebb_request_extras*)request->data;
	if (request_extras->dispatcher && request_extras->dispatcher->parse)
		request_extras->context = request_extras->dispatcher->parse(request_extras->dispatcher->context, request_extras->context, request_extras->resource, at, length, URI_QUERY_STRING, 0);
}

static void on_request_part_data(ebb_request* request, const char* at, size_t length)
{
	ebb_request_extras* request_extras = (ebb_request_extras*)request->data;
	if (request_extras->dispatcher && request_extras->dispatcher->parse)
		request_extras->context = request_extras->dispatcher->parse(request_extras->dispatcher->context, request_extras->context, request_extras->resource, at, length, URI_MULTIPART_DATA, -1);
}

static void on_request_multipart_header_field(ebb_request* request, const char* at, size_t length, int header_index)
{
	ebb_request_extras* request_extras = (ebb_request_extras*)request->data;
	if (request_extras->dispatcher && request_extras->dispatcher->parse)
		request_extras->context = request_extras->dispatcher->parse(request_extras->dispatcher->context, request_extras->context, request_extras->resource, at, length, URI_MULTIPART_HEADER_FIELD, header_index);
}

static void on_request_multipart_header_value(ebb_request* request, const char* at, size_t length, int header_index)
{
	ebb_request_extras* request_extras = (ebb_request_extras*)request->data;
	if (request_extras->dispatcher && request_extras->dispatcher->parse)
		request_extras->context = request_extras->dispatcher->parse(request_extras->dispatcher->context, request_extras->context, request_extras->resource, at, length, URI_MULTIPART_HEADER_VALUE, header_index);
}

static void on_request_body(ebb_request* request, const char* at, size_t length)
{
	ebb_request_extras* request_extras = (ebb_request_extras*)request->data;
	if (request_extras->dispatcher && request_extras->dispatcher->parse && request->multipart_boundary_len == 0)
		request_extras->context = request_extras->dispatcher->parse(request_extras->dispatcher->context, request_extras->context, request_extras->resource, at, length, URI_CONTENT_BODY, -1);
}

static void on_connection_response_continue(ebb_connection* connection)
{
	ebb_connection_extras* connection_extras = (ebb_connection_extras*)(connection->data);
	ebb_request* request = connection_extras->request;
	ebb_request_extras* request_extras = (ebb_request_extras*)request->data;
	// call custom release function for the buffer
	if (request_extras->response.data && request_extras->response.on_release)
		request_extras->response.on_release(&request_extras->response);
	ebb_connection_schedule_close(connection);
	free(request);
}

static void on_request_response(void* context)
{
	ebb_request* request = (ebb_request*)context;
	ebb_request_extras* request_extras = (ebb_request_extras*)request->data;
	ebb_connection* connection = request_extras->connection;
	ebb_connection_write(connection, request_extras->response.data, request_extras->response.len, on_connection_response_continue);
}

static void on_request_execute(void* context)
{
	// this is called off-thread
	ebb_request* request = (ebb_request*)context;
	ebb_request_extras* request_extras = (ebb_request_extras*)request->data;
	const static char http_bad_request[] =
		"HTTP/1.1 400 Bad Request\r\nCache-Control: no-cache\r\nContent-Type: application/json; charset=utf8\r\nContent-Length: 6\r\n\r\n"
		"false\n";
	int response_code = 0;
	request_extras->response.on_release = 0;
	switch (request->method)
	{
		case EBB_POST:
			if (request_extras->dispatcher->post)
			{
				response_code = request_extras->dispatcher->post(request_extras->dispatcher->context, request_extras->context, &request_extras->response);
				break;
			}
		case EBB_GET:
			if (request_extras->dispatcher->get)
			{
				response_code = request_extras->dispatcher->get(request_extras->dispatcher->context, request_extras->context, &request_extras->response);
				break;
			}
		case EBB_DELETE:
			if (request_extras->dispatcher->delete)
			{
				response_code = request_extras->dispatcher->delete(request_extras->dispatcher->context, request_extras->context, &request_extras->response);
				break;
			}
		default:
			request_extras->response.data = (void*)ebb_http_404;
			request_extras->response.len = sizeof(ebb_http_404);
			break;
	}
	if (response_code != 0)
	{
		assert(request_extras->response.on_release == 0);
		request_extras->response.data = (void*)http_bad_request;
		request_extras->response.len = sizeof(http_bad_request);
	}
	main_async_f(request, on_request_response);
}

static void on_request_dispatch(ebb_request* request)
{
	ebb_request_extras* request_extras = (ebb_request_extras*)request->data;
	ebb_connection* connection = request_extras->connection;
	if (request_extras->dispatcher)
		dispatch_async_f(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), request, on_request_execute);
	else { // write 404
		request_extras->response.data = 0;
		ebb_connection_write(connection, ebb_http_404, sizeof(ebb_http_404), on_connection_response_continue);
	}
}

static ebb_request* new_request(ebb_connection* connection)
{
	ebb_request* request = (ebb_request*)malloc(sizeof(ebb_request) + sizeof(ebb_request_extras));
	ebb_request_init(request);
	ebb_connection_extras* connection_extras = (ebb_connection_extras*)(connection->data);
	connection_extras->request = request;
	ebb_request_extras* request_extras = (ebb_request_extras*)(request + 1);
	request_extras->connection = connection;
	request_extras->cursor = 0;
	memset(request_extras->uri, 0, sizeof(request_extras->uri));
	request_extras->dispatcher = 0;
	request->data = request_extras;
	request->on_path = on_request_path;
	request->on_part_data = on_request_part_data;
	request->on_multipart_header_field = on_request_multipart_header_field;
	request->on_multipart_header_value = on_request_multipart_header_value;
	request->on_body = on_request_body;
	request->on_query_string = on_request_query_string;
	request->on_headers_complete = on_request_headers_complete;
	request->on_complete = on_request_dispatch;
	return request;
}

static void on_connection_close(ebb_connection* connection)
{
	free(connection);
}

static ebb_connection* new_connection(ebb_server* server, struct sockaddr_in* addr)
{
	ebb_connection* connection = (ebb_connection*)malloc(sizeof(ebb_connection) + sizeof(ebb_connection_extras));
	ebb_connection_init(connection);
	ebb_connection_extras* connection_extras = (ebb_connection_extras*)(connection + 1);
	connection_extras->request = 0;
	connection->data = connection_extras;
	connection->new_request = new_request;
	connection->on_close = on_connection_close;
	return connection;
}

int main(int argc, char** argv)
{
	ebb_server server;
	ebb_server_init(&server, EV_DEFAULT);
	server.new_connection = new_connection;
	ebb_server_listen_on_port(&server, 3350);
	uri_init();
	main_async_init();
	main_async_start(EV_DEFAULT);
	printf("listen on 3350, http://localhost:3350/\n");
	ev_run(EV_DEFAULT_ 0);
	main_async_destroy();
	uri_destroy();
	return 0;
}
