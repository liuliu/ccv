#include <ccv.h>
#include <ev.h>
#include <dispatch/dispatch.h>
#include "ebb.h"
#include "uri.h"
#include "async.h"

static const char* ebb_http_404 = "HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\nContent-Length: 4\r\n\r\n404\n";

typedef struct {
	ebb_request* request;
} ebb_connection_extras;

typedef struct {
	ebb_connection* connection;
	ccv_uri_dispatch_t* dispatcher;
	char* query;
	ebb_buf response;
} ebb_request_extras;

static void on_request_path(ebb_request* request, const char* at, size_t length)
{
	ebb_request_extras* extras = (ebb_request_extras*)request->data;
	char* path = (char*)at;
	char eof = path[length];
	path[length] = '\0';
	extras->dispatcher = find_uri_dispatch(path);
	path[length] = eof;
}

static void on_request_query_string(ebb_request* request, const char* at, size_t length)
{
	ebb_request_extras* request_extras = (ebb_request_extras*)request->data;
	if (request_extras->dispatcher)
	{
	}
}

static void on_connection_response_continue(ebb_connection* connection)
{
	ebb_connection_schedule_close(connection);
}

static void on_request_response(void* context)
{
	ebb_request* request = (ebb_request*)context;
	ebb_request_extras* request_extras = (ebb_request_extras*)request->data;
	ebb_connection* connection = request_extras->connection;
	ebb_connection_write(connection, request_extras->response.data, request_extras->response.len, on_connection_response_continue);
	ccfree(request);
}

static void on_request_processing(void* context)
{
	// this is called off-thread
	ebb_request* request = (ebb_request*)context;
	ebb_request_extras* request_extras = (ebb_request_extras*)request->data;
	request_extras->response = request_extras->dispatcher->dispatch(0);
	dispatch_main_async_f(request, on_request_response);
}

static void on_request_dispatch(ebb_request* request)
{
	ebb_request_extras* request_extras = (ebb_request_extras*)request->data;
	ebb_connection* connection = request_extras->connection;
	if (request_extras->dispatcher)
		dispatch_async_f(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), request, on_request_processing);
	else // write 404
		ebb_connection_write(connection, ebb_http_404, sizeof(ebb_http_404), on_connection_response_continue);
}

static ebb_request* new_request(ebb_connection* connection)
{
	ebb_request* request = (ebb_request*)ccmalloc(sizeof(ebb_request) + sizeof(ebb_request_extras));
	ebb_request_init(request);
	ebb_connection_extras* connection_extras = (ebb_connection_extras*)(connection->data);
	connection_extras->request = request;
	ebb_request_extras* request_extras = (ebb_request_extras*)(request + 1);
	request_extras->connection = connection;
	request->data = request_extras;
	request->on_path = on_request_path;
	request->on_query_string = on_request_query_string;
	request->on_complete = on_request_dispatch;
	return request;
}

static void on_connection_close(ebb_connection* connection)
{
	ccfree(connection);
}

static ebb_connection* new_connection(ebb_server* server, struct sockaddr_in* addr)
{
	ebb_connection* connection = (ebb_connection*)ccmalloc(sizeof(ebb_connection) + sizeof(ebb_connection_extras));
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
	printf("Listen on 3350, http://localhost:3350/\n");
	main_async_init();
	main_async_start(EV_DEFAULT);
	ev_run(EV_DEFAULT_ 0);
	main_async_destroy();
	return 0;
}
