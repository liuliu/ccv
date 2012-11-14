#include <ccv.h>
#include <ev.h>
#include <dispatch/dispatch.h>
#include "ebb.h"

typedef struct {
	ebb_connection* connection;
	ebb_buf* responses;
} ebb_request_extras;

static ev_async main_async;

static void dispatch_main_async(dispatch_block_t block)
{
	dispatch_async(dispatch_get_main_queue(), block);
	ev_async_send(EV_DEFAULT_ &main_async);
}

static void main_async_dispatch(EV_P_ ev_async* w, int revents)
{
	dispatch_main();
}

static void on_request_path(ebb_request* request, const char* at, size_t length)
{
}

static void on_request_query_string(ebb_request* request, const char* at, size_t length)
{
}

static void on_request_dispatch(ebb_request* request)
{
	dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
		// do the real computation off thread
		dispatch_main_async(^{
		});
	});
}

static ebb_request* new_request(ebb_connection* connection)
{
	ebb_request* request = (ebb_request*)ccmalloc(sizeof(ebb_request) + sizeof(ebb_request_extras));
	ebb_request_init(request);
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
	ebb_connection* connection = (ebb_connection*)ccmalloc(sizeof(ebb_connection));
	ebb_connection_init(connection);
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
	ev_async_init(&main_async, main_async_dispatch);
	ev_run(EV_DEFAULT_ 0);
	return 0;
}
