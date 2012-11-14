/* libebb web server library
 * Copyright 2008 ryah dahl, ry at tiny clouds punkt org
 *
 * This software may be distributed under the "MIT" license included in the
 * README
 */

#include <assert.h>
#include <string.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/tcp.h> /* TCP_NODELAY */
#include <netinet/in.h>  /* inet_ntoa */
#include <arpa/inet.h>   /* inet_ntoa */
#include <unistd.h>
#include <stdio.h>      /* perror */
#include <errno.h>      /* perror */
#include <stdlib.h> /* for the default methods */
#include <ev.h>

#include "ebb.h"
#include "ebb_request_parser.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define error(FORMAT, ...) fprintf(stderr, "error: " FORMAT "\n", ##__VA_ARGS__)

#define CONNECTION_HAS_SOMETHING_TO_WRITE (connection->to_write != NULL)

static void 
set_nonblock (int fd)
{
  int flags = fcntl(fd, F_GETFL, 0);
  int r = fcntl(fd, F_SETFL, flags | O_NONBLOCK);
  assert(0 <= r && "Setting socket non-block failed!");
}

static ssize_t 
nosigpipe_push(int fd, const void *buf, size_t len)
{
  int flags = 0;
#ifdef MSG_NOSIGNAL
  flags = MSG_NOSIGNAL;
#endif
  return send(fd, buf, len, flags);
}

static void 
close_connection(ebb_connection *connection)
{
  ev_io_stop(connection->server->loop, &connection->read_watcher);
  ev_io_stop(connection->server->loop, &connection->write_watcher);
  ev_timer_stop(connection->server->loop, &connection->timeout_watcher);

  if(0 > close(connection->fd))
    error("problem closing connection fd");

  connection->open = 0;

  if(connection->on_close)
    connection->on_close(connection);
  /* No access to the connection past this point! 
   * The user is allowed to free in the callback
   */
}

/* Internal callback 
 * called by connection->timeout_watcher
 */
static void 
on_timeout(EV_P_ ev_timer *watcher, int revents)
{
  ebb_connection *connection = watcher->data;

  assert(watcher == &connection->timeout_watcher);

  //printf("on_timeout\n");

  /* if on_timeout returns true, we don't time out */
  if(connection->on_timeout) {
    int r = connection->on_timeout(connection);

    if(r == EBB_AGAIN) {
      ebb_connection_reset_timeout(connection);
      return;
    }
  }

  ebb_connection_schedule_close(connection);
}

/* Internal callback 
 * called by connection->read_watcher
 */
static void 
on_readable(EV_P_ ev_io *watcher, int revents)
{
  ebb_connection *connection = watcher->data;
  char base[TCP_MAXWIN];
  char *recv_buffer = base;
  size_t recv_buffer_size = TCP_MAXWIN;
  ssize_t recved;

  //printf("on_readable\n");

  // TODO -- why is this broken?
  //assert(ev_is_active(&connection->timeout_watcher));
  assert(watcher == &connection->read_watcher);

  if(EV_ERROR & revents) {
    error("on_readable() got error event, closing connection.");
    goto error;
  }

  ebb_buf *buf = NULL;
  if(connection->new_buf) {
    buf = connection->new_buf(connection);
    if(buf == NULL) return; 
    recv_buffer = buf->base;
    recv_buffer_size = buf->len;
  }


  recved = recv(connection->fd, recv_buffer, recv_buffer_size, 0);
  if(recved < 0) goto error;
  if(recved == 0) return;


  ebb_connection_reset_timeout(connection);

  ebb_request_parser_execute(&connection->parser, recv_buffer, recved);

  /* parse error? just drop the client. screw the 400 response */
  if(ebb_request_parser_has_error(&connection->parser)) goto error;

  if(buf && buf->on_release)
    buf->on_release(buf);

  return;
error:
  ebb_connection_schedule_close(connection);
}

/* Internal callback 
 * called by connection->write_watcher
 */
static void 
on_writable(EV_P_ ev_io *watcher, int revents)
{
  ebb_connection *connection = watcher->data;
  ssize_t sent;
  
  //printf("on_writable\n");

  assert(CONNECTION_HAS_SOMETHING_TO_WRITE);
  assert(connection->written <= connection->to_write_len);
  // TODO -- why is this broken?
  //assert(ev_is_active(&connection->timeout_watcher));
  assert(watcher == &connection->write_watcher);

  sent = nosigpipe_push( connection->fd
                       , connection->to_write + connection->written
                       , connection->to_write_len - connection->written
                       );
  if(sent < 0) goto error;
  if(sent == 0) return;

  ebb_connection_reset_timeout(connection);

  connection->written += sent;

  if(connection->written == connection->to_write_len) {
    ev_io_stop(EV_A_ watcher);
    connection->to_write = NULL;

    if(connection->after_write_cb)
      connection->after_write_cb(connection);
  }
  return;
error:
  error("close connection on write.");
  ebb_connection_schedule_close(connection);
}

static void 
on_goodbye(EV_P_ ev_timer *watcher, int revents)
{
  ebb_connection *connection = watcher->data;
  assert(watcher == &connection->goodbye_watcher);

  close_connection(connection);
}

static ebb_request* 
new_request_wrapper(void *data)
{
  ebb_connection *connection = data;
  if(connection->new_request)
    return connection->new_request(connection);
  return NULL;
}

/* Internal callback 
 * Called by server->connection_watcher.
 */
static void 
on_connection(EV_P_ ev_io *watcher, int revents)
{
  ebb_server *server = watcher->data;

  //printf("on connection!\n");

  assert(server->listening);
  assert(server->loop == EV_A);
  assert(&server->connection_watcher == watcher);
  
  if(EV_ERROR & revents) {
    error("on_connection() got error event, closing server.");
    ebb_server_stop(server);
    return;
  }

  
  struct sockaddr_in addr; // connector's address information
  socklen_t addr_len = sizeof(addr); 
  int fd = accept( server->fd
                 , (struct sockaddr*) & addr
                 , & addr_len
                 );
  if(fd < 0) {
    perror("accept()");
    return;
  }

  ebb_connection *connection = NULL;
  if(server->new_connection)
    connection = server->new_connection(server, &addr);
  if(connection == NULL) {
    close(fd);
    return;
  } 
  
  set_nonblock(fd);
  connection->fd = fd;
  connection->open = 1;
  connection->server = server;
  memcpy(&connection->sockaddr, &addr, addr_len);
  if(server->port[0] != '\0')
    connection->ip = inet_ntoa(connection->sockaddr.sin_addr);  

#ifdef SO_NOSIGPIPE
  int arg = 1;
  setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &arg, sizeof(int));
#endif

  /* Note: not starting the write watcher until there is data to be written */
  ev_io_set(&connection->write_watcher, connection->fd, EV_WRITE);
  ev_io_set(&connection->read_watcher, connection->fd, EV_READ);

  ev_timer_start(EV_A_ &connection->timeout_watcher);

  ev_io_start(EV_A_ &connection->read_watcher);
}

/**
 * Begin the server listening on a file descriptor.  This DOES NOT start the
 * event loop.  Start the event loop after making this call.
 */
int 
ebb_server_listen_on_fd(ebb_server *server, const int fd)
{
  assert(server->listening == 0);

  if (listen(fd, EBB_MAX_CONNECTIONS) < 0) {
    perror("listen()");
    return -1;
  }
  
  set_nonblock(fd); /* XXX superfluous? */
  
  server->fd = fd;
  server->listening = 1;
  
  ev_io_set (&server->connection_watcher, server->fd, EV_READ);
  ev_io_start (server->loop, &server->connection_watcher);
  
  return server->fd;
}

/**
 * Begin the server listening on a file descriptor This DOES NOT start the
 * event loop. Start the event loop after making this call.
 */
int 
ebb_server_listen_on_port(ebb_server *server, const int port)
{
  int fd = -1;
  struct linger ling = {0, 0};
  struct sockaddr_in addr;
  int flags = 1;
  
  if ((fd = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
    perror("socket()");
    goto error;
  }
  
  flags = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, (void *)&flags, sizeof(flags));
  setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, (void *)&flags, sizeof(flags));
  setsockopt(fd, SOL_SOCKET, SO_LINGER, (void *)&ling, sizeof(ling));

  /* XXX: Sending single byte chunks in a response body? Perhaps there is a
   * need to enable the Nagel algorithm dynamically. For now disabling.
   */
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (void *)&flags, sizeof(flags));
  
  /* the memset call clears nonstandard fields in some impementations that
   * otherwise mess things up.
   */
  memset(&addr, 0, sizeof(addr));
  
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  
  if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
    perror("bind()");
    goto error;
  }
  
  int ret = ebb_server_listen_on_fd(server, fd);
  if (ret >= 0) {
    sprintf(server->port, "%d", port);
  }
  return ret;
error:
  if(fd > 0) close(fd);
  return -1;
}

/**
 * Stops the server. Will not accept new connections.  Does not drop
 * existing connections.
 */
void 
ebb_server_stop(ebb_server *server)
{
  if(server->listening) {
    ev_io_stop(server->loop, &server->connection_watcher);
    close(server->fd);
    server->port[0] = '\0';
    server->listening = 0;
  }
}

/**
 * Initialize an ebb_server structure.  After calling ebb_server_init set
 * the callback server->new_connection and, optionally, callback data
 * server->data.  The new connection MUST be initialized with
 * ebb_connection_init before returning it to the server.
 *
 * @param server the server to initialize
 * @param loop a libev loop
 */
void 
ebb_server_init(ebb_server *server, struct ev_loop *loop)
{
  server->loop = loop;
  server->listening = 0;
  server->port[0] = '\0';
  server->fd = -1;
  server->connection_watcher.data = server;
  ev_init (&server->connection_watcher, on_connection);
  server->secure = 0;

  server->new_connection = NULL;
  server->data = NULL;
}

/**
 * Initialize an ebb_connection structure. After calling this function you
 * must setup callbacks for the different actions the server can take. See
 * server.h for which callbacks are availible. 
 * 
 * This should be called immediately after allocating space for a new
 * ebb_connection structure. Most likely, this will only be called within
 * the ebb_server->new_connection callback which you supply. 
 *
 * If using SSL do consider setting
 *   gnutls_db_set_retrieve_function (connection->session, _);
 *   gnutls_db_set_remove_function (connection->session, _);
 *   gnutls_db_set_store_function (connection->session, _);
 *   gnutls_db_set_ptr (connection->session, _);
 * To provide a better means of storing SSL session caches. libebb provides
 * only a simple default implementation. 
 *
 * @param connection the connection to initialize
 * @param timeout    the timeout in seconds
 */
void 
ebb_connection_init(ebb_connection *connection)
{
  connection->fd = -1;
  connection->server = NULL;
  connection->ip = NULL;
  connection->open = 0;

  ebb_request_parser_init( &connection->parser );
  connection->parser.data = connection;
  connection->parser.new_request = new_request_wrapper;
  
  ev_init (&connection->write_watcher, on_writable);
  connection->write_watcher.data = connection;
  connection->to_write = NULL;

  ev_init(&connection->read_watcher, on_readable);
  connection->read_watcher.data = connection;

  ev_timer_init(&connection->goodbye_watcher, on_goodbye, 0., 0.);
  connection->goodbye_watcher.data = connection;  

  ev_timer_init(&connection->timeout_watcher, on_timeout, EBB_DEFAULT_TIMEOUT, 0.);
  connection->timeout_watcher.data = connection;  

  connection->new_buf = NULL;
  connection->new_request = NULL;
  connection->on_timeout = NULL;
  connection->on_close = NULL;
  connection->data = NULL;
}

void 
ebb_connection_schedule_close (ebb_connection *connection)
{
  ev_timer_start(connection->server->loop, &connection->goodbye_watcher);
}

/* 
 * Resets the timeout to stay alive for another connection->timeout seconds
 */
void 
ebb_connection_reset_timeout(ebb_connection *connection)
{
  ev_timer_again(connection->server->loop, &connection->timeout_watcher);
}

/**
 * Writes a string to the socket. This is actually sets a watcher
 * which may take multiple iterations to write the entire string.
 *
 * The buf->on_release() callback will be made when the operation is complete.
 *
 * This can only be called once at a time. If you call it again
 * while the connection is writing another buffer the ebb_connection_write
 * will return 0 and ignore the request.
 */
int 
ebb_connection_write (ebb_connection *connection, const char *buf, size_t len, ebb_after_write_cb cb)
{
  if(ev_is_active(&connection->write_watcher))
    return 0;
  assert(!CONNECTION_HAS_SOMETHING_TO_WRITE);
  connection->to_write = buf;
  connection->to_write_len = len;
  connection->written = 0;
  connection->after_write_cb = cb;
  ev_io_start(connection->server->loop, &connection->write_watcher);
  return 1;
}
