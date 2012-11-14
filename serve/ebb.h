/* libebb web server library
 * Copyright 2008 ryah dahl, ry at tiny clouds punkt org
 *
 * This software may be distributed under the "MIT" license included in the
 * README
 */
#ifndef EBB_H
#define EBB_H

/* remove this if you want to embed libebb without GNUTLS */

#include <sys/socket.h>
#include <netinet/in.h>
#include <ev.h>
#include "ebb_request_parser.h"

#define EBB_MAX_CONNECTIONS 1024
#define EBB_DEFAULT_TIMEOUT 30.0

#define EBB_AGAIN 0
#define EBB_STOP 1

typedef struct ebb_buf        ebb_buf;
typedef struct ebb_server     ebb_server;
typedef struct ebb_connection ebb_connection;
typedef void (*ebb_after_write_cb) (ebb_connection *connection); 

typedef void (*ebb_connection_cb)(ebb_connection *connection, void *data);

struct ebb_buf {
  size_t written; /* private */

  /* public */
  char *base;
  size_t len;
  void (*on_release)(ebb_buf*);
  void *data;
};

struct ebb_server {
  int fd;                                       /* ro */
  struct sockaddr_in sockaddr;                  /* ro */
  socklen_t socklen;                            /* ro */
  char port[6];                                 /* ro */
  struct ev_loop *loop;                         /* ro */
  unsigned listening:1;                         /* ro */
  unsigned secure:1;                            /* ro */
  ev_io connection_watcher;                     /* private */

  /* Public */

  /* Allocates and initializes an ebb_connection.  NULL by default. */
  ebb_connection* (*new_connection) (ebb_server*, struct sockaddr_in*);

  void *data;
};

struct ebb_connection {
  int fd;                      /* ro */
  struct sockaddr_in sockaddr; /* ro */
  socklen_t socklen;           /* ro */ 
  ebb_server *server;          /* ro */
  char *ip;                    /* ro */
  unsigned open:1;             /* ro */

  const char *to_write;              /* ro */
  size_t to_write_len;               /* ro */
  size_t written;                    /* ro */ 
  ebb_after_write_cb after_write_cb; /* ro */

  ebb_request_parser parser;   /* private */
  ev_io write_watcher;         /* private */
  ev_io read_watcher;          /* private */
  ev_timer timeout_watcher;    /* private */
  ev_timer goodbye_watcher;    /* private */

  /* Public */

  ebb_request* (*new_request) (ebb_connection*); 

  /* The new_buf callback allocates and initializes an ebb_buf structure.
   * By default this is set to a simple malloc() based callback which always
   * returns 4 kilobyte bufs.  Write over it with your own to use your own
   * custom allocation
   *
   * new_buf is called each time there is data from a client connection to
   * be read. See on_readable() in server.c to see exactly how this is used.
   */
  ebb_buf* (*new_buf) (ebb_connection*); 

  /* Returns EBB_STOP or EBB_AGAIN 
   * NULL by default.
   */
  int (*on_timeout) (ebb_connection*); 

  /* The connection was closed */
  void (*on_close) (ebb_connection*); 

  void *data;
};

void ebb_server_init (ebb_server *server, struct ev_loop *loop);
int ebb_server_listen_on_port (ebb_server *server, const int port);
int ebb_server_listen_on_fd (ebb_server *server, const int sfd);
void ebb_server_stop (ebb_server *server);

void ebb_connection_init (ebb_connection *connection);
void ebb_connection_schedule_close (ebb_connection *);
void ebb_connection_reset_timeout (ebb_connection *connection);
int ebb_connection_write (ebb_connection *connection, const char *buf, size_t len, ebb_after_write_cb);

#endif
