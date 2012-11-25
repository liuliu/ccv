#include <stdlib.h>
#include <assert.h>
#include <ev.h>
#include <dispatch/dispatch.h>

typedef struct {
	void *context;
	void (*cb)(void*);
} main_async_t;

static dispatch_semaphore_t async_queue_semaphore;
static int queue_position = 0;
static int queue_pending = 0;
static int queue_length = 10;
static main_async_t* async_queue;
static ev_async main_async;

void main_async_f(void* context, void (*cb)(void*))
{
	assert(cb);
	dispatch_semaphore_wait(async_queue_semaphore, DISPATCH_TIME_FOREVER);
	++queue_pending;
	if (queue_pending > queue_length)
	{
		queue_length = (queue_length * 3 + 1) / 2;
		async_queue = (main_async_t*)realloc(async_queue, sizeof(main_async_t) * queue_length);
		// when expand the queue, the order of our circular buffer is not maintained
		// thus, have to reset the queue_postion here
		queue_position = queue_pending - 1;
	}
	async_queue[queue_position].context = context;
	async_queue[queue_position].cb = cb;
	queue_position = (queue_position + 1) % queue_length;
	dispatch_semaphore_signal(async_queue_semaphore);
	ev_async_send(EV_DEFAULT_ &main_async);
}

static void main_async_drain(EV_P_ ev_async* w, int revents)
{
	dispatch_semaphore_wait(async_queue_semaphore, DISPATCH_TIME_FOREVER);
	while (queue_pending > 0)
	{
		main_async_t async;
		queue_position = (queue_position + queue_length - 1) % queue_length;
		--queue_pending;
		async = async_queue[queue_position];
		dispatch_semaphore_signal(async_queue_semaphore);
		// call the async block outside the lock
		async.cb(async.context);
		// continue the lock so we can get correct queue_pending
		dispatch_semaphore_wait(async_queue_semaphore, DISPATCH_TIME_FOREVER);
	}
	dispatch_semaphore_signal(async_queue_semaphore);
}

void main_async_init(void)
{
	async_queue_semaphore = dispatch_semaphore_create(1);
	async_queue = (main_async_t*)malloc(sizeof(main_async_t) * queue_length);
	ev_async_init(&main_async, main_async_drain);
}

void main_async_start(EV_P)
{
	ev_async_start(EV_A_ &main_async);
}

void main_async_destroy(void)
{
	dispatch_release(async_queue_semaphore);
	free(async_queue);
}
