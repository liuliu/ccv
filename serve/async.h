#ifndef _GUARD_async_h_
#define _GUARD_async_h_

// it tries to maintain a FIFO order, but it may not be the case sometimes
void main_async_f(void* context, void (*cb)(void*));
void main_async_init(void);
void main_async_start(EV_P);
void main_async_destroy(void);

#endif
