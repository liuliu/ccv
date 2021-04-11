/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

/**********************************************************
 * CCV - Neural Network Collection
 **********************************************************/

#ifndef GUARD_ccv_nnc_stream_internal_h
#define GUARD_ccv_nnc_stream_internal_h

#include "ccv_nnc.h"
#include "co.h"
#include "3rdparty/khash/khash.h"
#include <pthread.h>

struct ccv_nnc_stream_signal_s {
	int type;
	ccv_nnc_stream_context_t* emit_context;
};

typedef struct {
	// Empty, this will hold things such as NCCL communicator in subclass.
} ccv_nnc_stream_resource_container_t;

typedef struct {
	pthread_mutex_t mutex;
	ccv_array_t* empty;
} ccv_nnc_signal_container_t;

struct ccv_nnc_stream_context_s {
	int type;
	// For resource container
	ccv_nnc_stream_resource_container_t* _inline_container[1];
	ccv_nnc_stream_resource_container_t** resource_container;
	// For scheduler
	co_routine_t* main; // main task.
	co_scheduler_t* scheduler;
	// For neighbor discovery
	ccv_nnc_stream_context_neighbor_discovery_f neighbor_discovery;
	void* neighbor_discovery_context;
	ccv_nnc_signal_container_t* container;
	// For hooks
	ccv_array_t* destructor_hooks;
	int reuse_destructor_hook;
	ccv_nnc_stream_signal_t* checkpoint;
};

typedef struct {
	int in_use;
	ccv_nnc_signal_container_t* container;
	ccv_nnc_stream_signal_t* signal;
} ccv_nnc_signal_handler_t;
KHASH_MAP_INIT_INT64(signal_container, ccv_nnc_signal_container_t*)
// Return the scheduler from a stream (if not created, create one).
CCV_WARN_UNUSED(co_scheduler_t*) ccv_nnc_stream_context_get_scheduler(ccv_nnc_stream_context_t* const stream_context);

#define co_stream_await(_stream) do { if (!_co_stream_await(_self_, _stream)) { return (co_state_t){ __LINE__, 0 }; } case __LINE__: ; } while (0)
int _co_stream_await(co_routine_t* const self, ccv_nnc_stream_context_t* const stream);

ccv_nnc_stream_signal_t* ccv_nnc_stream_context_emit_signal_new(ccv_nnc_stream_context_t* const stream);
/**
 * This is used in ccv_nnc_dynamic_graph_evaluate.c to workaround a particular CUDA ordering issue where when you have:
 * signal_a -> stream_0, stream_1, stream_2, stream_3
 * ... do some computations on stream_0, 1, 2, 3.
 * stream_1 -> signal_b, stream_2 -> signal_c, stream_3 -> signal_d
 * signal_b, signal_c, signal_d -> stream_0
 * stream_0 -> signal_e
 * signal_e -> stream_4
 * stream_4 -> signal_f // signal_f is redundant.
 * signal_f -> stream_5, stream_6, stream_7
 * where stream_0, stream_4 on device 0, stream_1, stream_5 on device 1, stream_2, stream_6 on device 3, stream_3, stream_7 on device 4.
 *
 * In above case, the signal_f can cause CUDA internal issues hence we need to retain signal_e somehow, passing
 * them directly to stream_5, stream_6, stream_7 to wait for. This checkpoint is used for that signal slot.
 */
ccv_nnc_stream_signal_t* ccv_nnc_stream_context_checkpoint(ccv_nnc_stream_context_t* const stream);
void ccv_nnc_stream_context_set_checkpoint(ccv_nnc_stream_context_t* const stream, ccv_nnc_stream_signal_t* const checkpoint);

typedef struct {
	ccv_nnc_callback_f fn;
	void* callback_context;
} ccv_nnc_async_callback_t;

typedef void(*ccv_nnc_async_callback_f)(ccv_nnc_async_callback_t* const async);

typedef struct {
	ccv_nnc_stream_context_t* stream;
	ccv_nnc_stream_signal_t* synced;
} ccv_nnc_synced_stream_t;

KHASH_MAP_INIT_INT(synced_stream, ccv_nnc_synced_stream_t);

#endif
