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

struct ccv_nnc_stream_signal_s {
	int type;
	ccv_nnc_stream_context_t* emit_context;
};

typedef struct {
	// Empty, this will hold things such as NCCL communicator in subclass.
} ccv_nnc_stream_resource_container_t;

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
	// For hooks
	ccv_array_t* destructor_hooks;
	int reuse_destructor_hook;
	ccv_nnc_stream_signal_t* event;
	ccv_nnc_stream_signal_t* checkpoint;
};

// Return the scheduler from a stream (if not created, create one).
CCV_WARN_UNUSED(co_scheduler_t*) ccv_nnc_stream_context_get_scheduler(ccv_nnc_stream_context_t* const stream_context);

#define co_stream_await(_stream) do { if (!_co_stream_await(_self_, _stream)) { return (co_state_t){ __LINE__, 0 }; } case __LINE__: ; } while (0)
int _co_stream_await(co_routine_t* const self, ccv_nnc_stream_context_t* const stream);

typedef struct {
	ccv_nnc_callback_f fn;
	void* callback_context;
} ccv_nnc_async_callback_t;

typedef void(*ccv_nnc_async_callback_f)(ccv_nnc_async_callback_t* const async);

KHASH_MAP_INIT_INT(stream_map, ccv_nnc_stream_context_t*);

#endif
