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
};

typedef struct {
	pthread_mutex_t mutex;
	ccv_array_t* empty;
} ccv_nnc_signal_container_t;

typedef struct {
	ccv_nnc_signal_container_t* container;
	ccv_nnc_stream_signal_t* signal;
} ccv_nnc_signal_handler_t;
KHASH_MAP_INIT_INT64(signal_container, ccv_nnc_signal_container_t*)
// Return the scheduler from a stream (if not created, create one).
CCV_WARN_UNUSED(co_scheduler_t*) ccv_nnc_stream_context_get_scheduler(ccv_nnc_stream_context_t* const stream_context);

#define co_stream_await(_stream) do { if (!_co_stream_await(_self_, _stream)) { return (co_state_t){ __LINE__, 0 }; } case __LINE__: ; } while (0)
int _co_stream_await(co_routine_t* const self, ccv_nnc_stream_context_t* const stream);

khash_t(signal_container)* ccv_nnc_signal_container_new(void);
ccv_nnc_stream_signal_t* ccv_nnc_emit_signal_from_container(khash_t(signal_container)* container, ccv_nnc_stream_context_t* const stream);
void ccv_nnc_signal_container_free(khash_t(signal_container)* signal_container);

#endif
