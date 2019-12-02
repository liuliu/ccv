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
#include "3rdparty/khash/khash.h"
#include "co.h"

struct ccv_nnc_stream_signal_s {
	int type;
	ccv_nnc_stream_context_t* emit_context;
};

#define CCV_NNC_TASK_STACK_SIZE (262144) // 256KiB

typedef struct {
	// Empty, this will likely hold things such as NCCL communicator.
} ccv_nnc_stream_resource_container_t;

KHASH_MAP_INIT_INT64(signal_container, ccv_nnc_stream_signal_t*)

struct ccv_nnc_stream_context_s {
	int type;
	// For resource container
	ccv_nnc_stream_resource_container_t* _inline_container[1];
	ccv_nnc_stream_resource_container_t** resource_container;
	// For stream signal
	khash_t(signal_container)* signal_container;
	// For scheduler
	co_routine_t* main; // main task.
	co_scheduler_t* scheduler;
	// For neighbor discovery
	ccv_nnc_stream_context_neighbor_discovery_f neighbor_discovery;
	void* neighbor_discovery_context;
};

// Return the scheduler from a stream (if not created, create one).
CCV_WARN_UNUSED(co_scheduler_t*) ccv_nnc_stream_context_get_scheduler(ccv_nnc_stream_context_t* const stream_context);

#define co_stream_await(_stream) if (!_co_stream_await(_self_, _stream)) { return (co_state_t){ __LINE__, 0 }; } case __LINE__:
int _co_stream_await(co_routine_t* const self, ccv_nnc_stream_context_t* const stream);

#endif
