#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/mps/ccv_nnc_mps.h>
#include <3rdparty/dsfmt/dSFMT.h>
#ifdef HAVE_MPS
#include <dispatch/dispatch.h>
#include <pthread.h>
#endif
#include <math.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_CASE("mps fork preserves default graph dependencies")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALAR_MUL_FORWARD, CCV_NNC_BACKEND_MPS));
#ifdef HAVE_MPS
	const int watermark = ccv_nnc_mps_queue_watermark();
	ccv_nnc_mps_set_queue_watermark(128);
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_dynamic_graph_set_no_grad(graph, 1);
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_tensor_variable_t x = ccv_nnc_tensor_variable_new(graph, GPU_TENSOR_NHWC(000, 32F, 4096));
	ccv_nnc_tensor_variable_t y = ccv_nnc_tensor_variable_new(graph, GPU_TENSOR_NHWC(000, 32F, 4096));
	ccv_nnc_dynamic_graph_exec(graph, CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(), TENSOR_VARIABLE_LIST(x, y), 0, 0);
	ccv_nnc_dynamic_graph_exec(graph, CMD_SET_FORWARD(3), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(), TENSOR_VARIABLE_LIST(x), 0, stream);
	dispatch_semaphore_t taken = dispatch_semaphore_create(0);
	dispatch_semaphore_t finish = dispatch_semaphore_create(0);
	dispatch_group_t group = dispatch_group_create();
	dispatch_group_async(group, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		ccv_nnc_fork();
		ccv_nnc_stream_context_t* const other_stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
		// Pause inside an independent operation, after command-buffer checkout.
		mtl_command_batch_t* const batch = ccv_nnc_stream_context_start_command_batch(other_stream);
		dispatch_semaphore_signal(taken);
		dispatch_semaphore_wait(finish, DISPATCH_TIME_FOREVER);
		ccv_nnc_stream_context_finish_command_batch(other_stream, batch);
		ccv_nnc_stream_context_wait(other_stream);
		ccv_nnc_stream_context_free(other_stream);
		ccv_nnc_join();
	});
	dispatch_semaphore_wait(taken, DISPATCH_TIME_FOREVER);
	ccv_nnc_dynamic_graph_exec(graph, CMD_SCALAR_MUL_FORWARD(2), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x), TENSOR_VARIABLE_LIST(y), 0, stream);
	ccv_nnc_stream_context_wait(stream);
	ccv_nnc_tensor_t* const result = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4096), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ccv_nnc_tensor_from_variable(graph, y)), TENSOR_LIST(result), 0);
	int i, mismatches = 0;
	for (i = 0; i < 4096; i++)
		mismatches += result->data.f32[i] != 6;
	dispatch_semaphore_signal(finish);
	dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
	dispatch_release(group);
	dispatch_release(finish);
	dispatch_release(taken);
	ccv_nnc_tensor_free(result);
	ccv_nnc_dynamic_graph_free(graph);
	ccv_nnc_stream_context_free(stream);
	ccv_nnc_mps_set_queue_watermark(watermark);
	REQUIRE_EQ(mismatches, 0, "another encoding thread must not delay this graph's producer past its consumer");
#endif
}

TEST_CASE("mps thread MFA scratch and shader caches are isolated")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALAR_MUL_FORWARD, CCV_NNC_BACKEND_MPS));
#ifdef HAVE_MPS
	ccv_nnc_mfa_context_t* contexts[2] = {};
	mtl_buffer_t* buffers[2] = {};
	ccv_nnc_mfa_context_t** const context_ptr = contexts;
	mtl_buffer_t** const buffer_ptr = buffers;
	dispatch_semaphore_t ready = dispatch_semaphore_create(0);
	dispatch_semaphore_t finish = dispatch_semaphore_create(0);
	dispatch_group_t group = dispatch_group_create();
	int i;
	for (i = 0; i < 2; i++)
	{
		const int index = i;
		dispatch_group_async(group, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
			ccv_nnc_fork();
			context_ptr[index] = ccv_nnc_default_mfa_context();
			buffer_ptr[index] = ccv_nnc_mfa_request_scratch(context_ptr[index], 4096);
			dispatch_semaphore_signal(ready);
			dispatch_semaphore_wait(finish, DISPATCH_TIME_FOREVER);
			ccv_nnc_join();
		});
	}
	dispatch_semaphore_wait(ready, DISPATCH_TIME_FOREVER);
	dispatch_semaphore_wait(ready, DISPATCH_TIME_FOREVER);
	const int isolated = contexts[0] && contexts[1] && contexts[0] != contexts[1] && buffers[0] && buffers[1] && buffers[0] != buffers[1];
	dispatch_semaphore_signal(finish);
	dispatch_semaphore_signal(finish);
	dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
	dispatch_release(group);
	dispatch_release(finish);
	dispatch_release(ready);
	REQUIRE(isolated, "concurrent callers must not share mutable MFA caches or scratch");
#endif
}

#ifdef HAVE_MPS
typedef struct {
	int seed;
	int mismatches;
	int fork;
} mps_thread_graph_result_t;

static void* _mps_thread_graph_run(void* const opaque)
{
	mps_thread_graph_result_t* const result = (mps_thread_graph_result_t*)opaque;
	if (result->fork)
		ccv_nnc_fork();
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, result->seed);
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_dynamic_graph_set_no_grad(graph, 1);
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	int iteration;
	for (iteration = 0; iteration < 32; iteration++)
	{
		const int m = 32 + 16 * ((iteration + result->seed) % 3);
		const int n = 64 + 16 * ((iteration + result->seed) % 5);
		const int k = 128;
		ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, m, k), 0);
		ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, n, k), 0);
		ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, m, n), 0);
		ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, m, n), 0);
		// Exactly representable inputs keep this sensitive to concurrency errors,
		// independently of the backend's mixed-precision GEMM policy.
		int i;
		for (i = 0; i < m * k; i++)
			a->data.f32[i] = floorf((dsfmt_genrand_open_close(&dsfmt) - 0.5) * 32) / 32;
		for (i = 0; i < n * k; i++)
			b->data.f32[i] = floorf((dsfmt_genrand_open_close(&dsfmt) - 0.5) * 32) / 32;
		ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(expected), 0);
		ccv_nnc_tensor_variable_t x = ccv_nnc_tensor_variable_new(graph, GPU_TENSOR_NHWC(000, 32F, m, k));
		ccv_nnc_tensor_variable_t w = ccv_nnc_tensor_variable_new(graph, GPU_TENSOR_NHWC(000, 32F, n, k));
		ccv_nnc_tensor_variable_t y = ccv_nnc_tensor_variable_new(graph, GPU_TENSOR_NHWC(000, 32F, m, n));
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(ccv_nnc_tensor_from_variable(graph, x), ccv_nnc_tensor_from_variable(graph, w)), stream);
		ccv_nnc_dynamic_graph_exec(graph, CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x, w), TENSOR_VARIABLE_LIST(y), 0, stream);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ccv_nnc_tensor_from_variable(graph, y)), TENSOR_LIST(actual), stream);
		ccv_nnc_stream_context_wait(stream);
		for (i = 0; i < m * n; i++)
		{
			result->mismatches += !isfinite(actual->data.f32[i]) || fabsf(actual->data.f32[i] - expected->data.f32[i]) > 1e-5;
		}
		ccv_nnc_tensor_variable_free(graph, x);
		ccv_nnc_tensor_variable_free(graph, w);
		ccv_nnc_tensor_variable_free(graph, y);
		ccv_nnc_tensor_free(a);
		ccv_nnc_tensor_free(b);
		ccv_nnc_tensor_free(expected);
		ccv_nnc_tensor_free(actual);
	}
	ccv_nnc_dynamic_graph_free(graph);
	ccv_nnc_stream_context_free(stream);
	return 0;
}
#endif

TEST_CASE("mps default and forked dynamic graphs match CPU across shape changes")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
#ifdef HAVE_MPS
	const int watermark = ccv_nnc_mps_queue_watermark();
	ccv_nnc_mps_set_queue_watermark(128);
	mps_thread_graph_result_t results[2] = {{.seed = 1}, {.seed = 2, .fork = 1}};
	mps_thread_graph_result_t serial = {.seed = 1};
	_mps_thread_graph_run(&serial);
	REQUIRE_EQ(serial.mismatches, 0, "serial graph must match CPU reference");
	pthread_t threads[2];
	int i;
	for (i = 0; i < 2; i++)
		pthread_create(&threads[i], 0, _mps_thread_graph_run, &results[i]);
	for (i = 0; i < 2; i++)
		pthread_join(threads[i], 0);
	ccv_nnc_mps_set_queue_watermark(watermark);
	REQUIRE_EQ(results[0].mismatches, 0, "first graph must match CPU reference");
	REQUIRE_EQ(results[1].mismatches, 0, "second graph must match CPU reference");
#endif
}

#ifdef HAVE_MPS
typedef struct {
	dispatch_queue_t queue;
	dispatch_semaphore_t producer_ready;
	dispatch_semaphore_t consumer_done;
	ccv_nnc_stream_context_t* stream;
	ccv_nnc_tensor_t* x;
	ccv_nnc_tensor_t* y;
	ccv_nnc_tensor_t* result;
	ccv_nnc_mfa_context_t* context;
	mtl_buffer_t* scratch;
	pthread_t producer_thread;
	pthread_t consumer_thread;
	int same_context;
} mps_queue_migration_t;

static void* _mps_queue_producer(void* const opaque)
{
	mps_queue_migration_t* const state = (mps_queue_migration_t*)opaque;
	dispatch_sync(state->queue, ^{
		state->producer_thread = pthread_self();
		state->context = ccv_nnc_default_mfa_context();
		state->scratch = ccv_nnc_mfa_request_scratch(state->context, 4096);
		ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
		ccv_nnc_cmd_exec(CMD_SET_FORWARD(3), ccv_nnc_no_hint, 0, 0, 0, TENSOR_LIST(state->x), stream);
		// Leave the batch pending; the consumer continues this stream on
		// another worker after this Dispatch block returns.
		state->stream = stream;
	});
	dispatch_semaphore_signal(state->producer_ready);
	// Keep this caller alive so the second block cannot reuse its pthread.
	dispatch_semaphore_wait(state->consumer_done, DISPATCH_TIME_FOREVER);
	return 0;
}

static void* _mps_queue_consumer(void* const opaque)
{
	mps_queue_migration_t* const state = (mps_queue_migration_t*)opaque;
	dispatch_semaphore_wait(state->producer_ready, DISPATCH_TIME_FOREVER);
	dispatch_sync(state->queue, ^{
		state->consumer_thread = pthread_self();
		ccv_nnc_mfa_context_t* const context = ccv_nnc_default_mfa_context();
		state->same_context = context == state->context && ccv_nnc_mfa_request_scratch(context, 4096) == state->scratch;
		ccv_nnc_stream_context_t* const stream = state->stream;
		ccv_nnc_cmd_exec(CMD_SCALAR_MUL_FORWARD(2), ccv_nnc_no_hint, 0, TENSOR_LIST(state->x), TENSOR_LIST(state->y), stream);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(state->y), TENSOR_LIST(state->result), stream);
		ccv_nnc_stream_context_wait(stream);
		ccv_nnc_stream_context_free(stream);
	});
	dispatch_semaphore_signal(state->consumer_done);
	return 0;
}
#endif

TEST_CASE("mps default execution still follows serial queue worker migration")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALAR_MUL_FORWARD, CCV_NNC_BACKEND_MPS));
#ifdef HAVE_MPS
	const int watermark = ccv_nnc_mps_queue_watermark();
	ccv_nnc_mps_set_queue_watermark(128);
	mps_queue_migration_t state = {
		.queue = dispatch_queue_create("ccv.test.migration", DISPATCH_QUEUE_SERIAL),
		.producer_ready = dispatch_semaphore_create(0),
		.consumer_done = dispatch_semaphore_create(0),
		.x = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4096), 0),
		.y = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4096), 0),
		.result = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4096), 0),
	};
	ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, 0, 0, TENSOR_LIST(state.x, state.y), 0);
	pthread_t producer, consumer;
	pthread_create(&producer, 0, _mps_queue_producer, &state);
	pthread_create(&consumer, 0, _mps_queue_consumer, &state);
	pthread_join(producer, 0);
	pthread_join(consumer, 0);
	int i, mismatches = 0;
	for (i = 0; i < 4096; i++)
		mismatches += state.result->data.f32[i] != 6;
	ccv_nnc_tensor_free(state.x);
	ccv_nnc_tensor_free(state.y);
	ccv_nnc_tensor_free(state.result);
	dispatch_release(state.queue);
	dispatch_release(state.producer_ready);
	dispatch_release(state.consumer_done);
	ccv_nnc_mps_set_queue_watermark(watermark);
	REQUIRE(!pthread_equal(state.producer_thread, state.consumer_thread), "the test must exercise two different queue workers");
	REQUIRE(state.same_context, "the serial queue must keep its caches and scratch when its worker changes");
	REQUIRE_EQ(mismatches, 0, "a migrated queue must submit its pending producer before its consumer");
#endif
}

TEST_CASE("mps fork is idempotent and join restores the default pending batch")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALAR_MUL_FORWARD, CCV_NNC_BACKEND_MPS));
#ifdef HAVE_MPS
	const int watermark = ccv_nnc_mps_queue_watermark();
	ccv_nnc_mps_set_queue_watermark(128);
	ccv_nnc_mfa_context_t* const original = ccv_nnc_default_mfa_context();
	mtl_buffer_t* const scratch = ccv_nnc_mfa_request_scratch(original, 4096);
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_tensor_t* const x = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4096), 0);
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4096), 0);
	ccv_nnc_tensor_t* const result = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4096), 0);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, 0, 0, TENSOR_LIST(x, y), 0);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD(3), ccv_nnc_no_hint, 0, 0, 0, TENSOR_LIST(x), stream);
	const int first = ccv_nnc_fork();
	ccv_nnc_mfa_context_t* const detached = ccv_nnc_default_mfa_context();
	const int isolated = detached != original && ccv_nnc_mfa_request_scratch(detached, 4096) != scratch;
	const int second = ccv_nnc_fork();
	const int unchanged = ccv_nnc_default_mfa_context() == detached;
	ccv_nnc_mps_clear_graph_executable_cache();
	ccv_nnc_join();
	const int restored = ccv_nnc_default_mfa_context() == original && ccv_nnc_mfa_request_scratch(original, 4096) == scratch;
	ccv_nnc_cmd_exec(CMD_SCALAR_MUL_FORWARD(2), ccv_nnc_no_hint, 0, TENSOR_LIST(x), TENSOR_LIST(y), stream);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y), TENSOR_LIST(result), stream);
	ccv_nnc_stream_context_wait(stream);
	int i, mismatches = 0;
	for (i = 0; i < 4096; i++)
		mismatches += result->data.f32[i] != 6;
	ccv_nnc_tensor_free(x);
	ccv_nnc_tensor_free(y);
	ccv_nnc_tensor_free(result);
	ccv_nnc_stream_context_free(stream);
	ccv_nnc_mps_set_queue_watermark(watermark);
	REQUIRE(first == 1 && second == 0 && isolated && unchanged && restored, "fork must detach once and join must restore existing default state");
	REQUIRE_EQ(mismatches, 0, "fork and join must not lose default execution's pending producer");
#endif
}

TEST_CASE("mps scoped forks on Dispatch workers match CPU and restore default state")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
#ifdef HAVE_MPS
	const int watermark = ccv_nnc_mps_queue_watermark();
	ccv_nnc_mps_set_queue_watermark(128);
	mps_thread_graph_result_t results[2] = {{.seed = 1}, {.seed = 2}};
	mps_thread_graph_result_t* const result_ptr = results;
	dispatch_queue_t queues[2] = {
		dispatch_queue_create("ccv.test.graph.first", DISPATCH_QUEUE_SERIAL),
		dispatch_queue_create("ccv.test.graph.second", DISPATCH_QUEUE_SERIAL),
	};
	dispatch_group_t group = dispatch_group_create();
	int i;
	for (i = 0; i < 2; i++)
	{
		const int index = i;
		dispatch_group_async(group, queues[i], ^{
			ccv_nnc_mfa_context_t* const original = ccv_nnc_default_mfa_context();
			const int detached = ccv_nnc_fork();
			_mps_thread_graph_run(&result_ptr[index]);
			if (detached)
				ccv_nnc_join();
			result_ptr[index].mismatches += ccv_nnc_default_mfa_context() != original;
		});
	}
	dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
	for (i = 0; i < 2; i++)
		dispatch_release(queues[i]);
	dispatch_release(group);
	ccv_nnc_mps_set_queue_watermark(watermark);
	REQUIRE_EQ(results[0].mismatches, 0, "first queue's graph must match CPU reference");
	REQUIRE_EQ(results[1].mismatches, 0, "second queue's graph must match CPU reference");
#endif
}

#ifdef HAVE_MPS
typedef struct {
	ccv_nnc_tensor_t* tensor;
	ccv_nnc_stream_context_t* stream;
} mps_fork_exit_t;

static void* _mps_fork_pending_at_exit(void* const opaque)
{
	ccv_nnc_fork();
	mps_fork_exit_t* const state = (mps_fork_exit_t*)opaque;
	ccv_nnc_tensor_t* const tensor = state->tensor;
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD(7), ccv_nnc_no_hint, 0, 0, 0, TENSOR_LIST(tensor), stream);
	// Keep the stream alive so its destructor cannot flush this pending batch.
	state->stream = stream;
	return 0;
}
#endif

TEST_CASE("mps fork thread destructor flushes pending work")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS));
#ifdef HAVE_MPS
	const int watermark = ccv_nnc_mps_queue_watermark();
	ccv_nnc_mps_set_queue_watermark(128);
	ccv_nnc_tensor_t* const tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4096), 0);
	ccv_nnc_tensor_t* const result = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4096), 0);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, 0, 0, TENSOR_LIST(tensor), 0);
	pthread_t thread;
	mps_fork_exit_t state = {.tensor = tensor};
	pthread_create(&thread, 0, _mps_fork_pending_at_exit, &state);
	pthread_join(thread, 0);
	ccv_nnc_stream_context_free(state.stream);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(result), 0);
	int i, mismatches = 0;
	for (i = 0; i < 4096; i++)
		mismatches += result->data.f32[i] != 7;
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(result);
	ccv_nnc_mps_set_queue_watermark(watermark);
	REQUIRE_EQ(mismatches, 0, "thread exit must submit pending commands and wait before freeing the fork");
#endif
}

#ifdef HAVE_MPS
static void _mps_fork_count_callback(void* const opaque)
{
	++*(int*)opaque;
}
#endif

TEST_CASE("mps fork callbacks accept work completed by a default wait")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS));
#ifdef HAVE_MPS
	const int detached = ccv_nnc_fork();
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_tensor_t* const tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4096), 0);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD(7), ccv_nnc_no_hint, 0, 0, 0, TENSOR_LIST(tensor), stream);
	ccv_nnc_stream_context_commit(stream);
	// Complete the GPU work on another thread without clearing this fork's fence.
	dispatch_group_t group = dispatch_group_create();
	dispatch_group_async(group, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		ccv_nnc_synchronize_stream_context(0);
	});
	dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
	dispatch_release(group);
	int callback_count = 0;
	ccv_nnc_stream_context_add_callback(stream, _mps_fork_count_callback, &callback_count);
	ccv_nnc_stream_context_add_callback(stream, _mps_fork_count_callback, &callback_count);
	ccv_nnc_stream_context_wait(stream);
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_stream_context_free(stream);
	if (detached)
		ccv_nnc_join();
	REQUIRE_EQ(callback_count, 2, "each callback must run once even when the retained command buffer has completed");
#endif
}

TEST_CASE("mps default and forked graph executable caches support concurrent shape changes")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
#ifdef HAVE_MPS
	const uint64_t flags = ccv_nnc_flags();
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	mps_thread_graph_result_t results[2] = {{.seed = 1}, {.seed = 2, .fork = 1}};
	pthread_t threads[2];
	int i;
	for (i = 0; i < 2; i++)
		pthread_create(&threads[i], 0, _mps_thread_graph_run, &results[i]);
	for (i = 0; i < 2; i++)
		pthread_join(threads[i], 0);
	if (!(flags & CCV_NNC_DISABLE_MFA))
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	REQUIRE_EQ(results[0].mismatches, 0, "default MPSGraph cache must match CPU reference");
	REQUIRE_EQ(results[1].mismatches, 0, "forked MPSGraph cache must match CPU reference");
#endif
}

#include "case_main.h"
