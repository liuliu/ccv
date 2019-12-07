#ifndef GUARD_co_h
#define GUARD_co_h

#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include "ccv.h"

typedef struct co_routine_s co_routine_t;

typedef struct {
	int active;
	int stream_await_count;
	co_routine_t* head;
	co_routine_t* tail;
	pthread_t thread;
	pthread_cond_t notify;
	pthread_cond_t wait;
	pthread_mutex_t mutex;
} co_scheduler_t;

typedef struct {
	int line;
	int done;
} co_state_t;

typedef co_state_t(*co_task_f)(struct co_routine_s* const self, void* const privates);

struct co_routine_s {
	int line;
	int done;
	int root;
	int other_size;
	co_scheduler_t* scheduler;
	co_routine_t* prev;
	co_routine_t* next;
	co_routine_t* notify_any;
	co_routine_t* const* others;
	co_routine_t* callee;
	co_routine_t* caller;
	co_task_f fn;
};

#define co_params_0(_0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12,_13,_14,_15,_16,_17,_18,_19,_20,_21,_22,_23,_24,_25,_26,_27,_28,_29,_30,_31,_32,_33,_34,_35,_36,_37,_38,_39,_40,_41,_42,_43,_44,_45,_46,_47,_48,_49,_50,_51,_52,_53,_54,_55,_56,_57,_58,_59,_60,_61,_62,_63,...) _0;_1;_2;_3;_4;_5;_6;_7;_8;_9;_10;_11;_12;_13;_14;_15;_16;_17;_18;_19;_20;_21;_22;_23;_24;_25;_26;_27;_28;_29;_30;_31;_32;_33;_34;_35;_36;_37;_38;_39;_40;_41;_42;_43;_44;_45;_46;_47;_48;_49;_50;_51;_52;_53;_54;_55;_56;_57;_58;_59;_60;_61;_62;_63;

#define co_params(...) co_params_0(__VA_ARGS__,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,)

#define co_escape(...) __VA_ARGS__

#define co_private(...) __VA_ARGS__

#define co_decl_1(_rettype, _func, _param) \
	co_state_t _func(co_routine_t* const _self_, void* const _privates_); \
	struct _func ## _param_s { \
		_rettype _co_ret; \
		struct { \
			co_params _param \
		} _co_params; \
	}; \
	size_t _func ## _stack_size(void);

#define co_decl_0(_func, _param) \
	co_state_t _func(co_routine_t* const _self, void* const _privates_); \
	struct _func ## _param_s { \
		struct { \
			co_params _param \
		} _co_params; \
	}; \
	size_t _func ## _stack_size(void);

#define co_decl_sel(_0, _1, _2, ...) _2

#define co_decl(_func_or_rettype, _param_or_func, ...) co_decl_sel(_0, ## __VA_ARGS__, co_decl_1, co_decl_0)(_func_or_rettype, _param_or_func, ## __VA_ARGS__)

#define co_task_1(_rettype, _func, _param, _private) \
	struct _func ## _private_s { \
		struct _func ## _param_s _co_params; \
		co_ ## _private \
	}; \
	size_t _func ## _stack_size(void) { return sizeof(struct _func ## _private_s); } \
	co_state_t _func(co_routine_t* const _self_, void* const _privates_) \
	{ \
	struct _private_s { \
		struct _func ## _param_s _co_params; \
		co_ ## _private \
	}; \
	switch (_self_->line) { \
		case 0:

#define co_task_0(_func, _param, _private) \
	struct _func ## _private_s { \
		struct _func ## _param_s _co_params; \
		co_ ## _private \
	}; \
	size_t _func ## _stack_size(void) { return sizeof(struct _func ## _private_s); } \
	co_state_t _func(co_routine_t* const _self_, void* const _privates_) \
	{ \
	struct _private_s { \
		struct _func ## _param_s _co_params; \
		co_ ## _private \
	}; \
	switch (_self_->line) { \
		case 0:

#define co_task_sel(_0, _1, _2, ...) _2

#define co_task(_func_or_rettype, _param_or_func, _private_or_param, ...) co_task_sel(_0, ## __VA_ARGS__, co_task_1, co_task_0)(_func_or_rettype, _param_or_func, _private_or_param, ## __VA_ARGS__)

#define co_decl_task_1(_rettype, _func, _param, _private) \
	co_decl_1(_rettype, _func, _param) \
	co_task_1(_rettype, _func, _param, _private)

#define co_decl_task_0(_func, _param, _private) \
	co_decl_0(_func, _param) \
	co_task_0(_func, _param, _private)

#define co_decl_task_sel(_0, _1, _2, ...) _2

#define co_decl_task(_func_or_rettype, _param_or_func, _private_or_param, ...) co_decl_task_sel(_0, ## __VA_ARGS__, co_decl_task_1, co_decl_task_0)(_func_or_rettype, _param_or_func, _private_or_param, ## __VA_ARGS__)

#define co_end() default: return (co_state_t){ __LINE__, 1 }; } }

#define CO_P(_x) (((struct _private_s*)(_privates_))->_co_params._co_params._x)
#define CO_V(_x) (((struct _private_s*)(_privates_))->_x)

#define co_self() (_self_)

#define co_yield(_val) ((struct _private_s*)(_privates_))->_co_params._co_ret = _val; return (co_state_t){ __LINE__, 0 }; case __LINE__:

#define co_return_1(_val) do { ((struct _private_s*)(_privates_))->_co_params._co_ret = _val; return (co_state_t){ __LINE__, 1 }; } while (0)

#define co_return_0() do { return (co_state_t){ __LINE__, 1 }; } while (0)

#define co_return_sel(_0, _1, _2, ...) _2

#define co_return(...) co_return_sel(_0, ## __VA_ARGS__, co_return_1, co_return_0)(__VA_ARGS__)

#define co_init(_task, _func, _param) do { \
	struct _func ## _param_s params = { \
		._co_params = { co_escape _param } \
	}; \
	_task->fn = _func; \
	_task->line = 0; \
	_task->done = 0; \
	_task->root = 0; \
	_task->other_size = 0; \
	_task->notify_any = 0; \
	_task->others = 0; \
	_task->caller = 0; \
	_task->callee = 0; \
	if (sizeof(params) > 0) \
		memcpy(_task + 1, &params, sizeof(params)); \
} while (0)

#define co_size(_func) (sizeof(co_routine_t) + _func ## _stack_size())

#define co_new(_func, _param) ({ \
	co_routine_t* const task = ccmalloc(co_size(_func)); \
	co_init(task, _func, _param); \
	task; \
})

#define co_retval_2(_task, _rettype) *(_rettype*)(_task + 1)

#define co_retval_0() ((struct _private_s*)(_privates_))->_co_params._co_ret

#define co_retval_sel(_0, _1, _2, _3, ...) _3

#define co_retval(...) co_retval_sel(_0, ## __VA_ARGS__, co_retval_2, co_retval_1, co_retval_0)(__VA_ARGS__)

#define co_await_any(_tasks, _task_size) if (!_co_await_any(_self_, _tasks, _task_size)) { return (co_state_t){ __LINE__, 0 }; } case __LINE__:

#define co_await_1(_task, _val) do { \
	co_await_any(&(_task), 1); \
	_val = co_retval(_task, typeof(_val)); \
} while (0)

#define co_await_0(_task) \
	co_await_any(&(_task), 1)

#define co_await_sel(_0, _1, _2, ...) _2

#define co_await(_task, ...) co_await_sel(_0, ## __VA_ARGS__, co_await_1, co_await_0)(_task, ## __VA_ARGS__)

#define co_apply_1(_func, _param, _val) do { \
	_self_->callee = co_new(_func, _param); \
	_co_apply(_self_, _self_->callee); \
	return (co_state_t){ __LINE__, 0 }; \
	case __LINE__: \
	_val = co_retval(&(_self_->callee), typeof(_val)); \
	co_free(_self_->callee); \
	_self_->callee = 0; \
} while (0)

#define co_apply_0(_func, _param) do { \
	_self_->callee = co_new(_func, _param); \
	_co_apply(_self_, _self_->callee); \
	return (co_state_t){ __LINE__, 0 }; \
	case __LINE__: \
	co_free(_self_->callee); \
	_self_->callee = 0; \
} while (0)

#define co_apply_sel(_0, _1, _2, ...) _2

#define co_apply(_func, _param, ...) co_apply_sel(_0, ## __VA_ARGS__, co_apply_1, co_apply_0)(_func, _param, ## __VA_ARGS__)

#define co_resume_1(_task, _val) do { \
	_co_resume(_self_, _task); \
	return (co_state_t){ __LINE__, 0 }; \
	case __LINE__: \
	_self_->callee = 0; \
	_val = co_retval(_task, typeof(_val)); \
} while (0)

#define co_resume_0(_task) do { \
	_co_resume(_self_, _task); \
	return (co_state_t){ __LINE__, 0 }; \
	case __LINE__: \
	_self_->callee = 0; \
} while (0)

#define co_resume_sel(_0, _1, _2, ...) _2

#define co_resume(_task, ...) co_resume_sel(_0, ## __VA_ARGS__, co_resume_1, co_resume_0)(_task, ## __VA_ARGS__)

void _co_prepend_task(co_scheduler_t* const scheduler, co_routine_t* const task);
void _co_apply(co_routine_t* const self, co_routine_t* const task);
void _co_resume(co_routine_t* const self, co_routine_t* const task);
int _co_await_any(co_routine_t* const self, co_routine_t* const* const tasks, const int task_size);

void co_free(co_routine_t* const task);
int co_is_done(const co_routine_t* const task);
co_scheduler_t* co_scheduler_new(void);
void co_scheduler_free(co_scheduler_t* const scheduler);
void co_schedule(co_scheduler_t* const scheduler, co_routine_t* const task);

#endif
