#ifndef _GUARD_uri_h_
#define _GUARD_uri_h_

#include "ebb.h"

typedef struct {
	char* uri;
	ebb_buf (*dispatch)(const void*);
} ccv_uri_dispatch_t;

ccv_uri_dispatch_t* find_uri_dispatch(const char* path);
ebb_buf ebb_bbf_detect_objects(const void* query);

#endif
