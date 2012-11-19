#ifndef _GUARD_uri_h_
#define _GUARD_uri_h_

#include "ebb.h"

typedef struct {
	char* uri;
	void* (*parse)(void*, const char*, const char*); // this runs on main thread
	ebb_buf (*get)(const void*); // this runs off thread
	ebb_buf (*post)(const void*); // this runs off thread
} ccv_uri_dispatch_t;

ccv_uri_dispatch_t* find_uri_dispatch(const char* path);
void uri_init(void);

ebb_buf uri_root_discovery(const void* query);

void* uri_bbf_detect_objects_parse(void* parsed, const char* key, const char* value);
ebb_buf uri_bbf_detect_objects_intro(const void* query);
ebb_buf uri_bbf_detect_objects(const void* query);

#endif
