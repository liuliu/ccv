#include "uri.h"
#include <string.h>

static uri_dispatch_t uri_map[] = {
	{
		.uri = "/",
		.init = 0,
		.parse = 0,
		.get = uri_root_discovery,
	},
	{
		.uri = "/bbf/detect.objects",
		.init = uri_bbf_detect_objects_init,
		.parse = uri_bbf_detect_objects_parse,
		.get = uri_bbf_detect_objects_intro,
		.post = uri_bbf_detect_objects,
	},
	{
		.uri = "/dpm/detect.objects",
		.init = 0,
		.parse = 0,
		.get = 0,
		.post = 0,
	},
	{
		.uri = "/swt/detect.words",
		.init = 0,
		.parse = 0,
		.get = 0,
		.post = 0,
	},
};

uri_dispatch_t* find_uri_dispatch(const char* path)
{
	uri_dispatch_t* low = (uri_dispatch_t*)uri_map;
	uri_dispatch_t* high = (uri_dispatch_t*)uri_map + sizeof(uri_map) / sizeof(uri_dispatch_t) - 1;
	while (low <= high)
	{
		uri_dispatch_t* middle = low + (high - low) / 2;
		int flag = strcmp(middle->uri, path);
		if (flag == 0)
			return middle;
		else if (flag < 0)
			low = middle + 1;
		else
			high = middle - 1;
	}
	return 0;
}

void uri_init(void)
{
	int i;
	size_t len = sizeof(uri_map) / sizeof(uri_dispatch_t);
	for (i = 0; i < len; i++)
		uri_map[i].context = (uri_map[i].init) ? uri_map[i].init() : 0;
}

ebb_buf uri_root_discovery(const void* context, const void* parsed)
{
	ebb_buf buf;
	const static char root_discovery[] = 
		"HTTP/1.1 200 OK\r\nCache-Control: no-cache\r\nContent-Type: text/plain\r\nContent-Length: 23\r\n\r\n"
		"[\"/bbf/detect.objects\"]\n";
	buf.data = (void*)root_discovery;
	buf.len = sizeof(root_discovery);
	return buf;
}
