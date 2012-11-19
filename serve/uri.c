#include "uri.h"
#include <string.h>

static const ccv_uri_dispatch_t uri_map[] = {
	{
		.uri = "/",
		.parse = 0,
		.get = uri_root_discovery,
	},
	{
		.uri = "/bbf/detect.objects",
		.parse = uri_bbf_detect_objects_parse,
		.get = uri_bbf_detect_objects_intro,
		.post = uri_bbf_detect_objects,
	},
	{
		.uri = "/dpm/detect.objects",
		.parse = 0,
		.get = 0,
		.post = 0,
	},
	{
		.uri = "/swt/detect.words",
		.parse = 0,
		.get = 0,
		.post = 0,
	},
};

ccv_uri_dispatch_t* find_uri_dispatch(const char* path)
{
	ccv_uri_dispatch_t* low = (ccv_uri_dispatch_t*)uri_map;
	ccv_uri_dispatch_t* high = (ccv_uri_dispatch_t*)uri_map + sizeof(uri_map) / sizeof(ccv_uri_dispatch_t) - 1;
	while (low <= high)
	{
		ccv_uri_dispatch_t* middle = low + (high - low) / 2;
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
}

ebb_buf uri_root_discovery(const void* query)
{
	ebb_buf buf;
	const static char root_discovery[] = 
		"HTTP/1.1 200 OK\r\nCache-Control: no-cache\r\nContent-Type: text/plain\r\nContent-Length: 23\r\n\r\n"
		"[\"/bbf/detect.objects\"]\n";
	buf.data = (void*)root_discovery;
	buf.len = sizeof(root_discovery);
	return buf;
}
