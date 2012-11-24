#include "uri.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

void uri_ebb_buf_free(ebb_buf* buf)
{
	free(buf->data);
	buf->data = 0;
	buf->len = buf->written = 0;
	buf->on_release = 0;
}

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
		.init = uri_dpm_detect_objects_init,
		.parse = uri_dpm_detect_objects_parse,
		.get = uri_dpm_detect_objects_intro,
		.post = uri_dpm_detect_objects,
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
		if (uri_map[i].init)
		{
			printf("init context for %s\n", uri_map[i].uri);
			uri_map[i].context = uri_map[i].init();
		} else
			uri_map[i].context = 0;
}

int uri_root_discovery(const void* context, const void* parsed, ebb_buf* buf)
{
	const static char root_discovery[] = 
		"HTTP/1.1 200 OK\r\nCache-Control: no-cache\r\nContent-Type: application/json\r\nContent-Length: 45\r\n\r\n"
		"[\"/bbf/detect.objects\",\"/dpm/detect.objects\"]\n";
	buf->data = (void*)root_discovery;
	buf->len = sizeof(root_discovery);
	return 0;
}
