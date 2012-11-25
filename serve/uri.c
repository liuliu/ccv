#include "uri.h"
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

const char ebb_http_header[] = "HTTP/1.1 201 Created\r\nCache-Control: no-cache\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: %zd\r\n\r\n";

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
	{
		.uri = "/tld/track.object",
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

static ebb_buf root_discovery;

void uri_init(void)
{
	int i;
	size_t len = sizeof(uri_map) / sizeof(uri_dispatch_t);
	assert(len > 1);
	root_discovery.len = 1;
	for (i = 0; i < len; i++)
	{
		if (uri_map[i].init)
		{
			printf("init context for %s\n", uri_map[i].uri);
			uri_map[i].context = uri_map[i].init();
		} else
			uri_map[i].context = 0;
		if (i > 0)
			root_discovery.len += strlen(uri_map[i].uri) + 3;
	}
	char* data = (char*)malloc(192 /* the head start for http header */ + root_discovery.len);
	snprintf(data, 192, ebb_http_header, root_discovery.len);
	root_discovery.written = strlen(data) + 2;
	data[root_discovery.written - 2] = '[';
	for (i = 1; i < len; i++)
	{
		size_t uri_len = strlen(uri_map[i].uri);
		data[root_discovery.written - 1] = '"';
		memcpy(data + root_discovery.written, uri_map[i].uri, uri_len);
		root_discovery.written += uri_len + 3;
		data[root_discovery.written - 3] = '"';
		data[root_discovery.written - 2] = (i == len - 1) ? ']' : ',';
	}
	data[root_discovery.written - 1] = '\n';
	root_discovery.len = root_discovery.written;
	root_discovery.data = data;
}

void uri_destroy(void)
{
	free(root_discovery.data);
}

int uri_root_discovery(const void* context, const void* parsed, ebb_buf* buf)
{
	buf->data = root_discovery.data;
	buf->len = root_discovery.len;
	return 0;
}
