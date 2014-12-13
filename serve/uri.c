#include "uri.h"
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

const char ebb_http_header[] = "HTTP/1.0 201 Created\r\nCache-Control: no-cache\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: %zd\r\n\r\n";

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
		.init = uri_root_init,
		.parse = 0,
		.get = uri_root_discovery,
		.post = 0,
		.delete = 0,
		.destroy = uri_root_destroy,
	},
	{
		.uri = "/bbf/detect.objects",
		.init = uri_bbf_detect_objects_init,
		.parse = uri_bbf_detect_objects_parse,
		.get = uri_bbf_detect_objects_intro,
		.post = uri_bbf_detect_objects,
		.delete = 0,
		.destroy = uri_bbf_detect_objects_destroy,
	},
	{
		.uri = "/convnet/classify",
		.init = uri_convnet_classify_init,
		.parse = uri_convnet_classify_parse,
		.get = uri_convnet_classify_intro,
		.post = uri_convnet_classify,
		.delete = 0,
		.destroy = uri_convnet_classify_destroy,
	},
	{
		.uri = "/dpm/detect.objects",
		.init = uri_dpm_detect_objects_init,
		.parse = uri_dpm_detect_objects_parse,
		.get = uri_dpm_detect_objects_intro,
		.post = uri_dpm_detect_objects,
		.delete = 0,
		.destroy = uri_dpm_detect_objects_destroy,
	},
	{
		.uri = "/icf/detect.objects",
		.init = uri_icf_detect_objects_init,
		.parse = uri_icf_detect_objects_parse,
		.get = uri_icf_detect_objects_intro,
		.post = uri_icf_detect_objects,
		.delete = 0,
		.destroy = uri_icf_detect_objects_destroy,
	},
	{
		.uri = "/scd/detect.objects",
		.init = uri_scd_detect_objects_init,
		.parse = uri_scd_detect_objects_parse,
		.get = uri_scd_detect_objects_intro,
		.post = uri_scd_detect_objects,
		.delete = 0,
		.destroy = uri_scd_detect_objects_destroy,
	},
	{
		.uri = "/sift",
		.init = uri_sift_init,
		.parse = uri_sift_parse,
		.get = uri_sift_intro,
		.post = uri_sift,
		.delete = 0,
		.destroy = uri_sift_destroy,
	},
	{
		.uri = "/swt/detect.words",
		.init = uri_swt_detect_words_init,
		.parse = uri_swt_detect_words_parse,
		.get = uri_swt_detect_words_intro,
		.post = uri_swt_detect_words,
		.delete = 0,
		.destroy = uri_swt_detect_words_destroy,
	},
	{
		.uri = "/tld/track.object",
		.init = uri_tld_track_object_init,
		.parse = uri_tld_track_object_parse,
		.get = uri_tld_track_object_intro,
		.post = uri_tld_track_object,
		.delete = uri_tld_track_object_free,
		.destroy = uri_tld_track_object_destroy,
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
	{
		if (uri_map[i].init)
		{
			printf("init context for %s\n", uri_map[i].uri);
			uri_map[i].context = uri_map[i].init();
		} else
			uri_map[i].context = 0;
	}
}

void uri_destroy(void)
{
	int i;
	size_t len = sizeof(uri_map) / sizeof(uri_dispatch_t);
	for (i = 0; i < len; i++)
	{
		if (uri_map[i].destroy)
		{
			printf("destroy context for %s\n", uri_map[i].uri);
			uri_map[i].destroy(uri_map[i].context);
		}
	}
}

void* uri_root_init(void)
{
	int i;
	size_t len = sizeof(uri_map) / sizeof(uri_dispatch_t);
	assert(len > 1);
	size_t root_len = 1;
	for (i = 1; i < len; i++)
		root_len += strlen(uri_map[i].uri) + 3;
	char* data = (char*)malloc(sizeof(ebb_buf) + 192 /* the head start for http header */ + root_len);
	ebb_buf* root_discovery = (ebb_buf*)data;
	data += sizeof(ebb_buf);
	snprintf(data, 192, ebb_http_header, root_len);
	root_discovery->written = strlen(data) + 2;
	data[root_discovery->written - 2] = '[';
	for (i = 1; i < len; i++)
	{
		size_t uri_len = strlen(uri_map[i].uri);
		data[root_discovery->written - 1] = '"';
		memcpy(data + root_discovery->written, uri_map[i].uri, uri_len);
		root_discovery->written += uri_len + 3;
		data[root_discovery->written - 3] = '"';
		data[root_discovery->written - 2] = (i == len - 1) ? ']' : ',';
	}
	data[root_discovery->written - 1] = '\n';
	root_discovery->len = root_discovery->written;
	root_discovery->data = data;
	return root_discovery;
}

int uri_root_discovery(const void* context, const void* parsed, ebb_buf* buf)
{
	ebb_buf* root_discovery = (ebb_buf*)context;
	buf->data = root_discovery->data;
	buf->len = root_discovery->len;
	return 0;
}

void uri_root_destroy(void* context)
{
	free(context);
}
