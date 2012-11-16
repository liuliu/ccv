#include "uri.h"
#include <string.h>

static const ccv_uri_dispatch_t uri_map[] = {
	{
		.uri = "/bbf/detect.objects",
		.dispatch = ebb_bbf_detect_objects,
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
