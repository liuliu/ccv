#include "uri.h"

#define MSG ("HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 12\r\n\r\nhello world\n")

ebb_buf ebb_bbf_detect_objects(const void* query)
{
	ebb_buf buf;
	buf.data = MSG;
	buf.len = sizeof(MSG);
	return buf;
}
