#include "uri.h"

void* uri_bbf_detect_objects_parse(void* parsed, const char* key, const char* value)
{
	return 0;
}

ebb_buf uri_bbf_detect_objects_intro(const void* query)
{
	ebb_buf buf;
	const static char bbf_intro[] = 
		"HTTP/1.1 200 OK\r\nCache-Control: no-cache\r\nAccept: \r\nContent-Type: text/html\r\nContent-Length: 156\r\n\r\n"
		"<html><body><form enctype='multipart/form-data' method='post'><input type='file' name='a'><input type='file' name='b'><input type='submit'></form>\n";
	buf.data = (void*)bbf_intro;
	buf.len = sizeof(bbf_intro);
	return buf;
}

ebb_buf uri_bbf_detect_objects(const void* query)
{
	ebb_buf buf;
	const static char bbf_intro[] = 
		"HTTP/1.1 201 Created\r\nCache-Control: no-cache\r\nContent-Type: text/plain\r\nContent-Length: 3\r\n\r\n"
		"OK\n";
	buf.data = (void*)bbf_intro;
	buf.len = sizeof(bbf_intro);
	return buf;
}
