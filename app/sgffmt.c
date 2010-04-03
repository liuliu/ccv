#include "ccv.h"

int main(int argc, char** argv)
{
	assert(argc >= 2);
	ccv_sgf_classifier_cascade_t* cascade = ccv_load_sgf_classifier_cascade(argv[1]);
	int len = ccv_sgf_classifier_cascade_write_binary(cascade, NULL, 0);
	char* s = malloc(len);
	ccv_sgf_classifier_cascade_write_binary(cascade, s, len);
	int i;
	for (i = 0; i < len; i++)
		printf("\\x%x", (unsigned char)s[i]);
	fflush(NULL);
	free(s);
	return 0;
}
