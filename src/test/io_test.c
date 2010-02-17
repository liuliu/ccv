#include "../ccv.h"

int main(int argc, char** argv)
{
	ccv_unserialize(argv[1], CCV_SERIAL_JPEG_FILE);
	return 0;
}
