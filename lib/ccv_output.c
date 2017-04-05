#include "ccv.h"

static int cli_output_levels = 0xffffffff & (~CCV_CLI_VERBOSE);

int ccv_cli_output_level_and_above(int level)
{
	if (level == CCV_CLI_NONE)
		return CCV_CLI_NONE;
	int i;
	uint32_t levels = 0;
	for (i = 0; i < 32; i++)
		if (level <= (1u << i))
			levels |= 1u << i;
	return (int)levels;
}

void ccv_cli_set_output_levels(int levels)
{
	cli_output_levels = levels;
}

int ccv_cli_get_output_levels(void)
{
	return cli_output_levels;
}
