#include "ccv.h"

static int cli_output_levels = CCV_CLI_INFO;

int ccv_cli_output_level_and_above(int level)
{
	if (level == CCV_CLI_NONE)
		return CCV_CLI_NONE;
	int i;
	int levels = 0;
	for (i = 0; i < 32; i++)
		if (level <= (1 << i))
			levels |= 1 << i;
	return levels;
}

void ccv_set_cli_output_levels(int levels)
{
	cli_output_levels = levels;
}

int ccv_get_cli_output_levels(void)
{
	return cli_output_levels;
}
