#include "ccv.h"
#include "case.h"
#include "ccv_case.h"

TEST_CASE("set and get cli output levels")
{
	ccv_set_cli_output_levels(CCV_CLI_INFO);
	REQUIRE_EQ(CCV_CLI_INFO, ccv_get_cli_output_levels(), "set output levels should be exactly the get cli output levels");
	ccv_set_cli_output_levels(ccv_cli_output_level_and_above(CCV_CLI_INFO));
	REQUIRE(CCV_CLI_OUTPUT_LEVEL_IS(CCV_CLI_INFO), "cli output levels set to CCV_CLI_INFO should contains CCV_CLI_INFO");
	REQUIRE(CCV_CLI_OUTPUT_LEVEL_IS(CCV_CLI_ERROR), "cli output levels set to CCV_CLI_INFO should contains CCV_CLI_ERROR");
	REQUIRE(!CCV_CLI_OUTPUT_LEVEL_IS(CCV_CLI_VERBOSE), "cli output levels set to CCV_CLI_INFO shouldn't contains CCV_CLI_VERBOSE");
	REQUIRE(!CCV_CLI_OUTPUT_LEVEL_IS(CCV_CLI_NONE), "cli output levels set to CCV_CLI_INFO shouldn't contains CCV_CLI_NONE");
	ccv_set_cli_output_levels(ccv_cli_output_level_and_above(CCV_CLI_NONE));
	REQUIRE(!CCV_CLI_OUTPUT_LEVEL_IS(CCV_CLI_INFO), "cli output levels set to CCV_CLI_NONE shouldn't contains CCV_CLI_INFO");
	REQUIRE(!CCV_CLI_OUTPUT_LEVEL_IS(CCV_CLI_ERROR), "cli output levels set to CCV_CLI_NONE shouldn't contains CCV_CLI_ERROR");
	REQUIRE(!CCV_CLI_OUTPUT_LEVEL_IS(CCV_CLI_VERBOSE), "cli output levels set to CCV_CLI_NONE shouldn't contains CCV_CLI_VERBOSE");
	REQUIRE(!CCV_CLI_OUTPUT_LEVEL_IS(CCV_CLI_NONE), "cli output levels set to CCV_CLI_NONE shouldn't contains CCV_CLI_NONE");
}

#include "case_main.h"
