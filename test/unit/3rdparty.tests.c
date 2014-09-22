#include "ccv.h"
#include "case.h"
#include "ccv_case.h"
#include "3rdparty/sfmt/SFMT.h"

TEST_CASE("SFMT shuffle")
{
	sfmt_t sfmt;
	sfmt_init_gen_rand(&sfmt, 11);
	int r[10];
	int i;
	for (i = 0; i < 10; i++)
		r[i] = i;
	sfmt_genrand_shuffle(&sfmt, r, 10, sizeof(int));
	int t[] = {4, 9, 1, 3, 5, 0, 8, 6, 2, 7};
	REQUIRE_ARRAY_EQ(int, r, t, 10, "SFMT shuffle error for int of 10");
}

#include "case_main.h"
