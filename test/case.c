#include "case.h"

TEST_CASE("fail test")
{
	REQUIRE_EQ(1, 2, "should fail %d", 1024);
}

TEST_CASE("fail array test")
{
	int a[3] = {1,2,3};
	int b[3] = {1,3,3};
	REQUIRE_ARRAY_EQ(int, a, b, 3, "a / b should be not equal");
}

TEST_CASE("tolerance test")
{
	double a = 1.0 / 3.0;
	REQUIRE_EQ_WITH_TOLERANCE(1, a * 2.999, 0.001, "should fail");
}

TEST_CASE("array tolerance test")
{
	double a[3] = {1,2,3};
	double b[3] = {1,2.001,3};
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(double, a, b, 3, 0.01, "a / b should be not equal");
}

TEST_CASE("pass test")
{
	REQUIRE_EQ(1, 1, "should pass");
}

TEST_CASE("pass array test")
{
	int a[3] = {1,2,3};
	int b[3] = {1,2,3};
	REQUIRE_ARRAY_EQ(int, a, b, 3, "a / b should be equal");
}

#include "case_main.h"
