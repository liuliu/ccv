#ifndef _GUARD_case_h_
#define _GUARD_case_h_

#include <stdio.h>
#include <math.h>

#define INTERNAL_CATCH_UNIQUE_NAME_LINE2( name, line ) name##line
#define INTERNAL_CATCH_UNIQUE_NAME_LINE( name, line ) INTERNAL_CATCH_UNIQUE_NAME_LINE2( name, line )
#define INTERNAL_CATCH_UNIQUE_NAME( name ) INTERNAL_CATCH_UNIQUE_NAME_LINE( name, __LINE__ )

typedef void (*case_f)(char*, int*);

typedef struct {
	case_f driver;
	unsigned long int sig;
	char* name;
} case_t;

#define TEST_CASE(desc) \
static void INTERNAL_CATCH_UNIQUE_NAME(__test_case_driver__) (char* __case_name__, int* __case_result__); \
extern case_t INTERNAL_CATCH_UNIQUE_NAME(__test_case_ctx__); \
case_t INTERNAL_CATCH_UNIQUE_NAME(__test_case_ctx__) = { .driver = INTERNAL_CATCH_UNIQUE_NAME(__test_case_driver__), .sig = 0x883253372849284B, .name = desc }; \
static void INTERNAL_CATCH_UNIQUE_NAME(__test_case_driver__) (char* __case_name__, int* __case_result__) 

#define REQUIRE_EQ(a, b, err) { \
if ((a) != (b)) \
{ \
	printf("\n\t\033[0;31mREQUEST_EQ\033[0;30m: %le != %le, %s", (double)(a), (double)(b), (err)); \
	*(__case_result__) = -1; \
	return; \
} }

#define REQUIRE_EQ_ARRAY(type, a, b, len, err) { \
int i; \
for (i = 0; i < (len); i++) \
	if (((type*)(a))[i] != ((type*)(b))[i]) \
	{ \
		printf("\n\t\033[0;31mREQUEST_EQ_ARRAY\033[0;30m: %le != %le at index %d, %s", (double)((type*)(a))[i], (double)((type*)(b))[i], i, (err)); \
		*(__case_result__) = -1; \
		return; \
	} }

#define REQUIRE_EQ_WITH_TOLERANCE(a, b, t, err) { \
if (fabs((double)((a) - (b))) > (t)) \
{ \
	printf("\n\t\033[0;31mREQUEST_EQ_WITH_TOLERANCE\033[0;30m: %le != %le (+-%le), %s", (double)(a), (double)(b), (double)(t), (err)); \
	*(__case_result__) = -1; \
	return; \
} }

#define REQUIRE_EQ_ARRAY_WITH_TOLERANCE(type, a, b, len, t, err) { \
int i; \
for (i = 0; i < (len); i++) \
	if (fabs((double)(((type*)(a))[i] - ((type*)(b))[i])) > (t)) \
	{ \
		printf("\n\t\033[0;31mREQUEST_EQ_ARRAY_WITH_TOLERANCE\033[0;30m: %le != %le (+-%le) at index %d, %s", (double)((type*)(a))[i], (double)((type*)(b))[i], (double)(t), i, (err)); \
		*(__case_result__) = -1; \
		return; \
	} }

#define REQUIRE_NOT_EQ(a, b, err) { \
if ((a) == (b)) \
{ \
	printf("\n\t\033[0;31mREQUEST_EQ\033[0;30m: %le != %le, %s", (double)(a), (double)(b), (err)); \
	*(__case_result__) = -1; \
	return; \
} }

#define REQUIRE_NOT_EQ_ARRAY(type, a, b, len, err) { \
int i; \
for (i = 0; i < (len); i++) \
	if (((type*)(a))[i] == ((type*)(b))[i]) \
	{ \
		printf("\n\t\033[0;31mREQUEST_EQ_ARRAY\033[0;30m: %le != %le at index %d, %s", (double)((type*)(a))[i], (double)((type*)(b))[i], i, (err)); \
		*(__case_result__) = -1; \
		return; \
	} }

#define REQUIRE_NOT_EQ_WITH_TOLERANCE(a, b, t, err) { \
if (fabs((double)((a) - (b))) <= (t)) \
{ \
	printf("\n\t\033[0;31mREQUEST_EQ_WITH_TLERANCE\033[0;30m: %le != %le (+-%le), %s", (double)(a), (double)(b), (double)(t), (err)); \
	*(__case_result__) = -1; \
	return; \
} }

#define REQUIRE_NOT_EQ_ARRAY_WITH_TOLERANCE(type, a, b, len, t, err) { \
int i; \
for (i = 0; i < (len); i++) \
	if (fabs((double)(((type*)(a))[i] - ((type*)(b))[i])) <= (t)) \
	{ \
		printf("\n\t\033[0;31mREQUEST_EQ_ARRAY_WITH_TOLERANCE\033[0;30m: %le != %le (+-%le) at index %d, %s", (double)((type*)(a))[i], (double)((type*)(b))[i], (double)(t), i, (err)); \
		*(__case_result__) = -1; \
		return; \
	} }

#endif
