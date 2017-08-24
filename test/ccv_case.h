#ifndef _GUARD_ccv_case_h_
#define _GUARD_ccv_case_h_

#include <math.h>
#include <assert.h>
#include <string.h>
#include <sys/stat.h>

#define REQUIRE_MATRIX_EQ(a, b, err, ...) { \
if (ccv_matrix_eq(a, b) != 0) \
{ \
	if (isatty(fileno(stdout))) \
		printf("\n\t\033[0;31mREQUIRE_MATRIX_EQ\033[0;0m: %s:%d: %s != %s, " err, __FILE__, __LINE__, #a, #b, ##__VA_ARGS__); \
	else \
		printf("\n\tREQUIRE_MATRIX_EQ: %s:%d: %s != %s, " err, __FILE__, __LINE__, #a, #b, ##__VA_ARGS__); \
	ABORT_CASE; \
} }

#define REQUIRE_MATRIX_FILE_EQ(a, f, err, ...) { \
ccv_dense_matrix_t* __case_b__ = 0; \
ccv_read(f, &__case_b__, CCV_IO_ANY_FILE); \
if (ccv_matrix_eq(a, __case_b__) != 0) \
{ \
	if (isatty(fileno(stdout))) \
		printf("\n\t\033[0;31mREQUIRE_MATRIX_FILE_EQ\033[0;0m: %s:%d: %s != %s, " err, __FILE__, __LINE__, #a, #f, ##__VA_ARGS__); \
	else \
		printf("\n\tREQUIRE_MATRIX_FILE_EQ: %s:%d: %s != %s, " err, __FILE__, __LINE__, #a, #f, ##__VA_ARGS__); \
	ccv_matrix_free(__case_b__); \
	ABORT_CASE; \
} \
ccv_matrix_free(__case_b__); }

#define REQUIRE_ARRAY_EQ_WITHIN_ANGLE_AND_MAGNITUDE(type, a, b, len, angle, magnitude, err, ...) { \
int __case_i__; \
double __dot_prod__ = 0, __mag_a__ = 0, __mag_b__ = 0; \
for (__case_i__ = 0; __case_i__ < (len); __case_i__++) \
{ \
	__dot_prod__ += (double)(((type*)(a))[__case_i__] * ((type*)(b))[__case_i__]); \
	__mag_a__ += (double)(((type*)(a))[__case_i__] * ((type*)(a))[__case_i__]); \
	__mag_b__ += (double)(((type*)(b))[__case_i__] * ((type*)(b))[__case_i__]); \
} \
__mag_a__ = sqrt(__mag_a__), __mag_b__ = sqrt(__mag_b__); \
if (acos(__dot_prod__ / (__mag_a__ * __mag_b__)) * 180 / 3.141592653 > angle || fabs(__mag_a__ - __mag_b__) / ccv_max(ccv_max(__mag_a__, __mag_b__), 1) > magnitude) \
{ \
	if (isatty(fileno(stdout))) \
		printf("\n\t\033[0;31mREQUIRE_ARRAY_EQ_WITHIN_ANGLE_AND_MAGNITUDE\033[0;0m: %s:%d: angle: %lg | %lg, magnitude: %lg != %lg | +-%lg, " err, __FILE__, __LINE__, (double)(acos(__dot_prod__ / (__mag_a__ * __mag_b__)) * 180 / 3.141592653), (double)angle, __mag_a__, __mag_b__, (double)(magnitude), ##__VA_ARGS__); \
	else \
		printf("\n\tREQUIRE_ARRAY_EQ_WITHIN_ANGLE_AND_MAGNITUDE: %s:%d: angle: %lg | %lg, magnitude: %lg != %lg | +-%lg, " err, __FILE__, __LINE__, (double)(acos(__dot_prod__ / (__mag_a__ * __mag_b__)) * 180 / 3.141592653), (double)angle, __mag_a__, __mag_b__, (double)(magnitude), ##__VA_ARGS__); \
	ABORT_CASE; \
} }

inline static FILE* _ccv_nnc_symbolic_graph_gen(const char* test_case_name)
{
	mkdir("gen", S_IRWXU | S_IRGRP | S_IROTH);
	mkdir("gen/symbolic", S_IRWXU | S_IRGRP | S_IROTH);
	char sanitized_test_case_name[1024] = "gen/symbolic/";
	strncpy(sanitized_test_case_name + 13, test_case_name, 1024 - 13);
	int i;
	for (i = 13; i < 1024 && sanitized_test_case_name[i]; i++)
		// If not A-Za-z0-9, replace with _
		if (!((sanitized_test_case_name[i] >= 'A' && sanitized_test_case_name[i] <= 'Z') ||
			 (sanitized_test_case_name[i] >= 'a' && sanitized_test_case_name[i] <= 'z') ||
			 (sanitized_test_case_name[i] >= '0' && sanitized_test_case_name[i] <= '9')))
			sanitized_test_case_name[i] = '_';
	assert(i < 1024);
	sanitized_test_case_name[i] = '.';
	sanitized_test_case_name[i + 1] = 'd';
	sanitized_test_case_name[i + 2] = 'o';
	sanitized_test_case_name[i + 3] = 't';
	sanitized_test_case_name[i + 4] = 0;
	return fopen(sanitized_test_case_name, "w+");
}

inline static FILE* _ccv_nnc_graph_gen(const char* test_case_name)
{
	mkdir("gen", S_IRWXU | S_IRGRP | S_IROTH);
	mkdir("gen/graph", S_IRWXU | S_IRGRP | S_IROTH);
	char sanitized_test_case_name[1024] = "gen/graph/";
	strncpy(sanitized_test_case_name + 10, test_case_name, 1024 - 10);
	int i;
	for (i = 10; i < 1024 && sanitized_test_case_name[i]; i++)
		// If not A-Za-z0-9, replace with _
		if (!((sanitized_test_case_name[i] >= 'A' && sanitized_test_case_name[i] <= 'Z') ||
			 (sanitized_test_case_name[i] >= 'a' && sanitized_test_case_name[i] <= 'z') ||
			 (sanitized_test_case_name[i] >= '0' && sanitized_test_case_name[i] <= '9')))
			sanitized_test_case_name[i] = '_';
	assert(i < 1024);
	sanitized_test_case_name[i] = '.';
	sanitized_test_case_name[i + 1] = 'd';
	sanitized_test_case_name[i + 2] = 'o';
	sanitized_test_case_name[i + 3] = 't';
	sanitized_test_case_name[i + 4] = 0;
	return fopen(sanitized_test_case_name, "w+");
}

// Generate dot graph into a designated directory.
#define SYMBOLIC_GRAPH_GEN(symbolic_graph, type) do { \
	FILE* _w_ = _ccv_nnc_symbolic_graph_gen(__case_name__); \
	ccv_nnc_symbolic_graph_dot(symbolic_graph, type, _w_); \
	fclose(_w_); \
} while (0)

#define GRAPH_GEN(graph, type) do { \
	FILE* _w_ = _ccv_nnc_graph_gen(__case_name__); \
	ccv_nnc_graph_dot(graph, type, _w_); \
	fclose(_w_); \
} while (0)


#endif
