#ifndef _GUARD_ccv_nnc_case_h_
#define _GUARD_ccv_nnc_case_h_

#include <math.h>

#define REQUIRE_TENSOR_EQ(a, b, err, ...) { \
if (ccv_nnc_tensor_eq(a, b) != 0) \
{ \
	printf("\n\t\033[0;31mREQUIRE_TENSOR_EQ\033[0;0m: %s:%d: %s != %s, " err, __FILE__, __LINE__, #a, #b, ##__VA_ARGS__); \
	ABORT_CASE; \
} }

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

inline static FILE* _ccv_nnc_dynamic_graph_gen(const char* test_case_name)
{
	mkdir("gen", S_IRWXU | S_IRGRP | S_IROTH);
	mkdir("gen/dynamic", S_IRWXU | S_IRGRP | S_IROTH);
	char sanitized_test_case_name[1024] = "gen/dynamic/";
	strncpy(sanitized_test_case_name + 12, test_case_name, 1024 - 12);
	int i;
	for (i = 12; i < 1024 && sanitized_test_case_name[i]; i++)
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

inline static FILE* _ccv_nnc_cnnp_model_gen(const char* test_case_name)
{
	mkdir("gen", S_IRWXU | S_IRGRP | S_IROTH);
	mkdir("gen/cnnp", S_IRWXU | S_IRGRP | S_IROTH);
	char sanitized_test_case_name[1024] = "gen/cnnp/";
	strncpy(sanitized_test_case_name + 9, test_case_name, 1024 - 9);
	int i;
	for (i = 12; i < 1024 && sanitized_test_case_name[i]; i++)
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
#define GRAPH_GEN(graph, type) do { \
	FILE* _w_ = _ccv_nnc_graph_gen(__case_name__); \
	ccv_nnc_graph_dot(graph, type, _w_); \
	fclose(_w_); \
} while (0)

#define SYMBOLIC_GRAPH_GEN(symbolic_graph, type) do { \
	FILE* _w_ = _ccv_nnc_symbolic_graph_gen(__case_name__); \
	ccv_nnc_symbolic_graph_dot(symbolic_graph, type, _w_); \
	fclose(_w_); \
} while (0)

#define DYNAMIC_GRAPH_GEN(dynamic_graph, type) do { \
	FILE* _w_ = _ccv_nnc_dynamic_graph_gen(__case_name__); \
	ccv_nnc_dynamic_graph_dot(dynamic_graph, type, _w_); \
	fclose(_w_); \
} while (0)

#define CNNP_MODEL_GEN(cnnp_model, type) do { \
	FILE* _w_ = _ccv_nnc_cnnp_model_gen(__case_name__); \
	ccv_cnnp_model_dot(cnnp_model, type, _w_); \
	fclose(_w_); \
} while (0)

#endif
