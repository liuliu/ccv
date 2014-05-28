/*************************************************************
 * C-based/Cached/Core Computer Vision Library with CUDA (CWC)
 * Liu Liu, 2013-12-01
 *************************************************************/

#ifndef GUARD_cwc_internal_h
#define GUARD_cwc_internal_h

// we are generating a lot of small kernel functions run ccv on CUDA, this function enables us to have
// different static functions, but we can dispatch them dynamically

#define cwc_vary_2_a(type, a, b, block, ...) { switch (type) { \
	case a: { block(__VA_ARGS__, a); break; } \
	case b: { block(__VA_ARGS__, b); break; } } }

#define cwc_vary_3_a(type, a, b, c, block, ...) { switch (type) { \
	case a: { block(__VA_ARGS__, a); break; } \
	case b: { block(__VA_ARGS__, b); break; } \
	case c: { block(__VA_ARGS__, c); break; } } }

#define cwc_vary_4_a(type, a, b, c, d, block, ...) { switch (type) { \
	case a: { block(__VA_ARGS__, a); break; } \
	case b: { block(__VA_ARGS__, b); break; } \
	case c: { block(__VA_ARGS__, c); break; } \
	case d: { block(__VA_ARGS__, d); break; } } }

#define cwc_vary_5_a(type, a, b, c, d, e, block, ...) { switch (type) { \
	case a: { block(__VA_ARGS__, a); break; } \
	case b: { block(__VA_ARGS__, b); break; } \
	case c: { block(__VA_ARGS__, c); break; } \
	case d: { block(__VA_ARGS__, d); break; } \
	case e: { block(__VA_ARGS__, e); break; } } }

#define cwc_vary_6_a(type, a, b, c, d, e, f, block, ...) { switch (type) { \
	case a: { block(__VA_ARGS__, a); break; } \
	case b: { block(__VA_ARGS__, b); break; } \
	case c: { block(__VA_ARGS__, c); break; } \
	case d: { block(__VA_ARGS__, d); break; } \
	case e: { block(__VA_ARGS__, e); break; } \
	case f: { block(__VA_ARGS__, f); break; } } }

#define cwc_vary_2_b(type, a, b, block, ...) { switch (type) { \
	case a: { block(__VA_ARGS__, a); break; } \
	case b: { block(__VA_ARGS__, b); break; } } }

#define cwc_vary_3_b(type, a, b, c, block, ...) { switch (type) { \
	case a: { block(__VA_ARGS__, a); break; } \
	case b: { block(__VA_ARGS__, b); break; } \
	case c: { block(__VA_ARGS__, c); break; } } }

#define cwc_vary_4_b(type, a, b, c, d, block, ...) { switch (type) { \
	case a: { block(__VA_ARGS__, a); break; } \
	case b: { block(__VA_ARGS__, b); break; } \
	case c: { block(__VA_ARGS__, c); break; } \
	case d: { block(__VA_ARGS__, d); break; } } }

#define cwc_vary_5_b(type, a, b, c, d, e, block, ...) { switch (type) { \
	case a: { block(__VA_ARGS__, a); break; } \
	case b: { block(__VA_ARGS__, b); break; } \
	case c: { block(__VA_ARGS__, c); break; } \
	case d: { block(__VA_ARGS__, d); break; } \
	case e: { block(__VA_ARGS__, e); break; } } }

#define cwc_vary_6_b(type, a, b, c, d, e, f, block, ...) { switch (type) { \
	case a: { block(__VA_ARGS__, a); break; } \
	case b: { block(__VA_ARGS__, b); break; } \
	case c: { block(__VA_ARGS__, c); break; } \
	case d: { block(__VA_ARGS__, d); break; } \
	case e: { block(__VA_ARGS__, e); break; } \
	case f: { block(__VA_ARGS__, f); break; } } }

#define cwc_vary_2_c(type, a, b, block, ...) { switch (type) { \
	case a: { block(__VA_ARGS__, a); break; } \
	case b: { block(__VA_ARGS__, b); break; } } }

#define cwc_vary_3_c(type, a, b, c, block, ...) { switch (type) { \
	case a: { block(__VA_ARGS__, a); break; } \
	case b: { block(__VA_ARGS__, b); break; } \
	case c: { block(__VA_ARGS__, c); break; } } }

#define cwc_vary_4_c(type, a, b, c, d, block, ...) { switch (type) { \
	case a: { block(__VA_ARGS__, a); break; } \
	case b: { block(__VA_ARGS__, b); break; } \
	case c: { block(__VA_ARGS__, c); break; } \
	case d: { block(__VA_ARGS__, d); break; } } }

#define cwc_vary_5_c(type, a, b, c, d, e, block, ...) { switch (type) { \
	case a: { block(__VA_ARGS__, a); break; } \
	case b: { block(__VA_ARGS__, b); break; } \
	case c: { block(__VA_ARGS__, c); break; } \
	case d: { block(__VA_ARGS__, d); break; } \
	case e: { block(__VA_ARGS__, e); break; } } }

#define cwc_vary_6_c(type, a, b, c, d, e, f, block, ...) { switch (type) { \
	case a: { block(__VA_ARGS__, a); break; } \
	case b: { block(__VA_ARGS__, b); break; } \
	case c: { block(__VA_ARGS__, c); break; } \
	case d: { block(__VA_ARGS__, d); break; } \
	case e: { block(__VA_ARGS__, e); break; } \
	case f: { block(__VA_ARGS__, f); break; } } }

#define cwc_vary_2_d(type, a, b, block, ...) { switch (type) { \
	case a: { block(__VA_ARGS__, a); break; } \
	case b: { block(__VA_ARGS__, b); break; } } }

#define cwc_vary_3_d(type, a, b, c, block, ...) { switch (type) { \
	case a: { block(__VA_ARGS__, a); break; } \
	case b: { block(__VA_ARGS__, b); break; } \
	case c: { block(__VA_ARGS__, c); break; } } }

#define cwc_vary_4_d(type, a, b, c, d, block, ...) { switch (type) { \
	case a: { block(__VA_ARGS__, a); break; } \
	case b: { block(__VA_ARGS__, b); break; } \
	case c: { block(__VA_ARGS__, c); break; } \
	case d: { block(__VA_ARGS__, d); break; } } }

#define cwc_vary_5_d(type, a, b, c, d, e, block, ...) { switch (type) { \
	case a: { block(__VA_ARGS__, a); break; } \
	case b: { block(__VA_ARGS__, b); break; } \
	case c: { block(__VA_ARGS__, c); break; } \
	case d: { block(__VA_ARGS__, d); break; } \
	case e: { block(__VA_ARGS__, e); break; } } }

#define cwc_vary_6_d(type, a, b, c, d, e, f, block, ...) { switch (type) { \
	case a: { block(__VA_ARGS__, a); break; } \
	case b: { block(__VA_ARGS__, b); break; } \
	case c: { block(__VA_ARGS__, c); break; } \
	case d: { block(__VA_ARGS__, d); break; } \
	case e: { block(__VA_ARGS__, e); break; } \
	case f: { block(__VA_ARGS__, f); break; } } }

// define the body of the function that bench / select best kernels

#define CWC_IMPLEMENT_VARY_STUB(config, vary_x, vary_y, vary_z, vary_func, ...) \
	if (config.x == 0 && config.y == 0 && config.z == 0) \
	{ \
		int i, j, k, t; \
		int x = 0, y = 0, z = 0; \
		float best_elapsed_time = FLT_MAX; \
		cudaEvent_t start; \
		cudaEvent_t stop; \
		cudaEventCreate(&start); \
		cudaEventCreate(&stop); \
		for (i = 0; i < sizeof(vary_x) / sizeof(int); i++) \
			for (j = 0; j < sizeof(vary_y) / sizeof(int); j++) \
				for (k = 0; k < sizeof(vary_z) / sizeof(int); k++) \
				{ \
					float elapsed_time_vary = 0; \
					cudaEventRecord(start, stream); \
					int success = vary_func(__VA_ARGS__, vary_x[i], vary_y[j], vary_z[k]); \
					cudaEventRecord(stop, stream); \
					cudaEventSynchronize(stop); \
					if (success != 0) \
						continue; \
					cudaEventElapsedTime(&elapsed_time_vary, start, stop); \
					for (t = 1; t < 2; t++) /* we do 2 runs and pick the best one */ \
					{ \
						cudaEventRecord(start, stream); \
						vary_func(__VA_ARGS__, vary_x[i], vary_y[j], vary_z[k]); \
						cudaEventRecord(stop, stream); \
						cudaEventSynchronize(stop); \
						float elapsed_time = 0; \
						cudaEventElapsedTime(&elapsed_time, start, stop); \
						if (elapsed_time < elapsed_time_vary) \
							elapsed_time_vary = elapsed_time; \
					} \
					if (elapsed_time_vary < best_elapsed_time) \
						best_elapsed_time = elapsed_time_vary, x = vary_x[i], y = vary_y[j], z = vary_z[k]; \
				} \
		cudaEventDestroy(start); \
		cudaEventDestroy(stop); \
		config.x = x; \
		config.y = y; \
		config.z = z; \
	} else /* we already have configuration, run it */ \
		vary_func(__VA_ARGS__, config.x, config.y, config.z);

#endif
