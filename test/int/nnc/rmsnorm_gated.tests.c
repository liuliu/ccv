#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <3rdparty/dsfmt/dSFMT.h>
#include <math.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_CASE("gpu rmsnorm gated matches CPU across precision, axes, views and in-place execution")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_RMSNORM_GATED_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_RMSNORM_GATED_FORWARD, CCV_NNC_BACKEND_MPS));
	const int datatypes[] = {CCV_32F, CCV_16F, CCV_16BF};
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 42);
	int dtype, variant, format;
	for (dtype = 0; dtype < 3; dtype++)
		for (variant = 0; variant < 4; variant++)
			for (format = 0; format < 2; format++)
			{
				const int columns = variant == 3 ? 258 : 129;
				ccv_nnc_tensor_param_t params = CPU_TENSOR_NHWC(32F, 3, 11, columns);
				params.format = format ? CCV_TENSOR_FORMAT_NCHW : CCV_TENSOR_FORMAT_NHWC;
				ccv_nnc_tensor_t* const source = ccv_nnc_tensor_new(0, params, 0);
				ccv_nnc_tensor_t* const rounded = ccv_nnc_tensor_new(0, params, 0);
				ccv_nnc_tensor_t* const gate = ccv_nnc_tensor_new(0, params, 0);
				ccv_nnc_tensor_t* const expected_storage = ccv_nnc_tensor_new(0, params, 0);
				ccv_nnc_tensor_t* const actual_storage = ccv_nnc_tensor_new(0, params, 0);
				ccv_nnc_tensor_param_t gpu_params = params;
				gpu_params.type = CCV_TENSOR_GPU_MEMORY;
				ccv_nnc_tensor_t* const gpu_source = ccv_nnc_tensor_new(0, gpu_params, 0);
				ccv_nnc_tensor_t* const gpu_gate_storage = ccv_nnc_tensor_new(0, gpu_params, 0);
				gpu_params.datatype = datatypes[dtype];
				ccv_nnc_tensor_t* const gpu_input = ccv_nnc_tensor_new(0, gpu_params, 0);
				ccv_nnc_tensor_t* const gpu_output = ccv_nnc_tensor_new(0, gpu_params, 0);
				int i;
				for (i = 0; i < 3 * 11 * columns; i++)
				{
					source->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) * 4;
					gate->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) * 12;
				}
				gate->data.f32[0] = -100;
				gate->data.f32[1] = 100;
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(source, gate), TENSOR_LIST(gpu_source, gpu_gate_storage), 0);
				ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_source), TENSOR_LIST(gpu_input), 0);
				ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_input), TENSOR_LIST(gpu_source), 0);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_source), TENSOR_LIST(rounded), 0);
				ccv_nnc_tensor_param_t view_params = params;
				view_params.dim[2] = 129;
				ccv_nnc_tensor_view_t* const a = ccv_nnc_tensor_view_new(rounded, view_params, DIM_ALLOC(0, 0, 0), DIM_ALLOC(11 * columns, columns, 1));
				ccv_nnc_tensor_view_t* const g = ccv_nnc_tensor_view_new(gate, view_params, DIM_ALLOC(0, 0, 0), DIM_ALLOC(11 * columns, columns, 1));
				ccv_nnc_tensor_view_t* const expected = ccv_nnc_tensor_view_new(expected_storage, view_params, DIM_ALLOC(0, 0, 0), DIM_ALLOC(11 * columns, columns, 1));
				view_params.type = CCV_TENSOR_GPU_MEMORY;
				ccv_nnc_tensor_view_t* const gg = ccv_nnc_tensor_view_new(gpu_gate_storage, view_params, DIM_ALLOC(0, 0, 0), DIM_ALLOC(11 * columns, columns, 1));
				view_params.datatype = datatypes[dtype];
				ccv_nnc_tensor_view_t* const ga = ccv_nnc_tensor_view_new(gpu_input, view_params, DIM_ALLOC(0, 0, 0), DIM_ALLOC(11 * columns, columns, 1));
				ccv_nnc_tensor_view_t* const gb = ccv_nnc_tensor_view_new(gpu_output, view_params, DIM_ALLOC(0, 0, 0), DIM_ALLOC(11 * columns, columns, 1));
				ccv_nnc_tensor_param_t scale_params = variant == 1 ? CPU_TENSOR_NHWC(32F, 1, 11, 1) : CPU_TENSOR_NHWC(32F, 1, 1, 129);
				ccv_nnc_tensor_t* const scale = ccv_nnc_tensor_new(0, scale_params, 0);
				scale_params.type = CCV_TENSOR_GPU_MEMORY;
				ccv_nnc_tensor_t* const gpu_scale = ccv_nnc_tensor_new(0, scale_params, 0);
				for (i = 0; i < ccv_nnc_tensor_count(scale->info); i++)
					scale->data.f32[i] = 0.5 + dsfmt_genrand_open_close(&dsfmt);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(scale), TENSOR_LIST(gpu_scale), 0);
				const int affine = variant != 1;
				ccv_nnc_cmd_t cmd = variant == 2 ? CMD_RMSNORM_GATED_FORWARD(1e-6, affine, 1, 2) : CMD_RMSNORM_GATED_FORWARD(1e-6, affine, variant == 1 ? 1 : 2);
				ccv_nnc_tensor_t* cpu_inputs[] = {(ccv_nnc_tensor_t*)a, (ccv_nnc_tensor_t*)g, scale};
				ccv_nnc_tensor_t* gpu_inputs[] = {(ccv_nnc_tensor_t*)ga, (ccv_nnc_tensor_t*)gg, gpu_scale};
				REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, cpu_inputs, affine ? 3 : 2, TENSOR_LIST((ccv_nnc_tensor_t*)expected), 0), CCV_NNC_EXEC_SUCCESS, "CPU gated norm should execute");
				REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, gpu_inputs, affine ? 3 : 2, TENSOR_LIST((ccv_nnc_tensor_t*)(variant == 3 ? ga : gb)), 0), CCV_NNC_EXEC_SUCCESS, "GPU gated norm should execute");
				ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(variant == 3 ? gpu_input : gpu_output), TENSOR_LIST(gpu_source), 0);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_source), TENSOR_LIST(actual_storage), 0);
				const float tolerance = datatypes[dtype] == CCV_32F ? 2e-5 : datatypes[dtype] == CCV_16F ? 1e-3 : 5e-3;
				for (i = 0; i < 3 * 11 * 129; i++)
				{
					const int offset = (i / 129) * columns + i % 129;
					const float expected_value = expected_storage->data.f32[offset];
					REQUIRE_EQ_WITH_TOLERANCE(actual_storage->data.f32[offset], expected_value, tolerance * fmaxf(1, fabsf(expected_value)), "GPU gated norm must match CPU for datatype=%d variant=%d format=%d", datatypes[dtype], variant, format);
				}
				ccv_nnc_tensor_view_free(a);
				ccv_nnc_tensor_view_free(g);
				ccv_nnc_tensor_view_free(expected);
				ccv_nnc_tensor_view_free(ga);
				ccv_nnc_tensor_view_free(gg);
				ccv_nnc_tensor_view_free(gb);
				ccv_nnc_tensor_free(source);
				ccv_nnc_tensor_free(rounded);
				ccv_nnc_tensor_free(gate);
				ccv_nnc_tensor_free(expected_storage);
				ccv_nnc_tensor_free(actual_storage);
				ccv_nnc_tensor_free(gpu_source);
				ccv_nnc_tensor_free(gpu_gate_storage);
				ccv_nnc_tensor_free(gpu_input);
				ccv_nnc_tensor_free(gpu_output);
				ccv_nnc_tensor_free(scale);
				ccv_nnc_tensor_free(gpu_scale);
			}
}

TEST_CASE("gpu rmsnorm gated contiguous rows match CPU")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_RMSNORM_GATED_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_RMSNORM_GATED_FORWARD, CCV_NNC_BACKEND_MPS));
	const int datatypes[] = {CCV_32F, CCV_16F, CCV_16BF};
	// Row widths for each register tile of the contiguous kernel, and one past the widest tile.
	const int columns[] = {64, 128, 200, 512, 1000, 1500};
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 7);
	int dtype, c, i;
	for (dtype = 0; dtype < 3; dtype++)
		for (c = 0; c < sizeof(columns) / sizeof(columns[0]); c++)
		{
			const int n = columns[c];
			const int inplace = c % 2;
			const ccv_nnc_tensor_param_t params = CPU_TENSOR_NHWC(32F, 7, 5, n);
			ccv_nnc_tensor_t* const source = ccv_nnc_tensor_new(0, params, 0);
			ccv_nnc_tensor_t* const rounded = ccv_nnc_tensor_new(0, params, 0);
			ccv_nnc_tensor_t* const gate = ccv_nnc_tensor_new(0, params, 0);
			ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, params, 0);
			ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, params, 0);
			ccv_nnc_tensor_t* const scale = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, n), 0);
			ccv_nnc_tensor_param_t gpu_params = params;
			gpu_params.type = CCV_TENSOR_GPU_MEMORY;
			ccv_nnc_tensor_t* const gpu_source = ccv_nnc_tensor_new(0, gpu_params, 0);
			ccv_nnc_tensor_t* const gpu_gate = ccv_nnc_tensor_new(0, gpu_params, 0);
			ccv_nnc_tensor_t* const gpu_scale = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, n), 0);
			gpu_params.datatype = datatypes[dtype];
			ccv_nnc_tensor_t* const gpu_input = ccv_nnc_tensor_new(0, gpu_params, 0);
			ccv_nnc_tensor_t* const gpu_output = ccv_nnc_tensor_new(0, gpu_params, 0);
			for (i = 0; i < 7 * 5 * n; i++)
			{
				source->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) * 4;
				gate->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) * 12;
			}
			for (i = 0; i < n; i++)
				scale->data.f32[i] = 0.5 + dsfmt_genrand_open_close(&dsfmt);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(source, gate, scale), TENSOR_LIST(gpu_source, gpu_gate, gpu_scale), 0);
			ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_source), TENSOR_LIST(gpu_input), 0);
			ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_input), TENSOR_LIST(gpu_source), 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_source), TENSOR_LIST(rounded), 0);
			const ccv_nnc_cmd_t cmd = CMD_RMSNORM_GATED_FORWARD(1e-6, 1, 2);
			REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(rounded, gate, scale), TENSOR_LIST(expected), 0), CCV_NNC_EXEC_SUCCESS, "CPU gated norm should execute");
			REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_input, gpu_gate, gpu_scale), TENSOR_LIST(inplace ? gpu_input : gpu_output), 0), CCV_NNC_EXEC_SUCCESS, "GPU gated norm should execute");
			ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(inplace ? gpu_input : gpu_output), TENSOR_LIST(gpu_source), 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_source), TENSOR_LIST(actual), 0);
			const float tolerance = datatypes[dtype] == CCV_32F ? 2e-5 : datatypes[dtype] == CCV_16F ? 1e-3 : 5e-3;
			for (i = 0; i < 7 * 5 * n; i++)
				REQUIRE_EQ_WITH_TOLERANCE(actual->data.f32[i], expected->data.f32[i], tolerance * fmaxf(1, fabsf(expected->data.f32[i])), "GPU gated norm must match CPU for datatype=%d n=%d inplace=%d", datatypes[dtype], n, inplace);
			ccv_nnc_tensor_free(source);
			ccv_nnc_tensor_free(rounded);
			ccv_nnc_tensor_free(gate);
			ccv_nnc_tensor_free(expected);
			ccv_nnc_tensor_free(actual);
			ccv_nnc_tensor_free(scale);
			ccv_nnc_tensor_free(gpu_source);
			ccv_nnc_tensor_free(gpu_gate);
			ccv_nnc_tensor_free(gpu_scale);
			ccv_nnc_tensor_free(gpu_input);
			ccv_nnc_tensor_free(gpu_output);
		}
}

TEST_CASE("cuda rmsnorm preserves output scaling used by Qwen query and key normalization")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_RMSNORM_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN));
	const int datatypes[] = {CCV_32F, CCV_16F, CCV_16BF};
	const float scales[] = {1.f / 128, 1.f / sqrtf(128), 2.f};
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 42);
	int dtype, affine, i, j;
	for (dtype = 0; dtype < 3; dtype++)
		for (affine = 0; affine < 2; affine++)
		{
			ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 31, 4, 128), 0);
			ccv_nnc_tensor_t* const scale = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1, 1, 128), 0);
			ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, a->info, 0);
			ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, a->info, 0);
			ccv_nnc_tensor_t* const saved = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 31, 4, 1), 0);
			ccv_nnc_tensor_t* const ga32 = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 31, 4, 128), 0);
			ccv_nnc_tensor_t* const gscale32 = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1, 1, 1, 128), 0);
			ccv_nnc_tensor_param_t params = ga32->info;
			params.datatype = datatypes[dtype];
			ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, params, 0);
			ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, params, 0);
			params = gscale32->info;
			params.datatype = datatypes[dtype];
			ccv_nnc_tensor_t* const gscale = ccv_nnc_tensor_new(0, params, 0);
			ccv_nnc_tensor_t* const gsave = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 31, 4, 1), 0);
			for (i = 0; i < ccv_nnc_tensor_count(a->info); i++)
				a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
			for (i = 0; i < 128; i++)
				scale->data.f32[i] = 0.5 + dsfmt_genrand_open_close(&dsfmt);
			// Round the inputs through the GPU datatype, so CPU and GPU see the same values.
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, scale), TENSOR_LIST(ga32, gscale32), 0);
			ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga32, gscale32), TENSOR_LIST(ga, gscale), 0);
			ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gscale), TENSOR_LIST(ga32, gscale32), 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga32, gscale32), TENSOR_LIST(a, scale), 0);
			const float tolerance = datatypes[dtype] == CCV_32F ? 1e-5 : datatypes[dtype] == CCV_16F ? 1e-3 : 5e-3;
			for (j = 0; j < 3; j++)
			{
				ccv_nnc_cmd_t cmd = CMD_RMSNORM_FORWARD(1e-6 / 128, affine, 3);
				cmd.info.rmsnorm.scale = scales[j];
				ccv_nnc_tensor_t* cpu_inputs[] = {a, scale};
				ccv_nnc_tensor_t* gpu_inputs[] = {ga, gscale};
				REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, cpu_inputs, affine ? 2 : 1, TENSOR_LIST(expected, saved), 0), CCV_NNC_EXEC_SUCCESS, "CPU scaled RMSNorm should execute");
				REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, gpu_inputs, affine ? 2 : 1, TENSOR_LIST(gb, gsave), 0), CCV_NNC_EXEC_SUCCESS, "CUDA scaled RMSNorm should execute");
				ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb), TENSOR_LIST(ga32), 0);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga32), TENSOR_LIST(actual), 0);
				for (i = 0; i < ccv_nnc_tensor_count(a->info); i++)
					REQUIRE_EQ_WITH_TOLERANCE(actual->data.f32[i], expected->data.f32[i], tolerance * fmaxf(1, fabsf(expected->data.f32[i])), "scaled RMSNorm should match CPU reference for datatype=%d affine=%d scale=%g", datatypes[dtype], affine, scales[j]);
			}
			ccv_nnc_tensor_free(a);
			ccv_nnc_tensor_free(scale);
			ccv_nnc_tensor_free(expected);
			ccv_nnc_tensor_free(actual);
			ccv_nnc_tensor_free(saved);
			ccv_nnc_tensor_free(ga32);
			ccv_nnc_tensor_free(gscale32);
			ccv_nnc_tensor_free(ga);
			ccv_nnc_tensor_free(gb);
			ccv_nnc_tensor_free(gscale);
			ccv_nnc_tensor_free(gsave);
		}
}

#include "case_main.h"
