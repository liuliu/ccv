#include <ctype.h>
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <3rdparty/dsfmt/dSFMT.h>

static void train_cifar_10(ccv_array_t* const training_set, const float mean[3], ccv_array_t* const test_set)
{
	ccv_cnnp_model_t* const sequential = ccv_cnnp_sequential_new(MODEL_LIST(
		ccv_cnnp_convolution(1, 32, DIM_ALLOC(5, 5), (ccv_cnnp_param_t){
			.norm = CCV_CNNP_BATCH_NORM,
			.activation = CCV_CNNP_ACTIVATION_RELU,
			.hint = HINT((1, 1), (2, 2)),
		}),
		ccv_cnnp_max_pool(DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.hint = HINT((2, 2), (0, 0)),
		}),
		ccv_cnnp_convolution(1, 32, DIM_ALLOC(5, 5), (ccv_cnnp_param_t){
			.norm = CCV_CNNP_BATCH_NORM,
			.activation = CCV_CNNP_ACTIVATION_RELU,
			.hint = HINT((1, 1), (2, 2)),
		}),
		ccv_cnnp_average_pool(DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.hint = HINT((2, 2), (0, 0)),
		}),
		ccv_cnnp_convolution(1, 64, DIM_ALLOC(5, 5), (ccv_cnnp_param_t){
			.norm = CCV_CNNP_BATCH_NORM,
			.activation = CCV_CNNP_ACTIVATION_RELU,
			.hint = HINT((1, 1), (2, 2)),
		}),
		ccv_cnnp_average_pool(DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.hint = HINT((2, 2), (0, 0)),
		}),
		ccv_cnnp_flatten(),
		ccv_cnnp_dense(256, (ccv_cnnp_param_t){
			.norm = CCV_CNNP_BATCH_NORM,
			.activation = CCV_CNNP_ACTIVATION_RELU,
		}),
		ccv_cnnp_dense(10, (ccv_cnnp_param_t){
			.activation = CCV_CNNP_ACTIVATION_SOFTMAX,
		})
	));
	const ccv_nnc_tensor_param_t input = GPU_TENSOR_NCHW(000, 128, 3, 31, 31);
	ccv_cnnp_model_compile(sequential, &input, 1, CMD_SGD_FORWARD(0.0001, 0.99, 0.9, 0.9), CMD_CATEGORICAL_CROSSENTROPY_FORWARD());
	ccv_nnc_tensor_t* const input_tensor = ccv_nnc_tensor_new(0, input, 0);
	ccv_nnc_tensor_t* const output_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 128, 10), 0);
	ccv_nnc_tensor_t* const fit_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 128, 1), 0);
	ccv_nnc_tensor_t* const cpu_input = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(128, 3, 31, 31), 0);
	ccv_nnc_tensor_t* const cpu_output = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(128, 10), 0);
	ccv_nnc_tensor_t* const cpu_fit = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(128, 1), 0);
	int i, j, k;
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int c[128];
	double correct_ratio = 0;
	for (i = 0; i < 10000; i++)
	{
		for (j = 0; j < 128; j++)
		{
			k = (int)(dsfmt_genrand_close_open(&dsfmt) * training_set->rnum);
			assert(k < training_set->rnum);
			ccv_categorized_t* const categorized = (ccv_categorized_t*)ccv_array_get(training_set, k);
			float* const ip = cpu_input->data.f32 + j * 31 * 31 * 3;
			float* const cp = categorized->matrix->data.f32;
			int fi, fj, fk;
			for (fi = 0; fi < 31; fi++)
				for (fj = 0; fj < 31; fj++)
					for (fk = 0; fk < 3; fk++)
						ip[fi * 31 + fj + fk * 31 * 31] = cp[fi * 31 * 3 + fj * 3 + fk] - mean[fk];
			assert(categorized->c >= 0 && categorized->c < 10);
			cpu_fit->data.f32[j] = c[j] = categorized->c;
		}
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(cpu_input, cpu_fit), TENSOR_LIST(input_tensor, fit_tensor), 0);
		ccv_cnnp_model_fit(sequential, TENSOR_LIST(input_tensor), TENSOR_LIST(fit_tensor), TENSOR_LIST(output_tensor));
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(output_tensor), TENSOR_LIST(cpu_output), 0);
		int correct = 0;
		for (j = 0; j < 128; j++)
		{
			float max = -FLT_MAX;
			int t = -1;
			for (k = 0; k < 10; k++)
				if (cpu_output->data.f32[j * 10 + k] > max)
					max = cpu_output->data.f32[j * 10 + k], t = k;
			if (c[j] == t)
				++correct;
		}
		correct_ratio = correct_ratio * 0.9 + correct * 0.1 / 128.;
		if (i % 11 == 0)
			FLUSH(CCV_CLI_INFO, "Batch %d, Correct %f", i + 1, correct_ratio);
	}
	ccv_cnnp_model_free(sequential);
	ccv_nnc_tensor_free(input_tensor);
	ccv_nnc_tensor_free(fit_tensor);
	ccv_nnc_tensor_free(output_tensor);
	ccv_nnc_tensor_free(cpu_input);
	ccv_nnc_tensor_free(cpu_fit);
	ccv_nnc_tensor_free(cpu_output);
}

int main(int argc, char** argv)
{
	ccv_nnc_init();
	assert(argc == 5);
	int num1 = atoi(argv[2]);
	int num2 = atoi(argv[4]);
	FILE* r1 = fopen(argv[1], "rb");
	FILE* r2 = fopen(argv[3], "rb");
	if (r1 && r2)
	{
		int i, j, k;
		unsigned char bytes[32 * 32 + 1];
		double mean[3] = {};
		ccv_array_t* categorizeds = ccv_array_new(sizeof(ccv_categorized_t), num1, 0);
		for (k = 0; k < num1; k++)
		{
			fread(bytes, 32 * 32 + 1, 1, r1);
			double per_mean[3] = {};
			int c = bytes[0];
			ccv_dense_matrix_t* a = ccv_dense_matrix_new(31, 31, CCV_32F | CCV_C3, 0, 0);
			for (i = 0; i < 31; i++)
				for (j = 0; j < 31; j++)
					per_mean[0] += (a->data.f32[(j + i * 31) * 3] = bytes[j + i * 32 + 1] * 2. / 255.);
			fread(bytes, 32 * 32, 1, r1);
			for (i = 0; i < 31; i++)
				for (j = 0; j < 31; j++)
					per_mean[1] += (a->data.f32[(j + i * 31) * 3 + 1] = bytes[j + i * 32] * 2. / 255.);
			fread(bytes, 32 * 32, 1, r1);
			for (i = 0; i < 31; i++)
				for (j = 0; j < 31; j++)
					per_mean[2] += (a->data.f32[(j + i * 31) * 3 + 2] = bytes[j + i * 32] * 2. / 255.);
			ccv_categorized_t categorized = ccv_categorized(c, a, 0);
			ccv_array_push(categorizeds, &categorized);
			mean[0] += per_mean[0] / (31 * 31);
			mean[1] += per_mean[1] / (31 * 31);
			mean[2] += per_mean[2] / (31 * 31);
		}
		ccv_array_t* tests = ccv_array_new(sizeof(ccv_categorized_t), num2, 0);
		for (k = 0; k < num2; k++)
		{
			fread(bytes, 32 * 32 + 1, 1, r2);
			int c = bytes[0];
			ccv_dense_matrix_t* a = ccv_dense_matrix_new(31, 31, CCV_32F | CCV_C3, 0, 0);
			for (i = 0; i < 31; i++)
				for (j = 0; j < 31; j++)
					a->data.f32[(j + i * 31) * 3] = bytes[j + i * 32 + 1] * 2. / 255.;
			fread(bytes, 32 * 32, 1, r2);
			for (i = 0; i < 31; i++)
				for (j = 0; j < 31; j++)
					a->data.f32[(j + i * 31) * 3 + 1] = bytes[j + i * 32] * 2. / 255.;
			fread(bytes, 32 * 32, 1, r2);
			for (i = 0; i < 31; i++)
				for (j = 0; j < 31; j++)
					a->data.f32[(j + i * 31) * 3 + 2] = bytes[j + i * 32] * 2. / 255.;
			ccv_categorized_t categorized = ccv_categorized(c, a, 0);
			ccv_array_push(tests, &categorized);
		}
		float meanf[3];
		meanf[0] = mean[0] / num1;
		meanf[1] = mean[1] / num1;
		meanf[2] = mean[2] / num1;
		train_cifar_10(categorizeds, meanf, tests);
	}
	if (r1)
		fclose(r1);
	if (r2)
		fclose(r2);
	return 0;
}
