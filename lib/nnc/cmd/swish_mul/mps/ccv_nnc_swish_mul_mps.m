#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

static int _ccv_nnc_swish_mul_mfa_datatype(const int datatype)
{
	switch (datatype) {
		case CCV_16F:
			return 16;
		case CCV_16BF:
			return 121;
		case CCV_32F:
			return 3;
		default:
			return -1;
	}
}

static int _ccv_nnc_swish_mul_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	const int weighted = cmd.info.swish_mul.weighted;
	assert(input_size == (weighted ? 3 : 2));
	assert(output_size == 1);
	const float beta = cmd.info.swish_mul.beta;
	const float scale = cmd.info.swish_mul.scale;
	const float limit = cmd.info.swish_mul.clamp;
	const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* const b = (const ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* const w = weighted ? (const ccv_nnc_tensor_view_t*)inputs[2] : 0;
	ccv_nnc_tensor_view_t* const c = (ccv_nnc_tensor_view_t*)outputs[0];
	if (a->info.dim[0] == 0 || b->info.dim[0] == 0 || c->info.dim[0] == 0)
		return CCV_NNC_EXEC_INVALID;
	assert(c->info.datatype == a->info.datatype);
	int i;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && a->info.dim[i] > 0; i++)
	{
		assert(a->info.dim[i] == b->info.dim[i]);
		assert(a->info.dim[i] == c->info.dim[i]);
	}
	@autoreleasepool {
		bool use_mfa = true;
		ccv_nnc_mfa_context_t* context = ccv_nnc_default_mfa_context();

		if (!ccv_nnc_mfa_context_supported(context) || (ccv_nnc_flags() & CCV_NNC_DISABLE_MFA))
			use_mfa = false;

		const int a_data_type = _ccv_nnc_swish_mul_mfa_datatype(a->info.datatype);
		const int b_data_type = _ccv_nnc_swish_mul_mfa_datatype(b->info.datatype);
		const int weight_data_type = w ? _ccv_nnc_swish_mul_mfa_datatype(w->info.datatype) : a_data_type;
		if (use_mfa && (a_data_type < 0 || b_data_type < 0 || weight_data_type < 0))
			use_mfa = false;

		if (use_mfa && (!CCV_IS_TENSOR_CONTIGUOUS(a) || !CCV_IS_TENSOR_CONTIGUOUS(b) || (w && !CCV_IS_TENSOR_CONTIGUOUS(w)) || !CCV_IS_TENSOR_CONTIGUOUS(c)))
			use_mfa = false;
		const uint32_t length = (uint32_t)ccv_nnc_tensor_count(a->info);
		const uint32_t weight_count = w ? (uint32_t)ccv_nnc_tensor_count(w->info) : 1;
		if (weighted && (weight_count == 0 || length % weight_count != 0))
			return CCV_NNC_EXEC_INVALID;

		if (use_mfa) {
			ccv_nnc_mfa_swish_mul_params_t params = {
				.beta = beta,
				.scale = scale,
				.clamp = limit,
				.a_data_type = (uint64_t)a_data_type,
				.b_data_type = (uint64_t)b_data_type,
				.weight_data_type = (uint64_t)weight_data_type,
				.length = length,
				.weight_count = weight_count,
				.weighted = weighted,
				.loadM = !!(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M),
			};
			ccv_nnc_mfa_prepare_swish_mul(context, params);

			mtl_command_batch_t* command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
			mtl_buffer_t* tensors[5] = {
				mpgetbuffer(inputs[0]), // value
				mpgetbuffer(inputs[1]), // gate
				weighted ? mpgetbuffer(inputs[2]) : mpgetbuffer(outputs[0]), // row weight or destination
				weighted ? mpgetbuffer(outputs[0]) : NULL, // weighted destination
				NULL,
			};
			size_t tensor_offsets[4] = {
				a->dataof,
				b->dataof,
				weighted ? w->dataof : c->dataof,
				weighted ? c->dataof : 0,
			};
			ccv_nnc_mfa_encode_swish_mul(context, params, command_batch, tensors, tensor_offsets);
			ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
		} else {
			MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
			ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
			int indices[3];
			MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
				MPSGraphTensor* mps_input_a;
				MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
				[inputTensors addObject:mps_input_a];
				MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, a->info.dim, a->stride);
				[inputShapedTypes addObject:mps_a_shape];
				MPSGraphTensor* mps_input_b;
				MPSGraphTensor* mps_b = ccv_nnc_mps_graph_tensor_input(graph, b, b->info.dim, b->stride, &mps_input_b);
				[inputTensors addObject:mps_input_b];
				MPSGraphShapedType* mps_b_shape = ccv_nnc_mps_graph_tensor_input_shape(b, b->info.dim, b->stride);
				[inputShapedTypes addObject:mps_b_shape];
				MPSGraphTensor* mps_a_f32 = mps_a;
				if (a->info.datatype != CCV_32F)
					mps_a_f32 = [graph castTensor:mps_a toType:MPSDataTypeFloat32 name:@"mps_a_float"];
				MPSGraphTensor* mps_b_f32 = mps_b;
				if (b->info.datatype != CCV_32F)
					mps_b_f32 = [graph castTensor:mps_b toType:MPSDataTypeFloat32 name:@"mps_b_float"];
				MPSGraphTensor* mps_w_f32 = nil;
				if (w)
				{
					MPSGraphTensor* mps_input_w;
					MPSGraphTensor* mps_w = ccv_nnc_mps_graph_tensor_input(graph, w, w->info.dim, w->stride, &mps_input_w);
					[inputTensors addObject:mps_input_w];
					[inputShapedTypes addObject:ccv_nnc_mps_graph_tensor_input_shape(w, w->info.dim, w->stride)];
					mps_w_f32 = w->info.datatype == CCV_32F ? mps_w : [graph castTensor:mps_w toType:MPSDataTypeFloat32 name:@"mps_w_float"];
				}
				if (limit > 0)
				{
					MPSGraphTensor* const mps_limit = [graph constantWithScalar:limit dataType:MPSDataTypeFloat32];
					MPSGraphTensor* const mps_negative_limit = [graph constantWithScalar:-limit dataType:MPSDataTypeFloat32];
					mps_a_f32 = [graph minimumWithPrimaryTensor:[graph maximumWithPrimaryTensor:mps_a_f32 secondaryTensor:mps_negative_limit name:nil] secondaryTensor:mps_limit name:nil];
					mps_b_f32 = [graph minimumWithPrimaryTensor:mps_b_f32 secondaryTensor:mps_limit name:nil];
				}
				MPSGraphTensor* mps_beta = [graph constantWithScalar:beta dataType:MPSDataTypeFloat32];
				MPSGraphTensor* mps_beta_b = [graph multiplicationWithPrimaryTensor:mps_b_f32 secondaryTensor:mps_beta name:nil];
				MPSGraphTensor* mps_neg = [graph negativeWithTensor:mps_beta_b name:nil];
				MPSGraphTensor* mps_exp = [graph exponentWithTensor:mps_neg name:nil];
				MPSGraphTensor* mps_one = [graph constantWithScalar:1.0 dataType:MPSDataTypeFloat32];
				MPSGraphTensor* mps_denom = [graph additionWithPrimaryTensor:mps_exp secondaryTensor:mps_one name:nil];
				MPSGraphTensor* mps_sigmoid = [graph divisionWithPrimaryTensor:mps_one secondaryTensor:mps_denom name:nil];
				MPSGraphTensor* mps_swish = [graph multiplicationWithPrimaryTensor:mps_b_f32 secondaryTensor:mps_sigmoid name:nil];
				MPSGraphTensor* mps_c = [graph multiplicationWithPrimaryTensor:mps_a_f32 secondaryTensor:mps_swish name:nil];
				if (scale != 1)
				{
					MPSGraphTensor* mps_scale = [graph constantWithScalar:scale dataType:MPSDataTypeFloat32];
					mps_c = [graph multiplicationWithPrimaryTensor:mps_c secondaryTensor:mps_scale name:nil];
				}
				if (mps_w_f32)
				{
					NSArray<NSNumber*>* const output_shape = mps_c.shape;
					mps_c = [graph reshapeTensor:mps_c withShape:@[@(weight_count), @(length / weight_count)] name:nil];
					mps_w_f32 = [graph reshapeTensor:mps_w_f32 withShape:@[@(weight_count), @1] name:nil];
					mps_c = [graph multiplicationWithPrimaryTensor:mps_c secondaryTensor:mps_w_f32 name:nil];
					mps_c = [graph reshapeTensor:mps_c withShape:output_shape name:nil];
				}
				if (c->info.datatype != CCV_32F)
					mps_c = [graph castTensor:mps_c toType:ccv_nnc_mps_datatype(c->info.datatype) name:@"mps_c"];
				[resultTensors addObject:mps_c];
			});
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
			MPSGraphTensorData* data_b = ccv_nnc_mps_graph_tensor_data(b, b->info.dim, b->stride);
			MPSGraphTensorData* data_w = w ? ccv_nnc_mps_graph_tensor_data(w, w->info.dim, w->stride) : nil;
			MPSGraphTensorData* data[] = {data_a, data_b, data_w};
			NSArray<MPSGraphTensorData*>* feeds = w ? @[data[indices[0]], data[indices[1]], data[indices[2]]] : @[data[indices[0]], data[indices[1]]];
			ccv_nnc_mps_graph_executable_result(executable, command_buffer, feeds, &c, (int*[]){ c->info.dim }, (int*[]){ c->stride }, 1, 0);
			ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_swish_mul_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 3);
	assert(output_size >= 1);
	const float beta = cmd.info.swish_mul.beta;
	const float scale = cmd.info.swish_mul.scale;
	const ccv_nnc_tensor_view_t* const g = (const ccv_nnc_tensor_view_t*)inputs[0]; // gradient
	const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[1]; // value
	const ccv_nnc_tensor_view_t* const b = (const ccv_nnc_tensor_view_t*)inputs[2]; // gate
	ccv_nnc_tensor_view_t* const da = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* const db = output_size > 1 ? (ccv_nnc_tensor_view_t*)outputs[1] : 0;
	assert(g);
	assert(b);
	assert(da || db);
	assert(CCV_IS_TENSOR_CONTIGUOUS(g));
	assert(!a || CCV_IS_TENSOR_CONTIGUOUS(a));
	assert(CCV_IS_TENSOR_CONTIGUOUS(b));
	assert(!da || CCV_IS_TENSOR_CONTIGUOUS(da));
	assert(!db || CCV_IS_TENSOR_CONTIGUOUS(db));
	if (db)
		assert(a);
	assert(_ccv_nnc_swish_mul_mfa_datatype(g->info.datatype) >= 0);
	assert(!a || _ccv_nnc_swish_mul_mfa_datatype(a->info.datatype) >= 0);
	assert(_ccv_nnc_swish_mul_mfa_datatype(b->info.datatype) >= 0);
	assert(!da || _ccv_nnc_swish_mul_mfa_datatype(da->info.datatype) >= 0);
	assert(!db || _ccv_nnc_swish_mul_mfa_datatype(db->info.datatype) >= 0);
	int i;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && g->info.dim[i] > 0; i++)
	{
		assert(b->info.dim[i] == g->info.dim[i]);
		if (a)
			assert(a->info.dim[i] == g->info.dim[i]);
		if (da)
			assert(da->info.dim[i] == g->info.dim[i]);
		if (db)
			assert(db->info.dim[i] == g->info.dim[i]);
	}
	@autoreleasepool {
		bool use_mfa = true;
		ccv_nnc_mfa_context_t* context = ccv_nnc_default_mfa_context();
		if (!ccv_nnc_mfa_context_supported(context) || (ccv_nnc_flags() & CCV_NNC_DISABLE_MFA))
			use_mfa = false;
		const int g_data_type = _ccv_nnc_swish_mul_mfa_datatype(g->info.datatype);
		const int a_data_type = a ? _ccv_nnc_swish_mul_mfa_datatype(a->info.datatype) : -1;
		const int b_data_type = _ccv_nnc_swish_mul_mfa_datatype(b->info.datatype);
		const int da_data_type = da ? _ccv_nnc_swish_mul_mfa_datatype(da->info.datatype) : -1;
		const int db_data_type = db ? _ccv_nnc_swish_mul_mfa_datatype(db->info.datatype) : -1;
		if (use_mfa && (g_data_type < 0 || b_data_type < 0 || (db && a_data_type < 0) || (da && da_data_type < 0) || (db && db_data_type < 0)))
			use_mfa = false;
		if (use_mfa && (!CCV_IS_TENSOR_CONTIGUOUS(g) || (a && !CCV_IS_TENSOR_CONTIGUOUS(a)) || !CCV_IS_TENSOR_CONTIGUOUS(b) || (da && !CCV_IS_TENSOR_CONTIGUOUS(da)) || (db && !CCV_IS_TENSOR_CONTIGUOUS(db))))
			use_mfa = false;
		if (use_mfa)
		{
			const uint8_t output_mask = (da ? 1 : 0) | (db ? 2 : 0);
			const int default_data_type = _ccv_nnc_swish_mul_mfa_datatype(CCV_32F);
			ccv_nnc_mfa_swish_mul_params_t params = {
				.beta = beta,
				.scale = scale,
				.a_data_type = (uint64_t)(a_data_type >= 0 ? a_data_type : default_data_type),
				.b_data_type = (uint64_t)b_data_type,
				.length = (uint32_t)ccv_nnc_tensor_count(g->info),
				.gradient = 1,
				.loadM = !!(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M),
				.output_mask = output_mask,
				.g_data_type = (uint64_t)g_data_type,
				.da_data_type = (uint64_t)(da_data_type >= 0 ? da_data_type : default_data_type),
				.db_data_type = (uint64_t)(db_data_type >= 0 ? db_data_type : default_data_type),
			};
			ccv_nnc_mfa_prepare_swish_mul(context, params);
			mtl_command_batch_t* command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
			mtl_buffer_t* tensors[6] = {};
			size_t tensor_offsets[5] = {};
			int tensor_count = 0;
			tensors[tensor_count] = mpgetbuffer(inputs[0]); // gradient
			tensor_offsets[tensor_count] = g->dataof;
			++tensor_count;
			if (da && db)
			{
				tensors[tensor_count] = mpgetbuffer(inputs[1]); // value
				tensor_offsets[tensor_count] = a->dataof;
				++tensor_count;
				tensors[tensor_count] = mpgetbuffer(inputs[2]); // gate
				tensor_offsets[tensor_count] = b->dataof;
				++tensor_count;
				tensors[tensor_count] = mpgetbuffer(outputs[0]); // value gradient
				tensor_offsets[tensor_count] = da->dataof;
				++tensor_count;
				tensors[tensor_count] = mpgetbuffer(outputs[1]); // gate gradient
				tensor_offsets[tensor_count] = db->dataof;
				++tensor_count;
			} else if (da) {
				tensors[tensor_count] = mpgetbuffer(inputs[2]); // gate
				tensor_offsets[tensor_count] = b->dataof;
				++tensor_count;
				tensors[tensor_count] = mpgetbuffer(outputs[0]); // value gradient
				tensor_offsets[tensor_count] = da->dataof;
				++tensor_count;
			} else {
				tensors[tensor_count] = mpgetbuffer(inputs[1]); // value
				tensor_offsets[tensor_count] = a->dataof;
				++tensor_count;
				tensors[tensor_count] = mpgetbuffer(inputs[2]); // gate
				tensor_offsets[tensor_count] = b->dataof;
				++tensor_count;
				tensors[tensor_count] = mpgetbuffer(outputs[1]); // gate gradient
				tensor_offsets[tensor_count] = db->dataof;
				++tensor_count;
			}
			tensors[tensor_count] = NULL;
			ccv_nnc_mfa_encode_swish_mul(context, params, command_batch, tensors, tensor_offsets);
			ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
		} else {
			MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
			if (da)
			{
				ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
				int indices[2];
				MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
					MPSGraphTensor* mps_input_g;
					MPSGraphTensor* mps_g = ccv_nnc_mps_graph_tensor_input(graph, g, g->info.dim, g->stride, &mps_input_g);
					[inputTensors addObject:mps_input_g];
					MPSGraphShapedType* mps_g_shape = ccv_nnc_mps_graph_tensor_input_shape(g, g->info.dim, g->stride);
					[inputShapedTypes addObject:mps_g_shape];
					MPSGraphTensor* mps_input_b;
					MPSGraphTensor* mps_b = ccv_nnc_mps_graph_tensor_input(graph, b, b->info.dim, b->stride, &mps_input_b);
					[inputTensors addObject:mps_input_b];
					MPSGraphShapedType* mps_b_shape = ccv_nnc_mps_graph_tensor_input_shape(b, b->info.dim, b->stride);
					[inputShapedTypes addObject:mps_b_shape];
					MPSGraphTensor* mps_g_f32 = mps_g;
					if (g->info.datatype != CCV_32F)
						mps_g_f32 = [graph castTensor:mps_g toType:MPSDataTypeFloat32 name:@"mps_g_float"];
					MPSGraphTensor* mps_b_f32 = mps_b;
					if (b->info.datatype != CCV_32F)
						mps_b_f32 = [graph castTensor:mps_b toType:MPSDataTypeFloat32 name:@"mps_b_float"];
					MPSGraphTensor* mps_beta = [graph constantWithScalar:beta dataType:MPSDataTypeFloat32];
					MPSGraphTensor* mps_beta_b = [graph multiplicationWithPrimaryTensor:mps_b_f32 secondaryTensor:mps_beta name:nil];
					MPSGraphTensor* mps_neg = [graph negativeWithTensor:mps_beta_b name:nil];
					MPSGraphTensor* mps_exp = [graph exponentWithTensor:mps_neg name:nil];
					MPSGraphTensor* mps_one = [graph constantWithScalar:1.0 dataType:MPSDataTypeFloat32];
					MPSGraphTensor* mps_denom = [graph additionWithPrimaryTensor:mps_exp secondaryTensor:mps_one name:nil];
					MPSGraphTensor* mps_sigmoid = [graph divisionWithPrimaryTensor:mps_one secondaryTensor:mps_denom name:nil];
					MPSGraphTensor* mps_da = mps_g_f32;
					if (scale != 1)
					{
						MPSGraphTensor* mps_scale = [graph constantWithScalar:scale dataType:MPSDataTypeFloat32];
						mps_da = [graph multiplicationWithPrimaryTensor:mps_da secondaryTensor:mps_scale name:nil];
					}
					mps_da = [graph multiplicationWithPrimaryTensor:mps_da secondaryTensor:mps_b_f32 name:nil];
					mps_da = [graph multiplicationWithPrimaryTensor:mps_da secondaryTensor:mps_sigmoid name:nil];
					if (da->info.datatype != CCV_32F)
						mps_da = [graph castTensor:mps_da toType:ccv_nnc_mps_datatype(da->info.datatype) name:@"mps_da"];
					[resultTensors addObject:mps_da];
				});
				MPSGraphTensorData* data_g = ccv_nnc_mps_graph_tensor_data(g, g->info.dim, g->stride);
				MPSGraphTensorData* data_b = ccv_nnc_mps_graph_tensor_data(b, b->info.dim, b->stride);
				MPSGraphTensorData* data[] = {data_g, data_b};
				ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]]], (ccv_nnc_tensor_view_t* []){ da }, (int*[]){ da->info.dim }, (int*[]){ da->stride }, 1, 0);
			}
			if (db)
			{
				ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 1, hint, flags, inputs, input_size, outputs, output_size);
				int indices[3];
				MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
					MPSGraphTensor* mps_input_g;
					MPSGraphTensor* mps_g = ccv_nnc_mps_graph_tensor_input(graph, g, g->info.dim, g->stride, &mps_input_g);
					[inputTensors addObject:mps_input_g];
					MPSGraphShapedType* mps_g_shape = ccv_nnc_mps_graph_tensor_input_shape(g, g->info.dim, g->stride);
					[inputShapedTypes addObject:mps_g_shape];
					MPSGraphTensor* mps_input_a;
					MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
					[inputTensors addObject:mps_input_a];
					MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, a->info.dim, a->stride);
					[inputShapedTypes addObject:mps_a_shape];
					MPSGraphTensor* mps_input_b;
					MPSGraphTensor* mps_b = ccv_nnc_mps_graph_tensor_input(graph, b, b->info.dim, b->stride, &mps_input_b);
					[inputTensors addObject:mps_input_b];
					MPSGraphShapedType* mps_b_shape = ccv_nnc_mps_graph_tensor_input_shape(b, b->info.dim, b->stride);
					[inputShapedTypes addObject:mps_b_shape];
					MPSGraphTensor* mps_g_f32 = mps_g;
					if (g->info.datatype != CCV_32F)
						mps_g_f32 = [graph castTensor:mps_g toType:MPSDataTypeFloat32 name:@"mps_g_float"];
					MPSGraphTensor* mps_a_f32 = mps_a;
					if (a->info.datatype != CCV_32F)
						mps_a_f32 = [graph castTensor:mps_a toType:MPSDataTypeFloat32 name:@"mps_a_float"];
					MPSGraphTensor* mps_b_f32 = mps_b;
					if (b->info.datatype != CCV_32F)
						mps_b_f32 = [graph castTensor:mps_b toType:MPSDataTypeFloat32 name:@"mps_b_float"];
					MPSGraphTensor* mps_beta = [graph constantWithScalar:beta dataType:MPSDataTypeFloat32];
					MPSGraphTensor* mps_beta_b = [graph multiplicationWithPrimaryTensor:mps_b_f32 secondaryTensor:mps_beta name:nil];
					MPSGraphTensor* mps_neg = [graph negativeWithTensor:mps_beta_b name:nil];
					MPSGraphTensor* mps_exp = [graph exponentWithTensor:mps_neg name:nil];
					MPSGraphTensor* mps_one = [graph constantWithScalar:1.0 dataType:MPSDataTypeFloat32];
					MPSGraphTensor* mps_denom = [graph additionWithPrimaryTensor:mps_exp secondaryTensor:mps_one name:nil];
					MPSGraphTensor* mps_sigmoid = [graph divisionWithPrimaryTensor:mps_one secondaryTensor:mps_denom name:nil];
					MPSGraphTensor* mps_one_minus_sigmoid = [graph subtractionWithPrimaryTensor:mps_one secondaryTensor:mps_sigmoid name:nil];
					MPSGraphTensor* mps_sigmoid_grad = [graph multiplicationWithPrimaryTensor:mps_sigmoid secondaryTensor:mps_one_minus_sigmoid name:nil];
					MPSGraphTensor* mps_swish_grad = [graph multiplicationWithPrimaryTensor:mps_beta_b secondaryTensor:mps_sigmoid_grad name:nil];
					mps_swish_grad = [graph additionWithPrimaryTensor:mps_sigmoid secondaryTensor:mps_swish_grad name:nil];
					MPSGraphTensor* mps_db = mps_g_f32;
					if (scale != 1)
					{
						MPSGraphTensor* mps_scale = [graph constantWithScalar:scale dataType:MPSDataTypeFloat32];
						mps_db = [graph multiplicationWithPrimaryTensor:mps_db secondaryTensor:mps_scale name:nil];
					}
					mps_db = [graph multiplicationWithPrimaryTensor:mps_db secondaryTensor:mps_a_f32 name:nil];
					mps_db = [graph multiplicationWithPrimaryTensor:mps_db secondaryTensor:mps_swish_grad name:nil];
					if (db->info.datatype != CCV_32F)
						mps_db = [graph castTensor:mps_db toType:ccv_nnc_mps_datatype(db->info.datatype) name:@"mps_db"];
					[resultTensors addObject:mps_db];
				});
				MPSGraphTensorData* data_g = ccv_nnc_mps_graph_tensor_data(g, g->info.dim, g->stride);
				MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
				MPSGraphTensorData* data_b = ccv_nnc_mps_graph_tensor_data(b, b->info.dim, b->stride);
				MPSGraphTensorData* data[] = {data_g, data_a, data_b};
				ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]], data[indices[2]]], (ccv_nnc_tensor_view_t* []){ db }, (int*[]){ db->info.dim }, (int*[]){ db->stride }, 1, 0);
			}
			ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SWISH_MUL_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_swish_mul_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SWISH_MUL_BACKWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_swish_mul_back;
}
