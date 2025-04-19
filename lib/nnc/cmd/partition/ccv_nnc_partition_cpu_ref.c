#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#ifdef USE_OPENMP
#include <omp.h>
#endif
#ifdef USE_DISPATCH
#include <dispatch/dispatch.h>
#endif

static int _ccv_nnc_partition_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	assert(output_size == 2);
	const ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[0];
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(ccv_nnc_tensor_nd(b->info.dim) == a_nd);
	ccv_nnc_tensor_view_t* const indices = (ccv_nnc_tensor_view_t*)outputs[1];
	assert(ccv_nnc_tensor_nd(indices->info.dim) == a_nd);
	assert(indices->info.datatype == CCV_32S);
	assert(CCV_IS_TENSOR_CONTIGUOUS(a));
	assert(CCV_IS_TENSOR_CONTIGUOUS(b));
	assert(CCV_IS_TENSOR_CONTIGUOUS(indices));
	assert(a->info.datatype == b->info.datatype);
	const int count = ccv_nnc_tensor_count(a->info);
	if (a_nd == 1)
	{
		int i, j;
		void* workmem = ccv_nnc_stream_context_get_workspace(stream_context, ((a->info.datatype == CCV_32F) ? sizeof(float) : sizeof(int)) * count + sizeof(int) * count, CCV_TENSOR_CPU_MEMORY);
		// This is the fast path, we just do a regular sort and extract the index.
		assert(ccv_nnc_tensor_count(b->info) == cmd.info.partition.kth);
		assert(ccv_nnc_tensor_count(indices->info) == cmd.info.partition.kth);
		if (a->info.datatype == CCV_32F)
		{
			memcpy(workmem, a->data.f32, sizeof(float) * count);
			float* const a_ptr = (float*)workmem;
			int* const idx_ptr = (int*)((char*)workmem + sizeof(float) * count);
			for (i = 0; i < count; i++)
				idx_ptr[i] = i;
			if (cmd.info.partition.descending)
			{
				for (i = 0; i < cmd.info.partition.kth; i++)
				{
					float k = a_ptr[i];
					int v = i;
					for (j = i + 1; j < count; j++)
						if (a_ptr[j] > k)
							k = a_ptr[j], v = j;
					b->data.f32[i] = k;
					indices->data.i32[i] = idx_ptr[v];
					if (i != v)
						a_ptr[v] = a_ptr[i], idx_ptr[v] = idx_ptr[i];
				}
			} else {
				for (i = 0; i < cmd.info.partition.kth; i++)
				{
					float k = a_ptr[i];
					int v = i;
					for (j = i + 1; j < count; j++)
						if (a_ptr[j] < k)
							k = a_ptr[j], v = j;
					b->data.f32[i] = k;
					indices->data.i32[i] = idx_ptr[v];
					if (i != v)
						a_ptr[v] = a_ptr[i], idx_ptr[v] = idx_ptr[i];
				}
			}
		} else {
			assert(a->info.datatype == CCV_32S);
			memcpy(workmem, a->data.i32, sizeof(int) * count);
			int* const a_ptr = (int*)workmem;
			int* const idx_ptr = (int*)((char*)workmem + sizeof(float) * count);
			for (i = 0; i < count; i++)
				idx_ptr[i] = i;
			if (cmd.info.partition.descending)
			{
				for (i = 0; i < cmd.info.partition.kth; i++)
				{
					int k = a_ptr[i];
					int v = i;
					for (j = i + 1; j < count; j++)
						if (a_ptr[j] > k)
							k = a_ptr[j], v = j;
					b->data.i32[i] = k;
					indices->data.i32[i] = idx_ptr[v];
					if (i != v)
						a_ptr[v] = a_ptr[i], idx_ptr[v] = idx_ptr[i];
				}
			} else {
				for (i = 0; i < cmd.info.partition.kth; i++)
				{
					int k = a_ptr[i];
					int v = i;
					for (j = i + 1; j < count; j++)
						if (a_ptr[j] < k)
							k = a_ptr[j], v = j;
					b->data.i32[i] = k;
					indices->data.i32[i] = idx_ptr[v];
					if (i != v)
						a_ptr[v] = a_ptr[i], idx_ptr[v] = idx_ptr[i];
				}
			}
		}
	} else {
		int i, j, k, f;
		int sort_runs = 1;
		int sort_stride = 1;
		for (i = 0; i < a_nd; i++)
		{
			if (i < cmd.info.partition.along_axis) // Skip this.
				sort_runs *= a->info.dim[i];
			else if (i > cmd.info.partition.along_axis)
				sort_stride *= a->info.dim[i];
		}
		const int skip_stride = sort_stride * a->info.dim[cmd.info.partition.along_axis];
		const int dim = a->info.dim[cmd.info.partition.along_axis];
		void* workmem = ccv_nnc_stream_context_get_workspace(stream_context, ((a->info.datatype == CCV_32F) ? sizeof(float) : sizeof(int)) * count + sizeof(int) * dim, CCV_TENSOR_CPU_MEMORY);
		if (a->info.datatype == CCV_32F)
		{
			memcpy(workmem, a->data.f32, sizeof(float) * count);
			float* const a_ptr = (float*)workmem;
			int* const idx_ptr = (int*)((char*)workmem + sizeof(float) * count);
			if (cmd.info.partition.descending)
				for (i = 0; i < sort_runs; i++)
					for (j = 0; j < sort_stride; j++)
					{
						float* const a_ptr_0 = a_ptr + skip_stride * i + j;
						for (k = 0; k < dim; k++)
							idx_ptr[k] = k;
						float* const b_ptr = b->data.f32 + sort_stride * cmd.info.partition.kth * i + j;
						int* const indices_ptr = indices->data.i32 + sort_stride * cmd.info.partition.kth * i + j;
						for (k = 0; k < cmd.info.partition.kth; k++)
						{
							float key = a_ptr_0[k * sort_stride];
							int val = k;
							for (f = k + 1; f < dim; f++)
								if (a_ptr_0[f * sort_stride] > key)
									key = a_ptr_0[f * sort_stride], val = f;
							b_ptr[k * sort_stride] = key;
							indices_ptr[k * sort_stride] = idx_ptr[val];
							if (k != val)
								a_ptr_0[val * sort_stride] = a_ptr_0[k * sort_stride], idx_ptr[val] = idx_ptr[k];
						}
					}
			else
				for (i = 0; i < sort_runs; i++)
					for (j = 0; j < sort_stride; j++)
					{
						float* const a_ptr_0 = a_ptr + skip_stride * i + j;
						for (k = 0; k < dim; k++)
							idx_ptr[k] = k;
						float* const b_ptr = b->data.f32 + sort_stride * cmd.info.partition.kth * i + j;
						int* const indices_ptr = indices->data.i32 + sort_stride * cmd.info.partition.kth * i + j;
						for (k = 0; k < cmd.info.partition.kth; k++)
						{
							float key = a_ptr_0[k * sort_stride];
							int val = k;
							for (f = k + 1; f < dim; f++)
								if (a_ptr_0[f * sort_stride] < key)
									key = a_ptr_0[f * sort_stride], val = f;
							b_ptr[k * sort_stride] = key;
							indices_ptr[k * sort_stride] = idx_ptr[val];
							if (k != val)
								a_ptr_0[val * sort_stride] = a_ptr_0[k * sort_stride], idx_ptr[val] = idx_ptr[k];
						}
					}
		} else {
			assert(a->info.datatype == CCV_32S);
			memcpy(workmem, a->data.f32, sizeof(int) * count);
			int* const a_ptr = (int*)workmem;
			int* const idx_ptr = (int*)((char*)workmem + sizeof(int) * count);
			if (cmd.info.partition.descending)
				for (i = 0; i < sort_runs; i++)
					for (j = 0; j < sort_stride; j++)
					{
						int* const a_ptr_0 = a_ptr + skip_stride * i + j;
						for (k = 0; k < dim; k++)
							idx_ptr[k] = k;
						int* const b_ptr = b->data.i32 + sort_stride * cmd.info.partition.kth * i + j;
						int* const indices_ptr = indices->data.i32 + sort_stride * cmd.info.partition.kth * i + j;
						for (k = 0; k < cmd.info.partition.kth; k++)
						{
							int key = a_ptr_0[k * sort_stride];
							int val = k;
							for (f = k + 1; f < dim; f++)
								if (a_ptr_0[f * sort_stride] > key)
									key = a_ptr_0[f * sort_stride], val = f;
							b_ptr[k * sort_stride] = key;
							indices_ptr[k * sort_stride] = idx_ptr[val];
							if (k != val)
								a_ptr_0[val * sort_stride] = a_ptr_0[k * sort_stride], idx_ptr[val] = idx_ptr[k];
						}
					}
			else
				for (i = 0; i < sort_runs; i++)
					for (j = 0; j < sort_stride; j++)
					{
						int* const a_ptr_0 = a_ptr + skip_stride * i + j;
						for (k = 0; k < dim; k++)
							idx_ptr[k] = k;
						int* const b_ptr = b->data.i32 + sort_stride * cmd.info.partition.kth * i + j;
						int* const indices_ptr = indices->data.i32 + sort_stride * cmd.info.partition.kth * i + j;
						for (k = 0; k < cmd.info.partition.kth; k++)
						{
							int key = a_ptr_0[k * sort_stride];
							int val = k;
							for (f = k + 1; f < dim; f++)
								if (a_ptr_0[f * sort_stride] < key)
									key = a_ptr_0[f * sort_stride], val = f;
							b_ptr[k * sort_stride] = key;
							indices_ptr[k * sort_stride] = idx_ptr[val];
							if (k != val)
								a_ptr_0[val * sort_stride] = a_ptr_0[k * sort_stride], idx_ptr[val] = idx_ptr[k];
						}
					}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_partition_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_INVALID;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_PARTITION_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_partition_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_PARTITION_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_partition_back;
}
