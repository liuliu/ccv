/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

/**********************************************************
 * CCV - Neural Network Collection
 **********************************************************/

#include "_ccv_nnc_tensor_tape.h"
#include "_ccv_nnc_graph.h"
#include "ccv_nnc_internal.h"
#include "ccv_nnc_easy.h"
#ifdef HAVE_CUDA
#include "gpu/ccv_nnc_compat.h"
#endif

ccv_nnc_tensor_tape_t* ccv_nnc_tensor_tape_new(void)
{
	ccv_nnc_tensor_tape_t* tape = (ccv_nnc_tensor_tape_t*)ccmalloc(sizeof(ccv_nnc_tensor_tape_t));
	tape->tensor_data = ccv_array_new(sizeof(ccv_nnc_tape_tensor_data_array_t), 0, 0);
	tape->exec_data = ccv_array_new(sizeof(ccv_nnc_tape_exec_data_array_t), 0, 0);
	return tape;
}

static ccv_nnc_tensor_t* _ccv_nnc_tensor_from_tensor_multiview(const ccv_nnc_graph_t* const* const graphs, const int graph_size, ccv_nnc_tensor_multiview_t* const mv)
{
	int i;
	ccv_nnc_tensor_t* tensor = (ccv_nnc_tensor_t*)mv;
	for (i = 0; CCV_IS_TENSOR_MULTIVIEW(tensor) && i < graph_size; i++)
	{
		const int count = (int)graphs[i]->while_count;
		while (CCV_IS_TENSOR_MULTIVIEW(tensor) &&
			   (((ccv_nnc_tensor_multiview_t*)tensor)->anchor == (intptr_t)graphs[i] ||
				((ccv_nnc_tensor_multiview_t*)tensor)->anchor == (intptr_t)graphs[i]->peer))
		{
			ccv_nnc_tensor_multiview_t* mv = (ccv_nnc_tensor_multiview_t*)tensor;
			const int off = mv->kind;
			const int mod = mv->repeat;
			// If reached the root.
			tensor = CCV_NNC_MULTIVIEW_DATA(mv)[count >= off ? ((count - off) % mod) + off : count]; // Unwrap.
		}
	}
	return tensor;
}

#define CCV_NNC_IS_TAPE_TENSOR_DATA_ARRAY_POS(ptr) ((uintptr_t)(ptr) & 1)
#define CCV_NUMERIC_DATA_NO_ALLOC(data) ((uintptr_t)(data.u8) & 1)
// Align integer to 16-bytes.
#define ALIGN_16(x) (((x) + 3) & -4)

// Simple allocator from ccv_array_t.
static void _ccv_nnc_tape_tensor_data_array_pos_new(ccv_array_t* const tensor_data, int* const pos_ref, ccv_nnc_tape_tensor_data_array_t** const tape_tensor_data_ref)
{
	int pos = tensor_data->rnum;
	ccv_array_resize(tensor_data, pos + 1);
	*pos_ref = (pos << 1) | 1;
	ccv_nnc_tape_tensor_data_array_t* const tape_tensor_data = (ccv_nnc_tape_tensor_data_array_t*)ccv_array_get(tensor_data, pos);
	memset(tape_tensor_data, 0, sizeof(ccv_nnc_tape_tensor_data_array_t));
	*tape_tensor_data_ref = tape_tensor_data;
}

static ccv_nnc_tape_tensor_data_array_t* _ccv_nnc_tape_tensor_data_array_get(const ccv_array_t* const tensor_data, const int pos)
{
	assert((pos >> 1) <= tensor_data->rnum);
	return (ccv_nnc_tape_tensor_data_array_t*)ccv_array_get(tensor_data, pos >> 1);
}

static void _ccv_nnc_tape_tensor_data_move(ccv_nnc_tape_tensor_data_t* const old_data, ccv_nnc_tape_tensor_data_t* const new_data, const int offset, const ccv_nnc_graph_t* const* const graphs, const int graph_size, const int* const dim, const int dim_count)
{
	int i;
	if (offset == ccv_max(dim_count, graph_size) - 1)
	{
		const int data_dim = offset < dim_count ? dim[offset] - 1 : 0;
		const int graph_dim = offset < graph_size ? graphs[offset]->while_count + 1 : 0;
		assert(old_data <= new_data);
		// Do the actual copy or set.
		if (!old_data)
			for (i = ccv_max(data_dim, graph_dim); i >= 0; i--)
				new_data[i].data.u8 = 0;
		else {
			for (i = graph_dim; i > data_dim; i--)
				new_data[i].data.u8 = 0;
			for (i = data_dim; i >= 0; i--)
				new_data[i] = old_data[i];
		}
	} else {
		int old_data_step = 1;
		for (i = offset + 1; i < dim_count; i++)
			old_data_step *= dim[i];
		const int new_dim_count = ccv_max(graph_size, dim_count);
		int new_data_step = 1;
		for (i = offset + 1; i < new_dim_count; i++)
		{
			int old_dim = (i < dim_count) ? dim[i] : 1;
			int graph_dim = (i < graph_size) ? (int)(graphs[i]->while_count + 2) : 1;
			new_data_step *= ccv_max(old_dim, graph_dim);
		}
		const int data_dim = offset < dim_count ? dim[offset] - 1 : 0;
		const int graph_dim = offset < graph_size ? graphs[offset]->while_count + 1 : 0;
		for (i = ccv_max(data_dim, graph_dim); i >= 0; i--)
			_ccv_nnc_tape_tensor_data_move((old_data && offset < dim_count && i < dim[offset]) ? old_data + i * old_data_step : 0, new_data + i * new_data_step, offset + 1, graphs, graph_size, dim, dim_count);
	}
}

static void _ccv_nnc_tape_tensor_data_array_resize(ccv_nnc_tape_tensor_data_array_t* const data_array, const ccv_nnc_graph_t* const* const graphs, const int graph_size)
{
	const int new_dim_count = ccv_max(graph_size, data_array->dim_count);
	int i;
	int size = 1;
	for (i = 0; i < new_dim_count; i++)
	{
		int old_dim = (i < data_array->dim_count) ? data_array->dim[i] : 1;
		int graph_dim = (i < graph_size) ? (int)(graphs[i]->while_count + 2) : 1;
		size *= ccv_max(old_dim, graph_dim);
	}
	data_array->dim = ccrealloc(data_array->dim, sizeof(int) * ALIGN_16(new_dim_count) + sizeof(ccv_nnc_tape_tensor_data_t) * size);
	ccv_nnc_tape_tensor_data_t* const old_data = (ccv_nnc_tape_tensor_data_t*)(data_array->dim + ALIGN_16(data_array->dim_count));
	ccv_nnc_tape_tensor_data_t* const new_data = (ccv_nnc_tape_tensor_data_t*)(data_array->dim + ALIGN_16(new_dim_count));
	// Note that both old_data and new_data occupies the same memory region, since the resize operation
	// is mono-increasing, we just need to move the data from the end to the beginning to avoid data
	// overwrite issues.
	assert(graph_size > 0);
	assert(data_array->dim_count > 0);
	_ccv_nnc_tape_tensor_data_move(old_data, new_data, 0, graphs, graph_size, data_array->dim, data_array->dim_count);
	data_array->data = new_data;
	// We are done, update the dim.
	for (i = 0; i < new_dim_count; i++)
	{
		int old_dim = (i < data_array->dim_count) ? data_array->dim[i] : 1;
		int graph_dim = (i < graph_size) ? (int)(graphs[i]->while_count + 2) : 1;
		data_array->dim[i] = ccv_max(old_dim, graph_dim);
	}
	data_array->dim_count = new_dim_count;
}

static void _ccv_nnc_tensor_from_tape(ccv_array_t* const tensor_data, ccv_nnc_tensor_t* const tensor, const int flags, const ccv_nnc_graph_t* const* const graphs, const int graph_size, const int create_if_missing)
{
	assert(graph_size > 0);
	ccv_nnc_tensor_t* tensor_ref = tensor;
	while (tensor_ref->alias_ref && !CCV_NNC_IS_TAPE_TENSOR_DATA_ARRAY_POS(tensor_ref->alias_ref))
	{
		tensor_ref = (ccv_nnc_tensor_t*)tensor->alias_ref;
		if (CCV_IS_TENSOR_MULTIVIEW(tensor_ref))
			tensor_ref = _ccv_nnc_tensor_from_tensor_multiview(graphs, graph_size, (ccv_nnc_tensor_multiview_t*)tensor_ref);
	}
	ccv_nnc_tape_tensor_data_array_t* data_array;
	if (!tensor_ref->alias_ref)
	{
		// Create data array.
		int pos;
		_ccv_nnc_tape_tensor_data_array_pos_new(tensor_data, &pos, &data_array);
		tensor_ref->alias_ref = pos;
	} else
		data_array = _ccv_nnc_tape_tensor_data_array_get(tensor_data, (int)tensor_ref->alias_ref);
	// Either the data exists, or it doesn't and we need to create one.
	int i;
	if (!data_array->dim)
	{
		int size = 1;
		for (i = 0; i < graph_size; i++)
			size *= (int)(graphs[i]->while_count + 2);
		data_array->dim_count = graph_size;
		data_array->dim = (int*)ccmalloc(sizeof(int) * ALIGN_16(graph_size) + sizeof(ccv_nnc_tape_tensor_data_t) * size);
		for (i = 0; i < graph_size; i++)
			data_array->dim[i] = (int)(graphs[i]->while_count + 2);
		data_array->data = (ccv_nnc_tape_tensor_data_t*)(data_array->dim + ALIGN_16(graph_size));
		for (i = 0; i < size; i++)
			data_array->data[i].data.u8 = 0;
	} else {
		int flag = (data_array->dim_count < graph_size);
		for (i = 0; !flag && i < graph_size; i++)
			flag = (data_array->dim[i] <= graphs[i]->while_count + 1);
		if (flag)
			_ccv_nnc_tape_tensor_data_array_resize(data_array, graphs, graph_size);
	}
	// Compute the index.
	int idx, step;
	idx = (graphs[graph_size - 1]->while_count + 1);
	step = data_array->dim[graph_size - 1];
	for (i = graph_size - 2; i >= 0; i--)
	{
		idx += (graphs[i]->while_count + 1) * step;
		step *= data_array->dim[i];
	}
	ccv_numeric_data_t data = data_array->data[idx].data;
	if (!data.u8)
	{
		// If we cannot create, loop back idx until we find one that exists.
		if (!create_if_missing)
		{
			if (data_array->data[idx].data.u8)
				data.u8 = (unsigned char*)((uintptr_t)data_array->data[idx].data.u8 | (uintptr_t)1);
			else
			// Now looped back to 0, if still cannot find, using the original pointer.
				data.u8 = data_array->data[idx].data.u8 = (unsigned char*)((uintptr_t)tensor_ref->data.u8 | (uintptr_t)1);
		} else {
			const size_t size = ccv_nnc_tensor_data_size(tensor->info);
			data_array->data[idx].type = tensor->info.type;
#ifdef HAVE_CUDA
			if (CCV_TENSOR_GET_MEMORY(tensor->info.type) == CCV_TENSOR_GPU_MEMORY)
				data_array->data[idx].data.u8 = (uint8_t*)cumalloc(CCV_TENSOR_GET_DEVICE_ID(tensor->info.type), size);
			else
				ccmemalign((void **)&data_array->data[idx].data.u8, 16, size);
#else
			assert(CCV_TENSOR_GET_MEMORY(tensor->info.type) == CCV_TENSOR_CPU_MEMORY);
			ccmemalign((void **)&data_array->data[idx].data.u8, 16, size);
#endif
			data = data_array->data[idx].data;
		}
	}
	tensor->data.u8 = (unsigned char*)((uintptr_t)data.u8 & ~(uintptr_t)1);
}

void ccv_nnc_tensor_tape_io(ccv_nnc_tensor_tape_t* const tape, const ccv_nnc_graph_t* const graph, const int* const input_flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, const int* const output_flags, ccv_nnc_tensor_t* const* const outputs, const int output_size)
{
	int i, tape_io = 0;
	for (i = 0; i < input_size && !tape_io; i++)
		if (inputs[i] && CCV_GET_TAPE_ALLOC(inputs[i]->type))
			tape_io = 1;
	for (i = 0; i < output_size && !tape_io; i++)
		if (outputs[i] && CCV_GET_TAPE_ALLOC(outputs[i]->type))
			tape_io = 1;
	// If doesn't need to update with tape io, just pointing to the inputs and outputs directly.
	if (!tape_io)
		return;
	// Go to the root graph, record which was taken along the way.
	// In this way, we can then unwrap multi-view tensors.
	assert(graph);
	const ccv_nnc_graph_t* curr_graph = graph;
	int d;
	for (d = 0; curr_graph; d++)
		curr_graph = curr_graph->p;
	curr_graph = graph;
	const int graph_size = d;
	assert(graph_size > 0);
	const ccv_nnc_graph_t* graphs[graph_size];
	for (d = graph_size - 1; curr_graph; d--, curr_graph = curr_graph->p)
		graphs[d] = curr_graph;
	// Now, go through the inputs / outputs and update.
	for (i = 0; i < input_size; i++)
		if (inputs[i] && CCV_GET_TAPE_ALLOC(inputs[i]->type))
			_ccv_nnc_tensor_from_tape(tape->tensor_data, inputs[i], input_flags ? input_flags[i] : 0, graphs, graph_size, 0);
	for (i = 0; i < output_size; i++)
		if (outputs[i] && CCV_GET_TAPE_ALLOC(outputs[i]->type))
			_ccv_nnc_tensor_from_tape(tape->tensor_data, outputs[i], output_flags ? output_flags[i] : 0, graphs, graph_size, 1); // Create if it is not found. This is OK for output tensor.
}

#define CCV_NNC_IS_TAPE_EXEC_DATA_ARRAY_POS(ptr) ((uintptr_t)(ptr) & 1)

// Simple allocator from ccv_array_t.
static void _ccv_nnc_tape_exec_data_array_pos_new(ccv_array_t* const exec_data, int* const pos_ref, ccv_nnc_tape_exec_data_array_t** const tape_exec_data_ref)
{
	int pos = exec_data->rnum;
	ccv_array_resize(exec_data, pos + 1);
	*pos_ref = (pos << 1) | 1;
	ccv_nnc_tape_exec_data_array_t* const tape_exec_data = (ccv_nnc_tape_exec_data_array_t*)ccv_array_get(exec_data, pos);
	memset(tape_exec_data, 0, sizeof(ccv_nnc_tape_exec_data_array_t));
	*tape_exec_data_ref = tape_exec_data;
}

static ccv_nnc_tape_exec_data_array_t* _ccv_nnc_tape_exec_data_array_get(const ccv_array_t* const exec_data, const int pos)
{
	assert((pos >> 1) <= exec_data->rnum);
	return (ccv_nnc_tape_exec_data_array_t*)ccv_array_get(exec_data, pos >> 1);
}

static void _ccv_nnc_tape_exec_data_move(uint64_t* const old_data, uint64_t* const new_data, const int offset, const uint64_t* const while_counts, const int graph_size, const int* const dim, const int dim_count)
{
	int i;
	if (offset == ccv_max(dim_count, graph_size) - 1)
	{
		const int data_dim = offset < dim_count ? dim[offset] - 1 : 0;
		const int graph_dim = offset < graph_size ? while_counts[offset] : 0;
		assert(old_data <= new_data);
		// Do the actual copy or set.
		if (!old_data)
			for (i = ccv_max(data_dim, graph_dim); i >= 0; i--)
				new_data[i] = 0;
		else {
			for (i = graph_dim; i > data_dim; i--)
				new_data[i] = 0;
			for (i = data_dim; i >= 0; i--)
				new_data[i] = old_data[i];
		}
	} else {
		int old_data_step = 1;
		for (i = offset + 1; i < dim_count; i++)
			old_data_step *= dim[i];
		const int new_dim_count = ccv_max(graph_size, dim_count);
		int new_data_step = 1;
		for (i = offset + 1; i < new_dim_count; i++)
		{
			int old_dim = (i < dim_count) ? dim[i] : 1;
			int graph_dim = (i < graph_size) ? (int)(while_counts[i] + 1) : 1;
			new_data_step *= ccv_max(old_dim, graph_dim);
		}
		const int data_dim = offset < dim_count ? dim[offset] - 1 : 0;
		const int graph_dim = offset < graph_size ? while_counts[offset] : 0;
		for (i = ccv_max(data_dim, graph_dim); i >= 0; i--)
			_ccv_nnc_tape_exec_data_move((old_data && offset < dim_count && i < dim[offset]) ? old_data + i * old_data_step : 0, new_data + i * new_data_step, offset + 1, while_counts, graph_size, dim, dim_count);
	}
}

static void _ccv_nnc_tape_exec_data_array_resize(ccv_nnc_tape_exec_data_array_t* const data_array, const uint64_t* const while_counts, const int graph_size)
{
	const int new_dim_count = ccv_max(graph_size, data_array->dim_count);
	int i;
	int size = 1;
	for (i = 0; i < new_dim_count; i++)
	{
		int old_dim = (i < data_array->dim_count) ? data_array->dim[i] : 1;
		int graph_dim = (i < graph_size) ? (int)(while_counts[i] + 1) : 1;
		size *= ccv_max(old_dim, graph_dim);
	}
	data_array->dim = ccrealloc(data_array->dim, sizeof(int) * ALIGN_16(new_dim_count) + sizeof(uint64_t) * size);
	uint64_t* const old_data = (uint64_t*)(data_array->dim + ALIGN_16(data_array->dim_count));
	uint64_t* const new_data = (uint64_t*)(data_array->dim + ALIGN_16(new_dim_count));
	// Note that both old_data and new_data occupies the same memory region, since the resize operation
	// is mono-increasing, we just need to move the data from the end to the beginning to avoid data
	// overwrite issues.
	assert(graph_size > 0);
	assert(data_array->dim_count > 0);
	_ccv_nnc_tape_exec_data_move(old_data, new_data, 0, while_counts, graph_size, data_array->dim, data_array->dim_count);
	data_array->data = new_data;
	// We are done, update the dim.
	for (i = 0; i < new_dim_count; i++)
	{
		int old_dim = (i < data_array->dim_count) ? data_array->dim[i] : 1;
		int graph_dim = (i < graph_size) ? (int)(while_counts[i] + 1) : 1;
		data_array->dim[i] = ccv_max(old_dim, graph_dim);
	}
	data_array->dim_count = new_dim_count;
}

uint64_t ccv_nnc_tensor_tape_numbering(ccv_nnc_tensor_tape_t* const tape, const ccv_nnc_graph_t* const graph, const ccv_nnc_graph_exec_t exec)
{
	assert(exec.graph == graph);
	ccv_nnc_graph_exec_info_t* exec_info = ccv_array_get(graph->exec_info, exec.d);
	if (!exec_info->alias_ref && exec_info->peer_ref)
		exec_info = ccv_array_get(graph->exec_info, exec_info->peer_ref - 1);
	ccv_nnc_tape_exec_data_array_t* const data_array = _ccv_nnc_tape_exec_data_array_get(tape->exec_data, (int)exec_info->alias_ref);
	const ccv_nnc_graph_t* curr_graph = graph;
	int i;
	for (i = 0; curr_graph; i++)
		curr_graph = curr_graph->p;
	curr_graph = graph;
	const int graph_size = i;
	uint64_t while_counts[graph_size];
	for (i = graph_size - 1; curr_graph; i--, curr_graph = curr_graph->p)
		while_counts[i] = curr_graph->while_count;
	assert(graph_size <= data_array->dim_count);
	int idx = 0, step = 1;
	for (i = graph_size - 1; i >= 0; i--)
	{
		assert(while_counts[i] < data_array->dim[i]);
		idx += while_counts[i] * step;
		step *= data_array->dim[i];
	}
	return data_array->data[idx];
}

void ccv_nnc_tensor_tape_set_numbering(ccv_nnc_tensor_tape_t* const tape, ccv_nnc_graph_t* const graph, const ccv_nnc_graph_exec_t exec, const uint64_t numbering)
{
	ccv_nnc_tape_exec_data_array_t* data_array;
	assert(exec.graph == graph);
	ccv_nnc_graph_exec_info_t* const exec_info = ccv_array_get(graph->exec_info, exec.d);
	if (exec_info->alias_ref)
	{
		assert(CCV_NNC_IS_TAPE_EXEC_DATA_ARRAY_POS(exec_info->alias_ref));
		data_array = _ccv_nnc_tape_exec_data_array_get(tape->exec_data, (int)exec_info->alias_ref);
	} else {
		int pos;
		_ccv_nnc_tape_exec_data_array_pos_new(tape->exec_data, &pos, &data_array);
		exec_info->alias_ref = pos;
	}
	const ccv_nnc_graph_t* curr_graph = graph;
	assert(curr_graph);
	int i;
	for (i = 0; curr_graph; i++)
		curr_graph = curr_graph->p;
	curr_graph = graph;
	const int graph_size = i;
	assert(graph_size > 0);
	uint64_t while_counts[graph_size];
	for (i = graph_size - 1; curr_graph; i--, curr_graph = curr_graph->p)
		while_counts[i] = curr_graph->while_count;
	if (!data_array->dim)
	{
		int size = 1;
		for (i = 0; i < graph_size; i++)
			size *= (int)(while_counts[i] + 1);
		data_array->dim_count = graph_size;
		data_array->dim = (int*)ccmalloc(sizeof(int) * ALIGN_16(graph_size) + sizeof(uint64_t) * size);
		for (i = 0; i < graph_size; i++)
			data_array->dim[i] = (int)(while_counts[i] + 1);
		data_array->data = (uint64_t*)(data_array->dim + ALIGN_16(graph_size));
		for (i = 0; i < size; i++)
			data_array->data[i] = 0;
	} else {
		int flag = (data_array->dim_count < graph_size);
		for (i = 0; !flag && i < graph_size; i++)
			flag = (data_array->dim[i] <= while_counts[i]);
		if (flag)
			_ccv_nnc_tape_exec_data_array_resize(data_array, while_counts, graph_size);
	}
	int idx = 0, step = 1;
	for (i = graph_size - 1; i >= 0; i--)
	{
		assert(while_counts[i] < data_array->dim[i]);
		idx += while_counts[i] * step;
		step *= data_array->dim[i];
	}
	data_array->data[idx] = numbering;
}

void ccv_nnc_tensor_tape_free(ccv_nnc_tensor_tape_t* const tape)
{
	int i, j;
	for (i = 0; i < tape->tensor_data->rnum; i++)
	{
		ccv_nnc_tape_tensor_data_array_t* const data_array = (ccv_nnc_tape_tensor_data_array_t*)ccv_array_get(tape->tensor_data, i);
		if (data_array->dim)
		{
			int size = 1;
			for (j = 0; j < data_array->dim_count; j++)
				size *= data_array->dim[j];
			for (j = 0; j < size; j++)
				if (data_array->data[j].data.u8 && !CCV_NUMERIC_DATA_NO_ALLOC(data_array->data[j].data))
				{
#ifdef HAVE_CUDA
					if (CCV_TENSOR_GET_MEMORY(data_array->data[j].type) == CCV_TENSOR_GPU_MEMORY)
						cufree(CCV_TENSOR_GET_DEVICE_ID(data_array->data[j].type), data_array->data[j].data.u8);
					else
						ccfree(data_array->data[j].data.u8);
#else
					ccfree(data_array->data[j].data.u8);
#endif
				}
			ccfree(data_array->dim);
		}
	}
	ccv_array_free(tape->tensor_data);
	for (i = 0; i < tape->exec_data->rnum; i++)
	{
		ccv_nnc_tape_exec_data_array_t* const data_array = (ccv_nnc_tape_exec_data_array_t*)ccv_array_get(tape->exec_data, i);
		if (data_array->dim)
			ccfree(data_array->dim);
	}
	ccv_array_free(tape->exec_data);
	ccfree(tape);
}
