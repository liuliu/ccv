#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#ifdef HAVE_CUDA
#include "gpu/ccv_nnc_compat.h"
#endif
#include "_ccv_nnc_symbolic_graph.h"

/**
 * Level-4 API
 */

typedef struct {
	int f_wrt; // Check if both f_symbols and wrt_symbols flow through this node.
	ccv_array_t* outgoings; // backward traverse nodes.
	uint64_t* input_bitmasks;
	int input_bitmask_size;
	uint64_t* output_bitmasks;
	int output_bitmask_size;
} ccv_nnc_graph_backward_info_t;

typedef struct {
	int input_size;
	int* inputs;
	int output;
	ccv_array_t* outgoings;
	float value;
	ccv_nnc_graph_exec_symbol_t symbol;
} ccv_nnc_graph_sum_or_set_exec_t;

typedef struct {
	int input_size;
	int output_size;
	int* inputs;
	int* outputs;
	ccv_array_t* outgoings;
	ccv_nnc_cmd_t cmd;
	ccv_nnc_graph_exec_symbol_t symbol;
} ccv_nnc_graph_autograd_exec_t;

typedef struct {
	int d; // The pointer to the forward level object.
	int alias_ref; // The alias ref to itself (autograd_tensor_symbol array).
	int flags; // Flags for this symbol.
	ccv_nnc_tensor_symbol_t symbol;
} ccv_nnc_autograd_tensor_symbol_t;

typedef struct {
	int d; // The tensor symbol ref.
	int x; // The exec symbol ref.
	ccv_array_t* exec_registry; // Additional exec symbol refs, similar to x, only useful for aliasing.
	ccv_array_t* alias_registry; // int point to all the alias (if this is not an alias). The alias is the object in autograd_tensor_symbol, you need another level of indirection to get the actual forward level alias.
} ccv_nnc_tensor_ref_t;

typedef struct {
	int c; // The start non-accumulated version.
	ccv_array_t* ref_version; // tensor ref point to the reverse tensor symbol.
} ccv_nnc_autograd_tensor_version_t;

typedef struct {
	int d;
	int alias_ref;
} ccv_nnc_sum_variable_t;

// This method tries to figure out if a set of aliases can cover the whole tensor dim.
// This is not a precise implementation though. The requirement is to answer this question
// with a given memory constraint, therefore, only allow up to 65536 different tensor locations.
// If you have more than that, it will assume that it doesn't have fully assigned aliases,
// and will return 0.

// Return 1 if inserted successfully.
static inline int _ccv_nnc_try_mix(int* const md, const int ins, const int c)
{
	if (!c)
	{
		md[0] = ins;
		return 1;
	}
	int ll = 0, uu = c - 1;
	int mm;
	do {
		mm = ll + ((uu - ll) >> 1);
		if (ins == md[mm])
			return 0;
		else if (ins < md[mm])
			uu = mm - 1;
		else if (ins > md[mm])
			ll = mm + 1;
	} while (ll <= uu);
	if (ll < c)
		memmove(md + ll + 1, md + ll, sizeof(int) * (c - ll));
	md[ll] = ins;
	return 1;
}

static inline int _ccv_nnc_mix_idx(const int* const md, const int ins, const int c)
{
	if (c <= 1)
		return 0;
	int ll = 0, uu = c - 1;
	int mm;
	do {
		mm = ll + ((uu - ll) >> 1);
		if (ins == md[mm])
			return mm;
		else if (ins < md[mm])
			uu = mm - 1;
		else if (ins > md[mm])
			ll = mm + 1;
	} while (ll <= uu);
	assert(0 && "Shouldn't reach here");
	return -1;
}

static inline void _ccv_nnc_try_set_pix_0(const int* const ofs, const int* const dim, const int* const tensor_dim, int* const* const scmd, const int* const cube_dim, const int* const cube_step, uint8_t* const cube, int offset)
{
	const int s = (ofs[0] == 0) ? 0 : _ccv_nnc_mix_idx(scmd[0], ofs[0], cube_dim[0]) + 1;
	const int d = ((ofs[0] + dim[0] == tensor_dim[0]) ? cube_dim[0] : _ccv_nnc_mix_idx(scmd[0], ofs[0] + ccv_max(1, dim[0]), cube_dim[0])) + 1;
	assert(s >= 0 && d > s);
	int i;
	for (i = s; i < d; i++)
		// Fill this pix. I can make this faster by loop through full ones (divided by 8), but too lazy.
		cube[(offset + i) >> 3] |= (1 << ((offset + i) & 0x7));
}

static inline void _ccv_nnc_try_set_pix_1(const int* const ofs, const int* const dim, const int* const tensor_dim, int* const* const scmd, const int* const cube_dim, const int* const cube_step, uint8_t* const cube, int offset)
{
	const int s0 = (ofs[0] == 0) ? 0 : _ccv_nnc_mix_idx(scmd[0], ofs[0], cube_dim[0]) + 1;
	const int d0 = ((ofs[0] + dim[0] == tensor_dim[0]) ? cube_dim[0] : _ccv_nnc_mix_idx(scmd[0], ofs[0] + ccv_max(1, dim[0]), cube_dim[0])) + 1;
	assert(s0 >= 0 && d0 > s0);
	const int s1 = (ofs[1] == 0) ? 0 : _ccv_nnc_mix_idx(scmd[1], ofs[1], cube_dim[1]) + 1;
	const int d1 = ((ofs[1] + dim[1] == tensor_dim[1]) ? cube_dim[1] : _ccv_nnc_mix_idx(scmd[1], ofs[1] + ccv_max(1, dim[1]), cube_dim[1])) + 1;
	assert(s1 >= 0 && d1 > s1);
	int i, j;
	const int step1 = cube_step[1];
	if (step1 == d0 - s0)
	{
		// Faster one, we can simply loop through.
		const int len = d1 + (d1 - s1) * step1;
		for (i = s1; i < len; i++)
			cube[(offset + i) >> 3] |= (1 << ((offset + i) & 0x7));
	} else {
		// There are gaps, slow one.
		for (i = s1; i < d1; i++, offset += step1)
			for (j = s0; j < d0; j++)
				cube[(offset + j) >> 3] |= (1 << ((offset + j) & 0x7));
	}
}

static inline void _ccv_nnc_try_set_pix(const int* const ofs, const int* const dim, const int* const tensor_dim, int* const* const scmd, const int* const cube_dim, const int* const cube_step, uint8_t* const cube, int offset, const int dim_idx)
{
	switch (dim_idx)
	{
		case 1:
			_ccv_nnc_try_set_pix_1(ofs, dim, tensor_dim, scmd, cube_dim, cube_step, cube, offset);
			return;
		case 0:
			_ccv_nnc_try_set_pix_0(ofs, dim, tensor_dim, scmd, cube_dim, cube_step, cube, offset);
			return;
	}
	int i;
	const int s = (ofs[dim_idx] == 0) ? 0 : _ccv_nnc_mix_idx(scmd[dim_idx], ofs[dim_idx], cube_dim[dim_idx]) + 1;
	const int d = ((ofs[dim_idx] + dim[dim_idx] == tensor_dim[dim_idx]) ? cube_dim[dim_idx] : _ccv_nnc_mix_idx(scmd[dim_idx], ofs[dim_idx] + ccv_max(1, dim[dim_idx]), cube_dim[dim_idx])) + 1;
	assert(s >= 0 && d > s);
	for (i = s; i < d; i++)
		_ccv_nnc_try_set_pix(ofs, dim, tensor_dim, scmd, cube_dim, cube_step, cube, offset + i * cube_step[dim_idx], dim_idx - 1);
}

static int _ccv_nnc_tensor_ref_fully_assigned_with_aliases(const ccv_nnc_tensor_ref_t* const tensor_ref, const ccv_array_t* const autograd_tensor_symbol, const ccv_nnc_tensor_symbol_info_t* const tensor_symbol_info)
{
	// Only work with tensor_ref of aliases.
	assert(tensor_ref->alias_registry);
	const ccv_nnc_autograd_tensor_symbol_t* autograd = (ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, tensor_ref->d);
	assert(tensor_symbol_info[autograd->d].alias_ref == 0);
	const int* tensor_dim = tensor_symbol_info[autograd->d].info.dim;
	int i, j;
	for (i = 0; i < tensor_ref->alias_registry->rnum; i++)
	{
		const int d = *(int*)ccv_array_get(tensor_ref->alias_registry, i);
		assert(d < autograd_tensor_symbol->rnum);
		const ccv_nnc_autograd_tensor_symbol_t* autograd = (ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, d);
		assert(tensor_symbol_info[autograd->d].alias_ref);
		const int* inc = tensor_symbol_info[autograd->d].inc;
		if (memcmp(inc, tensor_dim, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC) != 0)
			return 0;
	}
	/* We need a solid cube (potentially hyper dimensional) to compute if there are overlaps.
	 * To make this cube as small as possible, we need to map the actual tensor dimension
	 * (therefore, we don't actually allocate the whole tensor to compute overlaps) to a smaller
	 * cube given the ofs and dim size of its aliases.
	 *
	 * The following code generated the dimension mapping (using scratch space) with binary search + insertion
	 * and then we fill the cube with a given tensor alias's dimensional information (ofs, dim).
	 * Afterwards, we simply need to check if the cube is totally filled up to know if this tensor
	 * is fully assigned with its aliases (if that is the case, we can skip zeroing for this tensor).
	 *
	 * There are several restrictions though to make this faster: 1). I cannot handle any cube that all side
	 * lengths combined larger than 1023 (scm only have 1024 scratch space). 2). I cannot handle any cube
	 * that the total volume is larger than 2048 * 8 (I only allocate 2K on stack for this).
	 * */
	int scm[1024]; // Having 1024 int scratch space for mapping dimensions. (Or sparse coordinate mapping).
	int cube_dim[CCV_NNC_MAX_DIM_ALLOC] = {0}; // Mapping dimension size.
	int cube_size = 1;
	int* scmptr = scm;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && tensor_dim[i]; i++)
	{
		int head = 0, tail = 0; // Note that we touched both the head and tail (otherwise this dimension is not fully covered).
		int len = 0;
		for (j = 0; j < tensor_ref->alias_registry->rnum; j++)
		{
			const int d = *(int*)ccv_array_get(tensor_ref->alias_registry, j);
			assert(d < autograd_tensor_symbol->rnum);
			const ccv_nnc_autograd_tensor_symbol_t* autograd = (ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, d);
			assert(tensor_symbol_info[autograd->d].alias_ref);
			const int* ofs = tensor_symbol_info[autograd->d].ofs;
			const int* dim = tensor_symbol_info[autograd->d].info.dim;
			head = head || (ofs[i] == 0);
			tail = tail || (ofs[i] + ccv_max(1, dim[i]) == tensor_dim[i]);
			if (ofs[i] != 0)
				len += _ccv_nnc_try_mix(scmptr, ofs[i], len);
			if (scmptr - scm + len >= 1024) // Cannot handle that much, abort.
				return 0;
			if (ofs[i] + ccv_max(1, dim[i]) < tensor_dim[i])
				len += _ccv_nnc_try_mix(scmptr, ofs[i] + ccv_max(1, dim[i]), len);
			if (scmptr - scm + len >= 1024) // Cannot handle that much, abort.
				return 0;
		}
		if (!head || !tail)
			return 0;
		cube_size *= (len + 1);
		cube_dim[i] = len;
		scmptr += len; // Moving to next level.
	}
	// The cube map is too large, cannot do the computation, assume it is not fully assigned.
	if (cube_size > 2048 * 8)
		return 0;
	// binary map to see if it fills up.
	uint8_t* cube = (uint8_t*)alloca(sizeof(uint8_t) * ((cube_size + 7) >> 3));
	memset(cube, 0, sizeof(uint8_t) * ((cube_size + 7) >> 3));
	int* scmd[CCV_NNC_MAX_DIM_ALLOC] = {0}; // Sparse coordinate map at dimension x.
	int cube_step[CCV_NNC_MAX_DIM_ALLOC] = {0};
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && tensor_dim[i]; i++)
	{
		cube_step[i] = (i > 0) ? cube_step[i - 1] * (cube_dim[i - 1] + 1) : 1;
		scmd[i] = (i > 0) ? scmd[i - 1] + cube_dim[i - 1] : scm;
	}
	const int max_dim = i;
	for (i = 0; i < tensor_ref->alias_registry->rnum; i++)
	{
		const int d = *(int*)ccv_array_get(tensor_ref->alias_registry, i);
		assert(d < autograd_tensor_symbol->rnum);
		const ccv_nnc_autograd_tensor_symbol_t* autograd = (ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, d);
		assert(tensor_symbol_info[autograd->d].alias_ref);
		const int* ofs = tensor_symbol_info[autograd->d].ofs;
		const int* dim = tensor_symbol_info[autograd->d].info.dim;
		_ccv_nnc_try_set_pix(ofs, dim, tensor_dim, scmd, cube_dim, cube_step, cube, 0, max_dim - 1);
	}
	// Compare to see now if the binary map filled up. If it filled up, we know it is fully assigned.
	for (i = 0; i < (cube_size >> 3); i++)
		if (cube[i] < 0xff)
			return 0;
	if ((cube_size & 0x7) > 0)
	{
		// Fetch the rest.
		uint8_t r = 0;
		for (i = 0; i < (cube_size & 0x7); i++)
			r |= (1 << i);
		assert(cube[((cube_size + 7) >> 3) - 1] <= r);
		if (cube[((cube_size + 7) >> 3) - 1] < r)
			return 0;
	}
	return 1;
}

static void _ccv_nnc_graph_sum_autograd_tensor_versions(const int idx, const int d, const int exec_symbol_size, const ccv_nnc_tensor_symbol_info_t* const tensor_symbol_info, ccv_nnc_autograd_tensor_version_t* const tensor_ver, ccv_nnc_graph_autograd_exec_t* const autograd_exec, ccv_array_t* const autograd_tensor_symbol, ccv_array_t* const sum_or_set_exec)
{
	int i, j;
	assert(tensor_ver->c < tensor_ver->ref_version->rnum);
	const int input_size = tensor_ver->ref_version->rnum - tensor_ver->c;
	int* inputs = (int*)ccmalloc(sizeof(int) * input_size);
	for (i = tensor_ver->c; i < tensor_ver->ref_version->rnum; i++)
		inputs[i] = ((ccv_nnc_tensor_ref_t*)ccv_array_get(tensor_ver->ref_version, i))->d;
	ccv_nnc_autograd_tensor_symbol_t tensor_sym = {0};
	tensor_sym.d = d;
	ccv_array_push(autograd_tensor_symbol, &tensor_sym);
	ccv_nnc_graph_sum_or_set_exec_t sum_exec = {
		.input_size = input_size,
		.inputs = inputs,
		.output = autograd_tensor_symbol->rnum - 1
	};
	if (idx >= 0)
	{
		sum_exec.outgoings = ccv_array_new(sizeof(int), 1, 0);
		ccv_array_push(sum_exec.outgoings, &idx);
	}
	ccv_array_push(sum_or_set_exec, &sum_exec);
	const int outgoing = exec_symbol_size + sum_or_set_exec->rnum - 1;
	for (i = tensor_ver->c; i < tensor_ver->ref_version->rnum; i++)
	{
		const ccv_nnc_tensor_ref_t* tensor_ref = (ccv_nnc_tensor_ref_t*)ccv_array_get(tensor_ver->ref_version, i);
		const int x = tensor_ref->x;
		assert(x >= 0); /* Otherwise, this is initialization tensor, which is impossible to be summed up by. */
		if (x < exec_symbol_size)
		{
			ccv_nnc_graph_autograd_exec_t* back_exec = autograd_exec + x;
			if (!back_exec->outgoings)
				back_exec->outgoings = ccv_array_new(sizeof(int), 1, 0);
			ccv_array_replace_int(back_exec->outgoings, idx, outgoing);
			// If this tensor have associated alias, we need to init it to zeros when it is allocated (we only need to set a flag here)
			// it is handled at compilation phase.
			if (tensor_ref->alias_registry &&
				// Loop over to see if this tensor is fully occupied to avoid extra zero step.
				!_ccv_nnc_tensor_ref_fully_assigned_with_aliases(tensor_ref, autograd_tensor_symbol, tensor_symbol_info))
			{
				ccv_nnc_autograd_tensor_symbol_t* tensor_sym = (ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, tensor_ref->d);
				// By having alias_registry, what this symbol represents must not by an alias.
				assert(tensor_sym->alias_ref == 0);
				tensor_sym->flags = CCV_NNC_SYM_TENSOR_INIT_ZEROS;
			}
		} else {
			// This tensor_ref cannot be referenced by aliases at all (because it is generated by the sum operation, therefore, it is presented as a full in the first place).
			assert(!tensor_ref->alias_registry);
			ccv_nnc_graph_sum_or_set_exec_t* sum_or_set = (ccv_nnc_graph_sum_or_set_exec_t*)ccv_array_get(sum_or_set_exec, x - exec_symbol_size);
			ccv_array_replace_int(sum_or_set->outgoings, idx, outgoing);
		}
		if (tensor_ref->exec_registry)
			for (j = 0; j < tensor_ref->exec_registry->rnum; j++)
			{
				const int x = *(int*)ccv_array_get(tensor_ref->exec_registry, j);
				assert(x >= 0);
				// The exec_registry can only be generated by alias registry, therefore, it cannot reference to a sum operation.
				assert(x < exec_symbol_size);
				ccv_nnc_graph_autograd_exec_t* back_exec = autograd_exec + x;
				if (!back_exec->outgoings)
					back_exec->outgoings = ccv_array_new(sizeof(int), 1, 0);
				ccv_array_replace_int(back_exec->outgoings, idx, outgoing);
			}
	}
	const ccv_nnc_tensor_ref_t tensor_ref = {
		.d = autograd_tensor_symbol->rnum - 1,
		.x = outgoing
	};
	ccv_array_push(tensor_ver->ref_version, &tensor_ref);
	/* Move the c pointer up to the latest summed result. */
	tensor_ver->c = tensor_ver->ref_version->rnum - 1;
}

static int _ccv_nnc_tensor_ref_version_involve_alias(const ccv_nnc_tensor_ref_t* const tensor_ref, const ccv_array_t* const autograd_tensor_symbol, const ccv_nnc_tensor_symbol_info_t* const tensor_symbol_info, const ccv_nnc_tensor_symbol_info_t* const alias)
{
	assert(alias->alias_ref > 0);
	// No alias_registry, must conflict (owns the whole band).
	if (!tensor_ref->alias_registry)
		return 1;
	int i, j;
	for (i = 0; i < tensor_ref->alias_registry->rnum; i++)
	{
		const int d = *(int*)ccv_array_get(tensor_ref->alias_registry, i);
		assert(d < autograd_tensor_symbol->rnum);
		ccv_nnc_autograd_tensor_symbol_t* autograd = (ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, d);
		// This must reference to an alias.
		assert(tensor_symbol_info[autograd->d].alias_ref);
		const int* inc = tensor_symbol_info[autograd->d].inc;
		// Only can compare if the inc is the same, otherwise, we can only assume it overlaps.
		if (memcmp(inc, alias->inc, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC) != 0)
			return 1;
		const int* ofs = tensor_symbol_info[autograd->d].ofs;
		const int* dim = tensor_symbol_info[autograd->d].info.dim;
		int none = 0;
		for (j = 0; j < CCV_NNC_MAX_DIM_ALLOC && dim[j] && alias->info.dim[j]; j++)
			if (ccv_min(ofs[j] + dim[j], alias->ofs[j] + alias->info.dim[j]) <= ccv_max(ofs[j], alias->ofs[j]))
			{
				none = 1;
				break;
			}
		// If it overlaps with this alias, nope.
		if (!none)
			return 1;
	}
	// All aliases referenced by this ref_version doesn't overlap with the provided one, thus, there is no conflict at all.
	return 0;
}

static int _ccv_nnc_tensor_ref_version_find_alias(const ccv_nnc_tensor_ref_t* const tensor_ref, const ccv_array_t* const autograd_tensor_symbol, const ccv_nnc_tensor_symbol_info_t* const tensor_symbol_info, const ccv_nnc_tensor_symbol_info_t* const alias)
{
	assert(alias->alias_ref > 0);
	// No alias_registry, thus, cannot find the exact matched alias.
	if (!tensor_ref->alias_registry)
		return -1;
	int i;
	for (i = 0; i < tensor_ref->alias_registry->rnum; i++)
	{
		const int d = *(int*)ccv_array_get(tensor_ref->alias_registry, i);
		assert(d < autograd_tensor_symbol->rnum);
		ccv_nnc_autograd_tensor_symbol_t* autograd = (ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, d);
		// This must reference to an alias.
		assert(tensor_symbol_info[autograd->d].alias_ref);
		const int* inc = tensor_symbol_info[autograd->d].inc;
		const int* ofs = tensor_symbol_info[autograd->d].ofs;
		const int* dim = tensor_symbol_info[autograd->d].info.dim;
		// If everything matches, this is the required alias.
		if (memcmp(inc, alias->inc, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC) == 0 &&
			memcmp(ofs, alias->ofs, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC) == 0 &&
			memcmp(dim, alias->info.dim, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC) == 0)
			return d;
	}
	return -1;
}

static int _ccv_nnc_tensor_ref_version_has_this_alias_exclusively(const ccv_nnc_tensor_ref_t* tensor_ref, const ccv_array_t* autograd_tensor_symbol, const ccv_nnc_tensor_symbol_info_t* tensor_symbol_info, const ccv_nnc_tensor_symbol_info_t* alias)
{
	assert(alias->alias_ref > 0);
	// No alias_registry, thus, cannot find the exact matched alias.
	if (!tensor_ref->alias_registry)
		return 0;
	int i;
	for (i = 0; i < tensor_ref->alias_registry->rnum; i++)
	{
		const int d = *(int*)ccv_array_get(tensor_ref->alias_registry, i);
		assert(d < autograd_tensor_symbol->rnum);
		ccv_nnc_autograd_tensor_symbol_t* autograd = (ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, d);
		// This must reference to an alias.
		assert(tensor_symbol_info[autograd->d].alias_ref);
		const int* inc = tensor_symbol_info[autograd->d].inc;
		const int* ofs = tensor_symbol_info[autograd->d].ofs;
		const int* dim = tensor_symbol_info[autograd->d].info.dim;
		if (memcmp(inc, alias->inc, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC) != 0 ||
			memcmp(ofs, alias->ofs, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC) != 0 ||
			memcmp(dim, alias->info.dim, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC) != 0)
			return 0;
	}
	// If everything matches for every alias in registry, we can use any of the alias directly.
	return 1;
}

static int _ccv_nnc_graph_sum_autograd_tensor_versions_alias(const int idx, const int d, const ccv_nnc_tensor_symbol_info_t* const tensor_symbol_info, const int exec_symbol_size, const ccv_nnc_tensor_symbol_info_t* const alias, ccv_nnc_autograd_tensor_version_t* const tensor_ver, ccv_nnc_graph_autograd_exec_t* const autograd_exec, ccv_array_t* const autograd_tensor_symbol, ccv_array_t* const sum_or_set_exec)
{
	assert(tensor_ver->c < tensor_ver->ref_version->rnum);
	int i, j = 0;
	struct {
		int k;
		int i;
	} kd[tensor_ver->ref_version->rnum - tensor_ver->c];
	for (i = tensor_ver->c; i < tensor_ver->ref_version->rnum; i++)
	{
		ccv_nnc_tensor_ref_t* tensor_ref = (ccv_nnc_tensor_ref_t*)ccv_array_get(tensor_ver->ref_version, i);
		const int k = _ccv_nnc_tensor_ref_version_find_alias(tensor_ref, autograd_tensor_symbol, tensor_symbol_info, alias);
		if (k >= 0)
			kd[j++] = (typeof(kd[0])){
				.k = k, .i = i
			};
		else if (_ccv_nnc_tensor_ref_version_involve_alias(tensor_ref, autograd_tensor_symbol, tensor_symbol_info, alias))
			kd[j++] = (typeof(kd[0])) {
				.k = -1, .i = i // It has dependency to the original tensor (non-alias) now, label this with highest bit.
			};
	}
	// Can only find one. This is the easy case, we can simply return that symbol (or its alias).
	if (j == 1)
	{
		if (kd[0].k >= 0)
			return kd[0].k; // Only can find one alias, that is the one.
		// Otherwise, need to create a new alias.
		ccv_nnc_tensor_ref_t* tensor_ref = (ccv_nnc_tensor_ref_t*)ccv_array_get(tensor_ver->ref_version, kd[0].i);
		ccv_nnc_autograd_tensor_symbol_t* ref = (ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, tensor_ref->d);
		// Since we create new alias, we need to set the referenced one to be allocated with 0s.
		if (ref->alias_ref) // If this is an alias, it has to be zero initialized.
		{
			ref = (ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, ref->alias_ref - 1);
			assert(ref->alias_ref == 0); // This is original.
			ref->flags = CCV_NNC_SYM_TENSOR_INIT_ZEROS;
		} else if (tensor_ref->alias_registry && // Otherwise, to see if this symbol is fully occupied.
				// Loop over to see if this tensor is fully occupied to avoid extra zero step.
				!_ccv_nnc_tensor_ref_fully_assigned_with_aliases(tensor_ref, autograd_tensor_symbol, tensor_symbol_info)) {
			ref->flags = CCV_NNC_SYM_TENSOR_INIT_ZEROS;
		}
		ccv_nnc_autograd_tensor_symbol_t tensor_sym = {0};
		tensor_sym.d = d;
		tensor_sym.alias_ref = tensor_ref->d + 1;
		ccv_array_push(autograd_tensor_symbol, &tensor_sym);
		const int ad = autograd_tensor_symbol->rnum - 1;
		if (tensor_ref->alias_registry) // Only push this when it has an alias registry (otherwise it already conflict with everyone).
			ccv_array_push(tensor_ref->alias_registry, &ad);
		// The newly inserted tensor symbol.
		return ad;
	}
	// Otherwise, we need to create the sum operation out of these.
	const int input_size = j;
	int has_this_alias_exclusively = 1;
	int* inputs = input_size > 0 ? (int*)ccmalloc(sizeof(int) * input_size) : 0;
	for (i = 0; i < input_size; i++)
	{
		ccv_nnc_tensor_ref_t* tensor_ref = (ccv_nnc_tensor_ref_t*)ccv_array_get(tensor_ver->ref_version, kd[i].i);
		// Can take a fast path if every ref involved has the same alias, our sum operation can be faster (using alias directly).
		if (has_this_alias_exclusively && kd[i].k >= 0 && _ccv_nnc_tensor_ref_version_has_this_alias_exclusively(tensor_ref, autograd_tensor_symbol, tensor_symbol_info, alias))
			inputs[i] = *(int*)ccv_array_get(tensor_ref->alias_registry, 0); // Assigning the alias.
		else {
			if (has_this_alias_exclusively)
			{
				has_this_alias_exclusively = 0;
				for (j = 0; j < i; j++)
					inputs[j] = ((ccv_nnc_tensor_ref_t*)ccv_array_get(tensor_ver->ref_version, kd[i].i))->d;
			}
			inputs[i] = tensor_ref->d;
		}
	}
	ccv_nnc_autograd_tensor_symbol_t tensor_sym = {0};
	tensor_sym.d = alias->alias_ref - 1;
	const int tensor_ref_d = autograd_tensor_symbol->rnum - 1;
	ccv_array_push(autograd_tensor_symbol, &tensor_sym);
	tensor_sym.d = d;
	tensor_sym.alias_ref = tensor_ref_d + 1;
	ccv_array_push(autograd_tensor_symbol, &tensor_sym);
	const int ad = autograd_tensor_symbol->rnum - 1;
	ccv_nnc_graph_sum_or_set_exec_t sum_exec = {
		.input_size = input_size,
		.inputs = inputs,
		.output = ad
	};
	if (idx >= 0)
	{
		sum_exec.outgoings = ccv_array_new(sizeof(int), 1, 0);
		ccv_array_push(sum_exec.outgoings, &idx);
	}
	ccv_array_push(sum_or_set_exec, &sum_exec);
	const int outgoing = exec_symbol_size + sum_or_set_exec->rnum - 1;
	int no_alias_registry = 0;
	for (i = 0; i < input_size; i++)
	{
		ccv_nnc_tensor_ref_t* tensor_ref = (ccv_nnc_tensor_ref_t*)ccv_array_get(tensor_ver->ref_version, kd[i].i);
		if (!has_this_alias_exclusively)
		{
			// If the sum operation is not operating on one alias. I need to zero this tensor out when it is first
			// allocated (see discussions around the flags I use).
			ccv_nnc_autograd_tensor_symbol_t* tensor_sym = (ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, tensor_ref->d);
			if (tensor_sym->alias_ref)
			{
				// Find the original tensor_sym and set its flags (I prefer to set flags on its original).
				ccv_nnc_autograd_tensor_symbol_t* ref = (ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, tensor_sym->alias_ref - 1);
				assert(ref->alias_ref == 0); // This is original.
				ref->flags = CCV_NNC_SYM_TENSOR_INIT_ZEROS;
			} else if (tensor_ref->alias_registry && // Otherwise, to see if this symbol is fully occupied.
					// Loop over to see if this tensor is fully occupied to avoid extra zero step.
					!_ccv_nnc_tensor_ref_fully_assigned_with_aliases(tensor_ref, autograd_tensor_symbol, tensor_symbol_info)) {
				tensor_sym->flags = CCV_NNC_SYM_TENSOR_INIT_ZEROS;
			}
		}
		// Check to see if any of these tensors doesn't have alias.
		no_alias_registry |= (!tensor_ref->alias_registry);
		const int x = tensor_ref->x;
		assert(x >= 0); /* Otherwise, this is initialization tensor, which is impossible to be summed up by. */
		if (x < exec_symbol_size)
		{
			ccv_nnc_graph_autograd_exec_t* back_exec = autograd_exec + x;
			if (!back_exec->outgoings)
				back_exec->outgoings = ccv_array_new(sizeof(int), 1, 0);
			ccv_array_push(back_exec->outgoings, &outgoing);
		} else {
			ccv_nnc_graph_sum_or_set_exec_t* sum_or_set = (ccv_nnc_graph_sum_or_set_exec_t*)ccv_array_get(sum_or_set_exec, x - exec_symbol_size);
			ccv_array_push(sum_or_set->outgoings, &outgoing);
		}
		if (tensor_ref->exec_registry)
			for (j = 0; j < tensor_ref->exec_registry->rnum; j++)
			{
				const int x = *(int*)ccv_array_get(tensor_ref->exec_registry, j);
				assert(x >= 0); /* Otherwise, this is initialization tensor, which is impossible to be summed up by. */
				assert(x < exec_symbol_size); // exec_registry is only used by alias_registry, it simply cannot reference to a sum operation.
				ccv_nnc_graph_autograd_exec_t* back_exec = autograd_exec + x;
				if (!back_exec->outgoings)
					back_exec->outgoings = ccv_array_new(sizeof(int), 1, 0);
				ccv_array_push(back_exec->outgoings, &outgoing);
			}
	}
	const ccv_nnc_tensor_ref_t tensor_ref = {
		.d = tensor_ref_d,
		.x = outgoing,
		.exec_registry = 0, // I don't need to take execution dependencies because this tensor is generated by sum, therefore, we already take that dependency.
		.alias_registry = !no_alias_registry || has_this_alias_exclusively ? ccv_array_new(sizeof(int), 1, 0) : 0
	};
	// If there is no alias registry, then we take the whole tensor ref as one.
	if (!no_alias_registry || has_this_alias_exclusively)
	{
		// If this tensor ref contains multiple different types of alias, have to add them together (otherwise
		// the computation for if there is an empty slot in this tensor ref is not correct without all the
		// occupancy availability information).
		if (!has_this_alias_exclusively)
			for (i = 0; i < input_size; i++)
			{
				ccv_nnc_tensor_ref_t* ref = (ccv_nnc_tensor_ref_t*)ccv_array_get(tensor_ver->ref_version, kd[i].i);
				assert(ref->alias_registry);
				// It may get duplicates. But whatever, won't matter the computation.
				for (j = 0; j < ref->alias_registry->rnum; j++)
					ccv_array_push(tensor_ref.alias_registry, ccv_array_get(ref->alias_registry, j));
			}
		ccv_array_push(tensor_ref.alias_registry, &ad);
	}
	assert(input_size <= tensor_ver->ref_version->rnum - tensor_ver->c);
	ccv_nnc_tensor_ref_t x;
	for (i = 0; i < input_size; i++)
		// If the current one (i + tensor_ver->c) is smaller than the one referenced to, exchange.
		if (kd[i].i > i + tensor_ver->c)
			CCV_SWAP(*(ccv_nnc_tensor_ref_t*)ccv_array_get(tensor_ver->ref_version, i + tensor_ver->c), *(ccv_nnc_tensor_ref_t*)ccv_array_get(tensor_ver->ref_version, kd[i].i), x);
	ccv_array_push(tensor_ver->ref_version, &tensor_ref);
	// We've consumed input_size tensor refs, now move c up to the pointer of non-consumed tensors.
	tensor_ver->c += input_size;
	return ad;
}

void ccv_nnc_symbolic_graph_backward(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t* const sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* const destinations, const int destination_size, const ccv_nnc_tensor_symbol_t* const f_symbols, const int f_symbol_size, const ccv_nnc_tensor_symbol_t* const wrt_symbols, const int wrt_symbol_size)
{
	// First, fill all the "auto" holes.
	// This is the symbol table that with "auto" info filled up.
	const int tensor_symbol_size = graph->tensor_symbol_info->rnum;
	const int exec_symbol_size = graph->exec_symbol_info->rnum;
	assert(tensor_symbol_size > 0);
	assert(exec_symbol_size > 0);
	ccv_nnc_tensor_symbol_info_t* tensor_symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccmalloc(sizeof(ccv_nnc_tensor_symbol_info_t) * tensor_symbol_size);
	ccv_nnc_graph_exec_symbol_info_t* exec_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccmalloc(sizeof(ccv_nnc_graph_exec_symbol_info_t) * exec_symbol_size);
	ccv_nnc_symbolic_graph_symbol_infer(graph, sources, source_size, destinations, destination_size, 0, 0, tensor_symbol_info, exec_symbol_info);
	// Now, for each one of these, find a reverse graph.
	ccv_nnc_graph_backward_info_t* backward_info = (ccv_nnc_graph_backward_info_t*)cccalloc(exec_symbol_size, sizeof(ccv_nnc_graph_backward_info_t));
	int i, j;
	for (i = 0; i < f_symbol_size; i++)
	{
		// f symbols cannot be alias.
		assert(!tensor_symbol_info[f_symbols[i].d].alias_ref);
	}
#define visitor(node, idx, ...) \
	do { \
		assert(ccv_nnc_cmd_is_forward(node->cmd) || node->cmd.cmd == CCV_NNC_NOOP); \
		if (node->outgoings) \
			for (i = 0; i < node->outgoings->rnum; i++) \
			{ \
				int d = *(int*)ccv_array_get(node->outgoings, i); \
				if (backward_info[d].outgoings == 0) \
					backward_info[d].outgoings = ccv_array_new(sizeof(int32_t), 1, 0); \
				ccv_array_push(backward_info[d].outgoings, &idx); \
			} \
	} while (0)
	CCV_NNC_GRAPH_VISIT(graph, exec_symbol_info, exec_symbol_size, sources, source_size, destinations, destination_size, visitor);
#undef visitor
	// Find the f_symbols, and tag its flows.
#define visitor(node, idx, ...) \
	do { \
		int f = node->f_wrt & 0x1; \
		for (i = 0; i < exec_symbol_info[idx].output_size && !f; i++) \
		{ \
			int d = exec_symbol_info[idx].outputs[i]; \
			if (tensor_symbol_info[d].alias_ref) \
				d = tensor_symbol_info[d].alias_ref - 1; \
			for (j = 0; j < f_symbol_size && !f; j++) \
				if (d == f_symbols[j].d) \
					f = 1; \
		} \
		if (f) \
		{ \
			node->f_wrt |= f; \
			if (node->outgoings) \
				for (i = 0; i < node->outgoings->rnum; i++) \
				{ \
					int d = *(int*)ccv_array_get(node->outgoings, i); \
					backward_info[d].f_wrt |= f; \
				} \
		} \
	} while (0)
	CCV_NNC_GRAPH_VISIT(graph, backward_info, exec_symbol_size, destinations, destination_size, sources, source_size, visitor);
#undef visitor
	// Find the wrt_symbols, and tag its flows.
#define visitor(node, idx, ...) \
	do { \
		int wrt = backward_info[idx].f_wrt & 0x2; \
		for (i = 0; i < node->input_size && !wrt; i++) \
			for (j = 0; j < wrt_symbol_size && !wrt; j++) \
				if (node->inputs[i] == wrt_symbols[j].d) \
					wrt = 0x2; \
		if (wrt) \
		{ \
			backward_info[idx].f_wrt |= wrt; \
			if (node->outgoings) \
				for (i = 0; i < node->outgoings->rnum; i++) \
				{ \
					int d = *(int*)ccv_array_get(node->outgoings, i); \
					backward_info[d].f_wrt |= wrt; \
				} \
		} \
	} while (0)
	CCV_NNC_GRAPH_VISIT(graph, exec_symbol_info, exec_symbol_size, sources, source_size, destinations, destination_size, visitor);
#undef visitor
	// Also mark only the output bits that we use.
	for (i = 0; i < exec_symbol_size; i++)
	{
		backward_info[i].input_bitmask_size = ((exec_symbol_info[i].output_size * 2 + exec_symbol_info[i].input_size + 63) >> 6);
		backward_info[i].output_bitmask_size = ((exec_symbol_info[i].input_size + 63) >> 6);
		// Allocate input / output bitmasks
		if (backward_info[i].input_bitmask_size + backward_info[i].output_bitmask_size > 0)
		{
			backward_info[i].input_bitmasks = (uint64_t*)cccalloc(backward_info[i].input_bitmask_size + backward_info[i].output_bitmask_size, sizeof(uint64_t));
			if (backward_info[i].output_bitmask_size)
				backward_info[i].output_bitmasks = backward_info[i].input_bitmasks + backward_info[i].input_bitmask_size;
		}
	}
	enum {
		WRT_SYMBOL_USE = 1,
		F_SYMBOL_USE = 2
	};
	uint8_t* used = (uint8_t*)cccalloc(tensor_symbol_size, sizeof(uint8_t));
	// First, all f_symbols and wrt_symbols are used.
	for (i = 0; i < f_symbol_size; i++)
		used[tensor_symbol_info[f_symbols[i].d].alias_ref ? tensor_symbol_info[f_symbols[i].d].alias_ref - 1 : f_symbols[i].d] |= F_SYMBOL_USE;
	for (i = 0; i < wrt_symbol_size; i++)
		used[tensor_symbol_info[wrt_symbols[i].d].alias_ref ? tensor_symbol_info[wrt_symbols[i].d].alias_ref - 1 : wrt_symbols[i].d] |= WRT_SYMBOL_USE;
#define visitor(_, idx, ...) \
	do { \
		ccv_nnc_graph_backward_info_t* node = backward_info + idx; \
		/* Only interested in the ones on the f / wrt flow */ \
		if (node->f_wrt == 0x3) \
		{ \
			const ccv_nnc_graph_exec_symbol_info_t* forw_exec = exec_symbol_info + idx; \
			ccv_nnc_cmd_t cmd = forw_exec->cmd; \
			if (cmd.cmd != CCV_NNC_NOOP) \
				cmd.cmd += 1; /* Backward command is the one after forward command. */ \
			assert(ccv_nnc_cmd_is_backward(cmd) || cmd.cmd == CCV_NNC_NOOP); \
			for (i = 0; i < forw_exec->output_size * 2 + forw_exec->input_size; i++) \
				node->input_bitmasks[i >> 6] |= ((uint64_t)1 << i); \
			for (i = 0; i < forw_exec->input_size; i++) \
				node->output_bitmasks[i >> 6] |= ((uint64_t)1 << i); \
			int maybe_noop = 1; \
			for (i = 0; i < forw_exec->input_size; i++) \
				/* See if it is used. */ \
				if (used[tensor_symbol_info[forw_exec->inputs[i]].alias_ref ? tensor_symbol_info[forw_exec->inputs[i]].alias_ref - 1 : forw_exec->inputs[i]] & WRT_SYMBOL_USE) \
				{ \
					maybe_noop = 0; \
					break; \
				} \
			if (maybe_noop) \
			{ \
				node->output_bitmasks = 0; \
				node->output_bitmask_size = 0; \
			/* If this is ok, try something else. Otherwise, this is it, no trick. */ \
			} else if (ccv_nnc_cmd_bitmask(cmd, node->input_bitmasks, node->input_bitmask_size, node->output_bitmasks, node->output_bitmask_size)) { \
				int flag; /* Only continue if it changed */ \
				do { \
					flag = 0; \
					/* Check if the output first */ \
					for (i = 0; i < forw_exec->input_size; i++) \
						/* Only try to eliminate the one that is not used. */ \
						if (!(used[tensor_symbol_info[forw_exec->inputs[i]].alias_ref ? tensor_symbol_info[forw_exec->inputs[i]].alias_ref - 1 : forw_exec->inputs[i]] & WRT_SYMBOL_USE) && \
							(node->output_bitmasks[i >> 6] & ((uint64_t)1 << i))) \
						{ \
							node->output_bitmasks[i >> 6] &= ~((uint64_t)1 << i); \
							/* If it worked, mark it as flagged. */ \
							if (ccv_nnc_cmd_bitmask(cmd, node->input_bitmasks, node->input_bitmask_size, node->output_bitmasks, node->output_bitmask_size)) \
								flag = 1; \
							else /* Refit this with the bit back again. */ \
								node->output_bitmasks[i >> 6] |= ((uint64_t)1 << i); \
						} \
					for (i = 0; i < forw_exec->output_size * 2 + forw_exec->input_size; i++) \
						if ((i >= forw_exec->output_size || \
							 !(used[tensor_symbol_info[forw_exec->outputs[i]].alias_ref ? tensor_symbol_info[forw_exec->outputs[i]].alias_ref - 1 : forw_exec->outputs[i]] & F_SYMBOL_USE)) && \
							node->input_bitmasks[i >> 6] & ((uint64_t)1 << i)) \
						{ /* Try to eliminate one of the input. */ \
							node->input_bitmasks[i >> 6] &= ~((uint64_t)1 << i); \
							/* If it worked, mark it as flagged. */ \
							if (ccv_nnc_cmd_bitmask(cmd, node->input_bitmasks, node->input_bitmask_size, node->output_bitmasks, node->output_bitmask_size)) \
								flag = 1; \
							else /* Refit this with the bit back again. */ \
								node->input_bitmasks[i >> 6] |= ((uint64_t)1 << i); \
						} \
				} while (flag); \
				for (i = 0; i < forw_exec->output_size; i++) \
					if (node->input_bitmasks[i >> 6] & ((uint64_t)1 << i)) \
						/* Mark it as used. */ \
						used[tensor_symbol_info[forw_exec->outputs[i]].alias_ref ? tensor_symbol_info[forw_exec->outputs[i]].alias_ref - 1 : forw_exec->outputs[i]] |= WRT_SYMBOL_USE; \
			} \
		} \
	} while (0)
	CCV_NNC_GRAPH_VISIT(graph, exec_symbol_info, exec_symbol_size, sources, source_size, destinations, destination_size, visitor);
#undef visitor
	ccfree(used);
	// Now, only the flow from f_symbols back to wrt_symbols are interested to us.
	// Visit the graph in reverse order, build the AD nodes.
	ccv_nnc_graph_autograd_exec_t* autograd_exec = (ccv_nnc_graph_autograd_exec_t*)cccalloc(exec_symbol_size, sizeof(ccv_nnc_graph_autograd_exec_t));
	for (i = 0; i < exec_symbol_size; i++)
		if (backward_info[i].f_wrt == 0x3 && backward_info[i].outgoings)
		{
			// Copy over the outgoing bits.
			autograd_exec[i].outgoings = ccv_array_new(sizeof(int), backward_info[i].outgoings->rnum, 0);
			for (j = 0; j < backward_info[i].outgoings->rnum; j++)
			{
				const int d = *(int*)ccv_array_get(backward_info[i].outgoings, j);
				// Only push the outgoing node if it is in the f_wrt path.
				if (backward_info[d].f_wrt == 0x3)
					ccv_array_push(autograd_exec[i].outgoings, &d);
			}
		}
	ccv_nnc_autograd_tensor_version_t* autograd_tensor_version = (ccv_nnc_autograd_tensor_version_t*)cccalloc(tensor_symbol_size, sizeof(ccv_nnc_autograd_tensor_version_t));
	ccv_array_t* autograd_tensor_symbol = ccv_array_new(sizeof(ccv_nnc_autograd_tensor_symbol_t), tensor_symbol_size, 0);
	ccv_array_t* sum_or_set_exec = ccv_array_new(sizeof(ccv_nnc_graph_sum_or_set_exec_t), 0, 0);
#define visitor(node, idx, ...) \
	do { \
		/* This is required by both f flow and wrt flow, therefore, an interest to us */ \
		if (node->f_wrt == 0x3) \
		{ \
			const ccv_nnc_graph_exec_symbol_info_t* forw_exec = exec_symbol_info + idx; \
			const ccv_nnc_graph_backward_info_t* back_info = backward_info + idx; \
			ccv_nnc_graph_autograd_exec_t* back_exec = autograd_exec + idx; \
			back_exec->cmd = forw_exec->cmd; \
			if (back_exec->cmd.cmd != CCV_NNC_NOOP) \
				back_exec->cmd.cmd += 1; /* Backward command is the one after forward command. */ \
			assert(ccv_nnc_cmd_is_backward(back_exec->cmd) || back_exec->cmd.cmd == CCV_NNC_NOOP); \
			if (!back_info->output_bitmasks) /* This has no output, can be a noop. */ \
				back_exec->cmd.cmd = CCV_NNC_NOOP; \
			else { \
				back_exec->output_size = forw_exec->input_size; \
				back_exec->input_size = forw_exec->output_size; \
				back_exec->inputs = ccmalloc(sizeof(int) * (back_exec->input_size + back_exec->output_size)); \
				back_exec->outputs = back_exec->inputs + back_exec->input_size; \
				/* Need to compute input before we compute output */ \
				for (i = 0; i < forw_exec->output_size; i++) \
				{ \
					/* If we can skip this input, do that. */ \
					if (!(back_info->input_bitmasks[i >> 6] & ((uint64_t)1 << i))) \
						continue; \
					const int d = forw_exec->outputs[i]; \
					const int alias_ref = tensor_symbol_info[d].alias_ref; \
					ccv_nnc_autograd_tensor_version_t* tensor_ver = alias_ref ? autograd_tensor_version + (alias_ref - 1) : autograd_tensor_version + d; \
					/* Initialization tensor, should corresponding to f symbols */ \
					if (!tensor_ver->ref_version) \
					{ \
						ccv_nnc_autograd_tensor_symbol_t tensor_sym = {0}; \
						if (!alias_ref) \
						{ \
							tensor_sym.d = d; \
							ccv_array_push(autograd_tensor_symbol, &tensor_sym); \
							const ccv_nnc_tensor_ref_t tensor_ref = { \
								.d = autograd_tensor_symbol->rnum - 1, \
								.x = idx, \
								.alias_registry = 0 \
							}; \
							tensor_ver->ref_version = ccv_array_new(sizeof(ccv_nnc_tensor_ref_t), 1, 0); \
							ccv_array_push(tensor_ver->ref_version, &tensor_ref); \
						} else { \
							tensor_sym.d = alias_ref - 1; \
							ccv_array_push(autograd_tensor_symbol, &tensor_sym); \
							const ccv_nnc_tensor_ref_t tensor_ref = { \
								.d = autograd_tensor_symbol->rnum - 1, \
								.x = idx, \
								.alias_registry = ccv_array_new(sizeof(int), 1, 0) \
							}; \
							tensor_ver->ref_version = ccv_array_new(sizeof(ccv_nnc_tensor_ref_t), 1, 0); \
							ccv_array_push(tensor_ver->ref_version, &tensor_ref); \
							tensor_sym.d = d; /* set back */ \
							tensor_sym.alias_ref = tensor_ref.d + 1; \
							ccv_array_push(autograd_tensor_symbol, &tensor_sym); \
							const int ad = autograd_tensor_symbol->rnum - 1; \
							ccv_array_push(tensor_ref.alias_registry, &ad); \
						} \
					} \
					/* The simplest case (most common), it is not an alias. */ \
					if (!alias_ref) \
					{ \
						/* Even simpler, this only have one reference tensor, thus, pass this as input. */ \
						if (tensor_ver->c == tensor_ver->ref_version->rnum - 1) \
						{ \
							ccv_nnc_tensor_ref_t* tensor_ref = (ccv_nnc_tensor_ref_t*)ccv_array_get(tensor_ver->ref_version, tensor_ver->c); \
							/* There are alias associated with this tensor ref, zero it out when this tensor is allocated. */ \
							/* This is is required. Consider the case that we have an alias of this tensor used somehwere */ \
							/* on forward pass, when we compute backward, we have that alias computed first, however, its */ \
							/* underlying tensor is not zero initialized, and we will end up with garbage values here. */ \
							if (tensor_ref->alias_registry && \
								/* Loop over to see if this tensor is fully occupied to avoid extra zero step. */ \
								!_ccv_nnc_tensor_ref_fully_assigned_with_aliases(tensor_ref, autograd_tensor_symbol, tensor_symbol_info)) \
							{ \
								ccv_nnc_autograd_tensor_symbol_t* tensor_sym = (ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, tensor_ref->d); \
								assert(tensor_sym->alias_ref == 0); \
								tensor_sym->flags = CCV_NNC_SYM_TENSOR_INIT_ZEROS; \
							} \
							back_exec->inputs[i] = tensor_ref->d; \
						} else { \
							/* Otherwise, we need to sum them up, and then pass the summed result to the computation. */ \
							_ccv_nnc_graph_sum_autograd_tensor_versions(idx, d, exec_symbol_size, tensor_symbol_info, tensor_ver, autograd_exec, autograd_tensor_symbol, sum_or_set_exec); \
							ccv_nnc_tensor_ref_t* tensor_ref = (ccv_nnc_tensor_ref_t*)ccv_array_get(tensor_ver->ref_version, tensor_ver->c); \
							back_exec->inputs[i] = tensor_ref->d; \
						} \
					} else \
						/* If this is an alias, go through all available tensor ref versions */ \
						back_exec->inputs[i] = _ccv_nnc_graph_sum_autograd_tensor_versions_alias(idx, d, tensor_symbol_info, exec_symbol_size, tensor_symbol_info + d, tensor_ver, autograd_exec, autograd_tensor_symbol, sum_or_set_exec); \
				} \
				for (i = 0; i < forw_exec->input_size; i++) \
				{ \
					/* If we can skip this output, do that. */ \
					if (!(back_info->output_bitmasks[i >> 6] & ((uint64_t)1 << i))) \
						continue; \
					const int d = forw_exec->inputs[i]; \
					const int alias_ref = tensor_symbol_info[d].alias_ref; \
					ccv_nnc_autograd_tensor_symbol_t tensor_sym = {0}; \
					tensor_sym.d = d; \
					/* The simplest case (most common), it is not an alias. */ \
					if (!alias_ref) \
					{ \
						ccv_array_push(autograd_tensor_symbol, &tensor_sym); \
						const ccv_nnc_tensor_ref_t tensor_ref = { \
							.d = autograd_tensor_symbol->rnum - 1, \
							.x = idx, \
							.exec_registry = 0, \
							.alias_registry = 0 \
						}; \
						ccv_nnc_autograd_tensor_version_t* tensor_ver = autograd_tensor_version + d; \
						if (!tensor_ver->ref_version) \
							tensor_ver->ref_version = ccv_array_new(sizeof(ccv_nnc_tensor_ref_t), 1, 0); \
						ccv_array_push(tensor_ver->ref_version, &tensor_ref); \
						back_exec->outputs[i] = tensor_ref.d; \
					} else { \
						/* Otherwise, in case that this is an alias, we try to find the existing one (in tensor_ver),
						 * see if can meet the need (thus, for the tensor info / ofs, it fits). */ \
						ccv_nnc_autograd_tensor_version_t* tensor_ver = autograd_tensor_version + (alias_ref - 1); \
						if (!tensor_ver->ref_version) \
							tensor_ver->ref_version = ccv_array_new(sizeof(ccv_nnc_tensor_ref_t), 1, 0); \
						/* If already exists a ref version, check if any of these not-sealed tensors have free space. */ \
						int found = 0; \
						for (j = tensor_ver->c; j < tensor_ver->ref_version->rnum; j++) \
						{ \
							ccv_nnc_tensor_ref_t* tensor_ref = (ccv_nnc_tensor_ref_t*)ccv_array_get(tensor_ver->ref_version, j); \
							if (!_ccv_nnc_tensor_ref_version_involve_alias(tensor_ref, autograd_tensor_symbol, tensor_symbol_info, tensor_symbol_info + d)) \
							{ \
								tensor_sym.alias_ref = tensor_ref->d + 1; \
								ccv_array_push(autograd_tensor_symbol, &tensor_sym); \
								const int ad = autograd_tensor_symbol->rnum - 1; \
								ccv_array_push(tensor_ref->alias_registry, &ad); \
								if (!tensor_ref->exec_registry) \
									tensor_ref->exec_registry = ccv_array_new(sizeof(int), 1, 0); \
								ccv_array_push(tensor_ref->exec_registry, &idx); \
								back_exec->outputs[i] = ad; \
								found = 1; \
								break; \
							} \
						} \
						if (!found) /* Cannot find an tensor ref to insert, create one first */ \
						{ \
							tensor_sym.d = alias_ref - 1; /* Reference back to the non-alias. */ \
							ccv_array_push(autograd_tensor_symbol, &tensor_sym); \
							const ccv_nnc_tensor_ref_t tensor_ref = { \
								.d = autograd_tensor_symbol->rnum - 1, \
								.x = idx, \
								.exec_registry = 0, \
								.alias_registry = ccv_array_new(sizeof(int), 1, 0) \
							}; \
							ccv_array_push(tensor_ver->ref_version, &tensor_ref); \
							tensor_sym.d = d; /* set back */ \
							tensor_sym.alias_ref = tensor_ref.d + 1; \
							ccv_array_push(autograd_tensor_symbol, &tensor_sym); \
							const int ad = autograd_tensor_symbol->rnum - 1; \
							ccv_array_push(tensor_ref.alias_registry, &ad); \
							back_exec->outputs[i] = ad; \
						} \
					} \
				} \
			} \
		} \
	} while (0)
	CCV_NNC_GRAPH_VISIT(graph, backward_info, exec_symbol_size, destinations, destination_size, sources, source_size, visitor);
#undef visitor
	// Find all relevant wrt symbols, generate sum for them if needed.
	for (i = 0; i < wrt_symbol_size; i++)
	{
		const int d = wrt_symbols[i].d;
		ccv_nnc_autograd_tensor_version_t* tensor_ver = (!tensor_symbol_info[d].alias_ref) ? autograd_tensor_version + d : autograd_tensor_version + (tensor_symbol_info[d].alias_ref - 1);
		// If there are more than one tensor in the list, it is possible to sum them up.
		if (tensor_ver->c < tensor_ver->ref_version->rnum - 1)
			_ccv_nnc_graph_sum_autograd_tensor_versions(-1, d, exec_symbol_size, tensor_symbol_info, tensor_ver, autograd_exec, autograd_tensor_symbol, sum_or_set_exec);
		// The tensor version should have ref_version, and only one now (after sum up).
		assert(tensor_ver->c == tensor_ver->ref_version->rnum - 1);
	}
	// Generate required symbols based on the information gathered above.
	for (i = 0; i < autograd_tensor_symbol->rnum; i++)
	{
		ccv_nnc_autograd_tensor_symbol_t* symbol = (ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, i);
		assert(symbol->d >= 0);
		assert(symbol->d < tensor_symbol_size);
		ccv_nnc_tensor_symbol_info_t* forw_symbol = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, symbol->d);
		if (!symbol->alias_ref)
		{
			assert(!forw_symbol->alias_ref);
			symbol->symbol = ccv_nnc_tensor_symbol_new(graph, forw_symbol->info, 0);
			ccv_nnc_tensor_symbol_set_flags(graph, symbol->symbol, symbol->flags);
		} else {
			assert(forw_symbol->alias_ref);
			assert(symbol->flags == 0); // We don't set flags on alias.
			// Due to our generation order, this must be after the original symbol is created.
			ccv_nnc_autograd_tensor_symbol_t* ref = (ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, symbol->alias_ref - 1);
			symbol->symbol = ccv_nnc_tensor_symbol_alias_new(graph, ref->symbol, forw_symbol->ofs, forw_symbol->inc, forw_symbol->info, 0);
		}
	}
	// a no symbol.
	const ccv_nnc_tensor_symbol_t no_symbol = {
		.d = -1,
		.graph = graph
	};
	ccv_array_t* symbols = ccv_array_new(sizeof(ccv_nnc_tensor_symbol_t), 0, 0);
	for (i = 0; i < exec_symbol_size; i++)
	{
		// This is not going to be an interesting node. Skip.
		if (backward_info[i].f_wrt != 0x3)
			continue;
		ccv_nnc_graph_backward_info_t* back_info = backward_info + i;
		ccv_nnc_graph_autograd_exec_t* back_exec = autograd_exec + i;
		if (back_exec->cmd.cmd == CCV_NNC_NOOP)
		{
			back_exec->symbol = ccv_nnc_graph_exec_symbol_new(graph, back_exec->cmd, 0, 0, 0, 0, 0);
			continue;
		}
		ccv_array_clear(symbols);
		// Gradient inputs.
		for (j = 0; j < back_exec->input_size; j++)
			if (back_info->input_bitmasks[j >> 6] & ((uint64_t)1 << j))
				ccv_array_push(symbols, &(((ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, back_exec->inputs[j]))->symbol));
			else
				ccv_array_push(symbols, &no_symbol);
		ccv_nnc_graph_exec_symbol_info_t* forw_exec = exec_symbol_info + i;
		// Inputs from forward function.
		for (j = 0; j < forw_exec->input_size; j++)
			if (!(back_info->input_bitmasks[(j + back_exec->input_size) >> 6] & ((uint64_t)1 << (j + back_exec->input_size))))
				ccv_array_push(symbols, &no_symbol);
			else {
				ccv_nnc_tensor_symbol_t symbol = {
					.info = tensor_symbol_info[forw_exec->inputs[j]].info,
					.d = forw_exec->inputs[j],
					.graph = graph
				};
				ccv_array_push(symbols, &symbol);
			}
		// Outputs from forward function.
		for (j = 0; j < forw_exec->output_size; j++)
			if (!(back_info->input_bitmasks[(j + back_exec->input_size + forw_exec->input_size) >> 6] & ((uint64_t)1 << (j + back_exec->input_size + forw_exec->input_size))))
				ccv_array_push(symbols, &no_symbol);
			else {
				ccv_nnc_tensor_symbol_t symbol = {
					.info = tensor_symbol_info[forw_exec->outputs[j]].info,
					.d = forw_exec->outputs[j],
					.graph = graph
				};
				ccv_array_push(symbols, &symbol);
			}
		for (j = 0; j < back_exec->output_size; j++)
			if (back_info->output_bitmasks[j >> 6] & ((uint64_t)1 << j))
				ccv_array_push(symbols, &(((ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, back_exec->outputs[j]))->symbol));
			else
				ccv_array_push(symbols, &no_symbol);
		back_exec->symbol = ccv_nnc_graph_exec_symbol_new(graph, back_exec->cmd, ccv_array_get(symbols, 0), back_exec->input_size + forw_exec->input_size + forw_exec->output_size, ccv_array_get(symbols, back_exec->input_size + forw_exec->input_size + forw_exec->output_size), back_exec->output_size, 0);
	}
	for (i = 0; i < sum_or_set_exec->rnum; i++)
	{
		ccv_nnc_graph_sum_or_set_exec_t* exec = (ccv_nnc_graph_sum_or_set_exec_t*)ccv_array_get(sum_or_set_exec, i);
		if (exec->input_size)
		{
			ccv_array_clear(symbols);
			// This is to sum.
			for (j = 0; j < exec->input_size; j++)
				ccv_array_push(symbols, &(((ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, exec->inputs[j]))->symbol));
			ccv_nnc_cmd_t cmd = ccv_nnc_cmd(CCV_NNC_EWSUM_FORWARD, 0, CMD_GENERIC(), 0);
			exec->symbol = ccv_nnc_graph_exec_symbol_new(graph, cmd, ccv_array_get(symbols, 0), exec->input_size, &(((ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, exec->output))->symbol), 1, 0);
		} else {
			// This is to zero.
			ccv_nnc_cmd_t cmd = ccv_nnc_cmd(CCV_NNC_SET_FORWARD, 0, CMD_BLAS(0), 0);
			exec->symbol = ccv_nnc_graph_exec_symbol_new(graph, cmd, 0, 0, &(((ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, exec->output))->symbol), 1, 0);
		}
	}
	ccv_array_free(symbols);
	for (i = 0; i < exec_symbol_size; i++)
	{
		// This is not going to be an interesting node. Skip.
		if (backward_info[i].f_wrt != 0x3)
			continue;
		ccv_nnc_graph_exec_symbol_t forw_exec = {
			.d = i,
			.graph = graph
		};
		ccv_nnc_graph_autograd_exec_t* back_exec = autograd_exec + i;
		ccv_nnc_graph_exec_symbol_concat(graph, forw_exec, back_exec->symbol);
		if (back_exec->outgoings)
			for (j = 0; j < back_exec->outgoings->rnum; j++)
			{
				int d = *(int*)ccv_array_get(back_exec->outgoings, j);
				if (d < exec_symbol_size)
					ccv_nnc_graph_exec_symbol_concat(graph, back_exec->symbol, autograd_exec[d].symbol);
				else
					ccv_nnc_graph_exec_symbol_concat(graph, back_exec->symbol, ((ccv_nnc_graph_sum_or_set_exec_t*)ccv_array_get(sum_or_set_exec, d - exec_symbol_size))->symbol);
			}
	}
	for (i = 0; i < sum_or_set_exec->rnum; i++)
	{
		ccv_nnc_graph_sum_or_set_exec_t* exec = (ccv_nnc_graph_sum_or_set_exec_t*)ccv_array_get(sum_or_set_exec, i);
		if (exec->outgoings)
			for (j = 0; j < exec->outgoings->rnum; j++)
			{
				int d = *(int*)ccv_array_get(exec->outgoings, j);
				if (d < exec_symbol_size)
					ccv_nnc_graph_exec_symbol_concat(graph, exec->symbol, autograd_exec[d].symbol);
				else
					ccv_nnc_graph_exec_symbol_concat(graph, exec->symbol, ((ccv_nnc_graph_sum_or_set_exec_t*)ccv_array_get(sum_or_set_exec, d - exec_symbol_size))->symbol);
			}
	}
	int alias_size = 0;
	for (i = 0; i < wrt_symbol_size; i++)
		if (tensor_symbol_info[wrt_symbols[i].d].alias_ref)
			// I need to create alias for these, therefore, add to alias_size count.
			++alias_size;
	// Now, everything is done, set the metadata on graph so that we can lookup later for backward symbols
	if (graph->backward_tensor_symbols)
		graph->backward_tensor_symbols = (int*)ccrealloc(graph->backward_tensor_symbols, sizeof(int) * (graph->tensor_symbol_info->rnum + alias_size));
	else
		graph->backward_tensor_symbols = (int*)ccmalloc(sizeof(int) * (graph->tensor_symbol_info->rnum + alias_size));
	graph->backward_exec_symbols = graph->backward_tensor_symbols + tensor_symbol_size;
	graph->forward_symbol_size = tensor_symbol_size;
	graph->backward_symbol_size = (graph->tensor_symbol_info->rnum + alias_size) - tensor_symbol_size;
	for (i = 0; i < graph->forward_symbol_size; i++)
		graph->backward_tensor_symbols[i] = -1;
	for (i = 0; i < graph->backward_symbol_size; i++)
		graph->backward_exec_symbols[i] = -1;
	// Assigning for wrt symbols.
	for (i = 0; i < wrt_symbol_size; i++)
	{
		const int d = wrt_symbols[i].d;
		assert(d >= 0);
		assert(d < tensor_symbol_size);
		// If this wrt symbol is an alias, create extra alias for this.
		ccv_nnc_tensor_ref_t* tensor_ref;
		int dd;
		if (tensor_symbol_info[d].alias_ref)
		{
			ccv_nnc_autograd_tensor_version_t* tensor_ver = autograd_tensor_version + (tensor_symbol_info[d].alias_ref - 1);
			assert(tensor_ver->ref_version);
			tensor_ref = (ccv_nnc_tensor_ref_t*)ccv_array_get(tensor_ver->ref_version, tensor_ver->c);
			ccv_nnc_autograd_tensor_symbol_t* autograd_symbol = (ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, tensor_ref->d);
			ccv_nnc_tensor_symbol_t alias = ccv_nnc_tensor_symbol_alias_new(graph, autograd_symbol->symbol, tensor_symbol_info[d].ofs, tensor_symbol_info[d].inc, tensor_symbol_info[d].info, 0);
			graph->backward_tensor_symbols[d] = alias.d;
			assert(alias.d >= tensor_symbol_size);
			dd = alias.d - tensor_symbol_size;
		} else {
			ccv_nnc_autograd_tensor_version_t* tensor_ver = autograd_tensor_version + d;
			assert(tensor_ver->ref_version);
			tensor_ref = (ccv_nnc_tensor_ref_t*)ccv_array_get(tensor_ver->ref_version, tensor_ver->c);
			ccv_nnc_autograd_tensor_symbol_t* autograd_symbol = (ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, tensor_ref->d);
			graph->backward_tensor_symbols[d] = autograd_symbol->symbol.d;
			assert(autograd_symbol->symbol.d >= tensor_symbol_size);
			dd = autograd_symbol->symbol.d - tensor_symbol_size;
		}
		const int x = tensor_ref->x;
		if (tensor_ref->exec_registry && tensor_ref->exec_registry->rnum) // Create no-op node.
		{
			ccv_nnc_graph_exec_symbol_t noop = ccv_nnc_graph_exec_symbol_new(graph, ccv_nnc_cmd(CCV_NNC_NOOP, 0, CMD_GENERIC(), 0), 0, 0, 0, 0, 0);
			if (x < exec_symbol_size)
				ccv_nnc_graph_exec_symbol_concat(graph, autograd_exec[x].symbol, noop);
			else
				ccv_nnc_graph_exec_symbol_concat(graph, ((ccv_nnc_graph_sum_or_set_exec_t*)ccv_array_get(sum_or_set_exec, x - exec_symbol_size))->symbol, noop);
			for (j = 0; j < tensor_ref->exec_registry->rnum; j++)
			{
				const int x = *(int*)ccv_array_get(tensor_ref->exec_registry, j);
				assert(x >= 0); /* Otherwise, this is initialization tensor, which is impossible to be summed up by. */
				assert(x < exec_symbol_size); // exec_registry is only used by alias_registry, it simply cannot reference to a sum operation.
				ccv_nnc_graph_exec_symbol_concat(graph, autograd_exec[x].symbol, noop);
			}
			graph->backward_exec_symbols[dd] = noop.d;
		} else {
			if (x < exec_symbol_size)
				graph->backward_exec_symbols[dd] = autograd_exec[x].symbol.d;
			else
				graph->backward_exec_symbols[dd] = ((ccv_nnc_graph_sum_or_set_exec_t*)ccv_array_get(sum_or_set_exec, x - exec_symbol_size))->symbol.d;
		}
	}
	// Assigning for f symbols.
	for (i = 0; i < f_symbol_size; i++)
	{
		const int d = f_symbols[i].d;
		assert(d >= 0);
		assert(d < tensor_symbol_size);
		ccv_nnc_autograd_tensor_version_t* tensor_ver = autograd_tensor_version + d;
		assert(tensor_ver->ref_version);
		ccv_nnc_tensor_ref_t* tensor_ref = (ccv_nnc_tensor_ref_t*)ccv_array_get(tensor_ver->ref_version, tensor_ver->c);
		ccv_nnc_autograd_tensor_symbol_t* autograd_symbol = (ccv_nnc_autograd_tensor_symbol_t*)ccv_array_get(autograd_tensor_symbol, tensor_ref->d);
		graph->backward_tensor_symbols[d] = autograd_symbol->symbol.d;
		// Cannot find relevant backward exec symbols for f.
	}
	for (i = 0; i < exec_symbol_size; i++)
	{
		if (autograd_exec[i].inputs)
			ccfree(autograd_exec[i].inputs);
		if (autograd_exec[i].outgoings)
			ccv_array_free(autograd_exec[i].outgoings);
	}
	ccfree(autograd_exec);
	for (i = 0; i < tensor_symbol_size; i++)
	{
		if (autograd_tensor_version[i].ref_version)
		{
			for (j = 0; j < autograd_tensor_version[i].ref_version->rnum; j++)
			{
				ccv_nnc_tensor_ref_t* ref_version = (ccv_nnc_tensor_ref_t*)ccv_array_get(autograd_tensor_version[i].ref_version, j);
				if (ref_version->exec_registry)
					ccv_array_free(ref_version->exec_registry);
				if (ref_version->alias_registry)
					ccv_array_free(ref_version->alias_registry);
			}
			ccv_array_free(autograd_tensor_version[i].ref_version);
		}
	}
	ccfree(autograd_tensor_version);
	ccv_array_free(autograd_tensor_symbol);
	for (i = 0; i < sum_or_set_exec->rnum; i++)
	{
		ccv_nnc_graph_sum_or_set_exec_t* sum_or_set = (ccv_nnc_graph_sum_or_set_exec_t*)ccv_array_get(sum_or_set_exec, i);
		if (sum_or_set->inputs)
			ccfree(sum_or_set->inputs);
		if (sum_or_set->outgoings)
			ccv_array_free(sum_or_set->outgoings);
	}
	ccv_array_free(sum_or_set_exec);
	for (i = 0; i < exec_symbol_size; i++)
	{
		if (backward_info[i].outgoings)
			ccv_array_free(backward_info[i].outgoings);
		if (backward_info[i].input_bitmasks)
			ccfree(backward_info[i].input_bitmasks);
	}
	ccfree(backward_info);
	ccfree(exec_symbol_info);
	ccfree(tensor_symbol_info);
}

ccv_nnc_tensor_symbol_t ccv_nnc_tensor_symbol_for_backward(const ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t symbol)
{
	assert(symbol.d >= 0);
	assert(symbol.d < graph->forward_symbol_size);
	assert(graph->backward_tensor_symbols[symbol.d] >= 0);
	ccv_nnc_tensor_symbol_t tensor = {
		.d = graph->backward_tensor_symbols[symbol.d],
		.graph = graph,
		.info = symbol.info
	};
	return tensor;
}

ccv_nnc_graph_exec_symbol_t ccv_nnc_graph_exec_symbol_for_backward(const ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t symbol)
{
	assert(symbol.d >= graph->forward_symbol_size);
	assert(symbol.d < graph->forward_symbol_size + graph->backward_symbol_size);
	const int dd = symbol.d - graph->forward_symbol_size;
	assert(graph->backward_exec_symbols[dd] >= 0);
	ccv_nnc_graph_exec_symbol_t exec = {
		.d = graph->backward_exec_symbols[dd],
		.graph = graph
	};
	return exec;
}
