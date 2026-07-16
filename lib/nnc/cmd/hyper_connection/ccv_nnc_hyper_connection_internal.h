#ifndef GUARD_ccv_nnc_hyper_connection_internal_h
#define GUARD_ccv_nnc_hyper_connection_internal_h

#include "nnc/ccv_nnc_easy.h"

static inline int _ccv_nnc_hyper_connection_prefix_matches(const ccv_nnc_tensor_param_t a, const int a_tail, const ccv_nnc_tensor_param_t b, const int b_tail)
{
	const int a_nd = ccv_nnc_tensor_nd(a.dim);
	const int b_nd = ccv_nnc_tensor_nd(b.dim);
	if (a_nd < a_tail || b_nd < b_tail || a_nd - a_tail != b_nd - b_tail)
		return 0;
	int i;
	for (i = 0; i < a_nd - a_tail; i++)
		if (a.dim[i] != b.dim[i])
			return 0;
	return 1;
}

static inline int _ccv_nnc_hyper_connection_combination_matches(const ccv_nnc_tensor_param_t anchor, const int anchor_tail, const ccv_nnc_tensor_param_t combination, const int count)
{
	const int combination_nd = ccv_nnc_tensor_nd(combination.dim);
	if (_ccv_nnc_hyper_connection_prefix_matches(anchor, anchor_tail, combination, 1) && combination_nd >= 1 && combination.dim[combination_nd - 1] == count * count)
		return 1;
	return _ccv_nnc_hyper_connection_prefix_matches(anchor, anchor_tail, combination, 2) && combination_nd >= 2 && combination.dim[combination_nd - 2] == count && combination.dim[combination_nd - 1] == count;
}

static inline int _ccv_nnc_hyper_connection_split_shapes_are_valid(const int count, const ccv_nnc_tensor_param_t mix, const ccv_nnc_tensor_param_t scale, const ccv_nnc_tensor_param_t base, const ccv_nnc_tensor_param_t* const residual, const ccv_nnc_tensor_param_t* const pre, const ccv_nnc_tensor_param_t post, const ccv_nnc_tensor_param_t combination, const ccv_nnc_tensor_param_t* const weighted)
{
	if (count <= 0 || count > 16)
		return 0;
	const int mix_nd = ccv_nnc_tensor_nd(mix.dim);
	const int mix_dim = 2 * count + count * count;
	if (mix_nd < 1 || mix.dim[mix_nd - 1] != mix_dim || ccv_nnc_tensor_count(scale) != 3 || ccv_nnc_tensor_count(base) != mix_dim)
		return 0;
	const int post_nd = ccv_nnc_tensor_nd(post.dim);
	if (!_ccv_nnc_hyper_connection_prefix_matches(mix, 1, post, 1) || post_nd < 1 || post.dim[post_nd - 1] != count || !_ccv_nnc_hyper_connection_combination_matches(mix, 1, combination, count))
		return 0;
	if (residual)
	{
		if (!weighted || pre)
			return 0;
		const int residual_nd = ccv_nnc_tensor_nd(residual->dim);
		const int weighted_nd = ccv_nnc_tensor_nd(weighted->dim);
		if (!_ccv_nnc_hyper_connection_prefix_matches(mix, 1, *residual, 2) || residual_nd < 2 || residual->dim[residual_nd - 2] != count || residual->dim[residual_nd - 1] <= 0)
			return 0;
		return _ccv_nnc_hyper_connection_prefix_matches(mix, 1, *weighted, 1) && weighted_nd >= 1 && weighted->dim[weighted_nd - 1] == residual->dim[residual_nd - 1];
	}
	if (!pre || weighted)
		return 0;
	const int pre_nd = ccv_nnc_tensor_nd(pre->dim);
	return _ccv_nnc_hyper_connection_prefix_matches(mix, 1, *pre, 1) && pre_nd >= 1 && pre->dim[pre_nd - 1] == count;
}

static inline int _ccv_nnc_hyper_connection_expand_shapes_are_valid(const int count, const ccv_nnc_tensor_param_t block, const ccv_nnc_tensor_param_t residual, const ccv_nnc_tensor_param_t post, const ccv_nnc_tensor_param_t combination, const ccv_nnc_tensor_param_t expanded)
{
	if (count <= 0 || count > 16)
		return 0;
	const int residual_nd = ccv_nnc_tensor_nd(residual.dim);
	const int block_nd = ccv_nnc_tensor_nd(block.dim);
	const int post_nd = ccv_nnc_tensor_nd(post.dim);
	if (residual_nd < 2 || residual.dim[residual_nd - 2] != count || residual.dim[residual_nd - 1] <= 0)
		return 0;
	if (!_ccv_nnc_hyper_connection_prefix_matches(residual, 2, block, 1) || block_nd < 1 || block.dim[block_nd - 1] != residual.dim[residual_nd - 1])
		return 0;
	if (!_ccv_nnc_hyper_connection_prefix_matches(residual, 2, post, 1) || post_nd < 1 || post.dim[post_nd - 1] != count)
		return 0;
	if (!_ccv_nnc_hyper_connection_combination_matches(residual, 2, combination, count))
		return 0;
	const int expanded_nd = ccv_nnc_tensor_nd(expanded.dim);
	if (expanded_nd != residual_nd)
		return 0;
	int i;
	for (i = 0; i < residual_nd; i++)
		if (expanded.dim[i] != residual.dim[i])
			return 0;
	return 1;
}

#endif
