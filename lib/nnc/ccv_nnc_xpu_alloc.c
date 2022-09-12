#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_nnc_easy.h"
#include "ccv_internal.h"
#include "_ccv_nnc_xpu_alloc.h"
#ifdef HAVE_CUDA
#include "gpu/ccv_nnc_compat.h"
#include <stdbool.h>

static int dy_alloc_tree_cmp(const dy_alloc_metadata_t* const a_node, const dy_alloc_metadata_t* const b_node)
{
	return (a_node->size > b_node->size) - (b_node->size > a_node->size);
}

rb_gen(, dy_alloc_tree_, dy_alloc_tree_t, dy_alloc_metadata_t, link, dy_alloc_tree_cmp)

static void _ccv_nnc_xpu_metadata_free(dy_alloc_metadata_t* node, void* arg)
{
	do {
		dy_alloc_metadata_t* const next = node->next;
		cufree(node->device, node->ptr);
		ccfree(node);
		node = next;
	} while (node);
}

static void _ccv_nnc_xpu_alloc_drain(const int device, khash_t(dy_dev)* const dev, const ccv_nnc_stream_context_t* const stream)
{
	// Wait until the stream is free, and then do the free.
	if (stream)
		ccv_nnc_stream_context_wait(stream);
	khiter_t k;
	if (device >= 0)
	{
		k = kh_get(dy_dev, dev, device);
		if (k != kh_end(dev))
		{
			dy_alloc_tree_t* const tree = &kh_val(dev, k);
			dy_alloc_tree_destroy(tree, _ccv_nnc_xpu_metadata_free, 0);
			kh_del(dy_dev, dev, k);
		}
		return;
	}
	for (k = kh_begin(dev); k != kh_end(dev); ++k)
	{
		if (!kh_exist(dev, k))
			continue;
		dy_alloc_tree_t* const tree = &kh_val(dev, k);
		dy_alloc_tree_destroy(tree, _ccv_nnc_xpu_metadata_free, 0);
		kh_del(dy_dev, dev, k);
	}
}

static void _ccv_nnc_xpu_stream_destructor_hook(const ccv_nnc_stream_context_t* const stream, void* const context)
{
	ccv_nnc_xpu_alloc_t* const xpu_alloc = (ccv_nnc_xpu_alloc_t*)context;
	khash_t(dy_str)* const freed = xpu_alloc->freed;
	const int64_t str = (int64_t)(intptr_t)stream;
	khiter_t i = kh_get(dy_str, freed, str);
	assert(i != kh_end(freed));
	khash_t(dy_dev)* const dev = kh_val(freed, i).dev;
	_ccv_nnc_xpu_alloc_drain(-1, dev, stream);
	kh_destroy(dy_dev, dev);
	kh_del(dy_str, freed, i);
}

void* ccv_nnc_xpu_alloc(ccv_nnc_xpu_alloc_t* const xpu_alloc, const int device, ccv_nnc_stream_context_t* const stream, const size_t size)
{
	khash_t(dy_str)* const freed = xpu_alloc->freed;
	const int64_t str = (int64_t)(intptr_t)stream;
	int ret;
	khiter_t i = kh_put(dy_str, freed, str, &ret);
	assert(ret >= 0);
	dy_alloc_metadata_t* node = 0;
	if (ret == 0)
	{
		// If we can find stream related allocations, try to
		// find the suitable ones.
		khash_t(dy_dev)* const dev = kh_val(freed, i).dev;
		assert(dev);
		khiter_t j = kh_get(dy_dev, dev, device);
		if (j != kh_end(dev))
		{
			dy_alloc_tree_t* const tree = &kh_val(dev, j);
			dy_alloc_metadata_t key = {
				.size = size
			};
			node = dy_alloc_tree_nsearch(tree, &key);
			if (node)
			{
				if (node->next) // If it is a linked list, select the one.
				{
					dy_alloc_metadata_t* next_node = node->next;
					node->next = node->next->next;
					node = next_node;
				} else
					dy_alloc_tree_remove(tree, node);
			}
		}
	} else {
		// Otherwise, create it.
		kh_val(freed, i).dev = kh_init(dy_dev);
		kh_val(freed, i).hook_id = stream ? ccv_nnc_stream_context_add_destructor_hook(stream, _ccv_nnc_xpu_stream_destructor_hook, xpu_alloc) : -1;

	}
	if (!node)
	{
		node = (dy_alloc_metadata_t*)ccmalloc(sizeof(dy_alloc_metadata_t));
		if (xpu_alloc->mp_hdr < 0)
			xpu_alloc->mp_hdr = curegmp(device, (cump_f)ccv_nnc_xpu_gc, xpu_alloc);
		node->ptr = cumalloc(device, size);
		if (!node->ptr) // If cannot allocate, drain the pool first and then allocate.
		{
			ccfree(node);
			return 0;
		}
		node->device = device;
		node->size = size;
		node->str = str;
	} else {
		assert(node->size >= size);
		assert(node->device == device);
		assert(node->str == str);
	}
	node->next = 0;
	khash_t(dy_alloc)* const allocd = xpu_alloc->allocd;
	i = kh_put(dy_alloc, allocd, (int64_t)(intptr_t)node->ptr, &ret);
	assert(ret > 0);
	kh_val(allocd, i) = node;
	return node->ptr;
}

void ccv_nnc_xpu_free(ccv_nnc_xpu_alloc_t* const xpu_alloc, void* const ptr)
{
	khash_t(dy_alloc)* const allocd = xpu_alloc->allocd;
	khiter_t i = kh_get(dy_alloc, allocd, (int64_t)(intptr_t)ptr);
	assert(i != kh_end(allocd));
	dy_alloc_metadata_t* const node = kh_val(allocd, i);
	kh_del(dy_alloc, allocd, i);
	assert(node->ptr == ptr);
	khash_t(dy_str)* const freed = xpu_alloc->freed;
	i = kh_get(dy_str, freed, node->str);
	// If cannot find associated stream, that means this allocation associated
	// stream has been freed. I have to do synchronous free of this pointer.
	if (i == kh_end(freed))
	{
		cufree(node->device, node->ptr);
		ccfree(node);
		return;
	}
	khash_t(dy_dev)* const dev = kh_val(freed, i).dev;
	int ret;
	khiter_t j = kh_put(dy_dev, dev, node->device, &ret);
	assert(ret >= 0);
	dy_alloc_tree_t* const tree = &kh_val(dev, j);
	if (ret != 0)
		dy_alloc_tree_new(tree);
	dy_alloc_metadata_t* const canon_node = dy_alloc_tree_search(tree, node);
	if (!canon_node)
		dy_alloc_tree_insert(tree, node);
	else { // Insert into the linked list.
		node->next = canon_node->next;
		canon_node->next = node;
	}
}

void ccv_nnc_xpu_alloc_destroy(ccv_nnc_xpu_alloc_t* const xpu_alloc)
{
	khash_t(dy_alloc)* const allocd = xpu_alloc->allocd;
	khiter_t k;
	for (k = kh_begin(allocd); k != kh_end(allocd); ++k)
	{
		if (!kh_exist(allocd, k))
			continue;
		_ccv_nnc_xpu_metadata_free(kh_val(allocd, k), 0);
	}
	kh_destroy(dy_alloc, allocd);
	khash_t(dy_str)* const freed = xpu_alloc->freed;
	for (k = kh_begin(freed); k != kh_end(freed); ++k)
	{
		if (!kh_exist(freed, k))
			continue;
		khash_t(dy_dev)* const dev = kh_val(freed, k).dev;
		ccv_nnc_stream_context_t* const stream = (ccv_nnc_stream_context_t*)(intptr_t)kh_key(freed, k);
		_ccv_nnc_xpu_alloc_drain(-1, dev, stream);
		if (stream)
		{
			const int hook_id = kh_val(freed, k).hook_id;
			ccv_nnc_stream_context_remove_destructor_hook(stream, hook_id);
		}
		kh_destroy(dy_dev, dev);
	}
	kh_destroy(dy_str, freed);
	if (xpu_alloc->mp_hdr >= 0)
		cuunregmp(xpu_alloc->mp_hdr);
}

void ccv_nnc_xpu_gc(const int device, ccv_nnc_xpu_alloc_t* const xpu_alloc)
{
	khash_t(dy_str)* const freed = xpu_alloc->freed;
	khiter_t k;
	for (k = kh_begin(freed); k != kh_end(freed); ++k)
	{
		if (!kh_exist(freed, k))
			continue;
		khash_t(dy_dev)* const dev = kh_val(freed, k).dev;
		ccv_nnc_stream_context_t* const stream = (ccv_nnc_stream_context_t*)(intptr_t)kh_key(freed, k);
		_ccv_nnc_xpu_alloc_drain(device, dev, stream);
	}
}
#else
void* ccv_nnc_xpu_alloc(ccv_nnc_xpu_alloc_t* const xpu_alloc, const int device, ccv_nnc_stream_context_t* const stream, const size_t size)
{
	return 0;
}

void ccv_nnc_xpu_free(ccv_nnc_xpu_alloc_t* const xpu_alloc, void* const ptr)
{
}

void ccv_nnc_xpu_alloc_destroy(ccv_nnc_xpu_alloc_t* const xpu_alloc)
{
}

void ccv_nnc_xpu_gc(const int device, ccv_nnc_xpu_alloc_t* const dynamic_graph)
{
}
#endif
