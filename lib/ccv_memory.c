#include "ccv.h"
#include "ccv_internal.h"
#include "3rdparty/sha1/sha1.h"

static ccv_cache_t ccv_cache;

/* option to enable/disable cache */
static int ccv_cache_opt = 0;

ccv_dense_matrix_t* ccv_dense_matrix_new(int rows, int cols, int type, void* data, uint64_t sig)
{
	ccv_dense_matrix_t* mat;
	if (ccv_cache_opt && sig != 0)
	{
		mat = (ccv_dense_matrix_t*)ccv_cache_out(&ccv_cache, sig);
		if (mat)
		{
			mat->type |= CCV_GARBAGE;
			mat->refcount = 1;
			return mat;
		}
	}
	mat = (ccv_dense_matrix_t*)(data ? data : ccmalloc(ccv_compute_dense_matrix_size(rows, cols, type)));
	mat->sig = sig;
	mat->type = (type | CCV_MATRIX_DENSE) & ~CCV_GARBAGE;
	mat->type |= data ? CCV_UNMANAGED : CCV_REUSABLE;
	mat->rows = rows;
	mat->cols = cols;
	mat->step = (cols * CCV_GET_DATA_TYPE_SIZE(type) * CCV_GET_CHANNEL(type) + 3) & -4;
	mat->refcount = 1;
	mat->data.u8 = (unsigned char*)(mat + 1);
	return mat;
}

ccv_dense_matrix_t* ccv_dense_matrix_renew(ccv_dense_matrix_t* x, int rows, int cols, int types, int prefer_type, uint64_t sig)
{
	if (x != 0)
	{
		assert(x->rows == rows && x->cols == cols && (CCV_GET_DATA_TYPE(x->type) & types) && (CCV_GET_CHANNEL(x->type) == CCV_GET_CHANNEL(types)));
		prefer_type = CCV_GET_DATA_TYPE(x->type) | CCV_GET_CHANNEL(x->type);
	}
	sig = ccv_matrix_generate_signature((const char*)&prefer_type, sizeof(int), sig, 0);
	if (x == 0)
	{
		x = ccv_dense_matrix_new(rows, cols, prefer_type, 0, sig);
	} else {
		x->sig = sig;
	}
	return x;
}

ccv_dense_matrix_t ccv_dense_matrix(int rows, int cols, int type, void* data, uint64_t sig)
{
	ccv_dense_matrix_t mat;
	mat.sig = sig;
	mat.type = (type | CCV_MATRIX_DENSE | CCV_UNMANAGED) & ~CCV_GARBAGE;
	mat.rows = rows;
	mat.cols = cols;
	mat.step = (cols * CCV_GET_DATA_TYPE_SIZE(type) * CCV_GET_CHANNEL(type) + 3) & -4;
	mat.refcount = 1;
	mat.data.u8 = (unsigned char*)data;
	return mat;
}

ccv_sparse_matrix_t* ccv_sparse_matrix_new(int rows, int cols, int type, int major, uint64_t sig)
{
	ccv_sparse_matrix_t* mat;
	mat = (ccv_sparse_matrix_t*)ccmalloc(sizeof(ccv_sparse_matrix_t));
	mat->rows = rows;
	mat->cols = cols;
	mat->type = type | CCV_MATRIX_SPARSE | ((type & CCV_DENSE_VECTOR) ? CCV_DENSE_VECTOR : CCV_SPARSE_VECTOR);
	mat->major = major;
	mat->prime = 0;
	mat->load_factor = 0;
	mat->refcount = 1;
	mat->vector = (ccv_dense_vector_t*)ccmalloc(CCV_GET_SPARSE_PRIME(mat->prime) * sizeof(ccv_dense_vector_t));
	int i;
	for (i = 0; i < CCV_GET_SPARSE_PRIME(mat->prime); i++)
	{
		mat->vector[i].index = -1;
		mat->vector[i].length = 0;
		mat->vector[i].next = 0;
	}
	return mat;
}

void ccv_matrix_free_immediately(ccv_matrix_t* mat)
{
	int type = *(int*)mat;
	assert(!(type & CCV_UNMANAGED));
	if (type & CCV_MATRIX_DENSE)
	{
		ccv_dense_matrix_t* dmt = (ccv_dense_matrix_t*)mat;
		dmt->refcount = 0;
		ccfree(dmt);
	} else if (type & CCV_MATRIX_SPARSE) {
		ccv_sparse_matrix_t* smt = (ccv_sparse_matrix_t*)mat;
		int i;
		for (i = 0; i < CCV_GET_SPARSE_PRIME(smt->prime); i++)
			if (smt->vector[i].index != -1)
			{
				ccv_dense_vector_t* iter = &smt->vector[i];
				ccfree(iter->data.u8);
				iter = iter->next;
				while (iter != 0)
				{
					ccv_dense_vector_t* iter_next = iter->next;
					ccfree(iter->data.u8);
					ccfree(iter);
					iter = iter_next;
				}
			}
		ccfree(smt->vector);
		ccfree(smt);
	} else if ((type & CCV_MATRIX_CSR) || (type & CCV_MATRIX_CSC)) {
		ccv_compressed_sparse_matrix_t* csm = (ccv_compressed_sparse_matrix_t*)mat;
		csm->refcount = 0;
		ccfree(csm);
	}
}

void ccv_matrix_free(ccv_matrix_t* mat)
{
	int type = *(int*)mat;
	assert(!(type & CCV_UNMANAGED));
	if (type & CCV_MATRIX_DENSE)
	{
		ccv_dense_matrix_t* dmt = (ccv_dense_matrix_t*)mat;
		dmt->refcount = 0;
		if (!ccv_cache_opt || !(dmt->type & CCV_REUSABLE) || dmt->sig == 0)
			ccfree(dmt);
		else {
			size_t size = sizeof(ccv_dense_matrix_t) + ((dmt->cols * CCV_GET_DATA_TYPE_SIZE(dmt->type) * CCV_GET_CHANNEL(dmt->type) + 3) & -4) * dmt->rows;
			ccv_cache_put(&ccv_cache, dmt->sig, dmt, size);
		}
	} else if (type & CCV_MATRIX_SPARSE) {
		ccv_sparse_matrix_t* smt = (ccv_sparse_matrix_t*)mat;
		int i;
		for (i = 0; i < CCV_GET_SPARSE_PRIME(smt->prime); i++)
			if (smt->vector[i].index != -1)
			{
				ccv_dense_vector_t* iter = &smt->vector[i];
				ccfree(iter->data.u8);
				iter = iter->next;
				while (iter != 0)
				{
					ccv_dense_vector_t* iter_next = iter->next;
					ccfree(iter->data.u8);
					ccfree(iter);
					iter = iter_next;
				}
			}
		ccfree(smt->vector);
		ccfree(smt);
	} else if ((type & CCV_MATRIX_CSR) || (type & CCV_MATRIX_CSC)) {
		ccv_compressed_sparse_matrix_t* csm = (ccv_compressed_sparse_matrix_t*)mat;
		csm->refcount = 0;
		ccfree(csm);
	}
}

void ccv_drain_cache(void)
{
	if (ccv_cache.rnum > 0)
		ccv_cache_cleanup(&ccv_cache);
}

void ccv_disable_cache(void)
{
	ccv_cache_opt = 0;
	ccv_cache_close(&ccv_cache);
}

void ccv_enable_cache(size_t size)
{
	ccv_cache_opt = 1;
	ccv_cache_init(&ccv_cache, ccfree, size);
}

void ccv_enable_default_cache(void)
{
	ccv_enable_cache(CCV_DEFAULT_CACHE_SIZE);
}

uint64_t ccv_matrix_generate_signature(const char* msg, int len, uint64_t sig_start, ...)
{
	blk_SHA_CTX ctx;
	blk_SHA1_Init(&ctx);
	uint64_t sigi;
	va_list arguments;
	va_start(arguments, sig_start);
	for (sigi = sig_start; sigi != 0; sigi = va_arg(arguments, uint64_t))
		blk_SHA1_Update(&ctx, &sigi, 8);
	va_end(arguments);
	blk_SHA1_Update(&ctx, msg, len);
	union {
		uint64_t u;
		uint8_t chr[20];
	} sig;
	blk_SHA1_Final(sig.chr, &ctx);
	return sig.u;
}
