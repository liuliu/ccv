#include "ccv.h"
#include "ccv_internal.h"
#include "3rdparty/siphash/siphash24.h"

static __thread ccv_cache_t ccv_cache;

/**
 * For new typed cache object:
 * ccv_dense_matrix_t: type 0
 * ccv_array_t: type 1
 **/

/* option to enable/disable cache */
static __thread int ccv_cache_opt = 0;

ccv_dense_matrix_t* ccv_dense_matrix_new(int rows, int cols, int type, void* data, uint64_t sig)
{
	ccv_dense_matrix_t* mat;
	if (ccv_cache_opt && sig != 0 && !data && !(type & CCV_NO_DATA_ALLOC))
	{
		uint8_t type;
		mat = (ccv_dense_matrix_t*)ccv_cache_out(&ccv_cache, sig, &type);
		if (mat)
		{
			assert(type == 0);
			mat->type |= CCV_GARBAGE; // set the flag so the upper level function knows this is from recycle-bin
			mat->refcount = 1;
			return mat;
		}
	}
	if (type & CCV_NO_DATA_ALLOC)
	{
		mat = (ccv_dense_matrix_t*)ccmalloc(sizeof(ccv_dense_matrix_t));
		mat->type = (CCV_GET_CHANNEL(type) | CCV_GET_DATA_TYPE(type) | CCV_MATRIX_DENSE | CCV_NO_DATA_ALLOC) & ~CCV_GARBAGE;
		mat->data.u8 = data;
	} else {
		const size_t hdr_size = (sizeof(ccv_dense_matrix_t) + 15) & -16;
		mat = (ccv_dense_matrix_t*)(data ? data : ccmalloc(ccv_compute_dense_matrix_size(rows, cols, type)));
		mat->type = (CCV_GET_CHANNEL(type) | CCV_GET_DATA_TYPE(type) | CCV_MATRIX_DENSE) & ~CCV_GARBAGE;
		mat->type |= data ? CCV_UNMANAGED : CCV_REUSABLE; // it still could be reusable because the signature could be derived one.
		mat->data.u8 = (unsigned char*)mat + hdr_size;
	}
	mat->sig = sig;
#if CCV_NNC_TENSOR_TFB
	mat->reserved0 = 0;
	mat->resides = CCV_TENSOR_CPU_MEMORY;
	mat->format = CCV_TENSOR_FORMAT_NHWC;
	mat->datatype = CCV_GET_DATA_TYPE(type);
	mat->channels = CCV_GET_CHANNEL(type);
	mat->reserved1 = 0;
#endif
	mat->rows = rows;
	mat->cols = cols;
	mat->step = CCV_GET_STEP(cols, type);
	mat->refcount = 1;
	return mat;
}

ccv_dense_matrix_t* ccv_dense_matrix_renew(ccv_dense_matrix_t* x, int rows, int cols, int types, int prefer_type, uint64_t sig)
{
	if (x != 0)
	{
		assert(x->rows == rows && x->cols == cols && (CCV_GET_DATA_TYPE(x->type) & types) && (CCV_GET_CHANNEL(x->type) == CCV_GET_CHANNEL(types)));
		prefer_type = CCV_GET_DATA_TYPE(x->type) | CCV_GET_CHANNEL(x->type);
	}
	if (sig != 0)
		sig = ccv_cache_generate_signature((const char*)&prefer_type, sizeof(int), sig, CCV_EOF_SIGN);
	if (x == 0)
	{
		x = ccv_dense_matrix_new(rows, cols, prefer_type, 0, sig);
	} else {
		x->sig = sig;
	}
	return x;
}

void ccv_make_matrix_mutable(ccv_matrix_t* mat)
{
	int type = *(int*)mat;
	if (type & CCV_MATRIX_DENSE)
	{
		ccv_dense_matrix_t* dmt = (ccv_dense_matrix_t*)mat;
		dmt->sig = 0;
		dmt->type &= ~CCV_REUSABLE;
	}
}

void ccv_make_matrix_immutable(ccv_matrix_t* mat)
{
	int type = *(int*)mat;
	if (type & CCV_MATRIX_DENSE)
	{
		ccv_dense_matrix_t* dmt = (ccv_dense_matrix_t*)mat;
		assert(dmt->sig == 0); // you cannot make matrix with derived signature immutable (it is immutable already)
		/* immutable matrix made this way is not reusable (collected), because its signature
		 * only depends on the content, not the operation to generate it */
		dmt->type &= ~CCV_REUSABLE;
		dmt->sig = ccv_cache_generate_signature((char*)dmt->data.u8, dmt->rows * dmt->step, (uint64_t)dmt->type, CCV_EOF_SIGN);
	}
}

ccv_dense_matrix_t ccv_dense_matrix(int rows, int cols, int type, void* data, uint64_t sig)
{
	ccv_dense_matrix_t mat;
	mat.reserved0 = 0;
	mat.sig = sig;
	mat.type = (CCV_GET_CHANNEL(type) | CCV_GET_DATA_TYPE(type) | CCV_MATRIX_DENSE | CCV_NO_DATA_ALLOC | CCV_UNMANAGED) & ~CCV_GARBAGE;
	mat.rows = rows;
	mat.cols = cols;
	mat.step = CCV_GET_STEP(cols, type);
	mat.refcount = 1;
#if CCV_NNC_TENSOR_TFB
	mat.reserved0 = 0;
	mat.resides = CCV_TENSOR_CPU_MEMORY;
	mat.format = CCV_TENSOR_FORMAT_NHWC | CCV_GET_DATA_TYPE(type);
	mat.channels = CCV_GET_CHANNEL(type);
	mat.reserved1 = 0;
#endif
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
	mat->prime_index = 1; // See ccv_util.c to know why this is 1 and why size is 2.
	mat->size = 2;
	mat->rnum = 0;
	mat->refcount = 1;
	mat->index = (ccv_sparse_matrix_index_t*)cccalloc(sizeof(ccv_sparse_matrix_index_t), mat->size);
	mat->vector = (ccv_sparse_matrix_vector_t*)ccmalloc(sizeof(ccv_sparse_matrix_vector_t) * mat->size);
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
		for (i = 0; i < smt->size; i++)
			if (smt->index[i].ifbit)
				ccfree(smt->vector[i].data.u8);
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
		if (!ccv_cache_opt || // e don't enable cache
			!(dmt->type & CCV_REUSABLE) || // or this is not a reusable piece
			dmt->sig == 0 || // or this doesn't have valid signature
			(dmt->type & CCV_NO_DATA_ALLOC)) // or this matrix is allocated as header-only, therefore we cannot cache it
			ccfree(dmt);
		else {
			assert(CCV_GET_DATA_TYPE(dmt->type) == CCV_8U ||
				   CCV_GET_DATA_TYPE(dmt->type) == CCV_32S ||
				   CCV_GET_DATA_TYPE(dmt->type) == CCV_32F ||
				   CCV_GET_DATA_TYPE(dmt->type) == CCV_64S ||
				   CCV_GET_DATA_TYPE(dmt->type) == CCV_64F);
			size_t size = ccv_compute_dense_matrix_size(dmt->rows, dmt->cols, dmt->type);
			ccv_cache_put(&ccv_cache, dmt->sig, dmt, size, 0 /* type 0 */);
		}
	} else if (type & CCV_MATRIX_SPARSE) {
		ccv_sparse_matrix_t* smt = (ccv_sparse_matrix_t*)mat;
		int i;
		for (i = 0; i < smt->size; i++)
		{
			if (smt->index[i].ifbit > 1)
				ccfree(smt->vector[i].index); // It is a union of index / data, can just free them.
		}
		ccfree(smt->index);
		ccfree(smt->vector);
		ccfree(smt);
	} else if ((type & CCV_MATRIX_CSR) || (type & CCV_MATRIX_CSC)) {
		ccv_compressed_sparse_matrix_t* csm = (ccv_compressed_sparse_matrix_t*)mat;
		csm->refcount = 0;
		ccfree(csm);
	}
}

ccv_array_t* ccv_array_new(int rsize, int rnum, uint64_t sig)
{
	ccv_array_t* array;
	if (ccv_cache_opt && sig != 0)
	{
		uint8_t type;
		array = (ccv_array_t*)ccv_cache_out(&ccv_cache, sig, &type);
		if (array)
		{
			assert(type == 1);
			array->type |= CCV_GARBAGE;
			array->refcount = 1;
			return array;
		}
	}
	array = (ccv_array_t*)ccmalloc(sizeof(ccv_array_t));
	array->sig = sig;
	array->type = CCV_REUSABLE & ~CCV_GARBAGE;
	array->rnum = 0;
	array->rsize = rsize;
	array->size = ccv_max(rnum, 2 /* allocate memory for at least 2 items */);
	array->data = ccmalloc((size_t)array->size * (size_t)rsize);
	return array;
}

void ccv_make_array_mutable(ccv_array_t* array)
{
	array->sig = 0;
	array->type &= ~CCV_REUSABLE;
}

void ccv_make_array_immutable(ccv_array_t* array)
{
	assert(array->sig == 0);
	array->type &= ~CCV_REUSABLE;
	/* TODO: trim the array */
	array->sig = ccv_cache_generate_signature(array->data, array->size * array->rsize, (uint64_t)array->rsize, CCV_EOF_SIGN);
}

void ccv_array_free_immediately(ccv_array_t* array)
{
	array->refcount = 0;
	ccfree(array->data);
	ccfree(array);
}

void ccv_array_free(ccv_array_t* array)
{
	if (!ccv_cache_opt || !(array->type & CCV_REUSABLE) || array->sig == 0)
	{
		array->refcount = 0;
		ccfree(array->data);
		ccfree(array);
	} else {
		size_t size = sizeof(ccv_array_t) + array->size * array->rsize;
		ccv_cache_put(&ccv_cache, array->sig, array, size, 1 /* type 1 */);
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
	ccv_cache_init(&ccv_cache, size, 2, ccv_matrix_free_immediately, ccv_array_free_immediately);
}

void ccv_enable_default_cache(void)
{
	ccv_enable_cache(CCV_DEFAULT_CACHE_SIZE);
}

static uint8_t key_siphash[16] = "libccvky4siphash";

uint64_t ccv_cache_generate_signature(const char* msg, int len, uint64_t sig_start, ...)
{
	uint64_t sig_out, sig_in[2]; // 1 is in, 0 is out
	siphash((uint8_t*)&sig_out, (const uint8_t*)msg, len, key_siphash);
	va_list arguments;
	va_start(arguments, sig_start);
	sig_in[0] = sig_out;
	sig_in[1] = sig_start;
	while (sig_in[1] != 0)
	{
		siphash((uint8_t*)&sig_out, (const uint8_t*)sig_in, sizeof(uint64_t) * 2, key_siphash);
		sig_in[0] = sig_out;
		sig_in[1] = va_arg(arguments, uint64_t);
	}
	va_end(arguments);
	return sig_out;
}
