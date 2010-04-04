#include "ccv.h"
#include "3rdparty/sha1.h"

typedef struct ccv_memory_index_t {
	int i;
	ccv_dense_matrix_t* m;
	struct ccv_memory_index_t* con;
} ccv_memory_index_t;
 
typedef struct {
	uint64_t rnum;
	ccv_memory_index_t* con;
} ccv_memory_t;

static int __ccv_bits_in_16bits[0x1u << 16];
static int __ccv_bits_in_16bits_init = 0;

static int __ccv_sparse_bitcount(unsigned int n)
{
	int count = 0;
	while (n)
	{
		++count;
		n &= (n - 1);
	}
	return count;
}

static void __ccv_precomputed_16bits()
{
	int i;
	for (i = 0; i < (0x1u << 16); ++i)
		__ccv_bits_in_16bits[i] = __ccv_sparse_bitcount(i);
	__ccv_bits_in_16bits_init = 1;
}

static int __ccv_sig_match(int* sig1, int* sig2, int k)
{
	for (; k < 5; ++k)
		if (sig1[k] != sig2[k])
			return 0;
	return 1;
}

static int __ccv_memory_add_matrix_cache(ccv_memory_t* memory, ccv_dense_matrix_t* m)
{
	if (!__ccv_bits_in_16bits_init)
		__ccv_precomputed_16bits();
	unsigned char* ucsig = (unsigned char*)m->sig;
	if (memory->rnum == 0)
	{
		if (memory->con == NULL)
			memory->con = (ccv_memory_index_t*)malloc(sizeof(ccv_memory_index_t));
		memory->con->i = 0;
		memory->con->m = m;
		memory->con->con = NULL;
	} else {
		ccv_memory_index_t* con = memory->con;
		int k = 0;
		int spot = ucsig[0] & 0xf;
		while ((con->i & (1 << spot)) && (k < 40))
		{
			if (__ccv_sig_match(m->sig, con->m->sig, k >> 2))
				return 0;
			int p = __ccv_bits_in_16bits[con->i & ((1 << spot) - 1)];
			con = con->con + p;
			++k;
			spot = (ucsig[k >> 1] >> ((k & 0x1) << 2)) & 0xf;
		}
		if (k >= 40)
			return 0;
		if (con->con == NULL)
		{
			con->i = (1 << spot) | 0x10000;
			con->con = (ccv_memory_index_t*)malloc(sizeof(ccv_memory_index_t));
			con->con->i = 0;
			con->con->m = m;
			con->con->con = NULL;
		} else {
			int p = __ccv_bits_in_16bits[con->i & ((1 << spot) - 1)];
			int c = con->i >> 16;
			con->i |= (1 << spot);
			con->i += 0x10000;
			con->con = (ccv_memory_index_t*)realloc(con->con, sizeof(ccv_memory_index_t) * (c + 1));
			int i;
			for (i = c; i > p; --i)
				con->con[i] = con->con[i - 1];
			con->con[p].i = 0;
			con->con[p].m = m;
			con->con[p].con = NULL;
		}
	}
	++memory->rnum;
	return 1;
}

static ccv_dense_matrix_t* __ccv_memory_get_and_remove_matrix_cache(ccv_memory_t* memory, int* sig)
{
	if (memory->rnum == 0)
		return NULL;
	if (memory->rnum == 1)
	{
		if (__ccv_sig_match(sig, memory->con->m->sig, 0))
		{
			memory->rnum = 0;
			ccv_dense_matrix_t* m = memory->con->m;
			free(memory->con);
			memory->con = NULL;
			return m;
		} else
			return NULL;
	}
	if (!__ccv_bits_in_16bits_init)
		__ccv_precomputed_16bits();
	unsigned char* ucsig = (unsigned char*)sig;
	ccv_memory_index_t* con = memory->con;
	ccv_memory_index_t* pcon = NULL;
	ccv_memory_index_t* found = NULL;
	int k = 0;
	int spot = ucsig[0] & 0xf;
	if (!__ccv_sig_match(sig, con->m->sig, 0))
	{
		while ((con->i & (1 << spot)) && (k < 40))
		{
			int p = __ccv_bits_in_16bits[con->i & ((1 << spot) - 1)];
			pcon = con;
			con = con->con + p;
			++k;
			if (__ccv_sig_match(sig, con->m->sig, k >> 2))
			{
				found = con;
				break;
			}
			spot = (ucsig[k >> 1] >> ((k & 0x1) << 2)) & 0xf;
		}
	} else
		found = con;
	if (!found)
		return NULL;
	ccv_dense_matrix_t* m = found->m;
	while (((con->i >> 16) != 0) && (k < 40))
	{
		ccv_memory_index_t* tcon = NULL;
		int i, max = con->i >> 16;
		ccv_memory_index_t* coni = con->con;
		for (i = 0; i < max; ++i)
		{
			if ((coni->i >> 16) == 0)
			{
				tcon = coni;
				break;
			}
			if (tcon == NULL)
				tcon = coni;
			++coni;
		}
		pcon = con;
		con = tcon;
		++k;
	}
	found->m = con->m;
	unsigned char* mcsig = (unsigned char*)con->m->sig;
	--k;
	spot = (mcsig[k >> 1] >> ((k & 0x1) << 2)) & 0xf;
	pcon->i -= 0x10000;
	pcon->i &= ~(1 << spot);
	int p = __ccv_bits_in_16bits[pcon->i & ((1 << spot) - 1)];
	int c = pcon->i >> 16;
	if (c > 0)
	{
		int i;
		for (i = p; i < c; ++i)
			pcon->con[i] = pcon->con[i + 1];
		pcon->con = (ccv_memory_index_t*)realloc(pcon->con, sizeof(ccv_memory_index_t) * c);
	} else {
		free(pcon->con);
		pcon->con = NULL;
	}
	--memory->rnum;
	return m;
}

static ccv_memory_t memory = { 0, NULL };

ccv_dense_matrix_t* ccv_dense_matrix_new(int rows, int cols, int type, void* data, int* sig)
{
	ccv_dense_matrix_t* mat;
	if (sig != NULL)
	{
		mat = __ccv_memory_get_and_remove_matrix_cache(&memory, sig);
		if (mat)
		{
			mat->type |= CCV_GARBAGE;
			mat->refcount = 1;
			return mat;
		}
	}
	mat = (ccv_dense_matrix_t*)malloc((data) ? sizeof(ccv_dense_matrix_t) : (sizeof(ccv_dense_matrix_t) + ((cols * CCV_GET_DATA_TYPE_SIZE(type) * CCV_GET_CHANNEL_NUM(type) + 3) & -4) * rows));
	if (sig != NULL)
	{
		memcpy(mat->sig, sig, 20);
		mat->type |= CCV_REUSABLE;
	} else
		memset(mat->sig, 0, 20);
	mat->type = (type | CCV_MATRIX_DENSE) & ~CCV_GARBAGE;
	mat->rows = rows;
	mat->cols = cols;
	mat->step = (cols * CCV_GET_DATA_TYPE_SIZE(type) * CCV_GET_CHANNEL_NUM(type) + 3) & -4;
	mat->refcount = 1;
	mat->data.ptr = (data) ? (unsigned char*)data : (unsigned char*)(mat + 1);
	return mat;
}

ccv_dense_matrix_t ccv_dense_matrix(int rows, int cols, int type, void* data, int* sig)
{
	ccv_dense_matrix_t mat;
	if (sig != NULL)
		memcpy(mat.sig, sig, 20);
	else
		memset(mat.sig, 0, 20);
	mat.type = (type | CCV_MATRIX_DENSE) & ~CCV_GARBAGE;
	mat.rows = rows;
	mat.cols = cols;
	mat.step = (cols * CCV_GET_DATA_TYPE_SIZE(type) * CCV_GET_CHANNEL_NUM(type) + 3) & -4;
	mat.refcount = 1;
	mat.data.ptr = (unsigned char*)data;
	return mat;
}

ccv_sparse_matrix_t* ccv_sparse_matrix_new(int rows, int cols, int type, int major, int* sig)
{
	ccv_sparse_matrix_t* mat;
	mat = (ccv_sparse_matrix_t*)malloc(sizeof(ccv_sparse_matrix_t));
	mat->rows = rows;
	mat->cols = cols;
	mat->type = type | CCV_MATRIX_SPARSE | ((type & CCV_DENSE_VECTOR) ? CCV_DENSE_VECTOR : CCV_SPARSE_VECTOR);
	mat->major = major;
	mat->prime = 0;
	mat->load_factor = 0;
	mat->refcount = 1;
	mat->vector = (ccv_dense_vector_t*)malloc(CCV_GET_SPARSE_PRIME(mat->prime) * sizeof(ccv_dense_vector_t));
	int i;
	for (i = 0; i < CCV_GET_SPARSE_PRIME(mat->prime); i++)
	{
		mat->vector[i].index = -1;
		mat->vector[i].length = 0;
		mat->vector[i].next = NULL;
	}
	return mat;
}

void ccv_matrix_free(ccv_matrix_t* mat)
{
	int type = *(int*)mat;
	if (type & CCV_MATRIX_DENSE)
	{
		ccv_dense_matrix_t* dmt = (ccv_dense_matrix_t*)mat;
		dmt->refcount = 0;
		if (!(dmt->type & CCV_REUSABLE) || CCV_IS_EMPTY_SIGNATURE(dmt) || !__ccv_memory_add_matrix_cache(&memory, dmt))
			free(dmt);
	} else if (type & CCV_MATRIX_SPARSE) {
		ccv_sparse_matrix_t* smt = (ccv_sparse_matrix_t*)mat;
		int i;
		for (i = 0; i < CCV_GET_SPARSE_PRIME(smt->prime); i++)
			if (smt->vector[i].index != -1)
			{
				ccv_dense_vector_t* iter = &smt->vector[i];
				free(iter->data.ptr);
				iter = iter->next;
				while (iter != NULL)
				{
					ccv_dense_vector_t* iter_next = iter->next;
					free(iter->data.ptr);
					free(iter);
					iter = iter_next;
				}
			}
		free(smt->vector);
		free(smt);
	} else if ((type & CCV_MATRIX_CSR) || (type & CCV_MATRIX_CSC)) {
		ccv_compressed_sparse_matrix_t* csm = (ccv_compressed_sparse_matrix_t*)mat;
		csm->refcount = 0;
		free(csm);
	}
}

void ccv_matrix_generate_signature(const char* msg, int len, int* sig, int* sig_start, ...)
{
	ccv_SHA_CTX ctx;
	ccv_SHA1_Init(&ctx);
	int* sigi;
	va_list arguments;
	va_start(arguments, sig_start);
	for (sigi = sig_start; sigi != NULL; sigi = va_arg(arguments, int*))
		ccv_SHA1_Update(&ctx, sigi, 20);
	va_end(arguments);
	ccv_SHA1_Update(&ctx, msg, len);
	ccv_SHA1_Final((unsigned char*)sig, &ctx);
}

static void __ccv_garbage_collect_impl(ccv_memory_index_t* con)
{
	if (con != NULL)
	{
		int i, max = con->i >> 16;
		if (max > 0)
		{
			ccv_memory_index_t* coni = con->con;
			for (i = 0; i < max; ++i)
			{
				__ccv_garbage_collect_impl(coni);
				++coni;
			}
			free(con->con);
		}
		free(con->m);
	}
}

void ccv_garbage_collect()
{
	if (memory.rnum > 0)
	{
		__ccv_garbage_collect_impl(memory.con);
		free(memory.con);
		memory.con = NULL;
		memory.rnum = 0;
	}
}
