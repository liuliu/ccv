#include "ccv.h"
#include "ccv_internal.h"

ccv_dense_matrix_t* ccv_get_dense_matrix(ccv_matrix_t* mat)
{
	int type = *(int*)mat;
	if (type & CCV_MATRIX_DENSE)
		return (ccv_dense_matrix_t*)mat;
	return 0;
}

ccv_sparse_matrix_t* ccv_get_sparse_matrix(ccv_matrix_t* mat)
{
	int type = *(int*)mat;
	if (type & CCV_MATRIX_SPARSE)
		return (ccv_sparse_matrix_t*)mat;
	return 0;
}

void ccv_visualize(ccv_matrix_t* a, ccv_matrix_t** b, int type)
{
	ccv_dense_matrix_t* da = ccv_get_dense_matrix(a);
	ccv_declare_derived_signature(sig, da->sig != 0, ccv_sign_with_literal("ccv_visualize"), da->sig, CCV_EOF_SIGN);
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, da->rows, da->cols, CCV_8U | CCV_C1, CCV_8U | CCV_C1, sig);
	ccv_object_return_if_cached(, db);
	ccv_dense_matrix_t* dc = 0;
	if (CCV_GET_CHANNEL(da->type) > CCV_C1)
	{
		ccv_flatten(da, (ccv_matrix_t**)&dc, 0, 0);
		da = dc;
	}
	int i, j;
	double minval = DBL_MAX, maxval = -DBL_MAX;
	unsigned char* aptr = da->data.u8;
	unsigned char* bptr = db->data.u8;
#define for_block(_, _for_get) \
	for (i = 0; i < da->rows; i++) \
	{ \
		for (j = 0; j < da->cols; j++) \
		{ \
			minval = ccv_min(minval, _for_get(aptr, j)); \
			maxval = ccv_max(maxval, _for_get(aptr, j)); \
		} \
		aptr += da->step; \
	} \
	aptr = da->data.u8; \
	for (i = 0; i < da->rows; i++) \
	{ \
		for (j = 0; j < da->cols; j++) \
			bptr[j] = ccv_clamp((_for_get(aptr, j) - minval) * 255.0 / (maxval - minval), 0, 255); \
		aptr += da->step; \
		bptr += db->step; \
	}
	ccv_matrix_getter(da->type, for_block);
#undef for_block
	if (dc != 0)
		ccv_matrix_free(dc);
}

void ccv_zero(ccv_matrix_t* mat)
{
	ccv_dense_matrix_t* dmt = ccv_get_dense_matrix(mat);
	memset(dmt->data.u8, 0, dmt->step * dmt->rows);
}

int ccv_any_nan(ccv_matrix_t *a)
{
	ccv_dense_matrix_t* da = ccv_get_dense_matrix(a);
	assert((da->type & CCV_32F) || (da->type & CCV_64F));
	int ch = CCV_GET_CHANNEL(da->type);
	int i;
	if (da->type & CCV_32F)
	{
		for (i = 0; i < da->rows * da->cols * ch; i++)
#ifdef isnanf
			if (isnanf(da->data.f32[i]))
#else
			if (isnan(da->data.f32[i]))
#endif
				return i + 1;
	} else {
		for (i = 0; i < da->rows * da->cols * ch; i++)
			if (isnan(da->data.f64[i]))
				return i + 1;
	}
	return 0;
}

void ccv_flatten(ccv_matrix_t* a, ccv_matrix_t** b, int type, int flag)
{
	ccv_dense_matrix_t* da = ccv_get_dense_matrix(a);
	ccv_declare_derived_signature(sig, da->sig != 0, ccv_sign_with_format(64, "ccv_flatten(%d)", flag), da->sig, CCV_EOF_SIGN);
	int no_8u_type = (da->type & CCV_8U) ? CCV_32S : da->type;
	type = (type == 0) ? CCV_GET_DATA_TYPE(no_8u_type) | CCV_C1 : CCV_GET_DATA_TYPE(type) | CCV_C1;
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, da->rows, da->cols, CCV_ALL_DATA_TYPE | CCV_C1, type, sig);
	ccv_object_return_if_cached(, db);
	int i, j, k, ch = CCV_GET_CHANNEL(da->type);
	unsigned char* aptr = da->data.u8;
	unsigned char* bptr = db->data.u8;
#define for_block(_for_get, _for_type, _for_set) \
	for (i = 0; i < da->rows; i++) \
	{ \
		for (j = 0; j < da->cols; j++) \
		{ \
			_for_type sum = 0; \
			for (k = 0; k < ch; k++) \
				sum += _for_get(aptr, j * ch + k); \
			_for_set(bptr, j, sum); \
		} \
		aptr += da->step; \
		bptr += db->step; \
	}
	ccv_matrix_getter(da->type, ccv_matrix_typeof_setter, db->type, for_block);
#undef for_block
}

void ccv_border(ccv_matrix_t* a, ccv_matrix_t** b, int type, ccv_margin_t margin)
{
	ccv_dense_matrix_t* da = ccv_get_dense_matrix(a);
	ccv_declare_derived_signature(sig, da->sig != 0, ccv_sign_with_format(64, "ccv_border(%d,%d,%d,%d)", margin.left, margin.top, margin.right, margin.bottom), da->sig, CCV_EOF_SIGN);
	int ch = CCV_GET_CHANNEL(da->type);
	type = (type == 0) ? CCV_GET_DATA_TYPE(da->type) | ch : CCV_GET_DATA_TYPE(type) | ch;
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, da->rows + margin.top + margin.bottom, da->cols + margin.left + margin.right, CCV_ALL_DATA_TYPE | ch, type, sig);
	ccv_object_return_if_cached(, db);
	ccv_zero(db);
	int i, j;
	unsigned char* aptr = da->data.u8;
	unsigned char* bptr = db->data.u8 + margin.top * db->step + margin.left * ch * CCV_GET_DATA_TYPE_SIZE(db->type);
	if (CCV_GET_DATA_TYPE(da->type) == CCV_GET_DATA_TYPE(db->type))
	{
		// use memcpy to speedup
		size_t scan = da->cols * ch * CCV_GET_DATA_TYPE_SIZE(da->type);
		for (i = 0; i < da->rows; i++)
		{
			memcpy(bptr, aptr, scan);
			aptr += da->step;
			bptr += db->step;
		}
	} else {
#define for_block(_for_get, _for_type, _for_set) \
		for (i = 0; i < da->rows; i++) \
		{ \
			for (j = 0; j < da->cols * ch; j++) \
			{ \
				_for_set(bptr, j, _for_get(aptr, j)); \
			} \
			aptr += da->step; \
			bptr += db->step; \
		}
		ccv_matrix_getter(da->type, ccv_matrix_typeof_setter, db->type, for_block);
#undef for_block
	}
}

void ccv_shift(ccv_matrix_t* a, ccv_matrix_t** b, int type, int lr, int rr)
{
	ccv_dense_matrix_t* da = ccv_get_dense_matrix(a);
	ccv_declare_derived_signature(sig, da->sig != 0, ccv_sign_with_format(64, "ccv_shift(%d,%d)", lr, rr), da->sig, CCV_EOF_SIGN);
	type = (type == 0) ? CCV_GET_DATA_TYPE(da->type) | CCV_GET_CHANNEL(da->type) : CCV_GET_DATA_TYPE(type) | CCV_GET_CHANNEL(da->type);
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, da->rows, da->cols, CCV_ALL_DATA_TYPE | CCV_GET_CHANNEL(da->type), type, sig); 
	ccv_object_return_if_cached(, db);
	int i, j, ch = CCV_GET_CHANNEL(da->type);
	unsigned char* aptr = da->data.u8;
	unsigned char* bptr = db->data.u8;
#define for_block(_for_get, _for_set) \
	for (i = 0; i < da->rows; i++) \
	{ \
		for (j = 0; j < da->cols * ch; j++) \
		{ \
			_for_set(bptr, j, _for_get(aptr, j, lr), rr); \
		} \
		aptr += da->step; \
		bptr += db->step; \
	}
	ccv_matrix_getter(da->type, ccv_matrix_setter, db->type, for_block);
#undef for_block
}

// From https://github.com/skarupke/flat_hash_map with anything overflows 32bit removed.
static uint32_t _ccv_sparse_matrix_index_for_hash(const uint32_t hash, const int prime_index)
{
	switch(prime_index)
	{
	case 0:
		return 0u;
	case 1:
		return hash % 2u;
	case 2:
		return hash % 3u;
	case 3:
		return hash % 5u;
	case 4:
		return hash % 7u;
	case 5:
		return hash % 11u;
	case 6:
		return hash % 13u;
	case 7:
		return hash % 17u;
	case 8:
		return hash % 23u;
	case 9:
		return hash % 29u;
	case 10:
		return hash % 37u;
	case 11:
		return hash % 47u;
	case 12:
		return hash % 59u;
	case 13:
		return hash % 73u;
	case 14:
		return hash % 97u;
	case 15:
		return hash % 127u;
	case 16:
		return hash % 151u;
	case 17:
		return hash % 197u;
	case 18:
		return hash % 251u;
	case 19:
		return hash % 313u;
	case 20:
		return hash % 397u;
	case 21:
		return hash % 499u;
	case 22:
		return hash % 631u;
	case 23:
		return hash % 797u;
	case 24:
		return hash % 1009u;
	case 25:
		return hash % 1259u;
	case 26:
		return hash % 1597u;
	case 27:
		return hash % 2011u;
	case 28:
		return hash % 2539u;
	case 29:
		return hash % 3203u;
	case 30:
		return hash % 4027u;
	case 31:
		return hash % 5087u;
	case 32:
		return hash % 6421u;
	case 33:
		return hash % 8089u;
	case 34:
		return hash % 10193u;
	case 35:
		return hash % 12853u;
	case 36:
		return hash % 16193u;
	case 37:
		return hash % 20399u;
	case 38:
		return hash % 25717u;
	case 39:
		return hash % 32401u;
	case 40:
		return hash % 40823u;
	case 41:
		return hash % 51437u;
	case 42:
		return hash % 64811u;
	case 43:
		return hash % 81649u;
	case 44:
		return hash % 102877u;
	case 45:
		return hash % 129607u;
	case 46:
		return hash % 163307u;
	case 47:
		return hash % 205759u;
	case 48:
		return hash % 259229u;
	case 49:
		return hash % 326617u;
	case 50:
		return hash % 411527u;
	case 51:
		return hash % 518509u;
	case 52:
		return hash % 653267u;
	case 53:
		return hash % 823117u;
	case 54:
		return hash % 1037059u;
	case 55:
		return hash % 1306601u;
	case 56:
		return hash % 1646237u;
	case 57:
		return hash % 2074129u;
	case 58:
		return hash % 2613229u;
	case 59:
		return hash % 3292489u;
	case 60:
		return hash % 4148279u;
	case 61:
		return hash % 5226491u;
	case 62:
		return hash % 6584983u;
	case 63:
		return hash % 8296553u;
	case 64:
		return hash % 10453007u;
	case 65:
		return hash % 13169977u;
	case 66:
		return hash % 16593127u;
	case 67:
		return hash % 20906033u;
	case 68:
		return hash % 26339969u;
	case 69:
		return hash % 33186281u;
	case 70:
		return hash % 41812097u;
	case 71:
		return hash % 52679969u;
	case 72:
		return hash % 66372617u;
	case 73:
		return hash % 83624237u;
	case 74:
		return hash % 105359939u;
	case 75:
		return hash % 132745199u;
	case 76:
		return hash % 167248483u;
	case 77:
		return hash % 210719881u;
	case 78:
		return hash % 265490441u;
	case 79:
		return hash % 334496971u;
	case 80:
		return hash % 421439783u;
	case 81:
		return hash % 530980861u;
	case 82:
		return hash % 668993977u;
	case 83:
		return hash % 842879579u;
	case 84:
		return hash % 1061961721u;
	case 85:
		return hash % 1337987929u;
	case 86:
		return hash % 1685759167u;
	case 87:
		return hash % 2123923447u;
	case 88:
		return hash % 2675975881u;
	case 89:
		return hash % 3371518343u;
	case 90:
		return hash % 4247846927u;
	case 91:
		return hash % 4294967291u;
	default:
		return hash;
	}
}

// prime numbers generated by the following method:
// 1. start with a prime p = 2
// 2. go to wolfram alpha and get p = NextPrime(2 * p)
// 3. repeat 2. until you overflow 32 bits
// you now have large gaps which you would hit if somebody called reserve() with an unlucky number.
// 4. to fill the gaps for every prime p go to wolfram alpha and get ClosestPrime(p * 2^(1/3)) and ClosestPrime(p * 2^(2/3)) and put those in the gaps
// 5. get PrevPrime(2^32) and put it at the end
const static uint32_t _ccv_sparse_prime[] =
{
	2u, 3u, 5u, 7u, 11u, 13u, 17u, 23u, 29u, 37u, 47u,
	59u, 73u, 97u, 127u, 151u, 197u, 251u, 313u, 397u,
	499u, 631u, 797u, 1009u, 1259u, 1597u, 2011u, 2539u,
	3203u, 4027u, 5087u, 6421u, 8089u, 10193u, 12853u, 16193u,
	20399u, 25717u, 32401u, 40823u, 51437u, 64811u, 81649u,
	102877u, 129607u, 163307u, 205759u, 259229u, 326617u,
	411527u, 518509u, 653267u, 823117u, 1037059u, 1306601u,
	1646237u, 2074129u, 2613229u, 3292489u, 4148279u, 5226491u,
	6584983u, 8296553u, 10453007u, 13169977u, 16593127u, 20906033u,
	26339969u, 33186281u, 41812097u, 52679969u, 66372617u,
	83624237u, 105359939u, 132745199u, 167248483u, 210719881u,
	265490441u, 334496971u, 421439783u, 530980861u, 668993977u,
	842879579u, 1061961721u, 1337987929u, 1685759167u, 2123923447u,
	2675975881u, 3371518343u, 4247846927u, 4294967291u
};

static uint8_t _ccv_sparse_matrix_index_next_size_over(const uint32_t size)
{
	int i;
	assert(sizeof(_ccv_sparse_prime) / sizeof(uint32_t) == 91);
	for (i = 0; i < sizeof(_ccv_sparse_prime) / sizeof(uint32_t); i++)
		if (_ccv_sparse_prime[i] > size)
			return i + 1;
	return sizeof(_ccv_sparse_prime) / sizeof(uint32_t) + 1;
}

static uint32_t _ccv_sparse_matrix_size_for_index(const int prime_index)
{
	return _ccv_sparse_prime[prime_index - 1];
}

static void _ccv_init_sparse_matrix_vector(ccv_sparse_matrix_vector_t* vector, const ccv_sparse_matrix_t* const mat)
{
	if (mat->type & CCV_DENSE_VECTOR)
	{
		const int vlen = (mat->major == CCV_SPARSE_COL_MAJOR) ? mat->rows : mat->cols;
		const size_t cell_size = CCV_GET_DATA_TYPE_SIZE(mat->type) * CCV_GET_CHANNEL(mat->type);
		vector->size = vlen;
		vector->data.u8 = (uint8_t*)cccalloc(1, (cell_size * vlen + 15) & -16);
		return;
	}
	vector->prime_index = 1; // See ccv_util.c to know why this is 1 and why size is 2.
	vector->size = 2;
	vector->rnum = 0;
	const size_t cell_size_aligned = (CCV_GET_DATA_TYPE_SIZE(mat->type) * CCV_GET_CHANNEL(mat->type) + 3) & -4;
	vector->index = (ccv_sparse_matrix_index_t*)cccalloc(sizeof(ccv_sparse_matrix_index_t) + cell_size_aligned, vector->size);
}

static inline void _ccv_move_sparse_matrix_vector(ccv_sparse_matrix_index_t* const index, ccv_sparse_matrix_vector_t* const vector, const ccv_sparse_matrix_t* const mat, uint32_t k, uint32_t idx, int i, const uint32_t size, const int prime_index)
{
	uint32_t old_i = index[idx].i;
	index[idx].i = i;
	i = old_i;
	ccv_sparse_matrix_vector_t val = vector[idx];
	_ccv_init_sparse_matrix_vector(vector + idx, mat);
	// Move to next idx, this is already occupied.
	++k;
	++idx;
	for (; k < 255; ++idx, ++k)
	{
		if (idx >= size)
			idx = 0;
		if (!index[idx].ifbit)
		{
			index[idx].ifbit = k;
			index[idx].i = i;
			vector[idx] = val;
			return;
		}
		uint32_t j = index[idx].ifbit;
		if (k > j)
		{
			index[idx].ifbit = k;
			k = j;
			uint32_t old_i = index[idx].i;
			index[idx].i = i;
			i = old_i;
			ccv_sparse_matrix_vector_t old_val = vector[idx];
			vector[idx] = val;
			val = old_val;
		}
	}
	for (;; ++idx, ++k)
	{
		if (idx >= size)
			idx = 0;
		if (!index[idx].ifbit)
		{
			index[idx].ifbit = k > 0xff ? 0xff : k;
			index[idx].i = i;
			vector[idx] = val;
			return;
		}
		uint32_t j = index[idx].ifbit;
		if (j == 0xff)
			j = (size - _ccv_sparse_matrix_index_for_hash(index[idx].i + size - idx, prime_index)) + 2;
		if (k > j)
		{
			index[idx].ifbit = k > 0xff ? 0xff : k;
			k = j;
			uint32_t old_i = index[idx].i;
			index[idx].i = i;
			i = old_i;
			ccv_sparse_matrix_vector_t old_val = vector[idx];
			vector[idx] = val;
			val = old_val;
		}
	}
}

static ccv_sparse_matrix_vector_t* _ccv_get_or_add_sparse_matrix_vector(ccv_sparse_matrix_t* mat, int i)
{
	const uint32_t size = mat->size;
	ccv_sparse_matrix_index_t* const index = mat->index;
	ccv_sparse_matrix_vector_t* const vector = mat->vector;
	const int prime_index = mat->prime_index;
	uint32_t idx = _ccv_sparse_matrix_index_for_hash(i, prime_index);
	uint32_t k = 2;
	for (; k < 255; ++idx, ++k)
	{
		if (idx >= size)
			idx = 0;
		uint32_t j = index[idx].ifbit;
		if (k > j)
		{
			++mat->rnum;
			index[idx].ifbit = k;
			if (!j)
			{
				index[idx].i = i;
				_ccv_init_sparse_matrix_vector(vector + idx, mat);
			} else
				_ccv_move_sparse_matrix_vector(index, vector, mat, j /* This is j not k because we are replacing it. */, idx, i, size, prime_index);
			return vector + idx;
		}
		if (index[idx].i == i)
			return vector + idx;
	}
	// Above or equal to 255, we need to fetch the key to recompute the distance every time now.
	for (;; ++idx, ++k)
	{
		if (idx >= size)
			idx = 0;
		uint32_t j = index[idx].ifbit;
		if (j == 0xff)
			j = (size - _ccv_sparse_matrix_index_for_hash(index[idx].i + size - idx, prime_index)) + 2;
		if (k > j)
		{
			++mat->rnum;
			index[idx].ifbit = k > 0xff ? 0xff : k;
			if (!j)
			{
				index[idx].i = i;
				_ccv_init_sparse_matrix_vector(vector + idx, mat);
			} else
				_ccv_move_sparse_matrix_vector(index, vector, mat, j /* This is j not k because we are replacing it. */, idx, i, size, prime_index);
			return vector + idx;
		}
		if (index[idx].i == i)
			return vector + idx;
	}
	return 0;
}

ccv_sparse_matrix_vector_t* ccv_get_sparse_matrix_vector(const ccv_sparse_matrix_t* mat, int i)
{
	const uint32_t size = mat->size;
	ccv_sparse_matrix_index_t* index = mat->index;
	ccv_sparse_matrix_vector_t* vector = mat->vector;
	const int prime_index = mat->prime_index;
	uint32_t idx = _ccv_sparse_matrix_index_for_hash(i, prime_index);
	uint32_t k = 2;
	for (; k < 255; ++idx, ++k)
	{
		if (idx >= size)
			idx = 0;
		if (k > index[idx].ifbit)
			return 0;
		if (index[idx].i == i)
			return vector + idx;
	}
	// Above or equal to 255, we need to fetch the key to recompute the distance every time now.
	for (;; ++idx, ++k)
	{
		if (idx >= size)
			idx = 0;
		uint32_t j = index[idx].ifbit;
		if (j == 0xff)
			j = (size - _ccv_sparse_matrix_index_for_hash(index[idx].i + size - idx, prime_index)) + 2;
		if (k > j)
			return 0;
		if (index[idx].i == i)
			return vector + idx;
	}
	return 0;
}

ccv_numeric_data_t ccv_get_sparse_matrix_cell_from_vector(const ccv_sparse_matrix_t* mat, ccv_sparse_matrix_vector_t* vector, int vidx)
{
	ccv_numeric_data_t cell = {}; // zero-init.
	const size_t cell_size = CCV_GET_DATA_TYPE_SIZE(mat->type) * CCV_GET_CHANNEL(mat->type);
	if (mat->type & CCV_DENSE_VECTOR)
	{
		cell.u8 = vector->data.u8 + cell_size * vidx;
		return cell;
	}
	const size_t cell_size_aligned = (CCV_GET_DATA_TYPE_SIZE(mat->type) * CCV_GET_CHANNEL(mat->type) + 3) & -4;
	const size_t index_size = sizeof(ccv_sparse_matrix_index_t) + cell_size_aligned;
	const uint32_t size = vector->size;
	uint8_t* const index = (uint8_t*)vector->index;
	const int prime_index = vector->prime_index;
	uint32_t idx = _ccv_sparse_matrix_index_for_hash(vidx, prime_index);
	uint32_t k = 2;
	for (; k < 255; ++idx, ++k)
	{
		if (idx >= size)
			idx = 0;
		ccv_sparse_matrix_index_t* const index_idx = (ccv_sparse_matrix_index_t*)(index + index_size * idx);
		if (k > index_idx->ifbit)
			return cell;
		if (index_idx->i == vidx)
		{
			cell.u8 = (uint8_t*)(index_idx + 1);
			return cell;
		}
	}
	// Above or equal to 255, we need to fetch the key to recompute the distance every time now.
	for (;; ++idx, ++k)
	{
		if (idx >= size)
			idx = 0;
		ccv_sparse_matrix_index_t* const index_idx = (ccv_sparse_matrix_index_t*)(index + index_size * idx);
		uint32_t j = index_idx->ifbit;
		if (j == 0xff)
			j = (size - _ccv_sparse_matrix_index_for_hash(index_idx->i + size - idx, prime_index)) + 2;
		if (k > j)
			return cell;
		if (index_idx->i == vidx)
		{
			cell.u8 = (uint8_t*)(index_idx + 1);
			return cell;
		}
	}
}

ccv_numeric_data_t ccv_get_sparse_matrix_cell(const ccv_sparse_matrix_t* mat, int row, int col)
{
	ccv_sparse_matrix_vector_t* vector = ccv_get_sparse_matrix_vector(mat, (mat->major == CCV_SPARSE_COL_MAJOR) ? col : row);
	if (!vector || !vector->rnum)
		return (ccv_numeric_data_t){};
	const int vidx = (mat->major == CCV_SPARSE_COL_MAJOR) ? row : col;
	return ccv_get_sparse_matrix_cell_from_vector(mat, vector, vidx);
}

static void _ccv_sparse_matrix_vector_inc_size(const ccv_sparse_matrix_t* const mat, ccv_sparse_matrix_vector_t* const vector)
{
	assert(vector->prime_index >= 0 && vector->prime_index < sizeof(_ccv_sparse_prime) / sizeof(int));
	const uint32_t size = vector->size;
	const int prime_index = vector->prime_index = _ccv_sparse_matrix_index_next_size_over(size * 2);
	const uint32_t new_size = vector->size = _ccv_sparse_matrix_size_for_index(vector->prime_index);
	assert(vector->prime_index >= 0 && vector->prime_index < sizeof(_ccv_sparse_prime) / sizeof(int));
	const size_t cell_size_aligned = (CCV_GET_DATA_TYPE_SIZE(mat->type) * CCV_GET_CHANNEL(mat->type) + 3) & -4;
	const size_t index_size = sizeof(ccv_sparse_matrix_index_t) + cell_size_aligned;
	const size_t cell_rnum = cell_size_aligned / sizeof(uint32_t);
	vector->index = (ccv_sparse_matrix_index_t*)ccrealloc(vector->index, index_size * new_size);
	uint8_t* const index = (uint8_t*)vector->index;
	memset(index + index_size * size, 0, index_size * (new_size - size));
	uint32_t i, h;
	for (i = 0; i < size; i++)
	{
		ccv_sparse_matrix_index_t* index_i = (ccv_sparse_matrix_index_t*)(index + index_size * i);
		index_i->ifbit = !!index_i->ifbit; // Mark this as need to be moved around.
	}
	for (i = 0; i < size; i++)
	{
		ccv_sparse_matrix_index_t* index_i = (ccv_sparse_matrix_index_t*)(index + index_size * i);
		if (index_i->ifbit == 1) // Encountered one need to be moved.
		{
			index_i->ifbit = 0;
			// This item is from old hash table, need to find a new location for it.
			uint32_t key = index_i->i;
			uint32_t* val = (uint32_t*)(index_i + 1);
			uint32_t k = 2;
			uint32_t idx = _ccv_sparse_matrix_index_for_hash(key, prime_index);
			for (; k < 255; ++idx, ++k)
			{
				if (idx >= new_size)
					idx = 0;
				ccv_sparse_matrix_index_t* index_idx = (ccv_sparse_matrix_index_t*)(index + index_size * idx);
				uint32_t j = index_idx->ifbit;
				if (!j)
				{
					index_idx->ifbit = k;
					index_idx->i = key;
					memcpy(index_idx + 1, val, cell_size_aligned);
					break;
				}
				if (k <= j) // j could be either a valid one or 1, in any case, this will pass.
					continue;
				index_idx->ifbit = k;
				uint32_t old_key = index_idx->i;
				index_idx->i = key;
				key = old_key;
				uint32_t* old_val = (uint32_t*)(index_idx + 1);
				for (h = 0; h < cell_rnum; h++)
				{
					uint32_t v = old_val[h];
					old_val[h] = val[h];
					val[h] = v;
				}
				if (j != 1)
					k = j;
				else { // In this case, I cannot keep going with the idx, need to recompute idx as well restart k.
					idx = _ccv_sparse_matrix_index_for_hash(key, prime_index) - 1;
					k = 1; // Restart.
				}
			}
			if (k < 255)
				continue;
			for (;; ++idx, ++k)
			{
				if (idx >= new_size)
					idx = 0;
				ccv_sparse_matrix_index_t* index_idx = (ccv_sparse_matrix_index_t*)(index + index_size * idx);
				uint32_t j = index_idx->ifbit;
				if (!j)
				{
					index_idx->ifbit = k > 0xff ? 0xff : k;
					index_idx->i = key;
					memcpy(index_idx + 1, val, cell_size_aligned);
					break;
				}
				if (j == 0xff)
					j = (new_size - _ccv_sparse_matrix_index_for_hash((index_idx->i + new_size - idx), prime_index)) + 2; // This is valid because on overflow, unsigned is well defined.
				if (k <= j) // j could be either a valid one or 1, in any case, this will pass.
					continue;
				index_idx->ifbit = k > 0xff ? 0xff : k;
				uint32_t old_key = index_idx->i;
				index_idx->i = key;
				key = old_key;
				uint32_t* old_val = (uint32_t*)(index_idx + 1);
				for (h = 0; h < cell_rnum; h++)
				{
					uint32_t v = old_val[h];
					old_val[h] = val[h];
					val[h] = v;
				}
				if (j != 1)
					k = j;
				else { // In this case, I cannot keep going with the idx, need to recompute idx as well restart k.
					idx = _ccv_sparse_matrix_index_for_hash(key, prime_index) - 1;
					k = 1; // Restart.
				}
			}
		}
	}
}

static void _ccv_sparse_matrix_inc_size(ccv_sparse_matrix_t* mat)
{
	assert(mat->prime_index >= 0 && mat->prime_index < sizeof(_ccv_sparse_prime) / sizeof(int));
	const uint32_t size = mat->size;
	const int prime_index = mat->prime_index = _ccv_sparse_matrix_index_next_size_over(size * 2);
	const uint32_t new_size = mat->size = _ccv_sparse_matrix_size_for_index(mat->prime_index);
	assert(mat->prime_index >= 0 && mat->prime_index < sizeof(_ccv_sparse_prime) / sizeof(int));
	ccv_sparse_matrix_index_t* const index = mat->index = (ccv_sparse_matrix_index_t*)ccrealloc(mat->index, sizeof(ccv_sparse_matrix_index_t) * new_size);
	memset(index + size, 0, sizeof(ccv_sparse_matrix_index_t) * (new_size - size));
	ccv_sparse_matrix_vector_t* const vector = mat->vector = (ccv_sparse_matrix_vector_t*)ccrealloc(mat->vector, sizeof(ccv_sparse_matrix_vector_t) * new_size);
	uint32_t i;
	for (i = 0; i < size; i++)
		index[i].ifbit = !!index[i].ifbit; // Mark this as need to be moved around.
	for (i = 0; i < size; i++)
		if (index[i].ifbit == 1) // Encountered one need to be moved.
		{
			index[i].ifbit = 0;
			// This item is from old hash table, need to find a new location for it.
			uint32_t key = index[i].i;
			ccv_sparse_matrix_vector_t val = vector[i];
			uint32_t k = 2;
			uint32_t idx = _ccv_sparse_matrix_index_for_hash(key, prime_index);
			for (; k < 255; ++idx, ++k)
			{
				if (idx >= new_size)
					idx = 0;
				uint32_t j = index[idx].ifbit;
				if (!j)
				{
					index[idx].ifbit = k;
					index[idx].i = key;
					vector[idx] = val;
					break;
				}
				if (k <= j) // j could be either a valid one or 1, in any case, this will pass.
					continue;
				index[idx].ifbit = k;
				uint32_t old_key = index[idx].i;
				index[idx].i = key;
				key = old_key;
				ccv_sparse_matrix_vector_t old_val = vector[idx];
				vector[idx] = val;
				val = old_val;
				if (j != 1)
					k = j;
				else { // In this case, I cannot keep going with the idx, need to recompute idx as well restart k.
					idx = _ccv_sparse_matrix_index_for_hash(key, prime_index) - 1;
					k = 1; // Restart.
				}
			}
			if (k < 255)
				continue;
			for (;; ++idx, ++k)
			{
				if (idx >= new_size)
					idx = 0;
				uint32_t j = index[idx].ifbit;
				if (!j)
				{
					index[idx].ifbit = k > 0xff ? 0xff : k;
					index[idx].i = key;
					vector[idx] = val;
					break;
				}
				if (j == 0xff)
					j = (new_size - _ccv_sparse_matrix_index_for_hash((index[idx].i + new_size - idx), prime_index)) + 2; // This is valid because on overflow, unsigned is well defined.
				if (k <= j) // j could be either a valid one or 1, in any case, this will pass.
					continue;
				index[idx].ifbit = k > 0xff ? 0xff : k;
				uint32_t old_key = index[idx].i;
				index[idx].i = key;
				key = old_key;
				ccv_sparse_matrix_vector_t old_val = vector[idx];
				vector[idx] = val;
				val = old_val;
				if (j != 1)
					k = j;
				else { // In this case, I cannot keep going with the idx, need to recompute idx as well restart k.
					idx = _ccv_sparse_matrix_index_for_hash(key, prime_index) - 1;
					k = 1; // Restart.
				}
			}
		}
}

static inline void _ccv_move_sparse_matrix_cell(uint8_t* const index, uint32_t k, uint32_t idx, int i, const uint32_t size, const int prime_index, const size_t index_size, const size_t cell_size_aligned)
{
	ccv_sparse_matrix_index_t* const index_idx = (ccv_sparse_matrix_index_t*)(index + index_size * idx);
	const size_t cell_rnum = cell_size_aligned / sizeof(uint32_t);
	uint32_t* val = (uint32_t*)(index_idx + 1);
	uint32_t old_i = index_idx->i;
	index_idx->i = i;
	i = old_i;
	// Move to next idx, this is already occupied.
	++k;
	++idx;
	uint32_t h;
	for (; k < 255; ++idx, ++k)
	{
		if (idx >= size)
			idx = 0;
		ccv_sparse_matrix_index_t* const index_idx = (ccv_sparse_matrix_index_t*)(index + index_size * idx);
		if (!index_idx->ifbit)
		{
			index_idx->ifbit = k;
			index_idx->i = i;
			memcpy(index_idx + 1, val, cell_size_aligned);
			return;
		}
		uint32_t j = index_idx->ifbit;
		if (k > j)
		{
			index_idx->ifbit = k;
			k = j;
			uint32_t old_i = index_idx->i;
			index_idx->i = i;
			i = old_i;
			uint32_t* old_val = (uint32_t*)(index_idx + 1);
			for (h = 0; h < cell_rnum; h++)
			{
				uint32_t v = old_val[h];
				old_val[h] = val[h];
				val[h] = v;
			}
		}
	}
	for (;; ++idx, ++k)
	{
		if (idx >= size)
			idx = 0;
		ccv_sparse_matrix_index_t* const index_idx = (ccv_sparse_matrix_index_t*)(index + index_size * idx);
		if (!index_idx->ifbit)
		{
			index_idx->ifbit = k > 0xff ? 0xff : k;
			index_idx->i = i;
			memcpy(index_idx + 1, val, cell_size_aligned);
			return;
		}
		uint32_t j = index_idx->ifbit;
		if (j == 0xff)
			j = (size - _ccv_sparse_matrix_index_for_hash(index_idx->i + size - idx, prime_index)) + 2;
		if (k > j)
		{
			index_idx->ifbit = k > 0xff ? 0xff : k;
			k = j;
			uint32_t old_i = index_idx->i;
			index_idx->i = i;
			i = old_i;
			uint32_t* old_val = (uint32_t*)(index_idx + 1);
			for (h = 0; h < cell_rnum; h++)
			{
				uint32_t v = old_val[h];
				old_val[h] = val[h];
				val[h] = v;
			}
		}
	}
}

void ccv_set_sparse_matrix_cell_from_vector(ccv_sparse_matrix_t* mat, ccv_sparse_matrix_vector_t* const vector, int vidx, const void* data)
{
	const size_t cell_size = CCV_GET_DATA_TYPE_SIZE(mat->type) * CCV_GET_CHANNEL(mat->type);
	if (mat->type & CCV_DENSE_VECTOR)
	{
		memcpy(vector->data.u8 + vidx * cell_size, data, cell_size);
		return;
	}
	if ((vector->rnum + 1) * 10llu > vector->size * 8llu) // expand when reached 90%.
		_ccv_sparse_matrix_vector_inc_size(mat, vector);
	// Align to 4 bytes.
	const size_t cell_size_aligned = (CCV_GET_DATA_TYPE_SIZE(mat->type) * CCV_GET_CHANNEL(mat->type) + 3) & -4;
	const size_t index_size = sizeof(ccv_sparse_matrix_index_t) + cell_size_aligned;
	uint8_t* const index = (uint8_t*)vector->index;
	const int prime_index = vector->prime_index;
	const uint32_t size = vector->size;
	uint32_t idx = _ccv_sparse_matrix_index_for_hash(vidx, prime_index);
	uint32_t k = 2;
	for (; k < 255; ++idx, ++k)
	{
		if (idx >= size)
			idx = 0;
		ccv_sparse_matrix_index_t* const index_idx = (ccv_sparse_matrix_index_t*)(index + index_size * idx);
		uint32_t j = index_idx->ifbit;
		if (k > j)
		{
			++vector->rnum;
			index_idx->ifbit = k;
			if (!j)
			{
				index_idx->i = vidx;
				// Assign it out.
				memcpy(index_idx + 1, data, cell_size);
			} else {
				_ccv_move_sparse_matrix_cell(index, j /* This is j not k because we are replacing it. */, idx, vidx, size, prime_index, index_size, cell_size_aligned);
				memcpy(index_idx + 1, data, cell_size);
			}
			return;
		}
		if (index_idx->i == vidx)
		{
			memcpy(index_idx + 1, data, cell_size);
			return;
		}
	}
	// Above or equal to 255, we need to fetch the key to recompute the distance every time now.
	for (;; ++idx, ++k)
	{
		if (idx >= size)
			idx = 0;
		ccv_sparse_matrix_index_t* const index_idx = (ccv_sparse_matrix_index_t*)(index + index_size * idx);
		uint32_t j = index_idx->ifbit;
		if (j == 0xff)
			j = (size - _ccv_sparse_matrix_index_for_hash(index_idx->i + size - idx, prime_index)) + 2;
		if (k > j)
		{
			++vector->rnum;
			index_idx->ifbit = k > 0xff ? 0xff : k;
			if (!j)
			{
				index_idx->i = vidx;
				memcpy(index_idx + 1, data, cell_size);
			} else {
				_ccv_move_sparse_matrix_cell(index, j /* This is j not k because we are replacing it. */, idx, vidx, size, prime_index, index_size, cell_size_aligned);
				memcpy(index_idx + 1, data, cell_size);
			}
			return;
		}
		if (index_idx->i == vidx)
		{
			memcpy(index_idx + 1, data, cell_size);
			return;
		}
	}
}

void ccv_set_sparse_matrix_cell(ccv_sparse_matrix_t* mat, int row, int col, const void* data)
{
	assert(data);
	const int hidx = (mat->major == CCV_SPARSE_COL_MAJOR) ? col : row;
	const int vidx = (mat->major == CCV_SPARSE_COL_MAJOR) ? row : col;
	// This will handle initialize a vector if it doesn't already exist.
	if ((mat->rnum + 1) * 10llu > mat->size * 8llu) // expand when reached 90%.
		_ccv_sparse_matrix_inc_size(mat);
	ccv_sparse_matrix_vector_t* const vector = _ccv_get_or_add_sparse_matrix_vector(mat, hidx);
	const size_t cell_size = CCV_GET_DATA_TYPE_SIZE(mat->type) * CCV_GET_CHANNEL(mat->type);
	if (mat->type & CCV_DENSE_VECTOR)
	{
		memcpy(vector->data.u8 + vidx * cell_size, data, cell_size);
		return;
	}
	if ((vector->rnum + 1) * 10llu > vector->size * 8llu) // expand when reached 90%.
		_ccv_sparse_matrix_vector_inc_size(mat, vector);
	// Align to 4 bytes.
	const size_t cell_size_aligned = (CCV_GET_DATA_TYPE_SIZE(mat->type) * CCV_GET_CHANNEL(mat->type) + 3) & -4;
	const size_t index_size = sizeof(ccv_sparse_matrix_index_t) + cell_size_aligned;
	uint8_t* const index = (uint8_t*)vector->index;
	const int prime_index = vector->prime_index;
	const uint32_t size = vector->size;
	uint32_t idx = _ccv_sparse_matrix_index_for_hash(vidx, prime_index);
	uint32_t k = 2;
	for (; k < 255; ++idx, ++k)
	{
		if (idx >= size)
			idx = 0;
		ccv_sparse_matrix_index_t* const index_idx = (ccv_sparse_matrix_index_t*)(index + index_size * idx);
		uint32_t j = index_idx->ifbit;
		if (k > j)
		{
			++vector->rnum;
			index_idx->ifbit = k;
			if (!j)
			{
				index_idx->i = vidx;
				// Assign it out.
				memcpy(index_idx + 1, data, cell_size);
			} else {
				_ccv_move_sparse_matrix_cell(index, j /* This is j not k because we are replacing it. */, idx, vidx, size, prime_index, index_size, cell_size_aligned);
				memcpy(index_idx + 1, data, cell_size);
			}
			return;
		}
		if (index_idx->i == vidx)
		{
			memcpy(index_idx + 1, data, cell_size);
			return;
		}
	}
	// Above or equal to 255, we need to fetch the key to recompute the distance every time now.
	for (;; ++idx, ++k)
	{
		if (idx >= size)
			idx = 0;
		ccv_sparse_matrix_index_t* const index_idx = (ccv_sparse_matrix_index_t*)(index + index_size * idx);
		uint32_t j = index_idx->ifbit;
		if (j == 0xff)
			j = (size - _ccv_sparse_matrix_index_for_hash(index_idx->i + size - idx, prime_index)) + 2;
		if (k > j)
		{
			++vector->rnum;
			index_idx->ifbit = k > 0xff ? 0xff : k;
			if (!j)
			{
				index_idx->i = vidx;
				memcpy(index_idx + 1, data, cell_size);
			} else {
				_ccv_move_sparse_matrix_cell(index, j /* This is j not k because we are replacing it. */, idx, vidx, size, prime_index, index_size, cell_size_aligned);
				memcpy(index_idx + 1, data, cell_size);
			}
			return;
		}
		if (index_idx->i == vidx)
		{
			memcpy(index_idx + 1, data, cell_size);
			return;
		}
	}
}

#define _ccv_indice_less_than(i1, i2, aux) ((i1) < (i2))
#define _ccv_swap_indice_and_uchar_data(i1, i2, array, aux, t) {  \
	unsigned char td = (aux)[(int)(&(i1) - (array))];			  \
	(aux)[(int)(&(i1) - (array))] = (aux)[(int)(&(i2) - (array))]; \
	(aux)[(int)(&(i2) - (array))] = td;							\
	CCV_SWAP(i1, i2, t); }
#define _ccv_swap_indice_and_int_data(i1, i2, array, aux, t) {	\
	int td = (aux)[(int)(&(i1) - (array))];						\
	(aux)[(int)(&(i1) - (array))] = (aux)[(int)(&(i2) - (array))]; \
	(aux)[(int)(&(i2) - (array))] = td;							\
	CCV_SWAP(i1, i2, t); }
#define _ccv_swap_indice_and_float_data(i1, i2, array, aux, t) {  \
	float td = (aux)[(int)(&(i1) - (array))];					  \
	(aux)[(int)(&(i1) - (array))] = (aux)[(int)(&(i2) - (array))]; \
	(aux)[(int)(&(i2) - (array))] = td;							\
	CCV_SWAP(i1, i2, t); }
#define _ccv_swap_indice_and_double_data(i1, i2, array, aux, t) { \
	double td = (aux)[(int)(&(i1) - (array))];					 \
	(aux)[(int)(&(i1) - (array))] = (aux)[(int)(&(i2) - (array))]; \
	(aux)[(int)(&(i2) - (array))] = td;							\
	CCV_SWAP(i1, i2, t); }

static CCV_IMPLEMENT_QSORT_EX(_ccv_indice_uchar_sort, int, _ccv_indice_less_than, _ccv_swap_indice_and_uchar_data, unsigned char*);
static CCV_IMPLEMENT_QSORT_EX(_ccv_indice_int_sort, int, _ccv_indice_less_than, _ccv_swap_indice_and_int_data, int*);
static CCV_IMPLEMENT_QSORT_EX(_ccv_indice_float_sort, int, _ccv_indice_less_than, _ccv_swap_indice_and_float_data, float*);
static CCV_IMPLEMENT_QSORT_EX(_ccv_indice_double_sort, int, _ccv_indice_less_than, _ccv_swap_indice_and_double_data, double*);

void ccv_compress_sparse_matrix(const ccv_sparse_matrix_t* mat, ccv_compressed_sparse_matrix_t** csm)
{
	uint32_t i, j;
	int nnz = 0;
	const uint32_t size = mat->size;
	for (i = 0; i < size; i++)
	{
		ccv_sparse_matrix_index_t* index = mat->index + i;
		ccv_sparse_matrix_vector_t* vector = mat->vector + i;
		if (index->ifbit <= 1 || !vector->rnum)
			continue;
		if (mat->type & CCV_DENSE_VECTOR)
		{
#define while_block(_, _while_get) \
			for (j = 0; j < vector->rnum; j++) \
				if (_while_get(vector->data.u8, j, 0) != 0) \
					nnz++; \
			ccv_matrix_getter(mat->type, while_block);
#undef while_block
		} else
			nnz += vector->rnum;
	}
	ccv_compressed_sparse_matrix_t* cm = *csm = (ccv_compressed_sparse_matrix_t*)ccmalloc(sizeof(ccv_compressed_sparse_matrix_t) + nnz * sizeof(int) + nnz * CCV_GET_DATA_TYPE_SIZE(mat->type) + (((mat->major == CCV_SPARSE_COL_MAJOR) ? mat->cols : mat->rows) + 1) * sizeof(int));
	cm->type = (mat->type & ~CCV_MATRIX_SPARSE & ~CCV_SPARSE_VECTOR & ~CCV_DENSE_VECTOR) | ((mat->major == CCV_SPARSE_COL_MAJOR) ? CCV_MATRIX_CSC : CCV_MATRIX_CSR);
	cm->nnz = nnz;
	cm->rows = mat->rows;
	cm->cols = mat->cols;
	cm->index = (int*)(cm + 1);
	cm->offset = cm->index + nnz;
	cm->data.i32 = cm->offset + ((mat->major == CCV_SPARSE_COL_MAJOR) ? mat->cols : mat->rows) + 1;
	unsigned char* m_ptr = cm->data.u8;
	int* idx = cm->index;
	cm->offset[0] = 0;
	const size_t cell_size_aligned = (CCV_GET_DATA_TYPE_SIZE(mat->type) * CCV_GET_CHANNEL(mat->type) + 3) & -4;
	const size_t index_size = sizeof(ccv_sparse_matrix_index_t) + cell_size_aligned;
	for (i = 0; i < ((mat->major == CCV_SPARSE_COL_MAJOR) ? mat->cols : mat->rows); i++)
	{
		ccv_sparse_matrix_vector_t* vector = ccv_get_sparse_matrix_vector(mat, i);
		if (!vector || !vector->rnum)
			cm->offset[i + 1] = cm->offset[i];
		else {
			if (mat->type & CCV_DENSE_VECTOR)
			{
				int k = 0;
#define for_block(_, _for_set, _for_get) \
				for (j = 0; j < vector->rnum; j++) \
					if (_for_get(vector->data.u8, j, 0) != 0) \
					{ \
						_for_set(m_ptr, k, _for_get(vector->data.u8, j)); \
						idx[k] = j; \
						k++; \
					}
				ccv_matrix_setter_getter(mat->type, for_block);
#undef for_block
				cm->offset[i + 1] = cm->offset[i] + k;
				idx += k;
				m_ptr += k * CCV_GET_DATA_TYPE_SIZE(mat->type);
			} else {
				int k = 0;
				uint8_t* const index = (uint8_t*)vector->index;
#define for_block(_, _for_set, _for_get) \
				for (j = 0; j < vector->size; j++) \
				{ \
					ccv_sparse_matrix_index_t* const index_j = (ccv_sparse_matrix_index_t*)(index + index_size * j); \
					if (index_j->ifbit > 1) \
					{ \
						_for_set(m_ptr, k, _for_get((uint8_t*)(index_j + 1), 0)); \
						idx[k] = index_j->i; \
						k++; \
					} \
				}
				ccv_matrix_setter_getter(mat->type, for_block);
#undef for_block
				assert(k == vector->rnum);
				switch (CCV_GET_DATA_TYPE(mat->type))
				{
					case CCV_8U:
						_ccv_indice_uchar_sort(idx, vector->rnum, (unsigned char*)m_ptr);
						break;
					case CCV_32S:
						_ccv_indice_int_sort(idx, vector->rnum, (int*)m_ptr);
						break;
					case CCV_32F:
						_ccv_indice_float_sort(idx, vector->rnum, (float*)m_ptr);
						break;
					case CCV_64F:
						_ccv_indice_double_sort(idx, vector->rnum, (double*)m_ptr);
						break;
				}
				cm->offset[i + 1] = cm->offset[i] + vector->rnum;
				idx += vector->rnum;
				m_ptr += vector->rnum * CCV_GET_DATA_TYPE_SIZE(mat->type);
			}
		}
	}
}

void ccv_decompress_sparse_matrix(const ccv_compressed_sparse_matrix_t* csm, ccv_sparse_matrix_t** smt)
{
	ccv_sparse_matrix_t* mat = *smt = ccv_sparse_matrix_new(csm->rows, csm->cols, csm->type & ~CCV_MATRIX_CSR & ~CCV_MATRIX_CSC, (csm->type & CCV_MATRIX_CSR) ? CCV_SPARSE_ROW_MAJOR : CCV_SPARSE_COL_MAJOR, 0);
	int i, j;
	for (i = 0; i < ((mat->major == CCV_SPARSE_COL_MAJOR) ? mat->cols : mat->rows); i++)
		for (j = csm->offset[i]; j < csm->offset[i + 1]; j++)
			if (mat->major == CCV_SPARSE_COL_MAJOR)
				ccv_set_sparse_matrix_cell(mat, csm->index[j], i, csm->data.u8 + CCV_GET_DATA_TYPE_SIZE(csm->type) * j);
			else
				ccv_set_sparse_matrix_cell(mat, i, csm->index[j], csm->data.u8 + CCV_GET_DATA_TYPE_SIZE(csm->type) * j);
}

int ccv_matrix_eq(ccv_matrix_t* a, ccv_matrix_t* b)
{
	int a_type = *(int*)a;
	int b_type = *(int*)b;
	if ((a_type & CCV_MATRIX_DENSE) && (b_type & CCV_MATRIX_DENSE))
	{
		ccv_dense_matrix_t* da = (ccv_dense_matrix_t*)a;
		ccv_dense_matrix_t* db = (ccv_dense_matrix_t*)b;
		if (CCV_GET_DATA_TYPE(da->type) != CCV_GET_DATA_TYPE(db->type))
			return -1;
		if (CCV_GET_CHANNEL(da->type) != CCV_GET_CHANNEL(db->type))
			return -1;
		if (da->rows != db->rows)
			return -1;
		if (da->cols != db->cols)
			return -1;
		int i, j, ch = CCV_GET_CHANNEL(da->type);
		unsigned char* a_ptr = da->data.u8;
		unsigned char* b_ptr = db->data.u8;
		// Read: http://www.cygnus-software.com/papers/comparingfloats/comparingfloats.htm
		// http://floating-point-gui.de/errors/comparison/
		if (CCV_GET_DATA_TYPE(db->type) == CCV_32F)
		{
			static const float epsi = FLT_EPSILON;
			static const int32_t ulps = 128; // So that for 1 and 1.000015 will be treated as the same.
			for (i = 0; i < da->rows; i++)
			{
				int32_t* ia_ptr = (int32_t*)a_ptr;
				int32_t* ib_ptr = (int32_t*)b_ptr;
				for (j = 0; j < da->cols * ch; j++)
				{
					int32_t i32a = ia_ptr[j];
					if (i32a < 0)
						i32a = 0x80000000 - i32a;
					int32_t i32b = ib_ptr[j];
					if (i32b < 0)
						i32b = 0x80000000 - i32b;
					if (abs(i32a - i32b) > ulps && fabsf(((float*)a_ptr)[j] - ((float*)b_ptr)[j]) > epsi)
						return -1;
				}
				a_ptr += da->step;
				b_ptr += db->step;
			}
		} else if (CCV_GET_DATA_TYPE(db->type) == CCV_64F) {
			static const double epsi = FLT_EPSILON; // Using FLT_EPSILON because for most of ccv computations, we have 32-bit float point computation inside.
			static const int64_t ulps = 0x100000l; // So that for 1 and 1.000000001 will be treated as the same.
			for (i = 0; i < da->rows; i++)
			{
				int64_t* ia_ptr = (int64_t*)a_ptr;
				int64_t* ib_ptr = (int64_t*)b_ptr;
				for (j = 0; j < da->cols * ch; j++)
				{
					int64_t i64a = ia_ptr[j];
					if (i64a < 0)
						i64a = 0x8000000000000000l - i64a;
					int64_t i64b = ib_ptr[j];
					if (i64b < 0)
						i64b = 0x8000000000000000l - i64b;
					if (llabs(i64a - i64b) > ulps && fabsf(((float*)a_ptr)[j] - ((float*)b_ptr)[j]) > epsi)
						return -1;
				}
				a_ptr += da->step;
				b_ptr += db->step;
			}
		} else {
#define for_block(_, _for_get) \
			for (i = 0; i < da->rows; i++) \
			{ \
				for (j = 0; j < da->cols * ch; j++) \
				{ \
					if (llabs(_for_get(b_ptr, j) - _for_get(a_ptr, j)) > 1) \
						return -1; \
				} \
				a_ptr += da->step; \
				b_ptr += db->step; \
			}
			ccv_matrix_getter_integer_only(da->type, for_block);
#undef for_block
		}
	}
	return 0;
}

void ccv_slice(ccv_matrix_t* a, ccv_matrix_t** b, int btype, int y, int x, int rows, int cols)
{
	int type = *(int*)a;
	if (type & CCV_MATRIX_DENSE)
	{
		ccv_dense_matrix_t* da = ccv_get_dense_matrix(a);
		ccv_declare_derived_signature(sig, da->sig != 0, ccv_sign_with_format(128, "ccv_slice(%d,%d,%d,%d)", y, x, rows, cols), da->sig, CCV_EOF_SIGN);
		btype = (btype == 0) ? CCV_GET_DATA_TYPE(da->type) | CCV_GET_CHANNEL(da->type) : CCV_GET_DATA_TYPE(btype) | CCV_GET_CHANNEL(da->type);
		ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, rows, cols, CCV_ALL_DATA_TYPE | CCV_GET_CHANNEL(da->type), btype, sig);
		ccv_object_return_if_cached(, db);
		int i, j, ch = CCV_GET_CHANNEL(da->type);
		int dx = 0, dy = 0;
		if (!(y >= 0 && y + rows <= da->rows && x >= 0 && x + cols <= da->cols))
		{
			ccv_zero(db);
			if (y < 0) { rows += y; dy = -y; y = 0; }
			if (y + rows > da->rows) rows = da->rows - y;
			if (x < 0) { cols += x; dx = -x; x = 0; }
			if (x + cols > da->cols) cols = da->cols - x;
		}
		unsigned char* a_ptr = da->data.u8 + x * ch * CCV_GET_DATA_TYPE_SIZE(da->type) + y * da->step;
		unsigned char* b_ptr = db->data.u8 + dx * ch * CCV_GET_DATA_TYPE_SIZE(db->type) + dy * db->step;
#define for_block(_for_set, _for_get) \
		for (i = 0; i < rows; i++) \
		{ \
			for (j = 0; j < cols * ch; j++) \
			{ \
				_for_set(b_ptr, j, _for_get(a_ptr, j)); \
			} \
			a_ptr += da->step; \
			b_ptr += db->step; \
		}
		ccv_matrix_setter(db->type, ccv_matrix_getter, da->type, for_block);
#undef for_block
	} else if (type & CCV_MATRIX_SPARSE) {
	}
}

ccv_dense_matrix_t ccv_reshape(ccv_dense_matrix_t* a, int y, int x, int rows, int cols)
{
	assert(y + rows <= a->rows);
	assert(x + cols <= a->cols);
	assert(x >= 0 && y >= 0);
	ccv_dense_matrix_t b = {
		.type = (CCV_UNMANAGED | CCV_GET_DATA_TYPE(a->type) | CCV_GET_CHANNEL(a->type) | CCV_NO_DATA_ALLOC | CCV_MATRIX_DENSE) & ~CCV_GARBAGE,
		.rows = rows,
		.cols = cols,
		.step = a->step,
		.refcount = 0,
		.sig = 0,
		.tb.u8 = 0,
		.data.u8 = ccv_get_dense_matrix_cell(a, y, x, 0),
	};
	return b;
}

void ccv_move(ccv_matrix_t* a, ccv_matrix_t** b, int btype, int y, int x)
{
	int type = *(int*)a;
	if (type & CCV_MATRIX_DENSE)
	{
		ccv_dense_matrix_t* da = ccv_get_dense_matrix(a);
		ccv_declare_derived_signature(sig, da->sig != 0, ccv_sign_with_format(64, "ccv_move(%d,%d)", y, x), da->sig, CCV_EOF_SIGN);
		btype = (btype == 0) ? CCV_GET_DATA_TYPE(da->type) | CCV_GET_CHANNEL(da->type) : CCV_GET_DATA_TYPE(btype) | CCV_GET_CHANNEL(da->type);
		ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, da->rows, da->cols, CCV_ALL_DATA_TYPE | CCV_GET_CHANNEL(da->type), btype, sig);
		ccv_object_return_if_cached(, db);
		int i, j, ch = CCV_GET_CHANNEL(da->type);
		unsigned char* a_ptr = da->data.u8 + ccv_max(x, 0) * ch * CCV_GET_DATA_TYPE_SIZE(da->type) + ccv_max(y, 0) * da->step;
		unsigned char* b_ptr = db->data.u8 + ccv_max(-x, 0) * ch * CCV_GET_DATA_TYPE_SIZE(db->type) + ccv_max(-y, 0) * db->step;
#define for_block(_for_set, _for_get) \
		for (i = abs(y); i < db->rows; i++) \
		{ \
			for (j = abs(x) * ch; j < db->cols * ch; j++) \
			{ \
				_for_set(b_ptr, j, _for_get(a_ptr, j)); \
			} \
			a_ptr += da->step; \
			b_ptr += db->step; \
		}
		ccv_matrix_setter(db->type, ccv_matrix_getter, da->type, for_block);
#undef for_block
	} else if (type & CCV_MATRIX_SPARSE) {
	}
}

// the code are shamelessly copied from ftp://ftp.fox-toolkit.org/pub/fasthalffloatconversion.pdf, assuming in Public Domain

static uint16_t _ccv_base_table[512] = {
	0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
	0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
	0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
	0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
	0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
	0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
	0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80, 0x100,
	0x200, 0x400, 0x800, 0xc00, 0x1000, 0x1400, 0x1800, 0x1c00, 0x2000, 0x2400, 0x2800, 0x2c00, 0x3000, 0x3400, 0x3800, 0x3c00,
	0x4000, 0x4400, 0x4800, 0x4c00, 0x5000, 0x5400, 0x5800, 0x5c00, 0x6000, 0x6400, 0x6800, 0x6c00, 0x7000, 0x7400, 0x7800, 0x7c00,
	0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00,
	0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00,
	0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00,
	0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00,
	0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00,
	0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00,
	0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00,
	0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
	0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
	0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
	0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
	0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
	0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
	0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8001, 0x8002, 0x8004, 0x8008, 0x8010, 0x8020, 0x8040, 0x8080, 0x8100,
	0x8200, 0x8400, 0x8800, 0x8c00, 0x9000, 0x9400, 0x9800, 0x9c00, 0xa000, 0xa400, 0xa800, 0xac00, 0xb000, 0xb400, 0xb800, 0xbc00,
	0xc000, 0xc400, 0xc800, 0xcc00, 0xd000, 0xd400, 0xd800, 0xdc00, 0xe000, 0xe400, 0xe800, 0xec00, 0xf000, 0xf400, 0xf800, 0xfc00,
	0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00,
	0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00,
	0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00,
	0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00,
	0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00,
	0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00,
	0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00,
};

static uint8_t _ccv_shift_table[512] = {
	0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
	0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
	0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
	0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
	0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
	0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
	0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, 0x10, 0xf,
	0xe, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd,
	0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0x18,
	0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
	0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
	0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
	0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
	0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
	0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
	0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0xd,
	0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
	0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
	0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
	0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
	0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
	0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
	0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, 0x10, 0xf,
	0xe, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd,
	0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0xd, 0x18,
	0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
	0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
	0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
	0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
	0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
	0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
	0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0xd,
};

void ccv_float_to_half_precision(float* f, uint16_t* h, size_t len)
{
	int i;
	uint32_t* u = (uint32_t*)f;
	for (i = 0; i < len; i++)
		h[i] = _ccv_base_table[(u[i] >> 23) & 0x1ff] + ((u[i] & 0x007fffff) >> _ccv_shift_table[(u[i] >> 23) & 0x1ff]);
}

void ccv_double_to_half_precision(double* f, uint16_t* h, size_t len)
{
	int i;
	for (i = 0; i < len; i++)
	{
		union {
			float v;
			uint32_t p;
		} u = { .v = (float)f[i] };
		h[i] = _ccv_base_table[(u.p >> 23) & 0x1ff] + ((u.p & 0x007fffff) >> _ccv_shift_table[(u.p >> 23) & 0x1ff]);
	}
}

static uint32_t _ccv_mantissa_table[2048] = {
	0x0, 0x33800000, 0x34000000, 0x34400000, 0x34800000, 0x34a00000, 0x34c00000, 0x34e00000,
	0x35000000, 0x35100000, 0x35200000, 0x35300000, 0x35400000, 0x35500000, 0x35600000, 0x35700000,
	0x35800000, 0x35880000, 0x35900000, 0x35980000, 0x35a00000, 0x35a80000, 0x35b00000, 0x35b80000,
	0x35c00000, 0x35c80000, 0x35d00000, 0x35d80000, 0x35e00000, 0x35e80000, 0x35f00000, 0x35f80000,
	0x36000000, 0x36040000, 0x36080000, 0x360c0000, 0x36100000, 0x36140000, 0x36180000, 0x361c0000,
	0x36200000, 0x36240000, 0x36280000, 0x362c0000, 0x36300000, 0x36340000, 0x36380000, 0x363c0000,
	0x36400000, 0x36440000, 0x36480000, 0x364c0000, 0x36500000, 0x36540000, 0x36580000, 0x365c0000,
	0x36600000, 0x36640000, 0x36680000, 0x366c0000, 0x36700000, 0x36740000, 0x36780000, 0x367c0000,
	0x36800000, 0x36820000, 0x36840000, 0x36860000, 0x36880000, 0x368a0000, 0x368c0000, 0x368e0000,
	0x36900000, 0x36920000, 0x36940000, 0x36960000, 0x36980000, 0x369a0000, 0x369c0000, 0x369e0000,
	0x36a00000, 0x36a20000, 0x36a40000, 0x36a60000, 0x36a80000, 0x36aa0000, 0x36ac0000, 0x36ae0000,
	0x36b00000, 0x36b20000, 0x36b40000, 0x36b60000, 0x36b80000, 0x36ba0000, 0x36bc0000, 0x36be0000,
	0x36c00000, 0x36c20000, 0x36c40000, 0x36c60000, 0x36c80000, 0x36ca0000, 0x36cc0000, 0x36ce0000,
	0x36d00000, 0x36d20000, 0x36d40000, 0x36d60000, 0x36d80000, 0x36da0000, 0x36dc0000, 0x36de0000,
	0x36e00000, 0x36e20000, 0x36e40000, 0x36e60000, 0x36e80000, 0x36ea0000, 0x36ec0000, 0x36ee0000,
	0x36f00000, 0x36f20000, 0x36f40000, 0x36f60000, 0x36f80000, 0x36fa0000, 0x36fc0000, 0x36fe0000,
	0x37000000, 0x37010000, 0x37020000, 0x37030000, 0x37040000, 0x37050000, 0x37060000, 0x37070000,
	0x37080000, 0x37090000, 0x370a0000, 0x370b0000, 0x370c0000, 0x370d0000, 0x370e0000, 0x370f0000,
	0x37100000, 0x37110000, 0x37120000, 0x37130000, 0x37140000, 0x37150000, 0x37160000, 0x37170000,
	0x37180000, 0x37190000, 0x371a0000, 0x371b0000, 0x371c0000, 0x371d0000, 0x371e0000, 0x371f0000,
	0x37200000, 0x37210000, 0x37220000, 0x37230000, 0x37240000, 0x37250000, 0x37260000, 0x37270000,
	0x37280000, 0x37290000, 0x372a0000, 0x372b0000, 0x372c0000, 0x372d0000, 0x372e0000, 0x372f0000,
	0x37300000, 0x37310000, 0x37320000, 0x37330000, 0x37340000, 0x37350000, 0x37360000, 0x37370000,
	0x37380000, 0x37390000, 0x373a0000, 0x373b0000, 0x373c0000, 0x373d0000, 0x373e0000, 0x373f0000,
	0x37400000, 0x37410000, 0x37420000, 0x37430000, 0x37440000, 0x37450000, 0x37460000, 0x37470000,
	0x37480000, 0x37490000, 0x374a0000, 0x374b0000, 0x374c0000, 0x374d0000, 0x374e0000, 0x374f0000,
	0x37500000, 0x37510000, 0x37520000, 0x37530000, 0x37540000, 0x37550000, 0x37560000, 0x37570000,
	0x37580000, 0x37590000, 0x375a0000, 0x375b0000, 0x375c0000, 0x375d0000, 0x375e0000, 0x375f0000,
	0x37600000, 0x37610000, 0x37620000, 0x37630000, 0x37640000, 0x37650000, 0x37660000, 0x37670000,
	0x37680000, 0x37690000, 0x376a0000, 0x376b0000, 0x376c0000, 0x376d0000, 0x376e0000, 0x376f0000,
	0x37700000, 0x37710000, 0x37720000, 0x37730000, 0x37740000, 0x37750000, 0x37760000, 0x37770000,
	0x37780000, 0x37790000, 0x377a0000, 0x377b0000, 0x377c0000, 0x377d0000, 0x377e0000, 0x377f0000,
	0x37800000, 0x37808000, 0x37810000, 0x37818000, 0x37820000, 0x37828000, 0x37830000, 0x37838000,
	0x37840000, 0x37848000, 0x37850000, 0x37858000, 0x37860000, 0x37868000, 0x37870000, 0x37878000,
	0x37880000, 0x37888000, 0x37890000, 0x37898000, 0x378a0000, 0x378a8000, 0x378b0000, 0x378b8000,
	0x378c0000, 0x378c8000, 0x378d0000, 0x378d8000, 0x378e0000, 0x378e8000, 0x378f0000, 0x378f8000,
	0x37900000, 0x37908000, 0x37910000, 0x37918000, 0x37920000, 0x37928000, 0x37930000, 0x37938000,
	0x37940000, 0x37948000, 0x37950000, 0x37958000, 0x37960000, 0x37968000, 0x37970000, 0x37978000,
	0x37980000, 0x37988000, 0x37990000, 0x37998000, 0x379a0000, 0x379a8000, 0x379b0000, 0x379b8000,
	0x379c0000, 0x379c8000, 0x379d0000, 0x379d8000, 0x379e0000, 0x379e8000, 0x379f0000, 0x379f8000,
	0x37a00000, 0x37a08000, 0x37a10000, 0x37a18000, 0x37a20000, 0x37a28000, 0x37a30000, 0x37a38000,
	0x37a40000, 0x37a48000, 0x37a50000, 0x37a58000, 0x37a60000, 0x37a68000, 0x37a70000, 0x37a78000,
	0x37a80000, 0x37a88000, 0x37a90000, 0x37a98000, 0x37aa0000, 0x37aa8000, 0x37ab0000, 0x37ab8000,
	0x37ac0000, 0x37ac8000, 0x37ad0000, 0x37ad8000, 0x37ae0000, 0x37ae8000, 0x37af0000, 0x37af8000,
	0x37b00000, 0x37b08000, 0x37b10000, 0x37b18000, 0x37b20000, 0x37b28000, 0x37b30000, 0x37b38000,
	0x37b40000, 0x37b48000, 0x37b50000, 0x37b58000, 0x37b60000, 0x37b68000, 0x37b70000, 0x37b78000,
	0x37b80000, 0x37b88000, 0x37b90000, 0x37b98000, 0x37ba0000, 0x37ba8000, 0x37bb0000, 0x37bb8000,
	0x37bc0000, 0x37bc8000, 0x37bd0000, 0x37bd8000, 0x37be0000, 0x37be8000, 0x37bf0000, 0x37bf8000,
	0x37c00000, 0x37c08000, 0x37c10000, 0x37c18000, 0x37c20000, 0x37c28000, 0x37c30000, 0x37c38000,
	0x37c40000, 0x37c48000, 0x37c50000, 0x37c58000, 0x37c60000, 0x37c68000, 0x37c70000, 0x37c78000,
	0x37c80000, 0x37c88000, 0x37c90000, 0x37c98000, 0x37ca0000, 0x37ca8000, 0x37cb0000, 0x37cb8000,
	0x37cc0000, 0x37cc8000, 0x37cd0000, 0x37cd8000, 0x37ce0000, 0x37ce8000, 0x37cf0000, 0x37cf8000,
	0x37d00000, 0x37d08000, 0x37d10000, 0x37d18000, 0x37d20000, 0x37d28000, 0x37d30000, 0x37d38000,
	0x37d40000, 0x37d48000, 0x37d50000, 0x37d58000, 0x37d60000, 0x37d68000, 0x37d70000, 0x37d78000,
	0x37d80000, 0x37d88000, 0x37d90000, 0x37d98000, 0x37da0000, 0x37da8000, 0x37db0000, 0x37db8000,
	0x37dc0000, 0x37dc8000, 0x37dd0000, 0x37dd8000, 0x37de0000, 0x37de8000, 0x37df0000, 0x37df8000,
	0x37e00000, 0x37e08000, 0x37e10000, 0x37e18000, 0x37e20000, 0x37e28000, 0x37e30000, 0x37e38000,
	0x37e40000, 0x37e48000, 0x37e50000, 0x37e58000, 0x37e60000, 0x37e68000, 0x37e70000, 0x37e78000,
	0x37e80000, 0x37e88000, 0x37e90000, 0x37e98000, 0x37ea0000, 0x37ea8000, 0x37eb0000, 0x37eb8000,
	0x37ec0000, 0x37ec8000, 0x37ed0000, 0x37ed8000, 0x37ee0000, 0x37ee8000, 0x37ef0000, 0x37ef8000,
	0x37f00000, 0x37f08000, 0x37f10000, 0x37f18000, 0x37f20000, 0x37f28000, 0x37f30000, 0x37f38000,
	0x37f40000, 0x37f48000, 0x37f50000, 0x37f58000, 0x37f60000, 0x37f68000, 0x37f70000, 0x37f78000,
	0x37f80000, 0x37f88000, 0x37f90000, 0x37f98000, 0x37fa0000, 0x37fa8000, 0x37fb0000, 0x37fb8000,
	0x37fc0000, 0x37fc8000, 0x37fd0000, 0x37fd8000, 0x37fe0000, 0x37fe8000, 0x37ff0000, 0x37ff8000,
	0x38000000, 0x38004000, 0x38008000, 0x3800c000, 0x38010000, 0x38014000, 0x38018000, 0x3801c000,
	0x38020000, 0x38024000, 0x38028000, 0x3802c000, 0x38030000, 0x38034000, 0x38038000, 0x3803c000,
	0x38040000, 0x38044000, 0x38048000, 0x3804c000, 0x38050000, 0x38054000, 0x38058000, 0x3805c000,
	0x38060000, 0x38064000, 0x38068000, 0x3806c000, 0x38070000, 0x38074000, 0x38078000, 0x3807c000,
	0x38080000, 0x38084000, 0x38088000, 0x3808c000, 0x38090000, 0x38094000, 0x38098000, 0x3809c000,
	0x380a0000, 0x380a4000, 0x380a8000, 0x380ac000, 0x380b0000, 0x380b4000, 0x380b8000, 0x380bc000,
	0x380c0000, 0x380c4000, 0x380c8000, 0x380cc000, 0x380d0000, 0x380d4000, 0x380d8000, 0x380dc000,
	0x380e0000, 0x380e4000, 0x380e8000, 0x380ec000, 0x380f0000, 0x380f4000, 0x380f8000, 0x380fc000,
	0x38100000, 0x38104000, 0x38108000, 0x3810c000, 0x38110000, 0x38114000, 0x38118000, 0x3811c000,
	0x38120000, 0x38124000, 0x38128000, 0x3812c000, 0x38130000, 0x38134000, 0x38138000, 0x3813c000,
	0x38140000, 0x38144000, 0x38148000, 0x3814c000, 0x38150000, 0x38154000, 0x38158000, 0x3815c000,
	0x38160000, 0x38164000, 0x38168000, 0x3816c000, 0x38170000, 0x38174000, 0x38178000, 0x3817c000,
	0x38180000, 0x38184000, 0x38188000, 0x3818c000, 0x38190000, 0x38194000, 0x38198000, 0x3819c000,
	0x381a0000, 0x381a4000, 0x381a8000, 0x381ac000, 0x381b0000, 0x381b4000, 0x381b8000, 0x381bc000,
	0x381c0000, 0x381c4000, 0x381c8000, 0x381cc000, 0x381d0000, 0x381d4000, 0x381d8000, 0x381dc000,
	0x381e0000, 0x381e4000, 0x381e8000, 0x381ec000, 0x381f0000, 0x381f4000, 0x381f8000, 0x381fc000,
	0x38200000, 0x38204000, 0x38208000, 0x3820c000, 0x38210000, 0x38214000, 0x38218000, 0x3821c000,
	0x38220000, 0x38224000, 0x38228000, 0x3822c000, 0x38230000, 0x38234000, 0x38238000, 0x3823c000,
	0x38240000, 0x38244000, 0x38248000, 0x3824c000, 0x38250000, 0x38254000, 0x38258000, 0x3825c000,
	0x38260000, 0x38264000, 0x38268000, 0x3826c000, 0x38270000, 0x38274000, 0x38278000, 0x3827c000,
	0x38280000, 0x38284000, 0x38288000, 0x3828c000, 0x38290000, 0x38294000, 0x38298000, 0x3829c000,
	0x382a0000, 0x382a4000, 0x382a8000, 0x382ac000, 0x382b0000, 0x382b4000, 0x382b8000, 0x382bc000,
	0x382c0000, 0x382c4000, 0x382c8000, 0x382cc000, 0x382d0000, 0x382d4000, 0x382d8000, 0x382dc000,
	0x382e0000, 0x382e4000, 0x382e8000, 0x382ec000, 0x382f0000, 0x382f4000, 0x382f8000, 0x382fc000,
	0x38300000, 0x38304000, 0x38308000, 0x3830c000, 0x38310000, 0x38314000, 0x38318000, 0x3831c000,
	0x38320000, 0x38324000, 0x38328000, 0x3832c000, 0x38330000, 0x38334000, 0x38338000, 0x3833c000,
	0x38340000, 0x38344000, 0x38348000, 0x3834c000, 0x38350000, 0x38354000, 0x38358000, 0x3835c000,
	0x38360000, 0x38364000, 0x38368000, 0x3836c000, 0x38370000, 0x38374000, 0x38378000, 0x3837c000,
	0x38380000, 0x38384000, 0x38388000, 0x3838c000, 0x38390000, 0x38394000, 0x38398000, 0x3839c000,
	0x383a0000, 0x383a4000, 0x383a8000, 0x383ac000, 0x383b0000, 0x383b4000, 0x383b8000, 0x383bc000,
	0x383c0000, 0x383c4000, 0x383c8000, 0x383cc000, 0x383d0000, 0x383d4000, 0x383d8000, 0x383dc000,
	0x383e0000, 0x383e4000, 0x383e8000, 0x383ec000, 0x383f0000, 0x383f4000, 0x383f8000, 0x383fc000,
	0x38400000, 0x38404000, 0x38408000, 0x3840c000, 0x38410000, 0x38414000, 0x38418000, 0x3841c000,
	0x38420000, 0x38424000, 0x38428000, 0x3842c000, 0x38430000, 0x38434000, 0x38438000, 0x3843c000,
	0x38440000, 0x38444000, 0x38448000, 0x3844c000, 0x38450000, 0x38454000, 0x38458000, 0x3845c000,
	0x38460000, 0x38464000, 0x38468000, 0x3846c000, 0x38470000, 0x38474000, 0x38478000, 0x3847c000,
	0x38480000, 0x38484000, 0x38488000, 0x3848c000, 0x38490000, 0x38494000, 0x38498000, 0x3849c000,
	0x384a0000, 0x384a4000, 0x384a8000, 0x384ac000, 0x384b0000, 0x384b4000, 0x384b8000, 0x384bc000,
	0x384c0000, 0x384c4000, 0x384c8000, 0x384cc000, 0x384d0000, 0x384d4000, 0x384d8000, 0x384dc000,
	0x384e0000, 0x384e4000, 0x384e8000, 0x384ec000, 0x384f0000, 0x384f4000, 0x384f8000, 0x384fc000,
	0x38500000, 0x38504000, 0x38508000, 0x3850c000, 0x38510000, 0x38514000, 0x38518000, 0x3851c000,
	0x38520000, 0x38524000, 0x38528000, 0x3852c000, 0x38530000, 0x38534000, 0x38538000, 0x3853c000,
	0x38540000, 0x38544000, 0x38548000, 0x3854c000, 0x38550000, 0x38554000, 0x38558000, 0x3855c000,
	0x38560000, 0x38564000, 0x38568000, 0x3856c000, 0x38570000, 0x38574000, 0x38578000, 0x3857c000,
	0x38580000, 0x38584000, 0x38588000, 0x3858c000, 0x38590000, 0x38594000, 0x38598000, 0x3859c000,
	0x385a0000, 0x385a4000, 0x385a8000, 0x385ac000, 0x385b0000, 0x385b4000, 0x385b8000, 0x385bc000,
	0x385c0000, 0x385c4000, 0x385c8000, 0x385cc000, 0x385d0000, 0x385d4000, 0x385d8000, 0x385dc000,
	0x385e0000, 0x385e4000, 0x385e8000, 0x385ec000, 0x385f0000, 0x385f4000, 0x385f8000, 0x385fc000,
	0x38600000, 0x38604000, 0x38608000, 0x3860c000, 0x38610000, 0x38614000, 0x38618000, 0x3861c000,
	0x38620000, 0x38624000, 0x38628000, 0x3862c000, 0x38630000, 0x38634000, 0x38638000, 0x3863c000,
	0x38640000, 0x38644000, 0x38648000, 0x3864c000, 0x38650000, 0x38654000, 0x38658000, 0x3865c000,
	0x38660000, 0x38664000, 0x38668000, 0x3866c000, 0x38670000, 0x38674000, 0x38678000, 0x3867c000,
	0x38680000, 0x38684000, 0x38688000, 0x3868c000, 0x38690000, 0x38694000, 0x38698000, 0x3869c000,
	0x386a0000, 0x386a4000, 0x386a8000, 0x386ac000, 0x386b0000, 0x386b4000, 0x386b8000, 0x386bc000,
	0x386c0000, 0x386c4000, 0x386c8000, 0x386cc000, 0x386d0000, 0x386d4000, 0x386d8000, 0x386dc000,
	0x386e0000, 0x386e4000, 0x386e8000, 0x386ec000, 0x386f0000, 0x386f4000, 0x386f8000, 0x386fc000,
	0x38700000, 0x38704000, 0x38708000, 0x3870c000, 0x38710000, 0x38714000, 0x38718000, 0x3871c000,
	0x38720000, 0x38724000, 0x38728000, 0x3872c000, 0x38730000, 0x38734000, 0x38738000, 0x3873c000,
	0x38740000, 0x38744000, 0x38748000, 0x3874c000, 0x38750000, 0x38754000, 0x38758000, 0x3875c000,
	0x38760000, 0x38764000, 0x38768000, 0x3876c000, 0x38770000, 0x38774000, 0x38778000, 0x3877c000,
	0x38780000, 0x38784000, 0x38788000, 0x3878c000, 0x38790000, 0x38794000, 0x38798000, 0x3879c000,
	0x387a0000, 0x387a4000, 0x387a8000, 0x387ac000, 0x387b0000, 0x387b4000, 0x387b8000, 0x387bc000,
	0x387c0000, 0x387c4000, 0x387c8000, 0x387cc000, 0x387d0000, 0x387d4000, 0x387d8000, 0x387dc000,
	0x387e0000, 0x387e4000, 0x387e8000, 0x387ec000, 0x387f0000, 0x387f4000, 0x387f8000, 0x387fc000,
	0x38000000, 0x38002000, 0x38004000, 0x38006000, 0x38008000, 0x3800a000, 0x3800c000, 0x3800e000,
	0x38010000, 0x38012000, 0x38014000, 0x38016000, 0x38018000, 0x3801a000, 0x3801c000, 0x3801e000,
	0x38020000, 0x38022000, 0x38024000, 0x38026000, 0x38028000, 0x3802a000, 0x3802c000, 0x3802e000,
	0x38030000, 0x38032000, 0x38034000, 0x38036000, 0x38038000, 0x3803a000, 0x3803c000, 0x3803e000,
	0x38040000, 0x38042000, 0x38044000, 0x38046000, 0x38048000, 0x3804a000, 0x3804c000, 0x3804e000,
	0x38050000, 0x38052000, 0x38054000, 0x38056000, 0x38058000, 0x3805a000, 0x3805c000, 0x3805e000,
	0x38060000, 0x38062000, 0x38064000, 0x38066000, 0x38068000, 0x3806a000, 0x3806c000, 0x3806e000,
	0x38070000, 0x38072000, 0x38074000, 0x38076000, 0x38078000, 0x3807a000, 0x3807c000, 0x3807e000,
	0x38080000, 0x38082000, 0x38084000, 0x38086000, 0x38088000, 0x3808a000, 0x3808c000, 0x3808e000,
	0x38090000, 0x38092000, 0x38094000, 0x38096000, 0x38098000, 0x3809a000, 0x3809c000, 0x3809e000,
	0x380a0000, 0x380a2000, 0x380a4000, 0x380a6000, 0x380a8000, 0x380aa000, 0x380ac000, 0x380ae000,
	0x380b0000, 0x380b2000, 0x380b4000, 0x380b6000, 0x380b8000, 0x380ba000, 0x380bc000, 0x380be000,
	0x380c0000, 0x380c2000, 0x380c4000, 0x380c6000, 0x380c8000, 0x380ca000, 0x380cc000, 0x380ce000,
	0x380d0000, 0x380d2000, 0x380d4000, 0x380d6000, 0x380d8000, 0x380da000, 0x380dc000, 0x380de000,
	0x380e0000, 0x380e2000, 0x380e4000, 0x380e6000, 0x380e8000, 0x380ea000, 0x380ec000, 0x380ee000,
	0x380f0000, 0x380f2000, 0x380f4000, 0x380f6000, 0x380f8000, 0x380fa000, 0x380fc000, 0x380fe000,
	0x38100000, 0x38102000, 0x38104000, 0x38106000, 0x38108000, 0x3810a000, 0x3810c000, 0x3810e000,
	0x38110000, 0x38112000, 0x38114000, 0x38116000, 0x38118000, 0x3811a000, 0x3811c000, 0x3811e000,
	0x38120000, 0x38122000, 0x38124000, 0x38126000, 0x38128000, 0x3812a000, 0x3812c000, 0x3812e000,
	0x38130000, 0x38132000, 0x38134000, 0x38136000, 0x38138000, 0x3813a000, 0x3813c000, 0x3813e000,
	0x38140000, 0x38142000, 0x38144000, 0x38146000, 0x38148000, 0x3814a000, 0x3814c000, 0x3814e000,
	0x38150000, 0x38152000, 0x38154000, 0x38156000, 0x38158000, 0x3815a000, 0x3815c000, 0x3815e000,
	0x38160000, 0x38162000, 0x38164000, 0x38166000, 0x38168000, 0x3816a000, 0x3816c000, 0x3816e000,
	0x38170000, 0x38172000, 0x38174000, 0x38176000, 0x38178000, 0x3817a000, 0x3817c000, 0x3817e000,
	0x38180000, 0x38182000, 0x38184000, 0x38186000, 0x38188000, 0x3818a000, 0x3818c000, 0x3818e000,
	0x38190000, 0x38192000, 0x38194000, 0x38196000, 0x38198000, 0x3819a000, 0x3819c000, 0x3819e000,
	0x381a0000, 0x381a2000, 0x381a4000, 0x381a6000, 0x381a8000, 0x381aa000, 0x381ac000, 0x381ae000,
	0x381b0000, 0x381b2000, 0x381b4000, 0x381b6000, 0x381b8000, 0x381ba000, 0x381bc000, 0x381be000,
	0x381c0000, 0x381c2000, 0x381c4000, 0x381c6000, 0x381c8000, 0x381ca000, 0x381cc000, 0x381ce000,
	0x381d0000, 0x381d2000, 0x381d4000, 0x381d6000, 0x381d8000, 0x381da000, 0x381dc000, 0x381de000,
	0x381e0000, 0x381e2000, 0x381e4000, 0x381e6000, 0x381e8000, 0x381ea000, 0x381ec000, 0x381ee000,
	0x381f0000, 0x381f2000, 0x381f4000, 0x381f6000, 0x381f8000, 0x381fa000, 0x381fc000, 0x381fe000,
	0x38200000, 0x38202000, 0x38204000, 0x38206000, 0x38208000, 0x3820a000, 0x3820c000, 0x3820e000,
	0x38210000, 0x38212000, 0x38214000, 0x38216000, 0x38218000, 0x3821a000, 0x3821c000, 0x3821e000,
	0x38220000, 0x38222000, 0x38224000, 0x38226000, 0x38228000, 0x3822a000, 0x3822c000, 0x3822e000,
	0x38230000, 0x38232000, 0x38234000, 0x38236000, 0x38238000, 0x3823a000, 0x3823c000, 0x3823e000,
	0x38240000, 0x38242000, 0x38244000, 0x38246000, 0x38248000, 0x3824a000, 0x3824c000, 0x3824e000,
	0x38250000, 0x38252000, 0x38254000, 0x38256000, 0x38258000, 0x3825a000, 0x3825c000, 0x3825e000,
	0x38260000, 0x38262000, 0x38264000, 0x38266000, 0x38268000, 0x3826a000, 0x3826c000, 0x3826e000,
	0x38270000, 0x38272000, 0x38274000, 0x38276000, 0x38278000, 0x3827a000, 0x3827c000, 0x3827e000,
	0x38280000, 0x38282000, 0x38284000, 0x38286000, 0x38288000, 0x3828a000, 0x3828c000, 0x3828e000,
	0x38290000, 0x38292000, 0x38294000, 0x38296000, 0x38298000, 0x3829a000, 0x3829c000, 0x3829e000,
	0x382a0000, 0x382a2000, 0x382a4000, 0x382a6000, 0x382a8000, 0x382aa000, 0x382ac000, 0x382ae000,
	0x382b0000, 0x382b2000, 0x382b4000, 0x382b6000, 0x382b8000, 0x382ba000, 0x382bc000, 0x382be000,
	0x382c0000, 0x382c2000, 0x382c4000, 0x382c6000, 0x382c8000, 0x382ca000, 0x382cc000, 0x382ce000,
	0x382d0000, 0x382d2000, 0x382d4000, 0x382d6000, 0x382d8000, 0x382da000, 0x382dc000, 0x382de000,
	0x382e0000, 0x382e2000, 0x382e4000, 0x382e6000, 0x382e8000, 0x382ea000, 0x382ec000, 0x382ee000,
	0x382f0000, 0x382f2000, 0x382f4000, 0x382f6000, 0x382f8000, 0x382fa000, 0x382fc000, 0x382fe000,
	0x38300000, 0x38302000, 0x38304000, 0x38306000, 0x38308000, 0x3830a000, 0x3830c000, 0x3830e000,
	0x38310000, 0x38312000, 0x38314000, 0x38316000, 0x38318000, 0x3831a000, 0x3831c000, 0x3831e000,
	0x38320000, 0x38322000, 0x38324000, 0x38326000, 0x38328000, 0x3832a000, 0x3832c000, 0x3832e000,
	0x38330000, 0x38332000, 0x38334000, 0x38336000, 0x38338000, 0x3833a000, 0x3833c000, 0x3833e000,
	0x38340000, 0x38342000, 0x38344000, 0x38346000, 0x38348000, 0x3834a000, 0x3834c000, 0x3834e000,
	0x38350000, 0x38352000, 0x38354000, 0x38356000, 0x38358000, 0x3835a000, 0x3835c000, 0x3835e000,
	0x38360000, 0x38362000, 0x38364000, 0x38366000, 0x38368000, 0x3836a000, 0x3836c000, 0x3836e000,
	0x38370000, 0x38372000, 0x38374000, 0x38376000, 0x38378000, 0x3837a000, 0x3837c000, 0x3837e000,
	0x38380000, 0x38382000, 0x38384000, 0x38386000, 0x38388000, 0x3838a000, 0x3838c000, 0x3838e000,
	0x38390000, 0x38392000, 0x38394000, 0x38396000, 0x38398000, 0x3839a000, 0x3839c000, 0x3839e000,
	0x383a0000, 0x383a2000, 0x383a4000, 0x383a6000, 0x383a8000, 0x383aa000, 0x383ac000, 0x383ae000,
	0x383b0000, 0x383b2000, 0x383b4000, 0x383b6000, 0x383b8000, 0x383ba000, 0x383bc000, 0x383be000,
	0x383c0000, 0x383c2000, 0x383c4000, 0x383c6000, 0x383c8000, 0x383ca000, 0x383cc000, 0x383ce000,
	0x383d0000, 0x383d2000, 0x383d4000, 0x383d6000, 0x383d8000, 0x383da000, 0x383dc000, 0x383de000,
	0x383e0000, 0x383e2000, 0x383e4000, 0x383e6000, 0x383e8000, 0x383ea000, 0x383ec000, 0x383ee000,
	0x383f0000, 0x383f2000, 0x383f4000, 0x383f6000, 0x383f8000, 0x383fa000, 0x383fc000, 0x383fe000,
	0x38400000, 0x38402000, 0x38404000, 0x38406000, 0x38408000, 0x3840a000, 0x3840c000, 0x3840e000,
	0x38410000, 0x38412000, 0x38414000, 0x38416000, 0x38418000, 0x3841a000, 0x3841c000, 0x3841e000,
	0x38420000, 0x38422000, 0x38424000, 0x38426000, 0x38428000, 0x3842a000, 0x3842c000, 0x3842e000,
	0x38430000, 0x38432000, 0x38434000, 0x38436000, 0x38438000, 0x3843a000, 0x3843c000, 0x3843e000,
	0x38440000, 0x38442000, 0x38444000, 0x38446000, 0x38448000, 0x3844a000, 0x3844c000, 0x3844e000,
	0x38450000, 0x38452000, 0x38454000, 0x38456000, 0x38458000, 0x3845a000, 0x3845c000, 0x3845e000,
	0x38460000, 0x38462000, 0x38464000, 0x38466000, 0x38468000, 0x3846a000, 0x3846c000, 0x3846e000,
	0x38470000, 0x38472000, 0x38474000, 0x38476000, 0x38478000, 0x3847a000, 0x3847c000, 0x3847e000,
	0x38480000, 0x38482000, 0x38484000, 0x38486000, 0x38488000, 0x3848a000, 0x3848c000, 0x3848e000,
	0x38490000, 0x38492000, 0x38494000, 0x38496000, 0x38498000, 0x3849a000, 0x3849c000, 0x3849e000,
	0x384a0000, 0x384a2000, 0x384a4000, 0x384a6000, 0x384a8000, 0x384aa000, 0x384ac000, 0x384ae000,
	0x384b0000, 0x384b2000, 0x384b4000, 0x384b6000, 0x384b8000, 0x384ba000, 0x384bc000, 0x384be000,
	0x384c0000, 0x384c2000, 0x384c4000, 0x384c6000, 0x384c8000, 0x384ca000, 0x384cc000, 0x384ce000,
	0x384d0000, 0x384d2000, 0x384d4000, 0x384d6000, 0x384d8000, 0x384da000, 0x384dc000, 0x384de000,
	0x384e0000, 0x384e2000, 0x384e4000, 0x384e6000, 0x384e8000, 0x384ea000, 0x384ec000, 0x384ee000,
	0x384f0000, 0x384f2000, 0x384f4000, 0x384f6000, 0x384f8000, 0x384fa000, 0x384fc000, 0x384fe000,
	0x38500000, 0x38502000, 0x38504000, 0x38506000, 0x38508000, 0x3850a000, 0x3850c000, 0x3850e000,
	0x38510000, 0x38512000, 0x38514000, 0x38516000, 0x38518000, 0x3851a000, 0x3851c000, 0x3851e000,
	0x38520000, 0x38522000, 0x38524000, 0x38526000, 0x38528000, 0x3852a000, 0x3852c000, 0x3852e000,
	0x38530000, 0x38532000, 0x38534000, 0x38536000, 0x38538000, 0x3853a000, 0x3853c000, 0x3853e000,
	0x38540000, 0x38542000, 0x38544000, 0x38546000, 0x38548000, 0x3854a000, 0x3854c000, 0x3854e000,
	0x38550000, 0x38552000, 0x38554000, 0x38556000, 0x38558000, 0x3855a000, 0x3855c000, 0x3855e000,
	0x38560000, 0x38562000, 0x38564000, 0x38566000, 0x38568000, 0x3856a000, 0x3856c000, 0x3856e000,
	0x38570000, 0x38572000, 0x38574000, 0x38576000, 0x38578000, 0x3857a000, 0x3857c000, 0x3857e000,
	0x38580000, 0x38582000, 0x38584000, 0x38586000, 0x38588000, 0x3858a000, 0x3858c000, 0x3858e000,
	0x38590000, 0x38592000, 0x38594000, 0x38596000, 0x38598000, 0x3859a000, 0x3859c000, 0x3859e000,
	0x385a0000, 0x385a2000, 0x385a4000, 0x385a6000, 0x385a8000, 0x385aa000, 0x385ac000, 0x385ae000,
	0x385b0000, 0x385b2000, 0x385b4000, 0x385b6000, 0x385b8000, 0x385ba000, 0x385bc000, 0x385be000,
	0x385c0000, 0x385c2000, 0x385c4000, 0x385c6000, 0x385c8000, 0x385ca000, 0x385cc000, 0x385ce000,
	0x385d0000, 0x385d2000, 0x385d4000, 0x385d6000, 0x385d8000, 0x385da000, 0x385dc000, 0x385de000,
	0x385e0000, 0x385e2000, 0x385e4000, 0x385e6000, 0x385e8000, 0x385ea000, 0x385ec000, 0x385ee000,
	0x385f0000, 0x385f2000, 0x385f4000, 0x385f6000, 0x385f8000, 0x385fa000, 0x385fc000, 0x385fe000,
	0x38600000, 0x38602000, 0x38604000, 0x38606000, 0x38608000, 0x3860a000, 0x3860c000, 0x3860e000,
	0x38610000, 0x38612000, 0x38614000, 0x38616000, 0x38618000, 0x3861a000, 0x3861c000, 0x3861e000,
	0x38620000, 0x38622000, 0x38624000, 0x38626000, 0x38628000, 0x3862a000, 0x3862c000, 0x3862e000,
	0x38630000, 0x38632000, 0x38634000, 0x38636000, 0x38638000, 0x3863a000, 0x3863c000, 0x3863e000,
	0x38640000, 0x38642000, 0x38644000, 0x38646000, 0x38648000, 0x3864a000, 0x3864c000, 0x3864e000,
	0x38650000, 0x38652000, 0x38654000, 0x38656000, 0x38658000, 0x3865a000, 0x3865c000, 0x3865e000,
	0x38660000, 0x38662000, 0x38664000, 0x38666000, 0x38668000, 0x3866a000, 0x3866c000, 0x3866e000,
	0x38670000, 0x38672000, 0x38674000, 0x38676000, 0x38678000, 0x3867a000, 0x3867c000, 0x3867e000,
	0x38680000, 0x38682000, 0x38684000, 0x38686000, 0x38688000, 0x3868a000, 0x3868c000, 0x3868e000,
	0x38690000, 0x38692000, 0x38694000, 0x38696000, 0x38698000, 0x3869a000, 0x3869c000, 0x3869e000,
	0x386a0000, 0x386a2000, 0x386a4000, 0x386a6000, 0x386a8000, 0x386aa000, 0x386ac000, 0x386ae000,
	0x386b0000, 0x386b2000, 0x386b4000, 0x386b6000, 0x386b8000, 0x386ba000, 0x386bc000, 0x386be000,
	0x386c0000, 0x386c2000, 0x386c4000, 0x386c6000, 0x386c8000, 0x386ca000, 0x386cc000, 0x386ce000,
	0x386d0000, 0x386d2000, 0x386d4000, 0x386d6000, 0x386d8000, 0x386da000, 0x386dc000, 0x386de000,
	0x386e0000, 0x386e2000, 0x386e4000, 0x386e6000, 0x386e8000, 0x386ea000, 0x386ec000, 0x386ee000,
	0x386f0000, 0x386f2000, 0x386f4000, 0x386f6000, 0x386f8000, 0x386fa000, 0x386fc000, 0x386fe000,
	0x38700000, 0x38702000, 0x38704000, 0x38706000, 0x38708000, 0x3870a000, 0x3870c000, 0x3870e000,
	0x38710000, 0x38712000, 0x38714000, 0x38716000, 0x38718000, 0x3871a000, 0x3871c000, 0x3871e000,
	0x38720000, 0x38722000, 0x38724000, 0x38726000, 0x38728000, 0x3872a000, 0x3872c000, 0x3872e000,
	0x38730000, 0x38732000, 0x38734000, 0x38736000, 0x38738000, 0x3873a000, 0x3873c000, 0x3873e000,
	0x38740000, 0x38742000, 0x38744000, 0x38746000, 0x38748000, 0x3874a000, 0x3874c000, 0x3874e000,
	0x38750000, 0x38752000, 0x38754000, 0x38756000, 0x38758000, 0x3875a000, 0x3875c000, 0x3875e000,
	0x38760000, 0x38762000, 0x38764000, 0x38766000, 0x38768000, 0x3876a000, 0x3876c000, 0x3876e000,
	0x38770000, 0x38772000, 0x38774000, 0x38776000, 0x38778000, 0x3877a000, 0x3877c000, 0x3877e000,
	0x38780000, 0x38782000, 0x38784000, 0x38786000, 0x38788000, 0x3878a000, 0x3878c000, 0x3878e000,
	0x38790000, 0x38792000, 0x38794000, 0x38796000, 0x38798000, 0x3879a000, 0x3879c000, 0x3879e000,
	0x387a0000, 0x387a2000, 0x387a4000, 0x387a6000, 0x387a8000, 0x387aa000, 0x387ac000, 0x387ae000,
	0x387b0000, 0x387b2000, 0x387b4000, 0x387b6000, 0x387b8000, 0x387ba000, 0x387bc000, 0x387be000,
	0x387c0000, 0x387c2000, 0x387c4000, 0x387c6000, 0x387c8000, 0x387ca000, 0x387cc000, 0x387ce000,
	0x387d0000, 0x387d2000, 0x387d4000, 0x387d6000, 0x387d8000, 0x387da000, 0x387dc000, 0x387de000,
	0x387e0000, 0x387e2000, 0x387e4000, 0x387e6000, 0x387e8000, 0x387ea000, 0x387ec000, 0x387ee000,
	0x387f0000, 0x387f2000, 0x387f4000, 0x387f6000, 0x387f8000, 0x387fa000, 0x387fc000, 0x387fe000,
};

static uint32_t _ccv_exponent_table[64] = {
	0x0, 0x800000, 0x1000000, 0x1800000, 0x2000000, 0x2800000, 0x3000000, 0x3800000,
	0x4000000, 0x4800000, 0x5000000, 0x5800000, 0x6000000, 0x6800000, 0x7000000, 0x7800000,
	0x8000000, 0x8800000, 0x9000000, 0x9800000, 0xa000000, 0xa800000, 0xb000000, 0xb800000,
	0xc000000, 0xc800000, 0xd000000, 0xd800000, 0xe000000, 0xe800000, 0xf000000, 0x47800000,
	0x80000000, 0x80800000, 0x81000000, 0x81800000, 0x82000000, 0x82800000, 0x83000000, 0x83800000,
	0x84000000, 0x84800000, 0x85000000, 0x85800000, 0x86000000, 0x86800000, 0x87000000, 0x87800000,
	0x88000000, 0x88800000, 0x89000000, 0x89800000, 0x8a000000, 0x8a800000, 0x8b000000, 0x8b800000,
	0x8c000000, 0x8c800000, 0x8d000000, 0x8d800000, 0x8e000000, 0x8e800000, 0x8f000000, 0xc7800000,
};

static uint16_t _ccv_offset_table[64] = {
	0x0, 0x400, 0x400, 0x400, 0x400, 0x400, 0x400, 0x400,
	0x400, 0x400, 0x400, 0x400, 0x400, 0x400, 0x400, 0x400,
	0x400, 0x400, 0x400, 0x400, 0x400, 0x400, 0x400, 0x400,
	0x400, 0x400, 0x400, 0x400, 0x400, 0x400, 0x400, 0x400,
	0x0, 0x400, 0x400, 0x400, 0x400, 0x400, 0x400, 0x400,
	0x400, 0x400, 0x400, 0x400, 0x400, 0x400, 0x400, 0x400,
	0x400, 0x400, 0x400, 0x400, 0x400, 0x400, 0x400, 0x400,
	0x400, 0x400, 0x400, 0x400, 0x400, 0x400, 0x400, 0x400,
};

void ccv_half_precision_to_float(uint16_t* h, float* f, size_t len)
{
	int i;
	uint32_t* u = (uint32_t*)f;
	for (i = 0; i < len; i++)
		u[i] = _ccv_mantissa_table[_ccv_offset_table[h[i] >> 10] + (h[i] & 0x3ff)] + _ccv_exponent_table[h[i] >> 10];
}

void ccv_half_precision_to_double(uint16_t* h, double* f, size_t len)
{
	int i;
	for (i = 0; i < len; i++)
	{
		union {
			float v;
			uint32_t p;
		} u = {
			.p = _ccv_mantissa_table[_ccv_offset_table[h[i] >> 10] + (h[i] & 0x3ff)] + _ccv_exponent_table[h[i] >> 10]
		};
		f[i] = u.v;
	}
}

void ccv_array_push(ccv_array_t* array, const void* r)
{
	array->rnum++;
	if (array->rnum > array->size)
	{
		array->size = ccv_max(array->size * 3 / 2, array->size + 1);
		array->data = ccrealloc(array->data, (size_t)array->size * (size_t)array->rsize);
	}
	memcpy(ccv_array_get(array, array->rnum - 1), r, array->rsize);
}

void ccv_array_zero(ccv_array_t* array)
{
	memset(array->data, 0, (size_t)array->size * (size_t)array->rsize);
}

void ccv_array_resize(ccv_array_t* array, int rnum)
{
	if (rnum > array->size)
	{
		array->size = ccv_max(array->size * 3 / 2, rnum);
		array->data = ccrealloc(array->data, (size_t)array->size * (size_t)array->rsize);
	}
	if (rnum > array->rnum)
		memset(ccv_array_get(array, array->rnum), 0, (size_t)array->rsize * (size_t)(rnum - array->rnum));
	array->rnum = rnum;
}

void ccv_array_clear(ccv_array_t* array)
{
	array->rnum = 0;
}

typedef struct ccv_ptree_node_t
{
	struct ccv_ptree_node_t* parent;
	void* element;
	int rank;
} ccv_ptree_node_t;

/* the code for grouping array is adopted from OpenCV's cvSeqPartition func, it is essentially a find-union algorithm */
int ccv_array_group(ccv_array_t* array, ccv_array_t** index, ccv_array_group_f gfunc, void* data)
{
	int i, j;
	ccv_ptree_node_t* node = (ccv_ptree_node_t*)ccmalloc(array->rnum * sizeof(ccv_ptree_node_t));
	for (i = 0; i < array->rnum; i++)
	{
		node[i].parent = 0;
		node[i].element = ccv_array_get(array, i);
		node[i].rank = 0;
	}
	for (i = 0; i < array->rnum; i++)
	{
		if (!node[i].element)
			continue;
		ccv_ptree_node_t* root = node + i;
		while (root->parent)
			root = root->parent;
		for (j = 0; j < array->rnum; j++)
		{
			if( i != j && node[j].element && gfunc(node[i].element, node[j].element, data))
			{
				ccv_ptree_node_t* root2 = node + j;

				while(root2->parent)
					root2 = root2->parent;

				if(root2 != root)
				{
					if(root->rank > root2->rank)
						root2->parent = root;
					else
					{
						root->parent = root2;
						root2->rank += root->rank == root2->rank;
						root = root2;
					}

					/* compress path from node2 to the root: */
					ccv_ptree_node_t* node2 = node + j;
					while(node2->parent)
					{
						ccv_ptree_node_t* temp = node2;
						node2 = node2->parent;
						temp->parent = root;
					}

					/* compress path from node to the root: */
					node2 = node + i;
					while(node2->parent)
					{
						ccv_ptree_node_t* temp = node2;
						node2 = node2->parent;
						temp->parent = root;
					}
				}
			}
		}
	}
	if (*index == 0)
		*index = ccv_array_new(sizeof(int), array->rnum, 0);
	else
		ccv_array_clear(*index);
	ccv_array_t* idx = *index;

	int class_idx = 0;
	for(i = 0; i < array->rnum; i++)
	{
		j = -1;
		ccv_ptree_node_t* node1 = node + i;
		if(node1->element)
		{
			while(node1->parent)
				node1 = node1->parent;
			if(node1->rank >= 0)
				node1->rank = ~class_idx++;
			j = ~node1->rank;
		}
		ccv_array_push(idx, &j);
	}
	ccfree(node);
	return class_idx;
}

ccv_contour_t* ccv_contour_new(int set)
{
	ccv_contour_t* contour = (ccv_contour_t*)ccmalloc(sizeof(ccv_contour_t));
	contour->rect.x = contour->rect.y =
	contour->rect.width = contour->rect.height = 0;
	contour->size = 0;
	if (set)
		contour->set = ccv_array_new(sizeof(ccv_point_t), 5, 0);
	else
		contour->set = 0;
	return contour;
}

void ccv_contour_push(ccv_contour_t* contour, ccv_point_t point)
{
	if (contour->size == 0)
	{
		contour->rect.x = point.x;
		contour->rect.y = point.y;
		contour->rect.width = contour->rect.height = 1;
		contour->m10 = point.x;
		contour->m01 = point.y;
		contour->m11 = point.x * point.y;
		contour->m20 = point.x * point.x;
		contour->m02 = point.y * point.y;
		contour->size = 1;
	} else {
		if (point.x < contour->rect.x)
		{
			contour->rect.width += contour->rect.x - point.x;
			contour->rect.x = point.x;
		} else if (point.x > contour->rect.x + contour->rect.width - 1) {
			contour->rect.width = point.x - contour->rect.x + 1;
		}
		if (point.y < contour->rect.y)
		{
			contour->rect.height += contour->rect.y - point.y;
			contour->rect.y = point.y;
		} else if (point.y > contour->rect.y + contour->rect.height - 1) {
			contour->rect.height = point.y - contour->rect.y + 1;
		}
		contour->m10 += point.x;
		contour->m01 += point.y;
		contour->m11 += point.x * point.y;
		contour->m20 += point.x * point.x;
		contour->m02 += point.y * point.y;
		contour->size++;
	}
	if (contour->set)
		ccv_array_push(contour->set, &point);
}

void ccv_contour_free(ccv_contour_t* contour)
{
	if (contour->set)
		ccv_array_free(contour->set);
	ccfree(contour);
}
