#include "ccv.h"

int _ccv_get_sparse_prime[] = { 53, 97, 193, 389, 769, 1543, 3079, 6151, 12289, 24593, 49157, 98317, 196613, 393241, 786433, 1572869 };

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

void ccv_shift(ccv_matrix_t* a, ccv_matrix_t** b, int type, int lr, int rr)
{
	ccv_dense_matrix_t* da = ccv_get_dense_matrix(a);
	char identifier[64];
	memset(identifier, 0, 64);
	snprintf(identifier, 64, "ccv_shift(%d,%d)", lr, rr);
	uint64_t sig = ccv_matrix_generate_signature(identifier, 64, da->sig, 0);
	type = (type == 0) ? CCV_GET_DATA_TYPE(da->type) | CCV_GET_CHANNEL(da->type) : CCV_GET_DATA_TYPE(type) | CCV_GET_CHANNEL(da->type);
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, da->rows, da->cols, CCV_ALL_DATA_TYPE | CCV_GET_CHANNEL(da->type), type, sig); 
	ccv_cache_return(db, );
	int i, j, ch = CCV_GET_CHANNEL(da->type);
	unsigned char* aptr = da->data.ptr;
	unsigned char* bptr = db->data.ptr;
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

ccv_dense_vector_t* ccv_get_sparse_matrix_vector(ccv_sparse_matrix_t* mat, int index)
{
	if (mat->vector[(index * 33) % CCV_GET_SPARSE_PRIME(mat->prime)].index != -1)
	{
		ccv_dense_vector_t* vector = &mat->vector[(index * 33) % CCV_GET_SPARSE_PRIME(mat->prime)];
		while (vector != 0 && vector->index != index)
			vector = vector->next;
		return vector;
	}
	return 0;
}

ccv_matrix_cell_t ccv_get_sparse_matrix_cell(ccv_sparse_matrix_t* mat, int row, int col)
{
	ccv_dense_vector_t* vector = ccv_get_sparse_matrix_vector(mat, (mat->major == CCV_SPARSE_COL_MAJOR) ? col : row);
	ccv_matrix_cell_t cell;
	cell.ptr = 0;
	if (vector != 0 && vector->length > 0)
	{
		int cell_width = CCV_GET_DATA_TYPE_SIZE(mat->type) * CCV_GET_CHANNEL(mat->type);
		int vidx = (mat->major == CCV_SPARSE_COL_MAJOR) ? row : col;
		if (mat->type & CCV_DENSE_VECTOR)
		{
			cell.ptr = vector->data.ptr + cell_width * vidx;
		} else {
			int h = (vidx * 33) % vector->length, i = 0;
			while (vector->indice[(h + i * i) % vector->length] != vidx && vector->indice[(h + i * i) % vector->length] != -1)
				i++;
			i = (h + i * i) % vector->length;
			if (vector->indice[i] != -1)
				cell.ptr = vector->data.ptr + i * cell_width;
		}
	}
	return cell;
}

static void _ccv_dense_vector_expand(ccv_sparse_matrix_t* mat, ccv_dense_vector_t* vector)
{
	if (vector->prime == -1)
		return;
	vector->prime++;
	int new_length = CCV_GET_SPARSE_PRIME(vector->prime);
	int cell_width = CCV_GET_DATA_TYPE_SIZE(mat->type) * CCV_GET_CHANNEL(mat->type);
	int new_step = (new_length * cell_width + 3) & -4;
	ccv_matrix_cell_t new_data;
	new_data.ptr = (unsigned char*)ccmalloc(new_step + sizeof(int) * new_length);
	int* new_indice = (int*)(new_data.ptr + new_step);
	int i;
	for (i = 0; i < new_length; i++)
		new_indice[i] = -1;
	for (i = 0; i < vector->length; i++)
		if (vector->indice[i] != -1)
		{
			int index = vector->indice[i];
			int h = (index * 33) % new_length, j = 0;
			while (new_indice[(h + j * j) % new_length] != index && new_indice[(h + j * j) % new_length] != -1)
				j++;
			j = (h + j * j) % new_length;
			new_indice[j] = index;
			memcpy(new_data.ptr + j * cell_width, vector->data.ptr + i * cell_width, cell_width);
		}
	vector->length = new_length;
	ccfree(vector->data.ptr);
	vector->data = new_data;
	vector->indice = new_indice;
}

static void _ccv_sparse_matrix_expand(ccv_sparse_matrix_t* mat)
{
	int length = CCV_GET_SPARSE_PRIME(mat->prime);
	mat->prime++;
	int new_length = CCV_GET_SPARSE_PRIME(mat->prime);
	ccv_dense_vector_t* new_vector = (ccv_dense_vector_t*)ccmalloc(new_length * sizeof(ccv_dense_vector_t));
	int i;
	for (i = 0; i < new_length; i++)
	{
		new_vector[i].index = -1;
		new_vector[i].length = 0;
		new_vector[i].next = 0;
	}
	for (i = 0; i < length; i++)
		if (mat->vector[i].index != -1)
		{
			int h = (mat->vector[i].index * 33) % new_length;
			if (new_vector[h].length == 0)
			{
				memcpy(new_vector + h, mat->vector + i, sizeof(ccv_dense_vector_t));
				new_vector[h].next = 0;
			} else {
				ccv_dense_vector_t* t = (ccv_dense_vector_t*)ccmalloc(sizeof(ccv_dense_vector_t));
				memcpy(t, mat->vector + i, sizeof(ccv_dense_vector_t));
				t->next = new_vector[h].next;
				new_vector[h].next = t;
			}
			ccv_dense_vector_t* iter = mat->vector[i].next;
			while (iter != 0)
			{
				ccv_dense_vector_t* iter_next = iter->next;
				h = (iter->index * 33) % new_length;
				if (new_vector[h].length == 0)
				{
					memcpy(new_vector + h, iter, sizeof(ccv_dense_vector_t));
					new_vector[h].next = 0;
					ccfree(iter);
				} else {
					iter->next = new_vector[h].next;
					new_vector[h].next = iter;
				}
				iter = iter_next;
			}
		}
	ccfree(mat->vector);
	mat->vector = new_vector;
}

void ccv_set_sparse_matrix_cell(ccv_sparse_matrix_t* mat, int row, int col, void* data)
{
	int i;
	int index = (mat->major == CCV_SPARSE_COL_MAJOR) ? col : row;
	int vidx = (mat->major == CCV_SPARSE_COL_MAJOR) ? row : col;
	int length = CCV_GET_SPARSE_PRIME(mat->prime);
	ccv_dense_vector_t* vector = ccv_get_sparse_matrix_vector(mat, index);
	if (vector == 0)
	{
		mat->load_factor++;
		if (mat->load_factor * 4 > CCV_GET_SPARSE_PRIME(mat->prime) * 3)
		{
			_ccv_sparse_matrix_expand(mat);
			length = CCV_GET_SPARSE_PRIME(mat->prime);
		}
		vector = &mat->vector[(index * 33) % length];
		if (vector->index != -1)
		{
			vector = (ccv_dense_vector_t*)ccmalloc(sizeof(ccv_dense_vector_t));
			vector->index = -1;
			vector->length = 0;
			vector->next = mat->vector[(index * 33) % length].next;
			mat->vector[(index * 33) % length].next = vector;
		}
	}
	int cell_width = CCV_GET_DATA_TYPE_SIZE(mat->type) * CCV_GET_CHANNEL(mat->type);
	if (mat->type & CCV_DENSE_VECTOR)
	{
		if (vector->index == -1)
		{
			vector->prime = -1;
			vector->length = (mat->major == CCV_SPARSE_COL_MAJOR) ? mat->rows : mat->cols;
			vector->index = index;
			vector->step = (vector->length * cell_width + 3) & -4;
			vector->data.ptr = (unsigned char*)calloc(vector->step, 1);
		}
		if (data != 0)
			memcpy(vector->data.ptr + vidx * cell_width, data, cell_width);
	} else {
		if (vector->index == -1)
		{
			vector->prime = 0;
			vector->load_factor = 0;
			vector->length = CCV_GET_SPARSE_PRIME(vector->prime);
			vector->index = index;
			vector->step  = (vector->length * cell_width + 3) & -4;
			vector->data.ptr = (unsigned char*)ccmalloc(vector->step + sizeof(int) * vector->length);
			vector->indice = (int*)(vector->data.ptr + vector->step);
			for (i = 0; i < vector->length; i++)
				vector->indice[i] = -1;
		}
		vector->load_factor++;
		if (vector->load_factor * 2 > vector->length)
		{
			_ccv_dense_vector_expand(mat, vector);
		}
		i = 0;
		int h = (vidx * 33) % vector->length;
		while (vector->indice[(h + i * i) % vector->length] != vidx && vector->indice[(h + i * i) % vector->length] != -1)
			i++;
		i = (h + i * i) % vector->length;
		vector->indice[i] = vidx;
		if (data != 0)
			memcpy(vector->data.ptr + i * cell_width, data, cell_width);
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

CCV_IMPLEMENT_QSORT_EX(_ccv_indice_uchar_sort, int, _ccv_indice_less_than, _ccv_swap_indice_and_uchar_data, unsigned char*);
CCV_IMPLEMENT_QSORT_EX(_ccv_indice_int_sort, int, _ccv_indice_less_than, _ccv_swap_indice_and_int_data, int*);
CCV_IMPLEMENT_QSORT_EX(_ccv_indice_float_sort, int, _ccv_indice_less_than, _ccv_swap_indice_and_float_data, float*);
CCV_IMPLEMENT_QSORT_EX(_ccv_indice_double_sort, int, _ccv_indice_less_than, _ccv_swap_indice_and_double_data, double*);

void ccv_compress_sparse_matrix(ccv_sparse_matrix_t* mat, ccv_compressed_sparse_matrix_t** csm)
{
	int i, j;
	int nnz = 0;
	int length = CCV_GET_SPARSE_PRIME(mat->prime);
	for (i = 0; i < length; i++)
	{
		ccv_dense_vector_t* vector = &mat->vector[i];
#define while_block(_, _while_get) \
		while (vector != 0) \
		{ \
			if (vector->index != -1) \
			{ \
				if (mat->type & CCV_DENSE_VECTOR) \
				{ \
					for (j = 0; j < vector->length; j++) \
						if (_while_get(vector->data.ptr, j, 0) != 0) \
							nnz++; \
				} else { \
					nnz += vector->load_factor; \
				} \
			} \
			vector = vector->next; \
		}
		ccv_matrix_getter(mat->type, while_block);
#undef while_block
	}
	ccv_compressed_sparse_matrix_t* cm = *csm = (ccv_compressed_sparse_matrix_t*)ccmalloc(sizeof(ccv_compressed_sparse_matrix_t) + nnz * sizeof(int) + nnz * CCV_GET_DATA_TYPE_SIZE(mat->type) + (((mat->major == CCV_SPARSE_COL_MAJOR) ? mat->cols : mat->rows) + 1) * sizeof(int));
	cm->type = (mat->type & ~CCV_MATRIX_SPARSE & ~CCV_SPARSE_VECTOR & ~CCV_DENSE_VECTOR) | ((mat->major == CCV_SPARSE_COL_MAJOR) ? CCV_MATRIX_CSC : CCV_MATRIX_CSR);
	cm->nnz = nnz;
	cm->rows = mat->rows;
	cm->cols = mat->cols;
	cm->index = (int*)(cm + 1);
	cm->offset = cm->index + nnz;
	cm->data.i = cm->offset + ((mat->major == CCV_SPARSE_COL_MAJOR) ? mat->cols : mat->rows) + 1;
	unsigned char* m_ptr = cm->data.ptr;
	int* idx = cm->index;
	cm->offset[0] = 0;
	for (i = 0; i < ((mat->major == CCV_SPARSE_COL_MAJOR) ? mat->cols : mat->rows); i++)
	{
		ccv_dense_vector_t* vector = ccv_get_sparse_matrix_vector(mat, i);
		if (vector == 0)
			cm->offset[i + 1] = cm->offset[i];
		else {
			if (mat->type & CCV_DENSE_VECTOR)
			{
				int k = 0;
#define for_block(_for_set, _for_get) \
				for (j = 0; j < vector->length; j++) \
					if (_for_get(vector->data.ptr, j, 0) != 0) \
					{ \
						_for_set(m_ptr, k, _for_get(vector->data.ptr, j, 0), 0); \
						idx[k] = j; \
						k++; \
					}
				ccv_matrix_setter(mat->type, ccv_matrix_getter, mat->type, for_block);
#undef for_block
				cm->offset[i + 1] = cm->offset[i] + k;
				idx += k;
				m_ptr += k * CCV_GET_DATA_TYPE_SIZE(mat->type);
			} else {
				int k = 0;
#define for_block(_for_set, _for_get) \
				for (j = 0; j < vector->length; j++) \
					if (vector->indice[j] != -1) \
					{ \
						_for_set(m_ptr, k, _for_get(vector->data.ptr, j, 0), 0); \
						idx[k] = vector->indice[j]; \
						k++; \
					}
				ccv_matrix_setter(mat->type, ccv_matrix_getter, mat->type, for_block);
#undef for_block
				switch (CCV_GET_DATA_TYPE(mat->type))
				{
					case CCV_8U:
						_ccv_indice_uchar_sort(idx, vector->load_factor, (unsigned char*)m_ptr);
						break;
					case CCV_32S:
						_ccv_indice_int_sort(idx, vector->load_factor, (int*)m_ptr);
						break;
					case CCV_32F:
						_ccv_indice_float_sort(idx, vector->load_factor, (float*)m_ptr);
						break;
					case CCV_64F:
						_ccv_indice_double_sort(idx, vector->load_factor, (double*)m_ptr);
						break;
				}
				cm->offset[i + 1] = cm->offset[i] + vector->load_factor;
				idx += vector->load_factor;
				m_ptr += vector->load_factor * CCV_GET_DATA_TYPE_SIZE(mat->type);
			}
		}
	}
}

void ccv_decompress_sparse_matrix(ccv_compressed_sparse_matrix_t* csm, ccv_sparse_matrix_t** smt)
{
	ccv_sparse_matrix_t* mat = *smt = ccv_sparse_matrix_new(csm->rows, csm->cols, csm->type & ~CCV_MATRIX_CSR & ~CCV_MATRIX_CSC, (csm->type & CCV_MATRIX_CSR) ? CCV_SPARSE_ROW_MAJOR : CCV_SPARSE_COL_MAJOR, 0);
	int i, j;
	for (i = 0; i < ((mat->major == CCV_SPARSE_COL_MAJOR) ? mat->cols : mat->rows); i++)
		for (j = csm->offset[i]; j < csm->offset[i + 1]; j++)
			if (mat->major == CCV_SPARSE_COL_MAJOR)
				ccv_set_sparse_matrix_cell(mat, csm->index[j], i, csm->data.ptr + CCV_GET_DATA_TYPE_SIZE(csm->type) * j);
			else
				ccv_set_sparse_matrix_cell(mat, i, csm->index[j], csm->data.ptr + CCV_GET_DATA_TYPE_SIZE(csm->type) * j);
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
		unsigned char* a_ptr = da->data.ptr;
		unsigned char* b_ptr = db->data.ptr;
#define for_block(_, _for_get) \
		for (i = 0; i < da->rows; i++) \
		{ \
			for (j = 0; j < da->cols * ch; j++) \
			{ \
				if (fabs(_for_get(b_ptr, j, 0) - _for_get(a_ptr, j, 0)) > 1e-6) \
					return -1; \
			} \
			a_ptr += da->step; \
			b_ptr += db->step; \
		}
		ccv_matrix_getter(da->type, for_block);
#undef for_block
	}
	return 0;
}

void ccv_slice(ccv_matrix_t* a, ccv_matrix_t** b, int btype, int y, int x, int rows, int cols)
{
	int type = *(int*)a;
	if (type & CCV_MATRIX_DENSE)
	{
		ccv_dense_matrix_t* da = ccv_get_dense_matrix(a);
		assert(y >= 0 && y + rows <= da->rows && x >= 0 && x + cols <= da->cols);
		char identifier[128];
		memset(identifier, 0, 128);
		snprintf(identifier, 128, "ccv_slice(%d,%d,%d,%d)", y, x, rows, cols);
		uint64_t sig = (da->sig == 0) ? 0 : ccv_matrix_generate_signature(identifier, 128, da->sig, 0);
		btype = (btype == 0) ? CCV_GET_DATA_TYPE(da->type) | CCV_GET_CHANNEL(da->type) : CCV_GET_DATA_TYPE(btype) | CCV_GET_CHANNEL(da->type);
		ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, rows, cols, CCV_ALL_DATA_TYPE | CCV_GET_CHANNEL(da->type), btype, sig);
		ccv_cache_return(db, );
		int i, j, ch = CCV_GET_CHANNEL(da->type);
		unsigned char* a_ptr = da->data.ptr + x * ch * CCV_GET_DATA_TYPE_SIZE(da->type) + y * da->step;
		unsigned char* b_ptr = db->data.ptr;
#define for_block(_, _for_set, _for_get) \
		for (i = 0; i < rows; i++) \
		{ \
			for (j = 0; j < cols * ch; j++) \
			{ \
				_for_set(b_ptr, j, _for_get(a_ptr, j, 0), 0); \
			} \
			a_ptr += da->step; \
			b_ptr += db->step; \
		}
		ccv_matrix_setter_getter(da->type, for_block);
#undef for_block
	} else if (type & CCV_MATRIX_SPARSE) {
	}
}

void ccv_move(ccv_matrix_t* a, ccv_matrix_t** b, int btype, int y, int x)
{
	int type = *(int*)a;
	if (type & CCV_MATRIX_DENSE)
	{
		ccv_dense_matrix_t* da = ccv_get_dense_matrix(a);
		char identifier[128];
		memset(identifier, 0, 128);
		snprintf(identifier, 128, "ccv_move(%d,%d)", y, x);
		uint64_t sig = (da->sig == 0) ? 0 : ccv_matrix_generate_signature(identifier, 128, da->sig, 0);
		btype = (btype == 0) ? CCV_GET_DATA_TYPE(da->type) | CCV_GET_CHANNEL(da->type) : CCV_GET_DATA_TYPE(btype) | CCV_GET_CHANNEL(da->type);
		ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, da->rows, da->cols, CCV_ALL_DATA_TYPE | CCV_GET_CHANNEL(da->type), btype, sig);
		ccv_cache_return(db, );
		int i, j, ch = CCV_GET_CHANNEL(da->type);
		unsigned char* a_ptr = da->data.ptr + ccv_max(x, 0) * ch * CCV_GET_DATA_TYPE_SIZE(da->type) + ccv_max(y, 0) * da->step;
		unsigned char* b_ptr = db->data.ptr + ccv_max(-x, 0) * ch * CCV_GET_DATA_TYPE_SIZE(db->type) + ccv_max(-y, 0) * db->step;
#define for_block(_, _for_set, _for_get) \
		for (i = abs(y); i < db->rows; i++) \
		{ \
			for (j = abs(x) * ch; j < db->cols * ch; j++) \
			{ \
				_for_set(b_ptr, j, _for_get(a_ptr, j, 0), 0); \
			} \
			a_ptr += da->step; \
			b_ptr += db->step; \
		}
		ccv_matrix_setter_getter(da->type, for_block);
#undef for_block
	} else if (type & CCV_MATRIX_SPARSE) {
	}
}

ccv_array_t* ccv_array_new(int rnum, int rsize)
{
	ccv_array_t* array = (ccv_array_t*)ccmalloc(sizeof(ccv_array_t));
	array->rnum = 0;
	array->rsize = rsize;
	array->size = rnum;
	array->data = ccmalloc(rnum * rsize);
	return array;
}

void ccv_array_push(ccv_array_t* array, void* r)
{
	array->rnum++;
	if (array->rnum > array->size)
	{
		array->size = array->size * 2;
		array->data = ccrealloc(array->data, array->size * array->rsize);
	}
	memcpy(ccv_array_get(array, array->rnum - 1), r, array->rsize);
}

void ccv_array_zero(ccv_array_t* array)
{
	memset(array->data, 0, array->size * array->rsize);
}

void ccv_array_clear(ccv_array_t* array)
{
	array->rnum = 0;
}

void ccv_array_free(ccv_array_t* array)
{
	ccfree(array->data);
	ccfree(array);
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
		*index = ccv_array_new(array->rnum, sizeof(int));
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
		contour->set = ccv_array_new(5, sizeof(ccv_point_t));
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
