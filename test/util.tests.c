#include "ccv.h"
#include "case.h"
#include "ccv_case.h"

TEST_CASE("dynamic array")
{
	ccv_array_t* array = ccv_array_new(2, 4);
	int i;
	i = 1;
	ccv_array_push(array, &i);
	i = 2;
	ccv_array_push(array, &i);
	i = 3;
	ccv_array_push(array, &i);
	i = 4;
	ccv_array_push(array, &i);
	i = 5;
	ccv_array_push(array, &i);
	REQUIRE_EQ(5, array->rnum, "should have 5 elements pushed to array");
	for (i = 0; i < array->rnum; i++)
		REQUIRE_EQ(i + 1, ((int*)ccv_array_get(array, i))[0], "check element values in array");
	ccv_array_clear(array);
	i = 3;
	ccv_array_push(array, &i);
	i = 4;
	ccv_array_push(array, &i);
	i = 5;
	ccv_array_push(array, &i);
	REQUIRE_EQ(3, array->rnum, "should have 3 elements after clear");
	for (i = 0; i < array->rnum; i++)
		REQUIRE_EQ(i + 3, ((int*)ccv_array_get(array, i))[0], "check element values in array after clear at index %d", i);
	ccv_array_free(array);
}

int is_equal(const void* r1, const void* r2, void* data)
{
	int a = *(int*)r1;
	int b = *(int*)r2;
	return a == b;
}

TEST_CASE("group array with is_equal function")
{
	ccv_array_t* array = ccv_array_new(2, 4);
	int i;
	i = 1;
	ccv_array_push(array, &i);
	i = 2;
	ccv_array_push(array, &i);
	i = 2;
	ccv_array_push(array, &i);
	i = 2;
	ccv_array_push(array, &i);
	i = 5;
	ccv_array_push(array, &i);
	i = 3;
	ccv_array_push(array, &i);
	i = 4;
	ccv_array_push(array, &i);
	i = 5;
	ccv_array_push(array, &i);
	ccv_array_t* idx = 0;
	ccv_array_group(array, &idx, is_equal, 0);
	REQUIRE_EQ(((int*)ccv_array_get(idx, 1))[0], ((int*)ccv_array_get(idx, 2))[0], "element 2, 3 should in the same group");
	REQUIRE_EQ(((int*)ccv_array_get(idx, 2))[0], ((int*)ccv_array_get(idx, 3))[0], "element 3, 4 should in the same group");
	REQUIRE_EQ(((int*)ccv_array_get(idx, 4))[0], ((int*)ccv_array_get(idx, 7))[0], "element 4, 8 should in the same group");
	REQUIRE_NOT_EQ(((int*)ccv_array_get(idx, 0))[0], ((int*)ccv_array_get(idx, 1))[0], "element 1, 2 should not in the same group");
	REQUIRE_NOT_EQ(((int*)ccv_array_get(idx, 0))[0], ((int*)ccv_array_get(idx, 4))[0], "element 1, 5 should not in the same group");
	REQUIRE_NOT_EQ(((int*)ccv_array_get(idx, 0))[0], ((int*)ccv_array_get(idx, 5))[0], "element 1, 6 should not in the same group");
	REQUIRE_NOT_EQ(((int*)ccv_array_get(idx, 0))[0], ((int*)ccv_array_get(idx, 6))[0], "element 1, 7 should not in the same group");
	REQUIRE_NOT_EQ(((int*)ccv_array_get(idx, 1))[0], ((int*)ccv_array_get(idx, 4))[0], "element 2, 5 should not in the same group");
	REQUIRE_NOT_EQ(((int*)ccv_array_get(idx, 1))[0], ((int*)ccv_array_get(idx, 5))[0], "element 2, 6 should not in the same group");
	REQUIRE_NOT_EQ(((int*)ccv_array_get(idx, 1))[0], ((int*)ccv_array_get(idx, 6))[0], "element 2, 7 should not in the same group");
	REQUIRE_NOT_EQ(((int*)ccv_array_get(idx, 4))[0], ((int*)ccv_array_get(idx, 5))[0], "element 5, 6 should not in the same group");
	REQUIRE_NOT_EQ(((int*)ccv_array_get(idx, 4))[0], ((int*)ccv_array_get(idx, 6))[0], "element 5, 7 should not in the same group");
	REQUIRE_NOT_EQ(((int*)ccv_array_get(idx, 5))[0], ((int*)ccv_array_get(idx, 6))[0], "element 6, 7 should not in the same group");
	ccv_array_free(array);
	ccv_array_free(idx);
}

TEST_CASE("compress sparse matrix")
{
	ccv_sparse_matrix_t* mat = ccv_sparse_matrix_new(3, 3, CCV_32F | CCV_C1, CCV_SPARSE_ROW_MAJOR, 0);
	float cell;
	cell = 1.0;
	ccv_set_sparse_matrix_cell(mat, 0, 0, &cell);
	cell = 2.0;
	ccv_set_sparse_matrix_cell(mat, 0, 2, &cell);
	cell = 3.0;
	ccv_set_sparse_matrix_cell(mat, 1, 2, &cell);
	cell = 4.0;
	ccv_set_sparse_matrix_cell(mat, 2, 0, &cell);
	cell = 5.0;
	ccv_set_sparse_matrix_cell(mat, 2, 1, &cell);
	cell = 6.0;
	ccv_set_sparse_matrix_cell(mat, 2, 2, &cell);
	ccv_compressed_sparse_matrix_t* csm = 0;
	ccv_compress_sparse_matrix(mat, &csm);
	float dm[6] = {1, 2, 3, 4, 5, 6};
	int di[6] = {0, 2, 2, 0, 1, 2};
	REQUIRE_EQ(6, csm->nnz, "compress to non-zero factor of 6");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dm, csm->data.fl, 6, 1e-6, "actual element value should be the same");
	REQUIRE_ARRAY_EQ(int, di, csm->index, 6, "element index of CSR");
	REQUIRE_EQ(3, csm->rows, "compress should have the same number of rows (CSR)");
	int df[4] = {0, 2, 3, 6};
	REQUIRE_ARRAY_EQ(int, df, csm->offset, 4, "offset of leading element in each row");
	ccv_sparse_matrix_t* smt = 0;
	ccv_decompress_sparse_matrix(csm, &smt);
	float m[3][3] = {{1, 0, 2}, {0, 0, 3}, {4, 5, 6}};
	int i, j;
	for (i = 0; i < 3; i++)
		for (j = 0; j < 3; j++)
		{
			ccv_matrix_cell_t cell = ccv_get_sparse_matrix_cell(smt, i, j);
			REQUIRE_EQ_WITH_TOLERANCE(m[i][j], (cell.ptr != 0) ? cell.fl[0] : 0, 1e-6, "should have the same matrix after decompressed at row %d, col %d", i, j);
		}
	ccv_matrix_free(smt);
	ccv_matrix_free(mat);
	ccv_matrix_free(csm);
}

TEST_CASE("matrix slice")
{
	ccv_dense_matrix_t* image = 0;
	ccv_unserialize("../samples/chessbox.png", &image, CCV_SERIAL_ANY_FILE);
	ccv_dense_matrix_t* b = 0;
	ccv_slice(image, (ccv_matrix_t**)&b, 0, 33, 41, 111, 91);
	REQUIRE_MATRIX_FILE_EQ(b, "data/chessbox.slice.bin", "should have data/chessbox.png sliced at (33, 41) with 111 x 91");
	ccv_matrix_free(image);
	ccv_matrix_free(b);
}

#include "case_main.h"
