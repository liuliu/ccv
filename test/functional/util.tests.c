#include "ccv.h"
#include "case.h"
#include "ccv_case.h"

TEST_CASE("type size macro")
{
	REQUIRE_EQ(CCV_GET_DATA_TYPE_SIZE(CCV_8U), 1, "CCV_8U should have size 1");
	REQUIRE_EQ(CCV_GET_DATA_TYPE_SIZE(CCV_32S), 4, "CCV_32S should have size 4");
	REQUIRE_EQ(CCV_GET_DATA_TYPE_SIZE(CCV_32F), 4, "CCV_32F should have size 4");
	REQUIRE_EQ(CCV_GET_DATA_TYPE_SIZE(CCV_64S), 8, "CCV_64S should have size 8");
	REQUIRE_EQ(CCV_GET_DATA_TYPE_SIZE(CCV_64F), 8, "CCV_64F should have size 8");
}

TEST_CASE("dynamic array")
{
	ccv_array_t* array = ccv_array_new(4, 2, 0);
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
	ccv_array_t* array = ccv_array_new(4, 2, 0);
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

TEST_CASE("sparse matrix basic insertion")
{
	ccv_sparse_matrix_t* mat = ccv_sparse_matrix_new(1000, 1000, CCV_32S | CCV_C1, CCV_SPARSE_ROW_MAJOR, 0);
	int i, j, k, n;
	k = 0;
	for (i = 0; i < 1000; i++)
		for (j = 0; j < 1000; j++)
		{
			ccv_set_sparse_matrix_cell(mat, i, j, &k);
			k++;
		}
	REQUIRE_EQ(1543, CCV_GET_SPARSE_PRIME(mat->prime), "sparse matrix column size should be the prime number 1543");
	for (n = 0; n < 100; n++)
	{
		k = 0;
		for (i = 0; i < 1000; i++)
			for (j = 0; j < 1000; j++)
			{
				ccv_matrix_cell_t cell = ccv_get_sparse_matrix_cell(mat, i, j);
				REQUIRE(cell.u8 != 0, "cell at (%d, %d) doesn't contain any valid value", i, j);
				REQUIRE_EQ(k, cell.i32[0], "cell at (%d, %d) doesn't match inserted value", i, j);
				k++;
			}
	}
	ccv_matrix_free(mat);
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
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dm, csm->data.f32, 6, 1e-6, "actual element value should be the same");
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
			REQUIRE_EQ_WITH_TOLERANCE(m[i][j], (cell.u8 != 0) ? cell.f32[0] : 0, 1e-6, "should have the same matrix after decompressed at row %d, col %d", i, j);
		}
	ccv_matrix_free(smt);
	ccv_matrix_free(mat);
	ccv_matrix_free(csm);
}

TEST_CASE("matrix slice")
{
	ccv_dense_matrix_t* image = 0;
	ccv_read("../../samples/chessbox.png", &image, CCV_IO_ANY_FILE);
	ccv_dense_matrix_t* b = 0;
	ccv_slice(image, (ccv_matrix_t**)&b, 0, 33, 41, 111, 91);
	REQUIRE_MATRIX_FILE_EQ(b, "data/chessbox.slice.bin", "should have data/chessbox.png sliced at (33, 41) with 111 x 91");
	ccv_matrix_free(image);
	ccv_matrix_free(b);
}

TEST_CASE("matrix flatten")
{
	ccv_dense_matrix_t* dmt = ccv_dense_matrix_new(2, 2, CCV_8U | CCV_C2, 0, 0);
	dmt->data.u8[0] = 200;
	dmt->data.u8[1] = 100;
	dmt->data.u8[2] = 150;
	dmt->data.u8[3] = 50;
	dmt->data.u8[4] = 25;
	dmt->data.u8[5] = 20;
	dmt->data.u8[6] = 200;
	dmt->data.u8[7] = 250;
	ccv_dense_matrix_t* result = 0;
	ccv_flatten(dmt, (ccv_matrix_t**)&result, 0, 0);
	ccv_matrix_free(dmt);
	int rf[4] = {300, 200, 45, 450};
	REQUIRE_EQ(CCV_GET_CHANNEL(result->type), CCV_C1, "flatten matrix should have only one channel");
	REQUIRE_ARRAY_EQ(int, result->data.i32, rf, 4, "matrix flatten should have same value as reference array");
	ccv_matrix_free(result);
}

TEST_CASE("matrix border")
{
	ccv_dense_matrix_t* dmt = ccv_dense_matrix_new(1, 1, CCV_32F | CCV_C1, 0, 0);
	dmt->data.f32[0] = 2.0;
	ccv_dense_matrix_t* result = 0;
	ccv_margin_t margin = ccv_margin(2, 3, 1, 2);
	ccv_border(dmt, (ccv_matrix_t**)&result, 0, margin);
	ccv_matrix_free(dmt);
	float rf[24] = {
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 2, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
	};
	REQUIRE_EQ(margin.top + margin.bottom + 1, result->rows, "bordered matrix should have margin.top + margin.bottom + 1 rows");
	REQUIRE_EQ(margin.left + margin.right + 1, result->cols, "bordered matrix should have margin.left + margin.right + 1 cols");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, result->data.f32, rf, 24, 1e-5, "matrix border should have same value as reference array");
	ccv_matrix_free(result);
}

TEST_CASE("half precision and float-point conversion")
{
	uint16_t* h = (uint16_t*)ccmalloc(sizeof(uint16_t) * 0x10000);
	float* f = (float*)ccmalloc(sizeof(float) * 0x10000);
	uint16_t* b = (uint16_t*)ccmalloc(sizeof(uint16_t) * 0x10000);
	float* c = (float*)ccmalloc(sizeof(float) * 0x10000);
	int i;
	for (i = 0; i < 0x10000; i++)
		h[i] = i;
	ccv_half_precision_to_float(h, f, 0x10000);
	ccv_float_to_half_precision(f, b, 0x10000);
	REQUIRE_ARRAY_EQ(uint16_t, h, b, 0x10000, "half precision convert to float and then convert back should match exactly");
	for (i = 0; i <= 2048; i++)
		f[i] = i;
	ccv_float_to_half_precision(f, h, 2049);
	ccv_half_precision_to_float(h, c, 2049);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, f, c, 2049, 1e-5, "0-2048 integer to half precision and convert back should match exactly");
	for (i = 4097; i <= 8192; i++)
		f[i - 4097] = i;
	ccv_float_to_half_precision(f, h, 8192 - 4097 + 1);
	ccv_half_precision_to_float(h, c, 8192 - 4097 + 1);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, f, c, 8192 - 4097 + 1, 4 + 1e-5, "4097-8192 integer to half precision and convert back should round to multiple of 4");
	ccfree(h);
	ccfree(f);
	ccfree(b);
	ccfree(c);
}

#include "case_main.h"
