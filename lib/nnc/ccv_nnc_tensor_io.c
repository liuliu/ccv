#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_nnc_symbolic_graph.h"
#include "3rdparty/sqlite3/sqlite3.h"
#ifdef HAVE_CUDA
#include "gpu/ccv_nnc_compat.h"
#endif

#ifdef NDEBUG
#define SQLITE_ENFORCE(stmt) (void)(stmt)
#else
#define SQLITE_ENFORCE assert
#endif

// MARK - Level-1 API

int ccv_nnc_tensor_write(const ccv_nnc_tensor_t* const tensor, void* const handle, const char* const name)
{
	assert(!CCV_IS_TENSOR_VIEW(tensor));
	assert(name);
	sqlite3* conn = (sqlite3*)handle;
	if (!conn)
		return CCV_IO_ERROR;
	const char tensor_create_table_qs[] = "CREATE TABLE IF NOT EXISTS tensors "
		"(name TEXT, type INTEGER, format INTEGER, datatype INTEGER, "
		"dim BLOB, data BLOB, PRIMARY KEY (name))";
	SQLITE_ENFORCE(SQLITE_OK == sqlite3_exec(conn, tensor_create_table_qs, 0, 0, 0));
	const char tensor_insert_qs[] =
		"REPLACE INTO tensors "
		"(name, type, format, datatype, dim, data) VALUES ("
		"$name, $type, $format, $datatype, $dim, $data)";
	sqlite3_stmt* tensor_insert_stmt = 0;
	SQLITE_ENFORCE(SQLITE_OK == sqlite3_prepare_v2(conn, tensor_insert_qs, sizeof(tensor_insert_qs), &tensor_insert_stmt, 0));
	sqlite3_bind_text(tensor_insert_stmt, 1, name, -1, 0);
	sqlite3_bind_int(tensor_insert_stmt, 2, tensor->info.type);
	sqlite3_bind_int(tensor_insert_stmt, 3, tensor->info.format);
	sqlite3_bind_int(tensor_insert_stmt, 4, tensor->info.datatype);
	sqlite3_bind_blob(tensor_insert_stmt, 5, tensor->info.dim, sizeof(tensor->info.dim), 0);
	const size_t data_size = ccv_nnc_tensor_data_size(tensor->info);
	void* workspace = 0;
#ifdef HAVE_CUDA
	if (CCV_TENSOR_GET_MEMORY(tensor->info.type) == CCV_TENSOR_GPU_MEMORY)
	{
		workspace = ccmalloc(data_size);
		cumemcpy(workspace, CCV_TENSOR_CPU_MEMORY, tensor->data.u8, tensor->info.type, data_size);
		sqlite3_bind_blob(tensor_insert_stmt, 6, workspace, data_size, 0);
	} else
		sqlite3_bind_blob(tensor_insert_stmt, 6, tensor->data.u8, data_size, 0);
#else
	sqlite3_bind_blob(tensor_insert_stmt, 6, tensor->data.u8, data_size, 0);
#endif
	sqlite3_step(tensor_insert_stmt);
	sqlite3_reset(tensor_insert_stmt);
	sqlite3_clear_bindings(tensor_insert_stmt);
	sqlite3_finalize(tensor_insert_stmt);
	if (workspace)
		free(workspace);
	return CCV_IO_FINAL;
}

int ccv_nnc_tensor_read(void* const handle, const char* const name, ccv_nnc_tensor_t** const tensor_out)
{
	assert(name);
	sqlite3* conn = (sqlite3*)handle;
	if (!conn)
		return CCV_IO_ERROR;
	const char tensor_select_qs[] =
		"SELECT data, type, format, datatype, dim FROM tensors WHERE name=$name";
	sqlite3_stmt* tensor_select_stmt = 0;
	if (SQLITE_OK != sqlite3_prepare_v2(conn, tensor_select_qs, sizeof(tensor_select_qs), &tensor_select_stmt, 0))
		return CCV_IO_ERROR;
	sqlite3_bind_text(tensor_select_stmt, 1, name, -1, 0);
	if (SQLITE_ROW != sqlite3_step(tensor_select_stmt))
		return CCV_IO_ERROR;
	ccv_nnc_tensor_t* tensor = *tensor_out;
	int datatype = 0;
	if (!tensor) // If the tensor is not provided, we need to create one.
	{
		ccv_nnc_tensor_param_t info;
		info.type = sqlite3_column_int(tensor_select_stmt, 1);
		info.format = sqlite3_column_int(tensor_select_stmt, 2);
		datatype = info.datatype = sqlite3_column_int(tensor_select_stmt, 3);
		const void* const dim = sqlite3_column_blob(tensor_select_stmt, 4);
		memcpy(info.dim, dim, ccv_min(sizeof(info.dim), sqlite3_column_bytes(tensor_select_stmt, 4)));
		*tensor_out = tensor = ccv_nnc_tensor_new(0, info, 0);
	} else
		datatype = sqlite3_column_int(tensor_select_stmt, 3);
	const void* const data = sqlite3_column_blob(tensor_select_stmt, 0);
	if (datatype != tensor->info.datatype)
	{
		// Only ever works for 16F to 32F or 32F to 16F transparently.
		assert((datatype == CCV_16F && tensor->info.datatype == CCV_32F) || (datatype == CCV_32F && tensor->info.datatype == CCV_16F));
		const size_t tensor_count = ccv_nnc_tensor_count(tensor->info);
#ifdef HAVE_CUDA
		if (CCV_TENSOR_GET_MEMORY(tensor->info.type) == CCV_TENSOR_GPU_MEMORY)
		{
			size_t data_size = ccv_nnc_tensor_data_size(tensor->info);
			void* const workspace = ccmalloc(data_size);
			if (datatype == CCV_16F && tensor->info.datatype == CCV_32F)
				ccv_half_precision_to_float((uint16_t*)data, (float*)workspace, ccv_min(tensor_count, sqlite3_column_bytes(tensor_select_stmt, 0) / sizeof(uint16_t)));
			else
				ccv_float_to_half_precision((float*)data, (uint16_t*)workspace, ccv_min(tensor_count, sqlite3_column_bytes(tensor_select_stmt, 0) / sizeof(float)));
			if (CCV_TENSOR_GET_MEMORY(tensor->info.type) == CCV_TENSOR_GPU_MEMORY)
				cumemcpy(tensor->data.u8, tensor->info.type, workspace, CCV_TENSOR_CPU_MEMORY, data_size);
			ccfree(workspace);
		} else {
			if (datatype == CCV_16F && tensor->info.datatype == CCV_32F)
				ccv_half_precision_to_float((uint16_t*)data, tensor->data.f32, ccv_min(tensor_count, sqlite3_column_bytes(tensor_select_stmt, 0) / sizeof(uint16_t)));
			else
				ccv_float_to_half_precision((float*)data, (uint16_t*)tensor->data.f16, ccv_min(tensor_count, sqlite3_column_bytes(tensor_select_stmt, 0) / sizeof(float)));
		}
#else
		if (datatype == CCV_16F && tensor->info.datatype == CCV_32F)
			ccv_half_precision_to_float((uint16_t*)data, tensor->data.f32, ccv_min(tensor_count, sqlite3_column_bytes(tensor_select_stmt, 0) / sizeof(uint16_t)));
		else
			ccv_float_to_half_precision((float*)data, (uint16_t*)tensor->data.f16, ccv_min(tensor_count, sqlite3_column_bytes(tensor_select_stmt, 0) / sizeof(float)));
#endif
	} else {
		size_t data_size = ccv_nnc_tensor_data_size(tensor->info);
#ifdef HAVE_CUDA
		if (CCV_TENSOR_GET_MEMORY(tensor->info.type) == CCV_TENSOR_GPU_MEMORY)
			cumemcpy(tensor->data.u8, tensor->info.type, data, CCV_TENSOR_CPU_MEMORY, ccv_min(data_size, sqlite3_column_bytes(tensor_select_stmt, 0)));
		else
			memcpy(tensor->data.u8, data, ccv_min(data_size, sqlite3_column_bytes(tensor_select_stmt, 0)));
#else
		memcpy(tensor->data.u8, data, ccv_min(data_size, sqlite3_column_bytes(tensor_select_stmt, 0)));
#endif
	}
	tensor->type &= ~CCV_GARBAGE; // If it is marked as garbage, remove that mark now.
	sqlite3_reset(tensor_select_stmt);
	sqlite3_clear_bindings(tensor_select_stmt);
	sqlite3_finalize(tensor_select_stmt);
	return CCV_IO_FINAL;
}
