#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_cnnp_model.h"
#include "3rdparty/sqlite3/sqlite3.h"
#ifdef HAVE_CUDA
#include "gpu/ccv_nnc_compat.h"
#endif

#ifdef NDEBUG
#define SQLITE_ENFORCE(stmt) (void)(stmt)
#else
#define SQLITE_ENFORCE assert
#endif

void ccv_cnnp_model_checkpoint(ccv_cnnp_model_t* const model, const char* const fn, const int flags)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data); // The model has to be compiled.
	sqlite3* conn = 0;
	if (SQLITE_OK != sqlite3_open(fn, &conn))
		return;
	const int tensors_init = !!compiled_data->trainable_tensors;
	int i;
	const int parallel_count = ccv_max(compiled_data->parallel_count, 1);
	const int trainable_size = compiled_data->trainables->rnum;
	const int retain_size = compiled_data->retains->rnum * parallel_count;
	if (!tensors_init || flags == CCV_CNNP_MODEL_CHECKPOINT_READ_ONLY)
	{
		const char tensor_checkpoint_select_qs[] =
			"SELECT id, data FROM tensor_checkpoint ORDER BY id";
		sqlite3_stmt* tensor_checkpoint_select_stmt = 0;
		if (SQLITE_OK != sqlite3_prepare_v2(conn, tensor_checkpoint_select_qs, sizeof(tensor_checkpoint_select_qs), &tensor_checkpoint_select_stmt, 0))
		{
			sqlite3_close(conn);
			return;
		}
		if (!compiled_data->trainable_tensors)
			ccv_cnnp_model_tensors_init(model->graph, compiled_data);
		for (i = 0; i < trainable_size && SQLITE_ROW == sqlite3_step(tensor_checkpoint_select_stmt); i++)
		{
			const void* const data = sqlite3_column_blob(tensor_checkpoint_select_stmt, 1);
			ccv_nnc_tensor_t* const tensor = compiled_data->trainable_tensors[i * parallel_count];
			size_t data_size = ccv_nnc_tensor_data_size(tensor->info);
#ifdef HAVE_CUDA
			if (CCV_TENSOR_GET_MEMORY(tensor->info.type) == CCV_TENSOR_GPU_MEMORY)
				cumemcpy(tensor->data.u8, tensor->info.type, data, CCV_TENSOR_CPU_MEMORY, ccv_min(data_size, sqlite3_column_bytes(tensor_checkpoint_select_stmt, 1)));
			else
				memcpy(tensor->data.u8, data, ccv_min(data_size, sqlite3_column_bytes(tensor_checkpoint_select_stmt, 1)));
#else
			memcpy(tensor->data.u8, data, ccv_min(data_size, sqlite3_column_bytes(tensor_checkpoint_select_stmt, 1)));
#endif
		}
		for (i = 0; i < retain_size && SQLITE_ROW == sqlite3_step(tensor_checkpoint_select_stmt); i++)
		{
			const void* const data = sqlite3_column_blob(tensor_checkpoint_select_stmt, 1);
			if (!data)
				continue;
			ccv_nnc_tensor_t* const tensor = compiled_data->retain_tensors[i];
			size_t data_size = ccv_nnc_tensor_data_size(tensor->info);
#ifdef HAVE_CUDA
			if (CCV_TENSOR_GET_MEMORY(tensor->info.type) == CCV_TENSOR_GPU_MEMORY)
				cumemcpy(tensor->data.u8, tensor->info.type, data, CCV_TENSOR_CPU_MEMORY, ccv_min(data_size, sqlite3_column_bytes(tensor_checkpoint_select_stmt, 1)));
			else
				memcpy(tensor->data.u8, data, ccv_min(data_size, sqlite3_column_bytes(tensor_checkpoint_select_stmt, 1)));
#else
			memcpy(tensor->data.u8, data, ccv_min(data_size, sqlite3_column_bytes(tensor_checkpoint_select_stmt, 1)));
#endif
		}
		sqlite3_finalize(tensor_checkpoint_select_stmt);
		sqlite3_close(conn);
		return;
	}
	// If it is not init, nothing to checkpoint.
	if (!tensors_init)
		return;
	const char tensor_checkpoint_create_table_qs[] = "CREATE TABLE IF NOT EXISTS tensor_checkpoint "
		"(id INTEGER, type INTEGER, format INTEGER, datatype INTEGER, "
		"dim BLOB, data BLOB, PRIMARY KEY (id))";
	SQLITE_ENFORCE(SQLITE_OK == sqlite3_exec(conn, tensor_checkpoint_create_table_qs, 0, 0, 0));
	const char tensor_checkpoint_insert_qs[] =
		"REPLACE INTO tensor_checkpoint "
		"(id, type, format, datatype, dim, data) VALUES ("
		"$id, $type, $format, $datatype, $dim, $data)";
	sqlite3_stmt* tensor_checkpoint_insert_stmt = 0;
	SQLITE_ENFORCE(SQLITE_OK == sqlite3_prepare_v2(conn, tensor_checkpoint_insert_qs, sizeof(tensor_checkpoint_insert_qs), &tensor_checkpoint_insert_stmt, 0));
	SQLITE_ENFORCE(SQLITE_OK == sqlite3_exec(conn, "BEGIN", 0, 0, 0));
#ifdef HAVE_CUDA
	size_t workspace_size = 0;
	void* workspace = 0;
#endif
	for (i = 0; i < trainable_size; i++)
	{
		const ccv_nnc_tensor_t* const tensor = compiled_data->trainable_tensors[i * parallel_count];
		assert(!CCV_IS_TENSOR_VIEW(tensor));
		sqlite3_bind_int(tensor_checkpoint_insert_stmt, 1, i);
		sqlite3_bind_int(tensor_checkpoint_insert_stmt, 2, tensor->info.type);
		sqlite3_bind_int(tensor_checkpoint_insert_stmt, 3, tensor->info.format);
		sqlite3_bind_int(tensor_checkpoint_insert_stmt, 4, tensor->info.datatype);
		sqlite3_bind_blob(tensor_checkpoint_insert_stmt, 5, tensor->info.dim, sizeof(tensor->info.dim), 0);
		const size_t data_size = ccv_nnc_tensor_data_size(tensor->info);
#ifdef HAVE_CUDA
		if (CCV_TENSOR_GET_MEMORY(tensor->info.type) == CCV_TENSOR_GPU_MEMORY)
		{
			if (!workspace)
			{
				workspace = ccmalloc(data_size);
				workspace_size = data_size;
			} else if (data_size > workspace_size) {
				workspace = ccrealloc(workspace, data_size);
				workspace_size = data_size;
			}
			cumemcpy(workspace, CCV_TENSOR_CPU_MEMORY, tensor->data.u8, tensor->info.type, data_size);
			sqlite3_bind_blob(tensor_checkpoint_insert_stmt, 6, workspace, data_size, 0);
		} else
			sqlite3_bind_blob(tensor_checkpoint_insert_stmt, 6, tensor->data.u8, data_size, 0);
#else
		sqlite3_bind_blob(tensor_checkpoint_insert_stmt, 6, tensor->data.u8, data_size, 0);
#endif
		sqlite3_step(tensor_checkpoint_insert_stmt);
		sqlite3_reset(tensor_checkpoint_insert_stmt);
		sqlite3_clear_bindings(tensor_checkpoint_insert_stmt);
	}
	for (i = 0; i < retain_size; i++)
	{
		const ccv_nnc_tensor_t* const tensor = compiled_data->retain_tensors[i];
		if (!tensor)
		{
			// Inject empty one.
			sqlite3_bind_int(tensor_checkpoint_insert_stmt, 1, i + trainable_size);
			sqlite3_step(tensor_checkpoint_insert_stmt);
			sqlite3_reset(tensor_checkpoint_insert_stmt);
			sqlite3_clear_bindings(tensor_checkpoint_insert_stmt);
			continue;
		}
		assert(!CCV_IS_TENSOR_VIEW(tensor));
		sqlite3_bind_int(tensor_checkpoint_insert_stmt, 1, i + trainable_size);
		sqlite3_bind_int(tensor_checkpoint_insert_stmt, 2, tensor->info.type);
		sqlite3_bind_int(tensor_checkpoint_insert_stmt, 3, tensor->info.format);
		sqlite3_bind_int(tensor_checkpoint_insert_stmt, 4, tensor->info.datatype);
		sqlite3_bind_blob(tensor_checkpoint_insert_stmt, 5, tensor->info.dim, sizeof(tensor->info.dim), 0);
		const size_t data_size = ccv_nnc_tensor_data_size(tensor->info);
#ifdef HAVE_CUDA
		if (CCV_TENSOR_GET_MEMORY(tensor->info.type) == CCV_TENSOR_GPU_MEMORY)
		{
			if (!workspace)
			{
				workspace = ccmalloc(data_size);
				workspace_size = data_size;
			} else if (data_size > workspace_size) {
				workspace = ccrealloc(workspace, data_size);
				workspace_size = data_size;
			}
			cumemcpy(workspace, CCV_TENSOR_CPU_MEMORY, tensor->data.u8, tensor->info.type, data_size);
			sqlite3_bind_blob(tensor_checkpoint_insert_stmt, 6, workspace, data_size, 0);
		} else
			sqlite3_bind_blob(tensor_checkpoint_insert_stmt, 6, tensor->data.u8, data_size, 0);
#else
		sqlite3_bind_blob(tensor_checkpoint_insert_stmt, 6, tensor->data.u8, data_size, 0);
#endif
		sqlite3_step(tensor_checkpoint_insert_stmt);
		sqlite3_reset(tensor_checkpoint_insert_stmt);
		sqlite3_clear_bindings(tensor_checkpoint_insert_stmt);
	}
	sqlite3_finalize(tensor_checkpoint_insert_stmt);
#ifdef HAVE_CUDA
	if (workspace)
		ccfree(workspace);
#endif
	SQLITE_ENFORCE(SQLITE_OK == sqlite3_exec(conn, "COMMIT", 0, 0, 0));
	sqlite3_close(conn);
}
