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
	const int tensors_init = !!compiled_data->tensors_init.v;
	int i, j;
	const int parallel_count = ccv_max(model->parallel_count, 1);
	const int parameter_size = compiled_data->parameters->rnum;
	const int internal_size = compiled_data->internals->rnum;
	char internal_name[1024 + 16];
	if (!tensors_init || flags == CCV_CNNP_MODEL_CHECKPOINT_READ_ONLY)
	{
		if (!tensors_init)
			ccv_cnnp_model_tensors_init(model, compiled_data);
		for (i = 0; i < parameter_size; i++)
		{
			const char* const id = *(char**)ccv_array_get(compiled_data->ids.parameters, i);
			if (ccv_nnc_tensor_read(conn, id, compiled_data->tensors.parameters + i) == CCV_IO_FINAL)
			{
				const int d = ((ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->parameters, i))->d;
				compiled_data->tensors_init.v[d >> 5] |= (1u << (d & 0x1f));
			}
		}
		for (i = 0; i < parallel_count; i++)
			for (j = 0; j < internal_size; j++)
			{
				const char* const id = *(char**)ccv_array_get(compiled_data->ids.internals, j);
				snprintf(internal_name, 1024 + 16, "%s(%d)", id, i);
				if (ccv_nnc_tensor_read(conn, internal_name, compiled_data->tensors.internals + i * internal_size + j) == CCV_IO_FINAL)
				{
					const int d = ((ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->internals, i))->d;
					compiled_data->tensors_init.v[d >> 5] |= (1u << (d & 0x1f));
				}
			}
		sqlite3_close(conn);
		return;
	}
	SQLITE_ENFORCE(SQLITE_OK == sqlite3_exec(conn, "BEGIN", 0, 0, 0));
	for (i = 0; i < parameter_size; i++)
	{
		const char* const id = *(char**)ccv_array_get(compiled_data->ids.parameters, i);
		ccv_nnc_tensor_write(compiled_data->tensors.parameters[i], conn, id);
	}
	for (i = 0; i < parallel_count; i++)
		for (j = 0; j < internal_size; j++)
		{
			const char* const id = *(char**)ccv_array_get(compiled_data->ids.internals, j);
			snprintf(internal_name, 1024 + 16, "%s(%d)", id, i);
			ccv_nnc_tensor_write(compiled_data->tensors.internals[i * internal_size + j], conn, internal_name);
		}
	SQLITE_ENFORCE(SQLITE_OK == sqlite3_exec(conn, "COMMIT", 0, 0, 0));
	sqlite3_close(conn);
}
