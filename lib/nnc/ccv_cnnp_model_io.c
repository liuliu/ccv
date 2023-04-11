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

static inline int _model_tensor_write(const ccv_cnnp_model_t* const self, const ccv_nnc_tensor_t* const tensor, void* const handle, const char* const name, const ccv_nnc_tensor_io_option_t* const options)
{
	if (self->rw.writer)
		return self->rw.writer(tensor, handle, name);
	return ccv_nnc_tensor_write(tensor, handle, name, options);
}

int ccv_cnnp_model_write(const ccv_cnnp_model_t* const model, void* const handle, const char* const name, const ccv_nnc_tensor_io_option_t* const options)
{
	sqlite3* conn = (sqlite3*)handle;
	assert(conn);
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data); // The model has to be compiled.
	SQLITE_ENFORCE(SQLITE_OK == sqlite3_exec(conn, "BEGIN", 0, 0, 0));
	int i, j;
	const int parallel_count = ccv_max(model->parallel_count, 1);
	const int parameter_size = compiled_data->parameters->rnum;
	const int internal_size = compiled_data->internals->rnum;
	char internal_name[2048 + 16];
	for (i = 0; i < parameter_size; i++)
	{
		const char* const id = *(char**)ccv_array_get(compiled_data->ids.parameters, i);
		if (name)
			snprintf(internal_name, 2048 + 16, "__%s__[%s]", name, id);
		else
			snprintf(internal_name, 2048 + 16, "%s", id);
		_model_tensor_write(model, compiled_data->tensors.parameters[i], conn, internal_name, options);
	}
	for (i = 0; i < parallel_count; i++)
		for (j = 0; j < internal_size; j++)
		{
			const char* const id = *(char**)ccv_array_get(compiled_data->ids.internals, j);
			if (name)
				snprintf(internal_name, 2048 + 16, "__%s__[%s(%d)]", name, id, i);
			else
				snprintf(internal_name, 2048 + 16, "%s(%d)", id, i);
			_model_tensor_write(model, compiled_data->tensors.internals[i * internal_size + j], conn, internal_name, options);
		}
	SQLITE_ENFORCE(SQLITE_OK == sqlite3_exec(conn, "COMMIT", 0, 0, 0));
	return CCV_IO_FINAL;
}

static inline int _model_tensor_read(const ccv_cnnp_model_t* const self, void* const handle, const char* const name, const char* const dir, const ccv_nnc_tensor_io_option_t* const options, ccv_nnc_tensor_t** const tensor_out)
{
	if (self->rw.reader)
		return self->rw.reader(handle, name, dir, tensor_out);
	return ccv_nnc_tensor_read(handle, name, dir, options, tensor_out);
}

int ccv_cnnp_model_read(void* const handle, const char* const name, const ccv_nnc_tensor_io_option_t* const options, const ccv_cnnp_model_t* const model_out)
{
	sqlite3* conn = (sqlite3*)handle;
	assert(conn);
	ccv_cnnp_compiled_data_t* const compiled_data = model_out->compiled_data;
	assert(compiled_data); // The model has to be compiled.
	const int tensors_init = !!compiled_data->tensors_init.v;
	if (!tensors_init)
		ccv_cnnp_model_tensors_init(model_out, compiled_data);
	int i, j;
	const int parallel_count = ccv_max(model_out->parallel_count, 1);
	const int parameter_size = compiled_data->parameters->rnum;
	const int internal_size = compiled_data->internals->rnum;
	char internal_name[2048 + 16];
	char* file_backed_dir = model_out->file_backed_dir;
	for (i = 0; i < parameter_size; i++)
	{
		const char* const id = *(char**)ccv_array_get(compiled_data->ids.parameters, i);
		if (name)
			snprintf(internal_name, 2048 + 16, "__%s__[%s]", name, id);
		else
			snprintf(internal_name, 2048 + 16, "%s", id);
		if (_model_tensor_read(model_out, conn, internal_name, file_backed_dir, options, compiled_data->tensors.parameters + i) == CCV_IO_FINAL)
		{
			const int d = ((ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->parameters, i))->d;
			compiled_data->tensors_init.v[d >> 5] |= (1u << (d & 0x1f));
		}
	}
	for (i = 0; i < parallel_count; i++)
		for (j = 0; j < internal_size; j++)
		{
			const char* const id = *(char**)ccv_array_get(compiled_data->ids.internals, j);
			if (name)
				snprintf(internal_name, 2048 + 16, "__%s__[%s(%d)]", name, id, i);
			else
				snprintf(internal_name, 2048 + 16, "%s(%d)", id, i);
			if (_model_tensor_read(model_out, conn, internal_name, file_backed_dir, options, compiled_data->tensors.internals + i * internal_size + j) == CCV_IO_FINAL)
			{
				const int d = ((ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->internals, i))->d;
				compiled_data->tensors_init.v[d >> 5] |= (1u << (d & 0x1f));
			}
		}
	return CCV_IO_FINAL;
}

void ccv_cnnp_model_checkpoint(ccv_cnnp_model_t* const model, const char* const fn, const int flags, const ccv_nnc_tensor_io_option_t* const options)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data); // The model has to be compiled.
	sqlite3* conn = 0;
	if (SQLITE_OK != sqlite3_open(fn, &conn))
		return;
	const int tensors_init = !!compiled_data->tensors_init.v;
	if (!tensors_init || flags == CCV_CNNP_MODEL_CHECKPOINT_READ_ONLY)
	{
		ccv_cnnp_model_read(conn, 0, options, model);
		sqlite3_close(conn);
		return;
	}
	ccv_cnnp_model_write(model, conn, 0, options);
	sqlite3_close(conn);
}

void ccv_cnnp_model_set_io(ccv_cnnp_model_t* const model, ccv_cnnp_model_io_reader_f reader, ccv_cnnp_model_io_writer_f writer)
{
	model->rw.reader = reader;
	model->rw.writer = writer;
}
