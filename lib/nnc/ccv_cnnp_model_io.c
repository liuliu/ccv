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

static inline int _model_tensor_write(const ccv_cnnp_model_t* const self, const ccv_nnc_tensor_t* const tensor, const char* const sql, void* const handle, const char* const name, const ccv_nnc_tensor_io_option_t* const options)
{
	if (self->rw.writer)
		return self->rw.writer(tensor, sql, handle, name, options);
	if (sql)
	{
		sqlite3* conn = (sqlite3*)handle;
		SQLITE_ENFORCE(SQLITE_OK == sqlite3_exec(conn, sql, 0, 0, 0));
		return CCV_IO_FINAL;
	} else
		return ccv_nnc_tensor_write(tensor, handle, name, options);
}

int ccv_cnnp_model_write(const ccv_cnnp_model_t* const model, void* const handle, const char* const name, const ccv_nnc_tensor_io_option_t* const options)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data); // The model has to be compiled.
	_model_tensor_write(model, 0, "BEGIN", handle, 0, options);
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
		_model_tensor_write(model, CCV_NNC_TENSOR(compiled_data->tensors.parameters[i]), 0, handle, internal_name, options);
	}
	for (i = 0; i < parallel_count; i++)
		for (j = 0; j < internal_size; j++)
		{
			const char* const id = *(char**)ccv_array_get(compiled_data->ids.internals, j);
			if (name)
				snprintf(internal_name, 2048 + 16, "__%s__[%s(%d)]", name, id, i);
			else
				snprintf(internal_name, 2048 + 16, "%s(%d)", id, i);
			_model_tensor_write(model, compiled_data->tensors.internals[i * internal_size + j], 0, handle, internal_name, options);
		}
	_model_tensor_write(model, 0, "COMMIT", handle, 0, options);
	return CCV_IO_FINAL;
}

static inline int _model_tensor_read(const ccv_cnnp_model_t* const self, void* const handle, const char* const name, const ccv_nnc_tensor_io_option_t* const options, const ccv_nnc_tensor_param_t info, ccv_nnc_tensor_t** const tensor_out)
{
	if (self->rw.reader)
		return self->rw.reader(handle, name, options, info, tensor_out);
	return ccv_nnc_tensor_read(handle, name, options, 0, &info, tensor_out);
}

int ccv_cnnp_model_read(void* const handle, const char* const name, const ccv_nnc_tensor_io_option_t* const options, const ccv_cnnp_model_t* const model_out)
{
	sqlite3* conn = (sqlite3*)handle;
	assert(conn);
	ccv_cnnp_compiled_data_t* const compiled_data = model_out->compiled_data;
	assert(compiled_data); // The model has to be compiled.
	const int tensors_init = !!compiled_data->tensors_init.v;
	if (!tensors_init)
		ccv_cnnp_model_tensors_init_0(model_out, compiled_data);
	int i, j;
	const int parallel_count = ccv_max(model_out->parallel_count, 1);
	const int parameter_size = compiled_data->parameters->rnum;
	const int internal_size = compiled_data->internals->rnum;
	char internal_name[2048 + 16];
	uint32_t* const init_v = CCV_NNC_INIT_V(compiled_data->tensors_init.v);
	for (i = 0; i < parameter_size; i++)
	{
		const char* const id = *(char**)ccv_array_get(compiled_data->ids.parameters, i);
		if (name)
			snprintf(internal_name, 2048 + 16, "__%s__[%s]", name, id);
		else
			snprintf(internal_name, 2048 + 16, "%s", id);
		const ccv_nnc_tensor_symbol_t parameter = *(ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->parameters, i);
		const int d = parameter.d;
		ccv_nnc_tensor_param_t info = ccv_nnc_tensor_symbol_params(parameter.graph, parameter);
		if (CCV_TENSOR_GET_DEVICE(info.type) == CCV_COMPUTE_DEVICE_ANY)
			CCV_TENSOR_SET_DEVICE_ID(info.type, 0);
		const int device_id = CCV_TENSOR_GET_DEVICE_ID(info.type);
		if (compiled_data->tensors.parameters[i]) // Cannot be a shared parameter to read.
			{ assert(!((uintptr_t)compiled_data->tensors.parameters[i] & (uintptr_t)1)); }
		if (_model_tensor_read(model_out, conn, internal_name, options, info, compiled_data->tensors.parameters + i) == CCV_IO_FINAL)
		{
			init_v[d >> 5] |= (1u << (d & 0x1f));
			// Create this tensor for other data parallel allocations.
			info = compiled_data->tensors.parameters[i]->info; // In case we loaded a different info.
			for (j = 1; j < parallel_count; j++)
				if (!compiled_data->tensors.parameters[i + j * parameter_size])
				{
					if (j != device_id)
						CCV_TENSOR_SET_DEVICE_ID(info.type, j);
					else
						CCV_TENSOR_SET_DEVICE_ID(info.type, 0);
					compiled_data->tensors.parameters[i + j * parameter_size] = ccv_nnc_tensor_new(0, info, 0);
				}
				// No need to copy over, this is done in ccv_cnnp_model.c's copy_tensors method.
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
			const ccv_nnc_tensor_symbol_t retained = *(ccv_nnc_tensor_symbol_t*)ccv_array_get(compiled_data->internals, j);
			const int d = retained.d;
			ccv_nnc_tensor_param_t info = ccv_nnc_tensor_symbol_params(retained.graph, retained);
			if (CCV_TENSOR_GET_DEVICE(info.type) == CCV_COMPUTE_DEVICE_ANY)
				CCV_TENSOR_SET_DEVICE_ID(info.type, 0);
			if (i > 0)
			{
				const int device_id = CCV_TENSOR_GET_DEVICE_ID(info.type);
				if (i != device_id)
					CCV_TENSOR_SET_DEVICE_ID(info.type, i);
				else
					CCV_TENSOR_SET_DEVICE_ID(info.type, 0);
			}
			if (_model_tensor_read(model_out, conn, internal_name, options, info, compiled_data->tensors.internals + i * internal_size + j) == CCV_IO_FINAL)
				init_v[d >> 5] |= (1u << (d & 0x1f));
		}
	// Mark it as to have some other tensors to allocate.
	if (ccv_cnnp_model_tensors_any_to_alloc(model_out, compiled_data))
		compiled_data->tensors_init.v = (uint32_t*)((uintptr_t)compiled_data->tensors_init.v | (uintptr_t)1);
	else // Remove the flag.
		compiled_data->tensors_init.v = CCV_NNC_INIT_V(compiled_data->tensors_init.v);
	return CCV_IO_FINAL;
}

void ccv_cnnp_model_write_to_file(ccv_cnnp_model_t* const model, const char* const fn, const ccv_nnc_tensor_io_option_t* const options)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data); // The model has to be compiled.
	sqlite3* conn = 0;
	if (SQLITE_OK != sqlite3_open(fn, &conn))
		return;
	ccv_cnnp_model_write(model, conn, 0, options);
	sqlite3_close(conn);
}

void ccv_cnnp_model_read_from_file(const char* const fn, const ccv_nnc_tensor_io_option_t* const options, const ccv_cnnp_model_t* const model)
{
	ccv_cnnp_compiled_data_t* const compiled_data = model->compiled_data;
	assert(compiled_data); // The model has to be compiled.
	sqlite3* conn = 0;
	if (SQLITE_OK != sqlite3_open(fn, &conn))
		return;
	ccv_cnnp_model_read(conn, 0, options, model);
	sqlite3_close(conn);
}

void ccv_cnnp_model_set_io(ccv_cnnp_model_t* const model, ccv_cnnp_model_io_reader_f reader, ccv_cnnp_model_io_writer_f writer)
{
	model->rw.reader = reader;
	model->rw.writer = writer;
}
