#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_nnc_symbolic_graph.h"
#include "3rdparty/sqlite3/sqlite3.h"
#ifdef HAVE_CUDA
#include "gpu/ccv_nnc_compat.h"
#elif HAVE_MPS
#include "mps/ccv_nnc_mps.h"
#endif

#ifdef NDEBUG
#define SQLITE_ENFORCE(stmt) (void)(stmt)
#else
#define SQLITE_ENFORCE assert
#endif

// MARK - Level-1 API

int ccv_nnc_tensor_write(const ccv_nnc_tensor_t* const tensor, void* const handle, const char* const name, const ccv_nnc_tensor_io_option_t* const options)
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
	sqlite3_bind_int(tensor_insert_stmt, 3, tensor->info.format);
	sqlite3_bind_int64(tensor_insert_stmt, 4, ((sqlite_int64)tensor->info.reserved << 32) | tensor->info.datatype);
	sqlite3_bind_blob(tensor_insert_stmt, 5, tensor->info.dim, sizeof(tensor->info.dim), 0);
	const size_t data_size = ccv_nnc_tensor_data_size(tensor->info);
	unsigned char* workspace = 0;
	unsigned int identifier = 0;
#ifdef HAVE_CUDA
	if (CCV_TENSOR_GET_MEMORY(tensor->info.type) == CCV_TENSOR_GPU_MEMORY)
	{
		if (!options || !options->encode)
		{
			workspace = ccmalloc(data_size);
			cumemcpy(workspace, CCV_TENSOR_CPU_MEMORY, tensor->data.u8, tensor->info.type, data_size);
			sqlite3_bind_blob(tensor_insert_stmt, 6, workspace, data_size, 0);
		} else {
			workspace = ccmalloc(data_size * 2);
			cumemcpy(workspace, CCV_TENSOR_CPU_MEMORY, tensor->data.u8, tensor->info.type, data_size);
			size_t encoded_size = data_size;
			if (options->encode(workspace, data_size, tensor->info.datatype, tensor->info.dim, ccv_nnc_tensor_nd(tensor->info.dim), options->context, workspace + data_size, &encoded_size, &identifier))
				sqlite3_bind_blob(tensor_insert_stmt, 6, workspace + data_size, encoded_size, 0);
			else
				sqlite3_bind_blob(tensor_insert_stmt, 6, workspace, data_size, 0);
		}
	} else {
		if (!options || !options->encode)
			sqlite3_bind_blob(tensor_insert_stmt, 6, tensor->data.u8, data_size, 0);
		else {
			workspace = ccmalloc(data_size);
			size_t encoded_size = data_size;
			if (options->encode(tensor->data.u8, data_size, tensor->info.datatype, tensor->info.dim, ccv_nnc_tensor_nd(tensor->info.dim), options->context, workspace, &encoded_size, &identifier))
				sqlite3_bind_blob(tensor_insert_stmt, 6, workspace, encoded_size, 0);
			else
				sqlite3_bind_blob(tensor_insert_stmt, 6, tensor->data.u8, data_size, 0);
		}
	}
#elif defined(HAVE_MPS)
	if (CCV_TENSOR_GET_MEMORY(tensor->info.type) == CCV_TENSOR_GPU_MEMORY)
	{
		if (!options || !options->encode)
		{
			workspace = ccmalloc(data_size);
			mpmemcpy(workspace, 0, CCV_TENSOR_CPU_MEMORY, tensor->data.u8, tensor->dataof, tensor->info.type, data_size);
			sqlite3_bind_blob(tensor_insert_stmt, 6, workspace, data_size, 0);
		} else {
			workspace = ccmalloc(data_size * 2);
			mpmemcpy(workspace, 0, CCV_TENSOR_CPU_MEMORY, tensor->data.u8, tensor->dataof, tensor->info.type, data_size);
			size_t encoded_size = data_size;
			if (options->encode(workspace, data_size, tensor->info.datatype, tensor->info.dim, ccv_nnc_tensor_nd(tensor->info.dim), options->context, workspace + data_size, &encoded_size, &identifier))
				sqlite3_bind_blob(tensor_insert_stmt, 6, workspace + data_size, encoded_size, 0);
			else
				sqlite3_bind_blob(tensor_insert_stmt, 6, workspace, data_size, 0);
		}
	} else {
		if (!options || !options->encode)
			sqlite3_bind_blob(tensor_insert_stmt, 6, tensor->data.u8, data_size, 0);
		else {
			workspace = ccmalloc(data_size);
			size_t encoded_size = data_size;
			if (options->encode(tensor->data.u8, data_size, tensor->info.datatype, tensor->info.dim, ccv_nnc_tensor_nd(tensor->info.dim), options->context, workspace, &encoded_size, &identifier))
				sqlite3_bind_blob(tensor_insert_stmt, 6, workspace, encoded_size, 0);
			else
				sqlite3_bind_blob(tensor_insert_stmt, 6, tensor->data.u8, data_size, 0);
		}
	}
#else
	if (!options || !options->encode)
		sqlite3_bind_blob(tensor_insert_stmt, 6, tensor->data.u8, data_size, 0);
	else {
		workspace = ccmalloc(data_size);
		size_t encoded_size = data_size;
		if (options->encode(tensor->data.u8, data_size, tensor->info.datatype, tensor->info.dim, ccv_nnc_tensor_nd(tensor->info.dim), options->context, workspace, &encoded_size, &identifier))
			sqlite3_bind_blob(tensor_insert_stmt, 6, workspace, encoded_size, 0);
		else
			sqlite3_bind_blob(tensor_insert_stmt, 6, tensor->data.u8, data_size, 0);
	}
#endif
	sqlite3_bind_int64(tensor_insert_stmt, 2, ((sqlite_int64)identifier << 32) | tensor->info.type);
	sqlite3_step(tensor_insert_stmt);
	sqlite3_reset(tensor_insert_stmt);
	sqlite3_clear_bindings(tensor_insert_stmt);
	sqlite3_finalize(tensor_insert_stmt);
	if (workspace)
		free(workspace);
	return CCV_IO_FINAL;
}

int ccv_nnc_tensor_read(void* const handle, const char* const name, const char* const dir, const ccv_nnc_tensor_io_option_t* const options, const ccv_nnc_tensor_param_t* const tensor_params_optional, ccv_nnc_tensor_t** const tensor_out)
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
	{
		sqlite3_finalize(tensor_select_stmt);
		return CCV_IO_ERROR;
	}
	ccv_nnc_tensor_t* tensor = *tensor_out;
	ccv_nnc_tensor_param_t tensor_params;
	int datatype = 0;
	unsigned int identifier = 0;
	if (!tensor) // If the tensor is not provided, we need to create one.
	{
		if (tensor_params_optional)
			tensor_params = *tensor_params_optional;
		else {
			const sqlite_int64 type = sqlite3_column_int64(tensor_select_stmt, 1);
			identifier = (type >> 32) & 0xffffffff;
			tensor_params.type = (type & 0xffffffff);
			tensor_params.format = sqlite3_column_int(tensor_select_stmt, 2);
			const sqlite_int64 datatype_mix = sqlite3_column_int64(tensor_select_stmt, 3);
			datatype = tensor_params.datatype = (datatype_mix & 0xffffffff);
			tensor_params.reserved = (datatype_mix >> 32) & 0xffffffff;
			const void* const dim = sqlite3_column_blob(tensor_select_stmt, 4);
			memcpy(tensor_params.dim, dim, ccv_min(sizeof(tensor_params.dim), sqlite3_column_bytes(tensor_select_stmt, 4)));
		}
		if (!options || !options->decode)
			*tensor_out = tensor = ccv_nnc_tensor_new(0, tensor_params, 0);
	} else {
		identifier = (sqlite3_column_int64(tensor_select_stmt, 1) >> 32) & 0xffffffff;
		datatype = sqlite3_column_int(tensor_select_stmt, 3);
		tensor_params = tensor->info;
	}
	const void* const data = sqlite3_column_blob(tensor_select_stmt, 0);
	int dim[CCV_NNC_MAX_DIM_ALLOC];
	memcpy(dim, sqlite3_column_blob(tensor_select_stmt, 4), ccv_min(sizeof(dim), sqlite3_column_bytes(tensor_select_stmt, 4)));
	const int nd = ccv_nnc_tensor_nd(dim);
	if (datatype != tensor_params.datatype)
	{
		// Only ever works for 16F to 32F or 32F to 16F transparently.
		assert((datatype == CCV_16F && tensor_params.datatype == CCV_32F) || (datatype == CCV_32F && tensor_params.datatype == CCV_16F));
		const size_t tensor_count = ccv_nnc_tensor_count(tensor_params);
		ccv_nnc_tensor_param_t params = tensor_params;
		params.datatype = datatype;
		const size_t source_data_size = ccv_nnc_tensor_data_size(params);
#ifdef HAVE_CUDA
		if (CCV_TENSOR_GET_MEMORY(tensor_params.type) == CCV_TENSOR_GPU_MEMORY)
		{
			const size_t data_size = ccv_nnc_tensor_data_size(tensor_params);
			unsigned char* workspace;
			unsigned char* copying;
			if (!options || !options->decode)
			{
				copying = workspace = ccmalloc(data_size);
				if (datatype == CCV_16F && tensor_params.datatype == CCV_32F)
					ccv_half_precision_to_float((uint16_t*)data, (float*)workspace, ccv_min(tensor_count, sqlite3_column_bytes(tensor_select_stmt, 0) / sizeof(uint16_t)));
				else if (datatype == CCV_32F && tensor_params.datatype == CCV_16F)
					ccv_float_to_half_precision((float*)data, (uint16_t*)workspace, ccv_min(tensor_count, sqlite3_column_bytes(tensor_select_stmt, 0) / sizeof(float)));
				else
					{ assert(0); }
			} else {
				copying = workspace = ccmalloc(data_size + source_data_size);
				if (datatype == CCV_16F && tensor_params.datatype == CCV_32F)
				{
					size_t decoded_size = source_data_size;
					if (options->decode(data, sqlite3_column_bytes(tensor_select_stmt, 0), datatype, dim, nd, identifier, options->context, tensor_params, tensor_out, workspace + data_size, &decoded_size))
					{
						// If we loaded quantized tensor, don't do the conversion..
						if (CCV_GET_DATA_TYPE(tensor_out[0]->info.datatype) == CCV_QX)
							copying = workspace + data_size;
						else
							ccv_half_precision_to_float((uint16_t*)(workspace + data_size), (float*)workspace, ccv_min(tensor_count, ccv_min(source_data_size, decoded_size) / sizeof(uint16_t)));
					} else
						ccv_half_precision_to_float((uint16_t*)data, (float*)workspace, ccv_min(tensor_count, sqlite3_column_bytes(tensor_select_stmt, 0) / sizeof(uint16_t)));
				} else if (datatype == CCV_32F && tensor_params.datatype == CCV_16F) {
					size_t decoded_size = source_data_size;
					if (options->decode(data, sqlite3_column_bytes(tensor_select_stmt, 0), datatype, dim, nd, identifier, options->context, tensor_params, tensor_out, workspace + data_size, &decoded_size))
					{
						if (CCV_GET_DATA_TYPE(tensor_out[0]->info.datatype) == CCV_QX)
							copying = workspace + data_size;
						else
							ccv_float_to_half_precision((float*)(workspace + data_size), (uint16_t*)workspace, ccv_min(tensor_count, ccv_min(source_data_size, decoded_size) / sizeof(float)));
					} else
						ccv_float_to_half_precision((float*)data, (uint16_t*)workspace, ccv_min(tensor_count, sqlite3_column_bytes(tensor_select_stmt, 0) / sizeof(float)));
				} else
					{ assert(0); }
			}
			cumemcpy(tensor_out[0]->data.u8, tensor_out[0]->info.type, copying, CCV_TENSOR_CPU_MEMORY, data_size);
			ccfree(workspace);
		} else {
			if (!options || !options->decode)
			{
				if (datatype == CCV_16F && tensor_params.datatype == CCV_32F)
					ccv_half_precision_to_float((uint16_t*)data, tensor->data.f32, ccv_min(tensor_count, sqlite3_column_bytes(tensor_select_stmt, 0) / sizeof(uint16_t)));
				else if (datatype == CCV_32F && tensor_params.datatype == CCV_16F)
					ccv_float_to_half_precision((float*)data, (uint16_t*)tensor->data.f16, ccv_min(tensor_count, sqlite3_column_bytes(tensor_select_stmt, 0) / sizeof(float)));
				else
					{ assert(0); }
			} else {
				void* const workspace = ccmalloc(source_data_size);
				if (datatype == CCV_16F && tensor_params.datatype == CCV_32F)
				{
					size_t decoded_size = source_data_size;
					if (options->decode(data, sqlite3_column_bytes(tensor_select_stmt, 0), datatype, dim, nd, identifier, options->context, tensor_params, tensor_out, workspace, &decoded_size))
					{
						if (CCV_GET_DATA_TYPE(tensor_out[0]->info.datatype) == CCV_QX)
							memcpy(tensor_out[0]->data.f32, workspace, ccv_min(source_data_size, decoded_size));
						else
							ccv_half_precision_to_float((uint16_t*)workspace, tensor_out[0]->data.f32, ccv_min(tensor_count, ccv_min(source_data_size, decoded_size) / sizeof(uint16_t)));
					} else {
						if (!tensor)
							*tensor_out = tensor = ccv_nnc_tensor_new(0, tensor_params, 0);
						ccv_half_precision_to_float((uint16_t*)data, tensor->data.f32, ccv_min(tensor_count, sqlite3_column_bytes(tensor_select_stmt, 0) / sizeof(uint16_t)));
					}
				} else if (datatype == CCV_32F && tensor_params.datatype == CCV_16F) {
					size_t decoded_size = source_data_size;
					if (options->decode(data, sqlite3_column_bytes(tensor_select_stmt, 0), datatype, dim, nd, identifier, options->context, tensor_params, tensor_out, workspace, &decoded_size))
					{
						if (CCV_GET_DATA_TYPE(tensor_out[0]->info.datatype) == CCV_QX)
							memcpy(tensor_out[0]->data.f16, workspace, ccv_min(source_data_size, decoded_size));
						else
							ccv_float_to_half_precision((float*)workspace, (uint16_t*)tensor_out[0]->data.f16, ccv_min(tensor_count, ccv_min(source_data_size, decoded_size) / sizeof(float)));
					} else {
						if (!tensor)
							*tensor_out = tensor = ccv_nnc_tensor_new(0, tensor_params, 0);
						ccv_float_to_half_precision((float*)data, (uint16_t*)tensor->data.f16, ccv_min(tensor_count, sqlite3_column_bytes(tensor_select_stmt, 0) / sizeof(float)));
					}
				} else
					{ assert(0); }
				ccfree(workspace);
			}
		}
#elif defined(HAVE_MPS)
		if (CCV_TENSOR_GET_MEMORY(tensor_params.type) == CCV_TENSOR_GPU_MEMORY)
		{
			const size_t data_size = ccv_nnc_tensor_data_size(tensor_params);
			unsigned char* workspace;
			unsigned char* copying;
			if (!options || !options->decode)
			{
				copying = workspace = ccmalloc(data_size);
				if (datatype == CCV_16F && tensor_params.datatype == CCV_32F)
					ccv_half_precision_to_float((uint16_t*)data, (float*)workspace, ccv_min(tensor_count, sqlite3_column_bytes(tensor_select_stmt, 0) / sizeof(uint16_t)));
				else if (datatype == CCV_32F && tensor_params.datatype == CCV_16F)
					ccv_float_to_half_precision((float*)data, (uint16_t*)workspace, ccv_min(tensor_count, sqlite3_column_bytes(tensor_select_stmt, 0) / sizeof(float)));
				else
					{ assert(0); }
			} else {
				workspace = ccmalloc(data_size + source_data_size);
				if (datatype == CCV_16F && tensor_params.datatype == CCV_32F)
				{
					size_t decoded_size = source_data_size;
					if (options->decode(data, sqlite3_column_bytes(tensor_select_stmt, 0), datatype, dim, nd, identifier, options->context, tensor_params, tensor_out, workspace + data_size, &decoded_size))
					{
						if (CCV_GET_DATA_TYPE(tensor_out[0]->info.datatype) == CCV_QX)
							copying = workspace + data_size;
						else
							ccv_half_precision_to_float((uint16_t*)(workspace + data_size), (float*)workspace, ccv_min(tensor_count, ccv_min(source_data_size, decoded_size) / sizeof(uint16_t)));
					} else
						ccv_half_precision_to_float((uint16_t*)data, (float*)workspace, ccv_min(tensor_count, sqlite3_column_bytes(tensor_select_stmt, 0) / sizeof(uint16_t)));
				} else if (datatype == CCV_32F && tensor_params.datatype == CCV_16F) {
					size_t decoded_size = source_data_size;
					if (options->decode(data, sqlite3_column_bytes(tensor_select_stmt, 0), datatype, dim, nd, identifier, options->context, tensor_params, tensor_out, workspace + data_size, &decoded_size))
					{
						if (CCV_GET_DATA_TYPE(tensor_out[0]->info.datatype) == CCV_QX)
							copying = workspace + data_size;
						else
							ccv_float_to_half_precision((float*)(workspace + data_size), (uint16_t*)workspace, ccv_min(tensor_count, ccv_min(source_data_size, decoded_size) / sizeof(float)));
					} else
						ccv_float_to_half_precision((float*)data, (uint16_t*)workspace, ccv_min(tensor_count, sqlite3_column_bytes(tensor_select_stmt, 0) / sizeof(float)));
				} else
					{ assert(0); }
			}
			assert(tensor_out[0]->dataof == 0);
			if (dir)
				tensor_out[0]->data.u8 = mpmemmap(tensor_out[0]->data.u8, copying, data_size, data_size, dir, name);
			else
				mpmemcpy(tensor_out[0]->data.u8, tensor_out[0]->dataof, tensor_out[0]->info.type, copying, 0, CCV_TENSOR_CPU_MEMORY, data_size);
			ccfree(workspace);
		} else {
			if (!options || !options->decode)
			{
				if (datatype == CCV_16F && tensor_params.datatype == CCV_32F)
					ccv_half_precision_to_float((uint16_t*)data, tensor->data.f32, ccv_min(tensor_count, sqlite3_column_bytes(tensor_select_stmt, 0) / sizeof(uint16_t)));
				else if (datatype == CCV_32F && tensor_params.datatype == CCV_16F)
					ccv_float_to_half_precision((float*)data, (uint16_t*)tensor->data.f16, ccv_min(tensor_count, sqlite3_column_bytes(tensor_select_stmt, 0) / sizeof(float)));
				else
					{ assert(0); }
			} else {
				void* const workspace = ccmalloc(source_data_size);
				if (datatype == CCV_16F && tensor_params.datatype == CCV_32F)
				{
					size_t decoded_size = source_data_size;
					if (options->decode(data, sqlite3_column_bytes(tensor_select_stmt, 0), datatype, dim, nd, identifier, options->context, tensor_params, tensor_out, workspace, &decoded_size))
					{
						if (CCV_GET_DATA_TYPE(tensor_out[0]->info.datatype) == CCV_QX)
							memcpy(tensor_out[0]->data.f32, workspace, ccv_min(source_data_size, decoded_size));
						else
							ccv_half_precision_to_float((uint16_t*)workspace, tensor_out[0]->data.f32, ccv_min(tensor_count, ccv_min(source_data_size, decoded_size) / sizeof(uint16_t)));
					} else {
						if (!tensor)
							*tensor_out = tensor = ccv_nnc_tensor_new(0, tensor_params, 0);
						ccv_half_precision_to_float((uint16_t*)data, tensor->data.f32, ccv_min(tensor_count, sqlite3_column_bytes(tensor_select_stmt, 0) / sizeof(uint16_t)));
					}
				} else if (datatype == CCV_32F && tensor_params.datatype == CCV_16F) {
					size_t decoded_size = source_data_size;
					if (options->decode(data, sqlite3_column_bytes(tensor_select_stmt, 0), datatype, dim, nd, identifier, options->context, tensor_params, tensor_out, workspace, &decoded_size))
					{
						if (CCV_GET_DATA_TYPE(tensor_out[0]->info.datatype) == CCV_QX)
							memcpy(tensor_out[0]->data.f16, workspace, ccv_min(source_data_size, decoded_size));
						else
							ccv_float_to_half_precision((float*)workspace, (uint16_t*)tensor_out[0]->data.f16, ccv_min(tensor_count, ccv_min(source_data_size, decoded_size) / sizeof(float)));
					} else {
						if (!tensor)
							*tensor_out = tensor = ccv_nnc_tensor_new(0, tensor_params, 0);
						ccv_float_to_half_precision((float*)data, (uint16_t*)tensor->data.f16, ccv_min(tensor_count, sqlite3_column_bytes(tensor_select_stmt, 0) / sizeof(float)));
					}
				} else
					{ assert(0); }
				ccfree(workspace);
			}
		}
#else
		if (!options || !options->decode)
		{
			if (datatype == CCV_16F && tensor_params.datatype == CCV_32F)
				ccv_half_precision_to_float((uint16_t*)data, tensor->data.f32, ccv_min(tensor_count, sqlite3_column_bytes(tensor_select_stmt, 0) / sizeof(uint16_t)));
			else if (datatype == CCV_32F && tensor_params.datatype == CCV_16F)
				ccv_float_to_half_precision((float*)data, (uint16_t*)tensor->data.f16, ccv_min(tensor_count, sqlite3_column_bytes(tensor_select_stmt, 0) / sizeof(float)));
			else
				{ assert(0); }
		} else {
			void* const workspace = ccmalloc(source_data_size);
			if (datatype == CCV_16F && tensor_params.datatype == CCV_32F)
			{
				size_t decoded_size = source_data_size;
				if (options->decode(data, sqlite3_column_bytes(tensor_select_stmt, 0), datatype, dim, nd, identifier, options->context, tensor_params, tensor_out, workspace, &decoded_size))
				{
					if (CCV_GET_DATA_TYPE(tensor_out[0]->info.datatype) == CCV_QX)
						memcpy(tensor_out[0]->data.f32, workspace, ccv_min(source_data_size, decoded_size));
					else
						ccv_half_precision_to_float((uint16_t*)workspace, tensor_out[0]->data.f32, ccv_min(tensor_count, ccv_min(source_data_size, decoded_size) / sizeof(uint16_t)));
				} else {
					if (!tensor)
						*tensor_out = tensor = ccv_nnc_tensor_new(0, tensor_params, 0);
					ccv_half_precision_to_float((uint16_t*)data, tensor->data.f32, ccv_min(tensor_count, sqlite3_column_bytes(tensor_select_stmt, 0) / sizeof(uint16_t)));
				}
			} else if (datatype == CCV_32F && tensor_params.datatype == CCV_16F) {
				size_t decoded_size = source_data_size;
				if (options->decode(data, sqlite3_column_bytes(tensor_select_stmt, 0), datatype, dim, nd, identifier, options->context, tensor_params, tensor_out, workspace, &decoded_size))
				{
					if (CCV_GET_DATA_TYPE(tensor_out[0]->info.datatype) == CCV_QX)
						memcpy(tensor_out[0]->data.f16, workspace, ccv_min(source_data_size, decoded_size));
					else
						ccv_float_to_half_precision((float*)workspace, (uint16_t*)tensor_out[0]->data.f16, ccv_min(tensor_count, ccv_min(source_data_size, decoded_size) / sizeof(float)));
				} else {
					if (!tensor)
						*tensor_out = tensor = ccv_nnc_tensor_new(0, tensor_params, 0);
					ccv_float_to_half_precision((float*)data, (uint16_t*)tensor->data.f16, ccv_min(tensor_count, sqlite3_column_bytes(tensor_select_stmt, 0) / sizeof(float)));
				}
			} else
				{ assert(0); }
			ccfree(workspace);
		}
#endif
	} else {
		size_t data_size = ccv_nnc_tensor_data_size(tensor_params);
#ifdef HAVE_CUDA
		if (!options || !options->decode)
		{
			if (CCV_TENSOR_GET_MEMORY(tensor_params.type) == CCV_TENSOR_GPU_MEMORY)
				cumemcpy(tensor->data.u8, tensor->info.type, data, CCV_TENSOR_CPU_MEMORY, ccv_min(data_size, sqlite3_column_bytes(tensor_select_stmt, 0)));
			else
				memcpy(tensor->data.u8, data, ccv_min(data_size, sqlite3_column_bytes(tensor_select_stmt, 0)));
		} else {
			if (CCV_TENSOR_GET_MEMORY(tensor_params.type) == CCV_TENSOR_GPU_MEMORY)
			{
				void* const workspace = ccmalloc(data_size);
				size_t decoded_size = data_size;
				if (options->decode(data, sqlite3_column_bytes(tensor_select_stmt, 0), datatype, dim, nd, identifier, options->context, tensor_params, tensor_out, workspace, &decoded_size))
					cumemcpy(tensor_out[0]->data.u8, tensor_out[0]->info.type, workspace, CCV_TENSOR_CPU_MEMORY, ccv_min(data_size, decoded_size));
				else {
					if (!tensor)
						*tensor_out = tensor = ccv_nnc_tensor_new(0, tensor_params, 0);
					cumemcpy(tensor->data.u8, tensor->info.type, data, CCV_TENSOR_CPU_MEMORY, ccv_min(data_size, sqlite3_column_bytes(tensor_select_stmt, 0)));
				}
				ccfree(workspace);
			} else {
				size_t decoded_size = data_size;
				if (!options->decode(data, sqlite3_column_bytes(tensor_select_stmt, 0), datatype, dim, nd, identifier, options->context, tensor_params, tensor_out, tensor ? tensor->data.u8 : 0, &decoded_size))
				{
					if (!tensor)
						*tensor_out = tensor = ccv_nnc_tensor_new(0, tensor_params, 0);
					memcpy(tensor->data.u8, data, ccv_min(data_size, sqlite3_column_bytes(tensor_select_stmt, 0)));
				}
			}
		}
#elif defined(HAVE_MPS)
		if (!options || !options->decode)
		{
			if (CCV_TENSOR_GET_MEMORY(tensor_params.type) == CCV_TENSOR_GPU_MEMORY)
			{
				assert(tensor->dataof == 0);
				if (dir)
					tensor->data.u8 = mpmemmap(tensor->data.u8, data, ccv_min(data_size, sqlite3_column_bytes(tensor_select_stmt, 0)), data_size, dir, name);
				else
					mpmemcpy(tensor->data.u8, tensor->dataof, tensor->info.type, data, 0, CCV_TENSOR_CPU_MEMORY, ccv_min(data_size, sqlite3_column_bytes(tensor_select_stmt, 0)));
			} else
				memcpy(tensor->data.u8, data, ccv_min(data_size, sqlite3_column_bytes(tensor_select_stmt, 0)));
		} else {
			if (CCV_TENSOR_GET_MEMORY(tensor_params.type) == CCV_TENSOR_GPU_MEMORY)
			{
				if (tensor)
					{ assert(tensor->dataof == 0); }
				void* const workspace = ccmalloc(data_size);
				size_t decoded_size = data_size;
				if (options->decode(data, sqlite3_column_bytes(tensor_select_stmt, 0), datatype, dim, nd, identifier, options->context, tensor_params, tensor_out, workspace, &decoded_size)) {
					if (dir)
						tensor_out[0]->data.u8 = mpmemmap(tensor_out[0]->data.u8, workspace, ccv_min(data_size, decoded_size), data_size, dir, name);
					else
						mpmemcpy(tensor_out[0]->data.u8, tensor_out[0]->dataof, tensor_out[0]->info.type, workspace, 0, CCV_TENSOR_CPU_MEMORY, ccv_min(data_size, decoded_size));
				} else {
					if (!tensor)
						*tensor_out = tensor = ccv_nnc_tensor_new(0, tensor_params, 0);
					if (dir)
						tensor->data.u8 = mpmemmap(tensor->data.u8, data, ccv_min(data_size, sqlite3_column_bytes(tensor_select_stmt, 0)), data_size, dir, name);
					else
						mpmemcpy(tensor->data.u8, tensor->dataof, tensor->info.type, data, 0, CCV_TENSOR_CPU_MEMORY, ccv_min(data_size, sqlite3_column_bytes(tensor_select_stmt, 0)));
				}
				ccfree(workspace);
			} else {
				size_t decoded_size = data_size;
				if (!options->decode(data, sqlite3_column_bytes(tensor_select_stmt, 0), datatype, dim, nd, identifier, options->context, tensor_params, tensor_out, tensor ? tensor->data.u8 : 0, &decoded_size))
				{
					if (!tensor)
						*tensor_out = tensor = ccv_nnc_tensor_new(0, tensor_params, 0);
					memcpy(tensor->data.u8, data, ccv_min(data_size, sqlite3_column_bytes(tensor_select_stmt, 0)));
				}
			}
		}
#else
		if (!options || !options->decode)
			memcpy(tensor->data.u8, data, ccv_min(data_size, sqlite3_column_bytes(tensor_select_stmt, 0)));
		else {
			size_t decoded_size = data_size;
			if (!options->decode(data, sqlite3_column_bytes(tensor_select_stmt, 0), datatype, dim, nd, identifier, options->context, tensor_params, tensor_out, tensor ? tensor->data.u8 : 0, &decoded_size))
			{
				if (!tensor)
					*tensor_out = tensor = ccv_nnc_tensor_new(0, tensor_params, 0);
				memcpy(tensor->data.u8, data, ccv_min(data_size, sqlite3_column_bytes(tensor_select_stmt, 0)));
			}
		}
#endif
	}
	tensor_out[0]->type &= ~CCV_GARBAGE; // If it is marked as garbage, remove that mark now.
	sqlite3_reset(tensor_select_stmt);
	sqlite3_clear_bindings(tensor_select_stmt);
	sqlite3_finalize(tensor_select_stmt);
	return CCV_IO_FINAL;
}

int ccv_nnc_tensor_swap(ccv_nnc_tensor_t* const tensor, const char* const name, const char* const dir, const void* const data, const size_t data_size)
{
#ifdef HAVE_MPS
	if (!data || !data_size)
	{
		if (CCV_TENSOR_GET_MEMORY(tensor->info.type) == CCV_TENSOR_GPU_MEMORY && dir && name)
		{
			assert(tensor->dataof == 0);
			size_t data_size = ccv_nnc_tensor_data_size(tensor->info);
			void* const data = ccmalloc(data_size);
			mpmemcpy(data, 0, CCV_TENSOR_CPU_MEMORY, tensor->data.u8, tensor->dataof, tensor->info.type, data_size);
			tensor->data.u8 = mpmemmap(tensor->data.u8, data, data_size, data_size, dir, name);
			ccfree(data);
			return 0;
		}
		return -1;
	}
#endif
	size_t expected_size = ccv_nnc_tensor_data_size(tensor->info);
#ifdef HAVE_CUDA
	if (CCV_TENSOR_GET_MEMORY(tensor->info.type) == CCV_TENSOR_GPU_MEMORY)
		cumemcpy(tensor->data.u8, tensor->info.type, data, CCV_TENSOR_CPU_MEMORY, ccv_min(expected_size, data_size));
	else
		memcpy(tensor->data.u8, data, ccv_min(expected_size, data_size));
#elif defined(HAVE_MPS)
	if (CCV_TENSOR_GET_MEMORY(tensor->info.type) == CCV_TENSOR_GPU_MEMORY)
	{
		assert(tensor->dataof == 0);
		if (dir && name)
		{
			ccv_nnc_synchronize_stream_context(0); // To avoid if the data is coming from GPU and haven't finish writing.
			tensor->data.u8 = mpmemmap(tensor->data.u8, data, ccv_min(expected_size, data_size), expected_size, dir, name);
		} else
			mpmemcpy(tensor->data.u8, tensor->dataof, tensor->info.type, data, 0, CCV_TENSOR_CPU_MEMORY, ccv_min(expected_size, data_size));
	} else
		memcpy(tensor->data.u8, data, ccv_min(expected_size, data_size));
#else
	memcpy(tensor->data.u8, data, ccv_min(expected_size, data_size));
#endif
	return 0;
}
