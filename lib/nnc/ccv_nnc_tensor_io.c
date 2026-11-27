#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_nnc_symbolic_graph.h"
#include "3rdparty/sqlite3/sqlite3.h"
#include "ccv_nnc_codebook.h"
#include <limits.h>
#include <stdint.h>
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

// SQLite INTEGER is 64-bit. Tensor IO already packs side metadata into the
// high 32 bits of type / datatype. Tensor formats are low 32-bit int enum
// values (NCHW/NHWC/CHWN), so bit 32 can mark that tensors.data is the head
// chunk. Always mask this out before assigning to ccv_nnc_tensor_param_t.format.
#define CCV_NNC_TENSOR_IO_SPLIT_FORMAT (((sqlite_int64)1) << 32)
#define CCV_NNC_TENSOR_IO_FORMAT_MASK 0xffffffffll
#define CCV_NNC_TENSOR_IO_SPLIT_HEADROOM 1024

static int _ccv_nnc_tensor_io_blob_chunk_size(sqlite3* const conn)
{
	const int limit = sqlite3_limit(conn, SQLITE_LIMIT_LENGTH, -1);
	if (limit <= 0)
		return INT_MAX;
	if (limit > CCV_NNC_TENSOR_IO_SPLIT_HEADROOM * 2)
		return limit - CCV_NNC_TENSOR_IO_SPLIT_HEADROOM;
	return limit > 1 ? limit / 2 : 1;
}

static int _ccv_nnc_tensor_io_delete_splits(sqlite3* const conn, const char* const name)
{
	const char tensor_split_delete_qs[] = "DELETE FROM tensor_splits WHERE name=$name";
	sqlite3_stmt* tensor_split_delete_stmt = 0;
	int rc = sqlite3_prepare_v2(conn, tensor_split_delete_qs, sizeof(tensor_split_delete_qs), &tensor_split_delete_stmt, 0);
	if (rc != SQLITE_OK)
		return rc;
	sqlite3_bind_text(tensor_split_delete_stmt, 1, name, -1, SQLITE_STATIC);
	rc = sqlite3_step(tensor_split_delete_stmt);
	sqlite3_finalize(tensor_split_delete_stmt);
	return rc == SQLITE_DONE ? SQLITE_OK : rc;
}

static int _ccv_nnc_tensor_io_write_splits(sqlite3* const conn, const char* const name, const unsigned char* const data, const size_t data_size, const size_t offset, const int chunk_size)
{
	const char tensor_split_insert_qs[] =
		"REPLACE INTO tensor_splits "
		"(name, part, data) VALUES ($name, $part, $data)";
	sqlite3_stmt* tensor_split_insert_stmt = 0;
	int rc = sqlite3_prepare_v2(conn, tensor_split_insert_qs, sizeof(tensor_split_insert_qs), &tensor_split_insert_stmt, 0);
	if (rc != SQLITE_OK)
		return rc;
	size_t pos = offset;
	int part = 0;
	while (pos < data_size)
	{
		const size_t tail_size = data_size - pos;
		const int write_size = (int)ccv_min(tail_size, (size_t)chunk_size);
		sqlite3_bind_text(tensor_split_insert_stmt, 1, name, -1, SQLITE_STATIC);
		sqlite3_bind_int(tensor_split_insert_stmt, 2, part++);
		rc = sqlite3_bind_blob(tensor_split_insert_stmt, 3, data + pos, write_size, SQLITE_STATIC);
		if (rc != SQLITE_OK)
			break;
		rc = sqlite3_step(tensor_split_insert_stmt);
		if (rc != SQLITE_DONE)
			break;
		sqlite3_reset(tensor_split_insert_stmt);
		sqlite3_clear_bindings(tensor_split_insert_stmt);
		pos += write_size;
	}
	sqlite3_finalize(tensor_split_insert_stmt);
	return rc == SQLITE_DONE ? SQLITE_OK : rc;
}

static int _ccv_nnc_tensor_io_read_split_data(sqlite3* const conn, const char* const name, const void* const first_data, const size_t first_size, const void** const data_out, size_t* const data_size_out, unsigned char** const workspace_out)
{
	const char tensor_split_size_qs[] = "SELECT COUNT(*), SUM(length(data)) FROM tensor_splits WHERE name=$name";
	sqlite3_stmt* tensor_split_size_stmt = 0;
	int rc = sqlite3_prepare_v2(conn, tensor_split_size_qs, sizeof(tensor_split_size_qs), &tensor_split_size_stmt, 0);
	if (rc != SQLITE_OK)
		return rc;
	sqlite3_bind_text(tensor_split_size_stmt, 1, name, -1, SQLITE_STATIC);
	rc = sqlite3_step(tensor_split_size_stmt);
	if (rc != SQLITE_ROW)
	{
		sqlite3_finalize(tensor_split_size_stmt);
		return rc;
	}
	const sqlite_int64 split_count = sqlite3_column_int64(tensor_split_size_stmt, 0);
	const sqlite_int64 tail_size = sqlite3_column_int64(tensor_split_size_stmt, 1);
	sqlite3_finalize(tensor_split_size_stmt);
	if (split_count <= 0 || tail_size < 0 || (size_t)tail_size > SIZE_MAX - first_size)
		return SQLITE_CORRUPT;
	const size_t total_size = first_size + (size_t)tail_size;
	unsigned char* const workspace = (unsigned char*)ccmalloc(total_size);
	if (first_size > 0)
		memcpy(workspace, first_data, first_size);
	const char tensor_split_select_qs[] = "SELECT part, data FROM tensor_splits WHERE name=$name ORDER BY part";
	sqlite3_stmt* tensor_split_select_stmt = 0;
	rc = sqlite3_prepare_v2(conn, tensor_split_select_qs, sizeof(tensor_split_select_qs), &tensor_split_select_stmt, 0);
	if (rc != SQLITE_OK)
	{
		ccfree(workspace);
		return rc;
	}
	sqlite3_bind_text(tensor_split_select_stmt, 1, name, -1, SQLITE_STATIC);
	size_t offset = first_size;
	int expected_part = 0;
	while ((rc = sqlite3_step(tensor_split_select_stmt)) == SQLITE_ROW)
	{
		if (sqlite3_column_int(tensor_split_select_stmt, 0) != expected_part++)
		{
			sqlite3_finalize(tensor_split_select_stmt);
			ccfree(workspace);
			return SQLITE_CORRUPT;
		}
		const int split_size = sqlite3_column_bytes(tensor_split_select_stmt, 1);
		if (split_size < 0 || (size_t)split_size > total_size - offset)
		{
			sqlite3_finalize(tensor_split_select_stmt);
			ccfree(workspace);
			return SQLITE_CORRUPT;
		}
		const void* const split_data = sqlite3_column_blob(tensor_split_select_stmt, 1);
		if (split_size > 0)
			memcpy(workspace + offset, split_data, split_size);
		offset += split_size;
	}
	sqlite3_finalize(tensor_split_select_stmt);
	if (rc != SQLITE_DONE || offset != total_size)
	{
		ccfree(workspace);
		return rc == SQLITE_DONE ? SQLITE_CORRUPT : rc;
	}
	*data_out = workspace;
	*data_size_out = total_size;
	*workspace_out = workspace;
	return SQLITE_OK;
}

// MARK - Level-1 API

// The codebook slot of a row-wise-x weight stored with reserved field `reserved`: 0 (the built-in table), the inline
// slot 255 for a TMCQ weight with an inline codebook (its table is checked where it is decoded), or -1 for anything
// else: a slot with no inline codebook (a file-level codebook tensor is not supported), or an inline codebook on
// another codec.
static int _ccv_nnc_tensor_stored_codebook_slot(const int reserved)
{
	if (!ccv_nnc_8i_rowwise_x_codebook_is_valid(reserved))
		return -1;
	if (ccv_nnc_8i_rowwise_x_inline_codebook_entries(reserved))
		return CCV_NNC_CODEBOOK_INLINE_SLOT;
	return ccv_nnc_8i_rowwise_x_codebook_slot(reserved) ? -1 : 0;
}

// Whether a raw FTQ payload (inline codebook, if any, then the blob) of format `reserved` has a header whose state
// window is not the format's. A payload too short for a header, or without the FTQ magic, only fails a wide window.
static int _ccv_nnc_tensor_ftq_payload_window_mismatch(const void* const payload, const size_t payload_size, const int reserved)
{
	const size_t inline_size = ccv_nnc_8i_rowwise_x_inline_codebook_size(reserved);
	if (!payload || payload_size < inline_size + 32)
		return ccv_nnc_8i_rowwise_x_ftq_window(reserved) != 0;
	const uint8_t* const head = (const uint8_t*)payload + inline_size;
	if (memcmp(head, ccv_nnc_8i_rowwise_x_is_ftq_vr(reserved) ? "CCVFTQVR" : "CCVFTQSR", 8) != 0)
		return ccv_nnc_8i_rowwise_x_ftq_window(reserved) != 0;
	uint32_t field;
	memcpy(&field, head + 24, 4);
	return ccv_nnc_8i_rowwise_x_ftq_header_window_mismatch(field, reserved);
}

int ccv_nnc_tensor_write(const ccv_nnc_tensor_t* const tensor, void* const handle, const char* const name, const ccv_nnc_tensor_io_option_t* const options)
{
	assert(CCV_IS_TENSOR_CONTIGUOUS(tensor));
	assert(name);
	sqlite3* conn = (sqlite3*)handle;
	if (!conn)
		return CCV_IO_ERROR;
	// A row-wise-x weight is stored with the built-in table or an inline codebook; a slot registered in this process
	// means nothing in a file.
	if (CCV_GET_DATA_TYPE(tensor->info.datatype) == CCV_QX && (tensor->info.datatype & 0xf00) == CCV_NNC_QX_8I_ROWWISE_X &&
		_ccv_nnc_tensor_stored_codebook_slot(tensor->info.reserved) < 0)
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
	ccv_nnc_tensor_param_t params = tensor->info;
	const size_t data_size = ccv_nnc_tensor_data_size_without_padding(tensor->info);
	unsigned char* workspace = 0;
	const void* payload = tensor->data.u8;
	size_t payload_size = data_size;
	unsigned int identifier = 0;
#ifdef HAVE_CUDA
	if (CCV_TENSOR_GET_MEMORY(tensor->info.type) == CCV_TENSOR_GPU_MEMORY)
	{
		if (!options || !options->encode)
		{
			workspace = ccmalloc(data_size);
			cumemcpy(workspace, CCV_TENSOR_CPU_MEMORY, tensor->data.u8, tensor->info.type, data_size);
			payload = workspace;
		} else {
			workspace = ccmalloc(data_size * 2 + 4);
			cumemcpy(workspace, CCV_TENSOR_CPU_MEMORY, tensor->data.u8, tensor->info.type, data_size);
			size_t encoded_size = data_size + 4;
			if (options->encode(workspace, data_size, tensor->info.datatype, tensor->info.dim, ccv_nnc_tensor_nd(tensor->info.dim), options->context, workspace + data_size, &encoded_size, &params, &identifier))
			{
				payload = workspace + data_size;
				payload_size = encoded_size;
			}
			else
				payload = workspace;
		}
	} else {
		if (!options || !options->encode)
			payload = tensor->data.u8;
		else {
			workspace = ccmalloc(data_size + 4);
			size_t encoded_size = data_size + 4;
			if (options->encode(tensor->data.u8, data_size, tensor->info.datatype, tensor->info.dim, ccv_nnc_tensor_nd(tensor->info.dim), options->context, workspace, &encoded_size, &params, &identifier))
			{
				payload = workspace;
				payload_size = encoded_size;
			}
			else
				payload = tensor->data.u8;
		}
	}
#elif defined(HAVE_MPS)
	if (CCV_TENSOR_GET_MEMORY(tensor->info.type) == CCV_TENSOR_GPU_MEMORY)
	{
		if (!options || !options->encode)
		{
			workspace = ccmalloc(data_size);
			mpmemcpy(workspace, 0, CCV_TENSOR_CPU_MEMORY, tensor->data.u8, tensor->dataof, tensor->info.type, data_size);
			payload = workspace;
		} else {
			workspace = ccmalloc(data_size * 2 + 4);
			mpmemcpy(workspace, 0, CCV_TENSOR_CPU_MEMORY, tensor->data.u8, tensor->dataof, tensor->info.type, data_size);
			size_t encoded_size = data_size + 4;
			if (options->encode(workspace, data_size, tensor->info.datatype, tensor->info.dim, ccv_nnc_tensor_nd(tensor->info.dim), options->context, workspace + data_size, &encoded_size, &params, &identifier))
			{
				payload = workspace + data_size;
				payload_size = encoded_size;
			}
			else
				payload = workspace;
		}
	} else {
		if (!options || !options->encode)
			payload = tensor->data.u8;
		else {
			workspace = ccmalloc(data_size + 4); // Allocate extra 4 bytes in case we need to copy the QX tensor out.
			size_t encoded_size = data_size + 4;
			if (options->encode(tensor->data.u8, data_size, tensor->info.datatype, tensor->info.dim, ccv_nnc_tensor_nd(tensor->info.dim), options->context, workspace, &encoded_size, &params, &identifier))
			{
				payload = workspace;
				payload_size = encoded_size;
			}
			else
				payload = tensor->data.u8;
		}
	}
#else
	if (!options || !options->encode)
		payload = tensor->data.u8;
	else {
		workspace = ccmalloc(data_size + 4);
		size_t encoded_size = data_size + 4;
		if (options->encode(tensor->data.u8, data_size, tensor->info.datatype, tensor->info.dim, ccv_nnc_tensor_nd(tensor->info.dim), options->context, workspace, &encoded_size, &params, &identifier))
		{
			payload = workspace;
			payload_size = encoded_size;
		}
		else
			payload = tensor->data.u8;
	}
#endif
	// A raw FTQ blob names its state window in its header, which must be the format's: a reader checks the two
	// against each other, and a 16-bit reader would decode a wide blob as another one. Data encoded by an identifier
	// codec has no header a reader can check, so a wide window is only stored raw (ccv_nnc_tensor_read rejects it).
	const int ftq_payload = CCV_GET_DATA_TYPE(params.datatype) == CCV_QX && (params.datatype & 0xf00) == CCV_NNC_QX_8I_ROWWISE_X &&
		ccv_nnc_8i_rowwise_x_is_ftq_any(params.reserved);
	if (ftq_payload && (identifier ? ccv_nnc_8i_rowwise_x_ftq_window(params.reserved) != 0 : _ccv_nnc_tensor_ftq_payload_window_mismatch(payload, payload_size, params.reserved)))
	{
		sqlite3_finalize(tensor_insert_stmt);
		if (workspace)
			free(workspace);
		return CCV_IO_ERROR;
	}
	int result = SQLITE_TOOBIG;
	int bind_result = payload_size <= INT_MAX ? sqlite3_bind_blob(tensor_insert_stmt, 6, payload, (int)payload_size, 0) : SQLITE_TOOBIG;
	if (bind_result == SQLITE_OK)
	{
		sqlite3_bind_int64(tensor_insert_stmt, 2, ((sqlite_int64)identifier << 32) | params.type);
		sqlite3_bind_int(tensor_insert_stmt, 3, params.format);
		sqlite3_bind_int64(tensor_insert_stmt, 4, ((sqlite_int64)params.reserved << 32) | params.datatype);
		sqlite3_bind_blob(tensor_insert_stmt, 5, params.dim, sizeof(params.dim), 0);
		result = sqlite3_step(tensor_insert_stmt);
		if (result == SQLITE_DONE)
		{
			sqlite3_reset(tensor_insert_stmt);
			sqlite3_clear_bindings(tensor_insert_stmt);
			sqlite3_finalize(tensor_insert_stmt);
			if (workspace)
				free(workspace);
			return CCV_IO_FINAL;
		}
		sqlite3_reset(tensor_insert_stmt);
		sqlite3_clear_bindings(tensor_insert_stmt);
		if (result != SQLITE_TOOBIG)
		{
			sqlite3_finalize(tensor_insert_stmt);
			if (workspace)
				free(workspace);
			return CCV_IO_ERROR;
		}
	} else {
		sqlite3_reset(tensor_insert_stmt);
		sqlite3_clear_bindings(tensor_insert_stmt);
		if (bind_result != SQLITE_TOOBIG)
		{
			sqlite3_finalize(tensor_insert_stmt);
			if (workspace)
				free(workspace);
			return CCV_IO_ERROR;
		}
	}
	if (sqlite3_exec(conn, "SAVEPOINT ccv_nnc_tensor_write", 0, 0, 0) != SQLITE_OK)
	{
		sqlite3_finalize(tensor_insert_stmt);
		if (workspace)
			free(workspace);
		return CCV_IO_ERROR;
	}
	const char tensor_split_create_table_qs[] = "CREATE TABLE IF NOT EXISTS tensor_splits "
		"(name TEXT, part INTEGER, data BLOB, PRIMARY KEY (name, part))";
	if (sqlite3_exec(conn, tensor_split_create_table_qs, 0, 0, 0) != SQLITE_OK)
	{
		sqlite3_finalize(tensor_insert_stmt);
		sqlite3_exec(conn, "ROLLBACK TO ccv_nnc_tensor_write", 0, 0, 0);
		sqlite3_exec(conn, "RELEASE ccv_nnc_tensor_write", 0, 0, 0);
		if (workspace)
			free(workspace);
		return CCV_IO_ERROR;
	}
	sqlite3_bind_text(tensor_insert_stmt, 1, name, -1, 0);
	const int chunk_size = _ccv_nnc_tensor_io_blob_chunk_size(conn);
	assert(chunk_size > 0);
	const size_t first_size = ccv_min(payload_size, (size_t)chunk_size);
	// A reader checks a raw FTQ blob's header (and its inline codebook) from the first part alone, so a part too
	// short for both is not written.
	if (ftq_payload && !identifier && first_size < ccv_nnc_8i_rowwise_x_inline_codebook_size(params.reserved) + 32)
		bind_result = SQLITE_TOOBIG;
	else
		bind_result = first_size < payload_size ? sqlite3_bind_blob(tensor_insert_stmt, 6, payload, (int)first_size, SQLITE_STATIC) : SQLITE_TOOBIG;
	if (bind_result == SQLITE_OK)
		bind_result = _ccv_nnc_tensor_io_delete_splits(conn, name);
	if (bind_result == SQLITE_OK)
		bind_result = _ccv_nnc_tensor_io_write_splits(conn, name, (const unsigned char*)payload, payload_size, first_size, chunk_size);
	if (bind_result != SQLITE_OK)
	{
		sqlite3_finalize(tensor_insert_stmt);
		sqlite3_exec(conn, "ROLLBACK TO ccv_nnc_tensor_write", 0, 0, 0);
		sqlite3_exec(conn, "RELEASE ccv_nnc_tensor_write", 0, 0, 0);
		if (workspace)
			free(workspace);
		return CCV_IO_ERROR;
	}
	sqlite3_bind_int64(tensor_insert_stmt, 2, ((sqlite_int64)identifier << 32) | params.type);
	sqlite3_bind_int64(tensor_insert_stmt, 3, ((sqlite_int64)params.format & CCV_NNC_TENSOR_IO_FORMAT_MASK) | CCV_NNC_TENSOR_IO_SPLIT_FORMAT);
	sqlite3_bind_int64(tensor_insert_stmt, 4, ((sqlite_int64)params.reserved << 32) | params.datatype);
	sqlite3_bind_blob(tensor_insert_stmt, 5, params.dim, sizeof(params.dim), 0);
	result = sqlite3_step(tensor_insert_stmt);
	sqlite3_reset(tensor_insert_stmt);
	sqlite3_clear_bindings(tensor_insert_stmt);
	sqlite3_finalize(tensor_insert_stmt);
	if (result == SQLITE_DONE)
	{
		if (sqlite3_exec(conn, "RELEASE ccv_nnc_tensor_write", 0, 0, 0) != SQLITE_OK)
		{
			if (workspace)
				free(workspace);
			return CCV_IO_ERROR;
		}
	} else {
		sqlite3_exec(conn, "ROLLBACK TO ccv_nnc_tensor_write", 0, 0, 0);
		sqlite3_exec(conn, "RELEASE ccv_nnc_tensor_write", 0, 0, 0);
	}
	if (workspace)
		free(workspace);
	return result == SQLITE_DONE ? CCV_IO_FINAL : CCV_IO_ERROR;
}

// Whether row-wise-x params do not name the inline codebook bits stored_inline_bits of the stored weight (0: the
// stored weight has none, nothing to match), or name another FTQ state window than stored_window (the window of a
// stored row-wise-x weight, -1: not such a weight, nothing to match). Params of another datatype are left to the
// decode callback.
static int _ccv_nnc_tensor_inline_codebook_mismatch(const ccv_nnc_tensor_param_t params, const int stored_inline_bits, const int stored_window)
{
	if (CCV_GET_DATA_TYPE(params.datatype) != CCV_QX || (params.datatype & 0xf00) != CCV_NNC_QX_8I_ROWWISE_X)
		return 0;
	if (stored_window >= 0 && ccv_nnc_8i_rowwise_x_ftq_window(params.reserved) != stored_window)
		return 1;
	if (!stored_inline_bits)
		return 0;
	return (params.reserved & (CCV_NNC_CODEBOOK_INLINE_MASK | CCV_NNC_CODEBOOK_SLOT_MASK)) != stored_inline_bits;
}

int ccv_nnc_tensor_read(void* const handle, const char* const name, const ccv_nnc_tensor_io_option_t* const options, const int flags, const ccv_nnc_tensor_param_t* const tensor_params_optional, ccv_nnc_tensor_t** const tensor_out)
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
	const int tensor_provided = !!tensor;
	ccv_nnc_tensor_param_t tensor_params;
	int datatype = 0;
	unsigned int identifier = 0;
	// A row-wise-x weight: check its codebook first, so one that is not the built-in table or inline fails the read
	// instead of decoding with the built-in table. codebook_slot < 0: not such a weight.
	int codebook_slot = -1;
	// A variable-rate FTQ weight (its stored format with the resolved codebook slot): the whole blob is checked once
	// it is read. -1: not such a weight.
	int ftq_vr_format = -1;
	// A static-rate FTQ weight stored raw (its stored format with the resolved codebook slot): the whole container is
	// checked once it is read. -1: not such a weight (or stored by an identifier codec, whose data is not the blob).
	int ftq_sr_format = -1;
	// An FTQ weight with an inline codebook: the bytes of the codebook before the blob that the checks above look at.
	size_t ftq_inline_size = 0;
	// For a weight with an inline codebook, its codebook bits (slot 255 and P): a row-wise-x read into caller params or
	// an existing tensor must name the same, since the table is part of the data and sets its size.
	int stored_inline_bits = 0;
	// For a row-wise-x weight, its FTQ state window (0 for every other codec): params or an existing tensor must name
	// the same, since the blob is decoded with it. -1: not such a weight.
	int stored_window = -1;
	{
		const sqlite_int64 stored_datatype_mix = sqlite3_column_int64(tensor_select_stmt, 3);
		const int stored_datatype = (int)(stored_datatype_mix & 0xffffffff);
		if (CCV_GET_DATA_TYPE(stored_datatype) == CCV_QX && (stored_datatype & 0xf00) == CCV_NNC_QX_8I_ROWWISE_X)
		{
			const int stored_reserved = (int)((stored_datatype_mix >> 32) & 0xffffffff);
			codebook_slot = _ccv_nnc_tensor_stored_codebook_slot(stored_reserved);
			if (codebook_slot < 0)
			{
				sqlite3_finalize(tensor_select_stmt);
				return CCV_IO_ERROR;
			}
			stored_window = ccv_nnc_8i_rowwise_x_ftq_window(stored_reserved);
			if (ccv_nnc_8i_rowwise_x_inline_codebook_entries(stored_reserved))
				stored_inline_bits = stored_reserved & (CCV_NNC_CODEBOOK_INLINE_MASK | CCV_NNC_CODEBOOK_SLOT_MASK);
			// An FTQ blob says in its header (table bits 16 - 17) whether its table is quadrant folded; the kernels
			// take that from the slot, so a folded table maps to the slot registered with CCV_NNC_CODEBOOK_FOLD. The
			// header sits at the start of the stored data (the first part of a split tensor); data encoded by an
			// identifier codec is not looked into here and is rejected if it names a codebook.
			// With an inline codebook the blob follows the codebook: the table is resolved from the data (registered
			// folded when the header says so, ccv_nnc_8i_rowwise_x_resolve_inline) for the checks, and the tensor keeps
			// the inline slot like TMCQ, its kernels resolving the table again from the data they are handed.
			if (ccv_nnc_8i_rowwise_x_is_ftq_any(stored_reserved))
			{
				const int vr = ccv_nnc_8i_rowwise_x_is_ftq_vr(stored_reserved);
				const int stored_identifier = (int)((sqlite3_column_int64(tensor_select_stmt, 1) >> 32) & 0xffffffff);
				const uint8_t* head = (const uint8_t*)sqlite3_column_blob(tensor_select_stmt, 0);
				int head_bytes = sqlite3_column_bytes(tensor_select_stmt, 0);
				int check_slot = codebook_slot;
				ftq_inline_size = ccv_nnc_8i_rowwise_x_inline_codebook_size(stored_reserved);
				if (ftq_inline_size)
				{
					// Only a raw blob carries an inline codebook; the codebook and the blob header are in the first part.
					// A header naming another state window fails before its table is registered.
					uint32_t inline_field = 0;
					if (stored_identifier == 0 && head && (size_t)head_bytes >= ftq_inline_size + 32)
						memcpy(&inline_field, head + ftq_inline_size + 24, 4);
					const int resolved = stored_identifier == 0 && head && (size_t)head_bytes >= ftq_inline_size + 32 && !ccv_nnc_8i_rowwise_x_ftq_header_window_mismatch(inline_field, stored_reserved) ? ccv_nnc_8i_rowwise_x_resolve_inline(head, head_bytes, stored_reserved) : -1;
					if (resolved < 0)
					{
						sqlite3_finalize(tensor_select_stmt);
						return CCV_IO_ERROR;
					}
					check_slot = ccv_nnc_codebook_slot_of(resolved);
					head += ftq_inline_size;
					head_bytes -= (int)ftq_inline_size;
				}
				// Data encoded by an identifier codec is not the blob, so has no header to check a wide state window
				// against.
				if (stored_identifier != 0 && stored_window != 0)
				{
					sqlite3_finalize(tensor_select_stmt);
					return CCV_IO_ERROR;
				}
				uint32_t table_field = 0;
				if (vr)
				{
					// Variable-rate FTQ is only stored raw, so its whole blob can be checked below.
					if (stored_identifier != 0 || !head || head_bytes < 32 || memcmp(head, "CCVFTQVR", 8) != 0)
					{
						sqlite3_finalize(tensor_select_stmt);
						return CCV_IO_ERROR;
					}
					memcpy(&table_field, head + 24, 4);
				} else if (head && head_bytes >= 32 && memcmp(head, "CCVFTQSR", 8) == 0) {
					// Every tile has 16 n bits and nothing is dropped (see CCV_NNC_QX_8I_ROWWISE_FTQ_N16).
					uint32_t high_drop[2];
					memcpy(high_drop, head + 16, sizeof(high_drop));
					if (high_drop[0] != 16u * (uint32_t)ccv_nnc_8i_rowwise_x_ftq_rate(stored_reserved) || high_drop[1] != 0)
					{
						sqlite3_finalize(tensor_select_stmt);
						return CCV_IO_ERROR;
					}
					memcpy(&table_field, head + 24, 4);
				} else if (stored_identifier == 0 || codebook_slot != 0) {
					sqlite3_finalize(tensor_select_stmt);
					return CCV_IO_ERROR;
				}
				// The header names the format's state window (also for a metadata-only read, which checks no more).
				if (stored_identifier == 0 && ccv_nnc_8i_rowwise_x_ftq_header_window_mismatch(table_field, stored_reserved))
				{
					sqlite3_finalize(tensor_select_stmt);
					return CCV_IO_ERROR;
				}
				// An inline table is registered with the header's structure already. Without one the table is the
				// built-in one, which is never folded.
				if (((table_field >> 16) & 3) && !ftq_inline_size)
				{
					sqlite3_finalize(tensor_select_stmt);
					return CCV_IO_ERROR;
				}
				// The checks take the blob without the inline codebook, so a format without the inline bits.
				const int check_format = ccv_nnc_codebook_with_slot(stored_reserved & ~CCV_NNC_QX_8I_ROWWISE_CODEBOOK_INLINE_MASK, check_slot);
				if (vr)
					ftq_vr_format = check_format;
				else if (stored_identifier == 0 && head && head_bytes >= 32)
					ftq_sr_format = check_format;
			}
		}
	}
	const sqlite_int64 format_mix = sqlite3_column_int64(tensor_select_stmt, 2);
	const int has_splits = !!(format_mix & CCV_NNC_TENSOR_IO_SPLIT_FORMAT);
	if (!tensor) // If the tensor is not provided, we need to create one.
	{
		if (tensor_params_optional)
		{
			identifier = (sqlite3_column_int64(tensor_select_stmt, 1) >> 32) & 0xffffffff;
			datatype = sqlite3_column_int64(tensor_select_stmt, 3) & 0xffffffff;
			tensor_params = *tensor_params_optional;
			assert(!(flags & CCV_NNC_TENSOR_READ_METADATA_ONLY));
			if (_ccv_nnc_tensor_inline_codebook_mismatch(tensor_params, stored_inline_bits, stored_window))
			{
				sqlite3_finalize(tensor_select_stmt);
				return CCV_IO_ERROR;
			}
		} else {
			const sqlite_int64 type = sqlite3_column_int64(tensor_select_stmt, 1);
			identifier = (type >> 32) & 0xffffffff;
			tensor_params.type = (type & 0xffffffff);
			tensor_params.format = (int)(format_mix & CCV_NNC_TENSOR_IO_FORMAT_MASK);
			const sqlite_int64 datatype_mix = sqlite3_column_int64(tensor_select_stmt, 3);
			datatype = tensor_params.datatype = (datatype_mix & 0xffffffff);
			tensor_params.reserved = (datatype_mix >> 32) & 0xffffffff;
			if (codebook_slot >= 0)
				tensor_params.reserved = ccv_nnc_codebook_with_slot(tensor_params.reserved, codebook_slot);
			const void* const dim = sqlite3_column_blob(tensor_select_stmt, 4);
			memcpy(tensor_params.dim, dim, ccv_min(sizeof(tensor_params.dim), sqlite3_column_bytes(tensor_select_stmt, 4)));
		}
		if (flags & CCV_NNC_TENSOR_READ_CPU_MEMORY) // Reset type to CPU memory.
			tensor_params.type = (tensor_params.type & 0xfff00000) | CCV_TENSOR_CPU_MEMORY;
		if (!options || !options->decode)
		{
			if (flags & CCV_NNC_TENSOR_READ_METADATA_ONLY)
			{
				*tensor_out = tensor = ccv_nnc_tensor_new(0, tensor_params, CCV_NO_DATA_ALLOC); // Set the data point to 1 so it is allocated without data.
				assert(tensor->data.u8 == 0); // Set it back to 0.
				// Already done loading metadata, return.
				sqlite3_reset(tensor_select_stmt);
				sqlite3_clear_bindings(tensor_select_stmt);
				sqlite3_finalize(tensor_select_stmt);
				return CCV_IO_FINAL;
			} else
				*tensor_out = tensor = ccv_nnc_tensor_new(0, tensor_params, 0);
		} else {
			assert(!(flags & CCV_NNC_TENSOR_READ_METADATA_ONLY));
		}
	} else {
		identifier = (sqlite3_column_int64(tensor_select_stmt, 1) >> 32) & 0xffffffff;
		datatype = sqlite3_column_int64(tensor_select_stmt, 3) & 0xffffffff;
		tensor_params = tensor->info;
		assert(!(flags & CCV_NNC_TENSOR_READ_METADATA_ONLY));
		if (_ccv_nnc_tensor_inline_codebook_mismatch(tensor_params, stored_inline_bits, stored_window))
		{
			sqlite3_finalize(tensor_select_stmt);
			return CCV_IO_ERROR;
		}
	}
	const void* data = sqlite3_column_blob(tensor_select_stmt, 0);
	size_t data_bytes = sqlite3_column_bytes(tensor_select_stmt, 0);
	unsigned char* split_workspace = 0;
	if (has_splits)
	{
		const int split_result = _ccv_nnc_tensor_io_read_split_data(conn, name, data, data_bytes, &data, &data_bytes, &split_workspace);
		if (split_result != SQLITE_OK)
		{
			sqlite3_reset(tensor_select_stmt);
			sqlite3_clear_bindings(tensor_select_stmt);
			sqlite3_finalize(tensor_select_stmt);
			return CCV_IO_ERROR;
		}
	}
	// An inline codebook stored as is (no decoder takes these bytes) is checked here, once, on the host: magic, P,
	// crc32, zero padding and values. Backends that cannot read the table on the host per command (CUDA) rely on it;
	// a decoder that takes the bytes checks its output itself (ccv_nnc_codebook_resolve_inline). An FTQ inline codebook
	// was resolved with its blob header above.
	if (stored_inline_bits && !ftq_inline_size && (!identifier || !options || !options->decode) &&
		(data_bytes < ccv_nnc_8i_rowwise_x_inline_codebook_size(stored_inline_bits) || ccv_nnc_codebook_resolve_inline(data, stored_inline_bits) < 0))
	{
		if (split_workspace)
			ccfree(split_workspace);
		if (!tensor_provided && *tensor_out)
		{
			ccv_nnc_tensor_free(*tensor_out);
			*tensor_out = 0;
		}
		sqlite3_reset(tensor_select_stmt);
		sqlite3_clear_bindings(tensor_select_stmt);
		sqlite3_finalize(tensor_select_stmt);
		return CCV_IO_ERROR;
	}
	int dim[CCV_NNC_MAX_DIM_ALLOC];
	memcpy(dim, sqlite3_column_blob(tensor_select_stmt, 4), ccv_min(sizeof(dim), sqlite3_column_bytes(tensor_select_stmt, 4)));
	const int nd = ccv_nnc_tensor_nd(dim);
	if (ftq_vr_format >= 0 || ftq_sr_format >= 0)
	{
		// Variable-rate FTQ: every rule of CCV_NNC_QX_8I_ROWWISE_FTQ_VR_N16 on the stored blob, against the stored shape
		// and format. Static-rate FTQ: the whole CCVFTQSR container (ccv_nnc_8i_rowwise_x_ftq_load_check). The
		// stored data (all parts of a split tensor) is the whole blob, so both take its length as exactly the blob's.
		// An inline codebook before the blob was checked when it was resolved.
		const void* const blob = (const uint8_t*)data + ftq_inline_size;
		const size_t blob_bytes = data_bytes >= ftq_inline_size ? data_bytes - ftq_inline_size : 0;
		size_t count = 1;
		int i;
		for (i = 0; i < nd; i++)
			count *= dim[i] > 0 ? (size_t)dim[i] : 0;
		const size_t row_length = nd > 0 && dim[nd - 1] > 0 ? (size_t)dim[nd - 1] : 0;
		if (row_length == 0 || count % row_length != 0 ||
			(ftq_vr_format >= 0 && (datatype != (((CCV_16F >> 12) & 0xff) | CCV_QX | CCV_NNC_QX_8I_ROWWISE_X) ||
				ccv_nnc_8i_rowwise_x_ftq_vr_check(blob, blob_bytes, ftq_vr_format, count / row_length, row_length, 0) != 0)) ||
			// Static-rate FTQ has an FP16 or an FP32 base (the same blob, see CCV_NNC_QX_8I_ROWWISE_FTQ_N16).
			(ftq_sr_format >= 0 && ((datatype != (((CCV_16F >> 12) & 0xff) | CCV_QX | CCV_NNC_QX_8I_ROWWISE_X) &&
				datatype != (((CCV_32F >> 12) & 0xff) | CCV_QX | CCV_NNC_QX_8I_ROWWISE_X)) ||
				ccv_nnc_8i_rowwise_x_ftq_load_check(blob, blob_bytes, ftq_sr_format, count / row_length, row_length) != 0)))
		{
			if (!tensor_provided && *tensor_out)
			{
				ccv_nnc_tensor_free(*tensor_out);
				*tensor_out = 0;
			}
			if (split_workspace)
				ccfree(split_workspace);
			sqlite3_reset(tensor_select_stmt);
			sqlite3_clear_bindings(tensor_select_stmt);
			sqlite3_finalize(tensor_select_stmt);
			return CCV_IO_ERROR;
		}
	}
	if (datatype != tensor_params.datatype && CCV_GET_DATA_TYPE(tensor_params.datatype) != CCV_QX)
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
			size_t decoded_size = data_size;
			if (!options || !options->decode)
			{
				copying = workspace = ccmalloc(data_size);
				if (datatype == CCV_16F && tensor_params.datatype == CCV_32F)
					ccv_half_precision_to_float((uint16_t*)data, (float*)workspace, ccv_min(tensor_count, data_bytes / sizeof(uint16_t)));
				else if (datatype == CCV_32F && tensor_params.datatype == CCV_16F)
					ccv_float_to_half_precision((float*)data, (uint16_t*)workspace, ccv_min(tensor_count, data_bytes / sizeof(float)));
				else
					{ assert(0); }
			} else {
				copying = workspace = ccmalloc(data_size + source_data_size);
				if (datatype == CCV_16F && tensor_params.datatype == CCV_32F)
				{
					decoded_size = source_data_size;
					if (options->decode(data, data_bytes, datatype, dim, nd, identifier, options->context, tensor_params, tensor_out, workspace + data_size, &decoded_size))
					{
						// If we loaded quantized tensor, don't do the conversion.
						if (CCV_GET_DATA_TYPE(tensor_out[0]->info.datatype) == CCV_QX)
							copying = workspace + data_size;
						else {
							ccv_half_precision_to_float((uint16_t*)(workspace + data_size), (float*)workspace, ccv_min(tensor_count, ccv_min(source_data_size, decoded_size) / sizeof(uint16_t)));
							decoded_size = data_size;
						}
					} else {
						if (!tensor)
							*tensor_out = tensor = ccv_nnc_tensor_new(0, tensor_params, 0);
						ccv_half_precision_to_float((uint16_t*)data, (float*)workspace, ccv_min(tensor_count, data_bytes / sizeof(uint16_t)));
						decoded_size = data_size;
					}
				} else if (datatype == CCV_32F && tensor_params.datatype == CCV_16F) {
					decoded_size = source_data_size;
					if (options->decode(data, data_bytes, datatype, dim, nd, identifier, options->context, tensor_params, tensor_out, workspace + data_size, &decoded_size))
					{
						if (CCV_GET_DATA_TYPE(tensor_out[0]->info.datatype) == CCV_QX)
							copying = workspace + data_size;
						else {
							ccv_float_to_half_precision((float*)(workspace + data_size), (uint16_t*)workspace, ccv_min(tensor_count, ccv_min(source_data_size, decoded_size) / sizeof(float)));
							decoded_size = data_size;
						}
					} else {
						if (!tensor)
							*tensor_out = tensor = ccv_nnc_tensor_new(0, tensor_params, 0);
						ccv_float_to_half_precision((float*)data, (uint16_t*)workspace, ccv_min(tensor_count, data_bytes / sizeof(float)));
						decoded_size = data_size;
					}
				} else
					{ assert(0); }
			}
			cumemcpy(tensor_out[0]->data.u8, tensor_out[0]->info.type, copying, CCV_TENSOR_CPU_MEMORY, decoded_size);
			ccfree(workspace);
		} else {
			if (!options || !options->decode)
			{
				if (datatype == CCV_16F && tensor_params.datatype == CCV_32F)
					ccv_half_precision_to_float((uint16_t*)data, tensor->data.f32, ccv_min(tensor_count, data_bytes / sizeof(uint16_t)));
				else if (datatype == CCV_32F && tensor_params.datatype == CCV_16F)
					ccv_float_to_half_precision((float*)data, (uint16_t*)tensor->data.f16, ccv_min(tensor_count, data_bytes / sizeof(float)));
				else
					{ assert(0); }
			} else {
				void* const workspace = ccmalloc(source_data_size);
				if (datatype == CCV_16F && tensor_params.datatype == CCV_32F)
				{
					size_t decoded_size = source_data_size;
					if (options->decode(data, data_bytes, datatype, dim, nd, identifier, options->context, tensor_params, tensor_out, workspace, &decoded_size))
					{
						if (CCV_GET_DATA_TYPE(tensor_out[0]->info.datatype) == CCV_QX)
						{
							if (decoded_size > 0)
								memcpy(tensor_out[0]->data.f32, workspace, ccv_min(source_data_size, decoded_size));
						} else
							ccv_half_precision_to_float((uint16_t*)workspace, tensor_out[0]->data.f32, ccv_min(tensor_count, ccv_min(source_data_size, decoded_size) / sizeof(uint16_t)));
					} else {
						if (!tensor)
							*tensor_out = tensor = ccv_nnc_tensor_new(0, tensor_params, 0);
						ccv_half_precision_to_float((uint16_t*)data, tensor->data.f32, ccv_min(tensor_count, data_bytes / sizeof(uint16_t)));
					}
				} else if (datatype == CCV_32F && tensor_params.datatype == CCV_16F) {
					size_t decoded_size = source_data_size;
					if (options->decode(data, data_bytes, datatype, dim, nd, identifier, options->context, tensor_params, tensor_out, workspace, &decoded_size))
					{
						if (CCV_GET_DATA_TYPE(tensor_out[0]->info.datatype) == CCV_QX)
						{
							if (decoded_size > 0)
								memcpy(tensor_out[0]->data.f16, workspace, ccv_min(source_data_size, decoded_size));
						} else
							ccv_float_to_half_precision((float*)workspace, (uint16_t*)tensor_out[0]->data.f16, ccv_min(tensor_count, ccv_min(source_data_size, decoded_size) / sizeof(float)));
					} else {
						if (!tensor)
							*tensor_out = tensor = ccv_nnc_tensor_new(0, tensor_params, 0);
						ccv_float_to_half_precision((float*)data, (uint16_t*)tensor->data.f16, ccv_min(tensor_count, data_bytes / sizeof(float)));
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
			size_t decoded_size = data_size;
			if (!options || !options->decode)
			{
				copying = workspace = ccmalloc(data_size);
				if (datatype == CCV_16F && tensor_params.datatype == CCV_32F)
					ccv_half_precision_to_float((uint16_t*)data, (float*)workspace, ccv_min(tensor_count, data_bytes / sizeof(uint16_t)));
				else if (datatype == CCV_32F && tensor_params.datatype == CCV_16F)
					ccv_float_to_half_precision((float*)data, (uint16_t*)workspace, ccv_min(tensor_count, data_bytes / sizeof(float)));
				else
					{ assert(0); }
			} else {
				copying = workspace = ccmalloc(data_size + source_data_size);
				if (datatype == CCV_16F && tensor_params.datatype == CCV_32F)
				{
					decoded_size = source_data_size;
					if (options->decode(data, data_bytes, datatype, dim, nd, identifier, options->context, tensor_params, tensor_out, workspace + data_size, &decoded_size))
					{
						if (CCV_GET_DATA_TYPE(tensor_out[0]->info.datatype) == CCV_QX)
							copying = workspace + data_size;
						else {
							ccv_half_precision_to_float((uint16_t*)(workspace + data_size), (float*)workspace, ccv_min(tensor_count, ccv_min(source_data_size, decoded_size) / sizeof(uint16_t)));
							decoded_size = data_size;
						}
					} else {
						if (!tensor)
							*tensor_out = tensor = ccv_nnc_tensor_new(0, tensor_params, 0);
						ccv_half_precision_to_float((uint16_t*)data, (float*)workspace, ccv_min(tensor_count, data_bytes / sizeof(uint16_t)));
						decoded_size = data_size;
					}
				} else if (datatype == CCV_32F && tensor_params.datatype == CCV_16F) {
					decoded_size = source_data_size;
					if (options->decode(data, data_bytes, datatype, dim, nd, identifier, options->context, tensor_params, tensor_out, workspace + data_size, &decoded_size))
					{
						if (CCV_GET_DATA_TYPE(tensor_out[0]->info.datatype) == CCV_QX)
							copying = workspace + data_size;
						else {
							ccv_float_to_half_precision((float*)(workspace + data_size), (uint16_t*)workspace, ccv_min(tensor_count, ccv_min(source_data_size, decoded_size) / sizeof(float)));
							decoded_size = data_size;
						}
					} else {
						if (!tensor)
							*tensor_out = tensor = ccv_nnc_tensor_new(0, tensor_params, 0);
						ccv_float_to_half_precision((float*)data, (uint16_t*)workspace, ccv_min(tensor_count, data_bytes / sizeof(float)));
						decoded_size = data_size;
					}
				} else
					{ assert(0); }
			}
			assert(tensor_out[0]->dataof == 0);
			mpmemcpy(tensor_out[0]->data.u8, tensor_out[0]->dataof, tensor_out[0]->info.type, copying, 0, CCV_TENSOR_CPU_MEMORY, decoded_size);
			ccfree(workspace);
		} else {
			if (!options || !options->decode)
			{
				if (datatype == CCV_16F && tensor_params.datatype == CCV_32F)
					ccv_half_precision_to_float((uint16_t*)data, tensor->data.f32, ccv_min(tensor_count, data_bytes / sizeof(uint16_t)));
				else if (datatype == CCV_32F && tensor_params.datatype == CCV_16F)
					ccv_float_to_half_precision((float*)data, (uint16_t*)tensor->data.f16, ccv_min(tensor_count, data_bytes / sizeof(float)));
				else
					{ assert(0); }
			} else {
				void* const workspace = ccmalloc(source_data_size);
				if (datatype == CCV_16F && tensor_params.datatype == CCV_32F)
				{
					size_t decoded_size = source_data_size;
					if (options->decode(data, data_bytes, datatype, dim, nd, identifier, options->context, tensor_params, tensor_out, workspace, &decoded_size))
					{
						if (CCV_GET_DATA_TYPE(tensor_out[0]->info.datatype) == CCV_QX)
						{
							if (decoded_size > 0)
								memcpy(tensor_out[0]->data.f32, workspace, ccv_min(source_data_size, decoded_size));
						} else
							ccv_half_precision_to_float((uint16_t*)workspace, tensor_out[0]->data.f32, ccv_min(tensor_count, ccv_min(source_data_size, decoded_size) / sizeof(uint16_t)));
					} else {
						if (!tensor)
							*tensor_out = tensor = ccv_nnc_tensor_new(0, tensor_params, 0);
						ccv_half_precision_to_float((uint16_t*)data, tensor->data.f32, ccv_min(tensor_count, data_bytes / sizeof(uint16_t)));
					}
				} else if (datatype == CCV_32F && tensor_params.datatype == CCV_16F) {
					size_t decoded_size = source_data_size;
					if (options->decode(data, data_bytes, datatype, dim, nd, identifier, options->context, tensor_params, tensor_out, workspace, &decoded_size))
					{
						if (CCV_GET_DATA_TYPE(tensor_out[0]->info.datatype) == CCV_QX)
						{
							if (decoded_size > 0)
								memcpy(tensor_out[0]->data.f16, workspace, ccv_min(source_data_size, decoded_size));
						} else
							ccv_float_to_half_precision((float*)workspace, (uint16_t*)tensor_out[0]->data.f16, ccv_min(tensor_count, ccv_min(source_data_size, decoded_size) / sizeof(float)));
					} else {
						if (!tensor)
							*tensor_out = tensor = ccv_nnc_tensor_new(0, tensor_params, 0);
						ccv_float_to_half_precision((float*)data, (uint16_t*)tensor->data.f16, ccv_min(tensor_count, data_bytes / sizeof(float)));
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
				ccv_half_precision_to_float((uint16_t*)data, tensor->data.f32, ccv_min(tensor_count, data_bytes / sizeof(uint16_t)));
			else if (datatype == CCV_32F && tensor_params.datatype == CCV_16F)
				ccv_float_to_half_precision((float*)data, (uint16_t*)tensor->data.f16, ccv_min(tensor_count, data_bytes / sizeof(float)));
			else
				{ assert(0); }
		} else {
			void* const workspace = ccmalloc(source_data_size);
			if (datatype == CCV_16F && tensor_params.datatype == CCV_32F)
			{
				size_t decoded_size = source_data_size;
				if (options->decode(data, data_bytes, datatype, dim, nd, identifier, options->context, tensor_params, tensor_out, workspace, &decoded_size))
				{
					if (CCV_GET_DATA_TYPE(tensor_out[0]->info.datatype) == CCV_QX)
					{
						if (decoded_size > 0)
							memcpy(tensor_out[0]->data.f32, workspace, ccv_min(source_data_size, decoded_size));
					} else
						ccv_half_precision_to_float((uint16_t*)workspace, tensor_out[0]->data.f32, ccv_min(tensor_count, ccv_min(source_data_size, decoded_size) / sizeof(uint16_t)));
				} else {
					if (!tensor)
						*tensor_out = tensor = ccv_nnc_tensor_new(0, tensor_params, 0);
					ccv_half_precision_to_float((uint16_t*)data, tensor->data.f32, ccv_min(tensor_count, data_bytes / sizeof(uint16_t)));
				}
			} else if (datatype == CCV_32F && tensor_params.datatype == CCV_16F) {
				size_t decoded_size = source_data_size;
				if (options->decode(data, data_bytes, datatype, dim, nd, identifier, options->context, tensor_params, tensor_out, workspace, &decoded_size))
				{
					if (CCV_GET_DATA_TYPE(tensor_out[0]->info.datatype) == CCV_QX)
					{
						if (decoded_size > 0)
							memcpy(tensor_out[0]->data.f16, workspace, ccv_min(source_data_size, decoded_size));
					} else
						ccv_float_to_half_precision((float*)workspace, (uint16_t*)tensor_out[0]->data.f16, ccv_min(tensor_count, ccv_min(source_data_size, decoded_size) / sizeof(float)));
				} else {
					if (!tensor)
						*tensor_out = tensor = ccv_nnc_tensor_new(0, tensor_params, 0);
					ccv_float_to_half_precision((float*)data, (uint16_t*)tensor->data.f16, ccv_min(tensor_count, data_bytes / sizeof(float)));
				}
			} else
				{ assert(0); }
			ccfree(workspace);
		}
#endif
	} else {
		// If it is QX, we need to have a custom decoder to decode properly.
		if (datatype != tensor_params.datatype)
			{ assert(options && options->decode); }
		size_t data_size = ccv_nnc_tensor_data_size(tensor_params);
#ifdef HAVE_CUDA
		if (!options || !options->decode)
		{
			if (CCV_TENSOR_GET_MEMORY(tensor_params.type) == CCV_TENSOR_GPU_MEMORY)
				cumemcpy(tensor->data.u8, tensor->info.type, data, CCV_TENSOR_CPU_MEMORY, ccv_min(data_size, data_bytes));
			else
				memcpy(tensor->data.u8, data, ccv_min(data_size, data_bytes));
		} else {
			if (CCV_TENSOR_GET_MEMORY(tensor_params.type) == CCV_TENSOR_GPU_MEMORY)
			{
				void* const workspace = ccmalloc(data_size);
				size_t decoded_size = data_size;
				if (options->decode(data, data_bytes, datatype, dim, nd, identifier, options->context, tensor_params, tensor_out, workspace, &decoded_size))
					cumemcpy(tensor_out[0]->data.u8, tensor_out[0]->info.type, workspace, CCV_TENSOR_CPU_MEMORY, ccv_min(data_size, decoded_size));
				else {
					if (!tensor)
						*tensor_out = tensor = ccv_nnc_tensor_new(0, tensor_params, 0);
					cumemcpy(tensor->data.u8, tensor->info.type, data, CCV_TENSOR_CPU_MEMORY, ccv_min(data_size, data_bytes));
				}
				ccfree(workspace);
			} else {
				size_t decoded_size = data_size;
				if (!options->decode(data, data_bytes, datatype, dim, nd, identifier, options->context, tensor_params, tensor_out, tensor ? tensor->data.u8 : 0, &decoded_size))
				{
					if (!tensor)
						*tensor_out = tensor = ccv_nnc_tensor_new(0, tensor_params, 0);
					memcpy(tensor->data.u8, data, ccv_min(data_size, data_bytes));
				}
			}
		}
#elif defined(HAVE_MPS)
		if (!options || !options->decode)
		{
			if (CCV_TENSOR_GET_MEMORY(tensor_params.type) == CCV_TENSOR_GPU_MEMORY)
			{
				assert(tensor->dataof == 0);
				mpmemcpy(tensor->data.u8, tensor->dataof, tensor->info.type, data, 0, CCV_TENSOR_CPU_MEMORY, ccv_min(data_size, data_bytes));
			} else
				memcpy(tensor->data.u8, data, ccv_min(data_size, data_bytes));
		} else {
			if (CCV_TENSOR_GET_MEMORY(tensor_params.type) == CCV_TENSOR_GPU_MEMORY)
			{
				if (tensor)
					{ assert(tensor->dataof == 0); }
				void* const workspace = ccmalloc(data_size);
				size_t decoded_size = data_size;
				if (options->decode(data, data_bytes, datatype, dim, nd, identifier, options->context, tensor_params, tensor_out, workspace, &decoded_size)) {
					mpmemcpy(tensor_out[0]->data.u8, tensor_out[0]->dataof, tensor_out[0]->info.type, workspace, 0, CCV_TENSOR_CPU_MEMORY, ccv_min(data_size, decoded_size));
				} else {
					if (!tensor)
						*tensor_out = tensor = ccv_nnc_tensor_new(0, tensor_params, 0);
					mpmemcpy(tensor->data.u8, tensor->dataof, tensor->info.type, data, 0, CCV_TENSOR_CPU_MEMORY, ccv_min(data_size, data_bytes));
				}
				ccfree(workspace);
			} else {
				size_t decoded_size = data_size;
				if (!options->decode(data, data_bytes, datatype, dim, nd, identifier, options->context, tensor_params, tensor_out, tensor ? tensor->data.u8 : 0, &decoded_size))
				{
					if (!tensor)
						*tensor_out = tensor = ccv_nnc_tensor_new(0, tensor_params, 0);
					memcpy(tensor->data.u8, data, ccv_min(data_size, data_bytes));
				}
			}
		}
#else
		if (!options || !options->decode)
			memcpy(tensor->data.u8, data, ccv_min(data_size, data_bytes));
		else {
			size_t decoded_size = data_size;
			if (!options->decode(data, data_bytes, datatype, dim, nd, identifier, options->context, tensor_params, tensor_out, tensor ? tensor->data.u8 : 0, &decoded_size))
			{
				if (!tensor)
					*tensor_out = tensor = ccv_nnc_tensor_new(0, tensor_params, 0);
				memcpy(tensor->data.u8, data, ccv_min(data_size, data_bytes));
			}
		}
#endif
	}
	tensor_out[0]->type &= ~CCV_GARBAGE; // If it is marked as garbage, remove that mark now.
	// The weight takes its stored codebook (the built-in table or inline, checked against caller params above).
	if (codebook_slot >= 0 && CCV_GET_DATA_TYPE(tensor_out[0]->info.datatype) == CCV_QX && (tensor_out[0]->info.datatype & 0xf00) == CCV_NNC_QX_8I_ROWWISE_X)
		tensor_out[0]->info.reserved = ccv_nnc_codebook_with_slot(tensor_out[0]->info.reserved, codebook_slot);
	sqlite3_reset(tensor_select_stmt);
	sqlite3_clear_bindings(tensor_select_stmt);
	sqlite3_finalize(tensor_select_stmt);
	if (split_workspace)
		ccfree(split_workspace);
	return CCV_IO_FINAL;
}
