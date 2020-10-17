#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_cnnp_dataframe.h"
#include "3rdparty/sfmt/SFMT.h"

#include <sys/mman.h>

// MARK - Create Dataframe from Comma-separated-values Files

typedef struct {
	int even;
	int odd;
	int even_starter;
	int odd_starter;
	int quotes;
} csv_crlf_t;

#define ANY_ZEROS(v) ((v - (uint64_t)0x0101010101010101) & ((~v) & (uint64_t)0x8080808080808080))

static inline void _fix_double_quote(const char* src, int count, char* dest)
{
	if (!src || count <= 0)
		return;
	char prev_char = src[0];
	dest[0] = src[0];
	++dest;
	int pos = 1;
	while (pos < count)
	{
		// double-quote, skip.
		if (prev_char == '"' && src[pos] == '"')
			++pos;
		dest[0] = src[pos];
		prev_char = src[pos];
		++dest;
		++pos;
	}
	dest[0] = '\0';
}

typedef struct {
	const char* data;
	void* mmap;
	size_t file_size;
	int column_size;
	int include_header;
	char delim;
	char quote;
} ccv_cnnp_csv_t;

typedef struct {
	// This need to be compressed to 64-bit. If we expand this to 128-bit. It will double the memory-bandwidth, and
	// slows the whole process down.
	uint64_t str:48;
	uint16_t count:15;
	uint8_t no_double_quote:1;
} ccv_cnnp_csv_str_view_t;

void _ccv_cnnp_csv_enum(const int column_idx, const int* const row_idxs, const int row_size, void** const data, void* const context, ccv_nnc_stream_context_t* const stream_context)
{
	ccv_cnnp_csv_t* const csv = (ccv_cnnp_csv_t*)context;
	const int column_size = csv->column_size;
	const int include_header = csv->include_header;
	const char quote = csv->quote;
	const char delim = csv->delim;
	const ccv_cnnp_csv_str_view_t* const sp = (const ccv_cnnp_csv_str_view_t*)(csv + 1) + column_idx + include_header * column_size;
	int i;
	for (i = 0; i < row_size; i++)
	{
		const int row_idx = row_idxs[i];
		const ccv_cnnp_csv_str_view_t* const csp = sp + row_idx * column_size;
		// This is the same as (csp->str == 0 && csp->no_double_quote = 0 && csp->count == 0)
		// If a string has 0 length, it cannot contain double quote, therefore, this condition
		// implies the pointer is null.
		if (((uint64_t*)csp)[0] == 0)
		{
			if (data[i])
			{
				int* hdr = (int*)data[i] - 1;
				ccfree(hdr);
			}
			data[i] = 0;
			continue;
		}
		const char* str = csv->data + csp->str;
		int count = 0;
		if (csp->count == 0x7fff) // We don't know the count yet. In this case, go over to find it.
		{
			const char* const p_end = csv->data + csv->file_size;
			int quotes = (str > csv->data && str[-1] == quote) ? 1 : 0;
			const char* p = str;
			const char* quote_end = 0;
			for (; p < p_end; p++)
			{
				if (p[0] == quote)
				{
					++quotes;
					quote_end = p;
				} else if (!(quotes & 1)) {
					if (p[0] == delim || p[0] == '\r' || p[0] == '\n')
					{
						if (quote_end >= str)
							count = quote_end - str;
						else
							count = p - str;
						break;
					}
				}
			}
		} else
			count = csp->count;
		if (!data[i])
		{
			int* const hdr = (int*)ccmalloc(sizeof(int) + count + 1);
			hdr[0] = count + 1;
			data[i] = (char*)(hdr + 1);
		} else {
			int* hdr = (int*)data[i] - 1;
			if (hdr[0] < count + 1)
			{
				hdr = (int*)ccrealloc(hdr, sizeof(int) + count + 1);
				hdr[0] = count + 1;
				data[i] = (char*)(hdr + 1);
			}
		}
		if (csp->no_double_quote)
		{
			memcpy(data[i], str, count);
			((char*)data[i])[count] = '\0';
		} else
			_fix_double_quote(str, count, (char*)data[i]);
	}
}

void _ccv_cnnp_csv_data_deinit(void* const data, void* const context)
{
	if (data)
	{
		int* hdr = (int*)data - 1;
		ccfree(hdr);
	}
}

void _ccv_cnnp_csv_deinit(void* const context)
{
	ccv_cnnp_csv_t* const csv = (ccv_cnnp_csv_t*)context;
	if (csv->mmap)
		munmap(csv->mmap, csv->file_size);
	ccfree(csv);
}

ccv_cnnp_dataframe_t* ccv_cnnp_dataframe_from_csv_new(void* const input, const int type, const size_t len, const char _delim, const char _quote, const int include_header, int* const column_size)
{
	assert(input);
	assert(column_size);
	size_t file_size;
	char* data;
	assert(type == CCV_CNNP_DATAFRAME_CSV_FILE || type == CCV_CNNP_DATAFRAME_CSV_MEMORY);
	if (type == CCV_CNNP_DATAFRAME_CSV_FILE)
	{
		FILE* file = (FILE*)input;
		const int fd = fileno(file);
		if (fd == -1)
			return 0;
		fseek(file, 0, SEEK_END);
		file_size = ftell(file);
		fseek(file, 0, SEEK_SET);
		if (file_size < 2)
			return 0;
		data = mmap((caddr_t)0, file_size, PROT_READ, MAP_SHARED, fd, 0);
		if (!data)
			return 0;
	} else {
		file_size = len;
		assert(len > 0);
		if (len < 2)
			return 0;
		data = input;
	}
	// We cannot handle file size larger than 2^48, which is around 281TB.
	assert(file_size <= 0xffffffffffffllu);
	const char delim = _delim ? _delim : ',';
	const char quote = _quote ? _quote : '"';
	const size_t chunk_size = 1024 * 1024;
	const int aligned_chunks = file_size / chunk_size;
	const int total_chunks = (file_size + chunk_size - 1) / chunk_size;
	// Get number of rows.
	csv_crlf_t* const crlf = cccalloc(total_chunks, sizeof(csv_crlf_t));
#define CSV_QUOTE_BR(c, n) \
	do { \
		if (c##n == quote) \
			++quotes; \
		else if (c##n == '\n') { \
			++count[quotes & 1]; \
			if (starter[quotes & 1] == -1) \
				starter[quotes & 1] = (int)(p - p_start) + n; \
		} \
	} while (0)
	parallel_for(i, aligned_chunks) {
		const uint64_t* pd = (const uint64_t*)(data + i * chunk_size);
		const char* const p_start = (const char*)pd;
		const uint64_t* const pd_end = pd + chunk_size / sizeof(uint64_t);
		int quotes = 0;
		int starter[2] = {-1, -1};
		int count[2] = {0, 0};
		for (; pd < pd_end; pd++)
		{
			// Load 8-bytes at batch.
			const char* const p = (const char*)pd;
			char c0, c1, c2, c3, c4, c5, c6, c7;
			c0 = p[0], c1 = p[1], c2 = p[2], c3 = p[3], c4 = p[4], c5 = p[5], c6 = p[6], c7 = p[7];
			CSV_QUOTE_BR(c, 0);
			CSV_QUOTE_BR(c, 1);
			CSV_QUOTE_BR(c, 2);
			CSV_QUOTE_BR(c, 3);
			CSV_QUOTE_BR(c, 4);
			CSV_QUOTE_BR(c, 5);
			CSV_QUOTE_BR(c, 6);
			CSV_QUOTE_BR(c, 7);
		}
		crlf[i].even = count[0];
		crlf[i].odd = count[1];
		crlf[i].even_starter = starter[0];
		crlf[i].odd_starter = starter[1];
		crlf[i].quotes = quotes;
	} parallel_endfor
	if (total_chunks > aligned_chunks)
	{
		const int residual_size = file_size - chunk_size * aligned_chunks;
		const uint64_t* pd = (const uint64_t*)(data + chunk_size * aligned_chunks);
		const char* const p_start = (const char*)pd;
		const uint64_t* const pd_end = pd + residual_size / sizeof(uint64_t);
		int quotes = 0;
		int starter[2] = {-1, -1};
		int count[2] = {0, 0};
		for (; pd < pd_end; pd++)
		{
			const char* const p = (const char*)pd;
			// Load 8-bytes at batch.
			char c0, c1, c2, c3, c4, c5, c6, c7;
			c0 = p[0], c1 = p[1], c2 = p[2], c3 = p[3], c4 = p[4], c5 = p[5], c6 = p[6], c7 = p[7];
			CSV_QUOTE_BR(c, 0);
			CSV_QUOTE_BR(c, 1);
			CSV_QUOTE_BR(c, 2);
			CSV_QUOTE_BR(c, 3);
			CSV_QUOTE_BR(c, 4);
			CSV_QUOTE_BR(c, 5);
			CSV_QUOTE_BR(c, 6);
			CSV_QUOTE_BR(c, 7);
		}
		const char* const p_end = data + file_size;
		const char* p = (const char*)pd_end;
		for (; p < p_end; p++)
		{
			const char c0 = p[0];
			CSV_QUOTE_BR(c, 0);
		}
		crlf[aligned_chunks].even = count[0];
		crlf[aligned_chunks].odd = count[1];
		crlf[aligned_chunks].even_starter = starter[0] < 0 ? residual_size : starter[0];
		crlf[aligned_chunks].odd_starter = starter[1] < 0 ? residual_size : starter[1];
		crlf[aligned_chunks].quotes = quotes;
	}
#undef CSV_QUOTE_BR
	int row_count = crlf[0].even;
	int quotes = crlf[0].quotes;
	crlf[0].odd_starter = 0;
	int i;
	// Go through all chunks serially to find exactly how many line ends in each chunk, moving that information to even*.
	// The odd_starter will record which row it currently at for this chunk.
	for (i = 1; i < total_chunks; i++)
	{
		if (quotes & 1)
		{
			// Even is the correct one, we will use that throughout.
			crlf[i].even = crlf[i].odd;
			crlf[i].even_starter = crlf[i].odd_starter;
		}
		crlf[i].odd_starter = row_count + 1;
		row_count += crlf[i].even;
		quotes += crlf[i].quotes;
	}
	// Didn't end with newline, one more row.
	if (!(data[file_size - 1] == '\n' || (data[file_size - 2] == '\n' && data[file_size - 1] == '\r')))
		++row_count;
	// Get number of columns.
	int column_count = 0;
	const uint64_t* pd = (const uint64_t*)data;
	int first_line_len = file_size;
	for (i = 0; i < total_chunks; i++)
		if (crlf[i].even_starter >= 0)
		{
			first_line_len = i * chunk_size + crlf[i].even_starter;
			break;
		}
	const uint64_t* const pd_end = pd + first_line_len / sizeof(uint64_t);
#define CSV_QUOTE_BR(cn) \
	do { \
		if (cn == quote) \
			++quotes; \
		else if (!(quotes & 1)) { \
			if (cn == delim) \
				++column_count; \
		} \
	} while (0)
	quotes = 0;
	for (; pd < pd_end; pd++)
	{
		const char* const p = (const char*)pd;
		char c0, c1, c2, c3, c4, c5, c6, c7;
		c0 = p[0], c1 = p[1], c2 = p[2], c3 = p[3], c4 = p[4], c5 = p[5], c6 = p[6], c7 = p[7];
		CSV_QUOTE_BR(c0);
		CSV_QUOTE_BR(c1);
		CSV_QUOTE_BR(c2);
		CSV_QUOTE_BR(c3);
		CSV_QUOTE_BR(c4);
		CSV_QUOTE_BR(c5);
		CSV_QUOTE_BR(c6);
		CSV_QUOTE_BR(c7);
	}
	// If haven't reached the flag yet (i.e., haven't reached a new line).
	const char* p = (const char*)pd;
	const char* const p_end = data + first_line_len;
	for (; p < p_end; p++)
	{
		const char c0 = p[0];
		CSV_QUOTE_BR(c0);
	}
#undef CSV_QUOTE_BR
	++column_count; // column count is 1 more than delimiter.
	if (row_count == 0) // This is possible because you have an open quote, and then \n is inside the open quote, which won't be recognized.
	{
		ccfree(crlf);
		if (type == CCV_CNNP_DATAFRAME_CSV_FILE)
			munmap(data, file_size);
		return 0;
	}
	// We only mark the beginning and the end of a cell. Removing double-quote etc will be left when iterating.
	ccv_cnnp_csv_t* const csv = (ccv_cnnp_csv_t*)ccmalloc(sizeof(ccv_cnnp_csv_t) + sizeof(ccv_cnnp_csv_str_view_t) * row_count * column_count);
	csv->column_size = column_count;
	csv->include_header = !!include_header;
	ccv_cnnp_csv_str_view_t* const sp = (ccv_cnnp_csv_str_view_t*)(csv + 1);
	memset(sp, 0, sizeof(ccv_cnnp_csv_str_view_t) * row_count * column_count);
	const uint64_t delim_mask = (uint64_t)0x0101010101010101 * (uint64_t)delim;
	const uint64_t quote_mask = (uint64_t)0x0101010101010101 * (uint64_t)quote;
	const uint64_t lf_mask = (uint64_t)0x0101010101010101 * (uint64_t)'\n';
	const uint64_t cr_mask = (uint64_t)0x0101010101010101 * (uint64_t)'\r';
#define CSV_QUOTE_BR(c, n) \
	do { \
		if (c##n == quote) \
		{ \
			/* If the preceding one is not a quote. Set it to be null-terminator temporarily. */ \
			++quotes; \
			quote_end = p + n; \
			if (!preceding_quote) \
				preceding_quote = 1; \
			else \
				double_quote = 1; \
		} else { \
			preceding_quote = 0; \
			if (!(quotes & 1)) \
			{ \
				if (c##n == delim) \
				{ \
					if (chunk_row_count < row_count) \
					{ \
						if (chunk_column_count < column_count) \
						{ \
							int count; \
							if (quote_end > 0 && quote_end - data >= csp[chunk_column_count].str) \
								count = (int)((quote_end - data) - csp[chunk_column_count].str); \
							else \
								count = (int)((p + n - data) - csp[chunk_column_count].str); \
							csp[chunk_column_count].count = ccv_min(count, 0x7fff); \
							csp[chunk_column_count].no_double_quote = !double_quote; \
						} \
						++chunk_column_count; \
						if (chunk_column_count < column_count) \
							/* Skip quote if presented. */ \
							csp[chunk_column_count].str = (p + (n + 1) < p_end && p[n + 1] == quote ? p + (n + 2) : p + (n + 1)) - data; \
					} \
					double_quote = 0; \
				} else if (c##n == '\n') { \
					if (chunk_row_count < row_count && chunk_column_count < column_count) \
					{ \
						int count; \
						if (quote_end > 0 && quote_end - data >= csp[chunk_column_count].str) \
							count = (int)((quote_end - data) - csp[chunk_column_count].str); \
						else if (p + n > data && p[n - 1] == '\r') \
							count = (int)((p + n - 1 - data) - csp[chunk_column_count].str); \
						else \
							count = (int)((p + n - data) - csp[chunk_column_count].str); \
						csp[chunk_column_count].count = ccv_min(count, 0x7fff); \
						csp[chunk_column_count].no_double_quote = !double_quote; \
					} \
					++chunk_row_count; \
					csp += column_count; \
					chunk_column_count = 0; \
					if (chunk_row_count < row_count) \
					{ \
						if (p + (n + 1) < p_end && p[n + 1] == '\r') \
							csp[0].str = (p + (n + 2) < p_end && p[n + 2] == quote ? p + (n + 3) : p + (n + 2)) - data; \
						else \
							csp[0].str = (p + (n + 1) < p_end && p[n + 1] == quote ? p + (n + 2) : p + (n + 1)) - data; \
					} \
					double_quote = 0; \
				} \
			} \
		} \
	} while (0)
	parallel_for(i, total_chunks) {
		// Skip if existing one don't have a line starter.
		if (i > 0 && crlf[i].even_starter < 0)
			continue;
		const char* p = (i == 0) ? data : data + i * chunk_size + crlf[i].even_starter + 1;
		const char* p_end = data + file_size;
		int j;
		for (j = i + 1; j < total_chunks; j++)
			if (crlf[j].even_starter >= 0)
			{
				p_end = data + j * chunk_size + crlf[j].even_starter;
				break;
			}
		if (p_end <= p)
			continue;
		int chunk_row_count = crlf[i].odd_starter;
		ccv_cnnp_csv_str_view_t* csp = sp + (uintptr_t)column_count * chunk_row_count;
		if (chunk_row_count < row_count)
		{
			if (p[0] == '\r')
				csp[0].str = (p + 1 < p_end && p[1] == quote ? p + 2 : p + 1) - data;
			else
				csp[0].str = (p[0] == quote ? p + 1 : p) - data;
		}
		int chunk_column_count = 0;
		int quotes = 0;
		int preceding_quote = 0;
		int double_quote = 0;
		const char* quote_end = 0;
		const int padding = ccv_min(0x7 - (((uintptr_t)p - 1) & 0x7), (int)(p_end - p));
		for (j = 0; j < padding; j++, p++)
		{
			char c0 = p[0];
			CSV_QUOTE_BR(c, 0);
		}
		const size_t cur_chunk_size = (size_t)(p_end - p);
		const uint64_t* pd = (const uint64_t*)p;
		const uint64_t* pd_end = pd + cur_chunk_size / sizeof(uint64_t);
		for (; pd < pd_end; pd++)
		{
			const uint64_t v = *pd;
			const uint64_t delim_v = v ^ delim_mask;
			const uint64_t quote_v = v ^ quote_mask;
			const uint64_t lf_v = v ^ lf_mask;
			const uint64_t cr_v = v ^ cr_mask;
			// If it doesn't contain any zeros, skip the logic.
			if (!ANY_ZEROS(delim_v) && !ANY_ZEROS(quote_v) && !ANY_ZEROS(lf_v) && !ANY_ZEROS(cr_v))
				continue;
			// Need to check and assign the length and starting point.
			p = (const char*)pd;
			// Load 8-bytes at batch.
			char c0, c1, c2, c3, c4, c5, c6, c7;
			c0 = p[0], c1 = p[1], c2 = p[2], c3 = p[3], c4 = p[4], c5 = p[5], c6 = p[6], c7 = p[7];
			CSV_QUOTE_BR(c, 0);
			CSV_QUOTE_BR(c, 1);
			CSV_QUOTE_BR(c, 2);
			CSV_QUOTE_BR(c, 3);
			CSV_QUOTE_BR(c, 4);
			CSV_QUOTE_BR(c, 5);
			CSV_QUOTE_BR(c, 6);
			CSV_QUOTE_BR(c, 7);
		}
		p = (const char*)pd;
		for (; p < p_end; p++)
		{
			char c0 = p[0];
			CSV_QUOTE_BR(c, 0);
		}
		if (chunk_row_count < row_count && chunk_column_count < column_count)
		{
			int count;
			if (quote_end > 0 && quote_end - data >= csp[chunk_column_count].str)
				count = (int)(quote_end - data - csp[chunk_column_count].str);
			else
				count = (int)(p - data - csp[chunk_column_count].str);
			csp[chunk_column_count].count = ccv_min(count, 0x7fff);
			csp[chunk_column_count].no_double_quote = !double_quote;
		}
	} parallel_endfor
#undef CSV_QUOTE_BR
	ccfree(crlf);
	csv->data = data;
	assert(file_size > 0);
	csv->file_size = file_size;
	csv->delim = delim;
	csv->quote = quote;
	if (type == CCV_CNNP_DATAFRAME_CSV_FILE)
		csv->mmap = data;
	*column_size = column_count;
	assert(column_count > 0);
	ccv_cnnp_column_data_t* const column_data = (ccv_cnnp_column_data_t*)cccalloc(column_count, sizeof(ccv_cnnp_column_data_t));
	for (i = 0; i < column_count; i++)
	{
		column_data[i].data_enum = _ccv_cnnp_csv_enum;
		column_data[i].context = csv;
		column_data[i].data_deinit = _ccv_cnnp_csv_data_deinit;
	}
	if (include_header)
		for (i = 0; i < column_count; i++)
			if (((uint64_t*)sp)[i] != 0)
			{
				column_data[i].name = (char*)ccmalloc(sp[i].count + 1);
				const char* str = data + sp[i].str;
				if (sp[i].no_double_quote)
				{
					memcpy(column_data[i].name, str, sp[i].count);
					column_data[i].name[sp[i].count] = '\0';
				} else
					_fix_double_quote(str, sp[i].count, column_data[i].name);
			}
	column_data[0].context_deinit = _ccv_cnnp_csv_deinit;
	ccv_cnnp_dataframe_t* dataframe = ccv_cnnp_dataframe_new(column_data, column_count, row_count - !!include_header);
	if (include_header)
		for (i = 0; i < column_count; i++)
			ccfree(column_data[i].name);
	ccfree(column_data);
	return dataframe;
}
