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

static inline void _fix_double_quote(char* src)
{
	if (!src)
		return;
	if (src[0] == '\0')
		return;
	char prev_char = src[0];
	char* dest = src;
	++src;
	while (src[1] != '\0')
	{
		// double-quote, skip.
		if (prev_char == '"' && src[0] == '"')
			++src;
		dest[0] = src[0];
		prev_char = src[0];
		++src;
	}
}

typedef struct {
	int column_size;
	int reserved;
} ccv_cnnp_csv_t;

void _ccv_cnnp_csv_enum(const int column_idx, const int* const row_idxs, const int row_size, void** const data, void* const context, ccv_nnc_stream_context_t* const stream_context)
{
	ccv_cnnp_csv_t* const csv = (ccv_cnnp_csv_t*)context;
	const int column_size = csv->column_size;
	const char** const sp = (const char**)(csv + 1) + column_idx;
	int i;
	for (i = 0; i < row_size; i++)
	{
		const int row_idx = row_idxs[i];
		data[i] = (void*)sp[row_idx * column_size];
	}
}

ccv_cnnp_dataframe_t* ccv_cnnp_dataframe_from_csv_new(FILE* const file, const char delim, const char quote, const int include_header, int* const column_size)
{
	assert(column_size);
	const int fd = fileno(file);
	fseek(file, 0, SEEK_END);
	const size_t file_size = ftell(file);
	fseek(file, 0, SEEK_SET);
	if (file_size < 2)
		return 0;
	// Note that we use p + 2 to check whether there is a quote after \r, hence, map 2 more bytes.
	char* const data = mmap((caddr_t)0, file_size + 2, PROT_READ, MAP_PRIVATE, fd, 0);
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
	crlf[0].even_starter = 0;
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
		assert(crlf[i].even_starter >= 0);
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
	const int first_line_len = crlf[0].even_starter;
	const uint64_t* const pd_end = pd + first_line_len / sizeof(uint64_t);
#define CSV_QUOTE_BR(cn) \
	if (cn == quote) \
		++quotes; \
	else if (!(quote & 1)) { \
		if (cn == delim) \
			++column_count; \
	}
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
	// Allocating memory that holds both start pointers to the string, and the actual string. Also responsible to copy over the string.
	// Note that we did file_size + 3 in case we don't have \n at the end of the file to host the null-terminator. Also, we may assign
	// q + (n + 3) as the end, hence, 3 more null bytes at the end.
	ccv_cnnp_csv_t* const csv = (ccv_cnnp_csv_t*)ccmalloc(sizeof(ccv_cnnp_csv_t) + sizeof(char*) * row_count * column_count + file_size + 3);
	csv->column_size = column_count;
	char** const sp = (char**)(csv + 1); 
	memset(sp, 0, sizeof(char*) * row_count * column_count);
	char* const dd = (char*)(sp + row_count * column_count);
	dd[file_size] = dd[file_size + 1] = dd[file_size + 2] = '\0';
	const uint64_t delim_mask = (uint64_t)0x0101010101010101 * (uint64_t)delim;
	const uint64_t quote_mask = (uint64_t)0x0101010101010101 * (uint64_t)quote;
	const uint64_t lf_mask = (uint64_t)0x0101010101010101 * (uint64_t)'\n';
	const uint64_t cr_mask = (uint64_t)0x0101010101010101 * (uint64_t)'\r';
#define CSV_QUOTE_BR(c, n) \
	do { \
		if (c##n == quote) \
		{ \
			/* If the preceding one is not a quote. Set it to be null-terminator temporarily. */ \
			if (!preceding_quote) \
			{ \
				c##n = 0; \
				preceding_quote = 1; \
			} else { /* This is double quote, mark it as a normal quote. */ \
				double_quote = 1; \
				q[n - 1] = p[n - 1]; \
			} \
			++quotes; \
		} else { \
			preceding_quote = 0; \
			if (!(quotes & 1)) \
			{ \
				if (c##n == delim) \
				{ \
					if (chunk_column_count < column_count && double_quote) \
						_fix_double_quote(sp[column_count * chunk_row_count + chunk_column_count]); \
					++chunk_column_count; \
					double_quote = 0; \
					if (chunk_column_count < column_count) \
						/* Skip quote if presented. */ \
						sp[column_count * chunk_row_count + chunk_column_count] = p[n + 1] == quote ? q + (n + 2) : q + (n + 1); \
				} else if (c##n == '\n') { \
					++chunk_row_count; \
					chunk_column_count = 0; \
					if (chunk_row_count < row_count) \
					{ \
						if (double_quote) \
							_fix_double_quote(sp[column_count * (chunk_row_count - 1) + column_count - 1]); \
						if (p[n + 1] == '\r') \
							sp[column_count * chunk_row_count] = p[n + 2] == quote ? q + (n + 3) : q + (n + 2); \
						else \
							sp[column_count * chunk_row_count] = p[n + 1] == quote ? q + (n + 2) : q + (n + 1); \
					} \
					double_quote = 0; \
				} \
				/* If c0 is delim out of quote, we set it to be null-terminator. */ \
				if (c##n == delim || c##n == '\r' || c##n == '\n') \
					c##n = 0; \
			} \
		} \
	} while (0)
	parallel_for(i, total_chunks) {
		// Skip if existing one don't have a line starter.
		if (crlf[i].even_starter < 0)
			continue;
		const char* p = (i == 0) ? data : data + i * chunk_size + crlf[i].even_starter + 1;
		const char* p_end = data + file_size;
		int j;
		for (j = i + 1; j < total_chunks; j++)
			if (crlf[i].even_starter >= 0)
			{
				p_end = data + j * chunk_size + crlf[j].even_starter;
				break;
			}
		if (p_end == p)
			continue;
		char* q = (i == 0) ? dd : dd + i * chunk_size + crlf[i].even_starter + 1;
		int chunk_row_count = crlf[i].odd_starter;
		if (p[0] == '\r')
			sp[column_count * chunk_row_count] = p[1] == quote ? q + 2 : q + 1;
		else
			sp[column_count * chunk_row_count] = p[0] == quote ? q + 1 : q;
		int chunk_column_count = 0;
		int quotes = 0;
		int double_quote = 0;
		int preceding_quote = 0;
		const int padding = ccv_min(0x7 - (((uintptr_t)p - 1) & 0x7), (int)(p_end - p));
		for (j = 0; j < padding; j++, q++, p++)
		{
			char c0 = p[0];
			CSV_QUOTE_BR(c, 0);
			q[0] = c0;
		}
		const size_t cur_chunk_size = (size_t)(p_end - p);
		const uint64_t* pd = (const uint64_t*)p;
		const uint64_t* pd_end = pd + cur_chunk_size / sizeof(uint64_t);
		uint64_t* qd = (uint64_t*)q;
		for (; pd < pd_end; pd++, qd++)
		{
			const uint64_t v = *pd;
			const uint64_t delim_v = v ^ delim_mask;
			const uint64_t quote_v = v ^ quote_mask;
			const uint64_t lf_v = v ^ lf_mask;
			const uint64_t cr_v = v ^ cr_mask;
			// If it doesn't contain any zeros, skip the logic.
			if (!ANY_ZEROS(delim_v) && !ANY_ZEROS(quote_v) && !ANY_ZEROS(lf_v) && !ANY_ZEROS(cr_v))
			{
				*qd = *pd;
				continue;
			}
			// Need to check and clean up.
			p = (const char*)pd;
			q = (char*)qd;
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
			q[0] = c0, q[1] = c1, q[2] = c2, q[3] = c3, q[4] = c4, q[5] = c5, q[6] = c6, q[7] = c7;
		}
		p = (const char*)pd;
		q = (char*)qd;
		for (; p < p_end; p++, q++)
		{
			char c0 = p[0];
			CSV_QUOTE_BR(c, 0);
			q[0] = c0;
		}
	} parallel_endfor
#undef CSV_QUOTE_BR
	ccfree(crlf);
	munmap(data, file_size + 2);
	*column_size = column_count;
	assert(column_count > 0);
	ccv_cnnp_column_data_t* const column_data = (ccv_cnnp_column_data_t*)cccalloc(column_count, sizeof(ccv_cnnp_column_data_t));
	for (i = 0; i < column_count; i++)
	{
		column_data[i].data_enum = _ccv_cnnp_csv_enum;
		column_data[i].context = csv;
	}
	column_data[0].context_deinit = (ccv_cnnp_column_data_context_deinit_f)ccfree;
	ccv_cnnp_dataframe_t* dataframe = ccv_cnnp_dataframe_new(column_data, column_count, row_count);
	ccfree(column_data);
	return dataframe;
}
