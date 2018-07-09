#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_nnc_symbolic_graph.h"
#include "3rdparty/sqlite3/sqlite3.h"

#ifdef NDEBUG
#define assert_sqlite(stmt) (void)(stmt)
#else
#define assert_sqlite assert
#endif

static int _ccv_nnc_symbolic_graph_index_in_repo(const ccv_nnc_symbolic_graph_t* const graph, const ccv_array_t* const repo)
{
	if (!graph)
		return -1;
	int i;
	for (i = 0; i < repo->rnum; i++)
		if (*(ccv_nnc_symbolic_graph_t**)ccv_array_get(repo, i) == graph)
			return i;
	return -1;
}

static void _ccv_nnc_symbolic_graph_write(const ccv_nnc_symbolic_graph_t* const graph, const ccv_array_t* const repo, const int graph_idx, sqlite3_stmt* const tensor_symbol_insert_stmt, sqlite3_stmt* const exec_symbol_insert_stmt, sqlite3_stmt* const graph_insert_stmt, ccv_array_t* const ws)
{
	int i;
	for (i = 0; i < graph->tensor_symbol_info->rnum; i++)
	{
		const ccv_nnc_tensor_symbol_info_t* const symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, i);
		sqlite3_bind_int(tensor_symbol_insert_stmt, 1, i);
		sqlite3_bind_int(tensor_symbol_insert_stmt, 2, graph_idx);
		sqlite3_bind_int(tensor_symbol_insert_stmt, 3, symbol_info->assign_ref);
		sqlite3_bind_int(tensor_symbol_insert_stmt, 4, symbol_info->r_assign_ref);
		sqlite3_bind_int(tensor_symbol_insert_stmt, 5, symbol_info->bypass_ref);
		sqlite3_bind_int(tensor_symbol_insert_stmt, 6, symbol_info->r_bypass_ref);
		sqlite3_bind_int(tensor_symbol_insert_stmt, 7, symbol_info->p_ref);
		sqlite3_bind_int(tensor_symbol_insert_stmt, 8, symbol_info->alias_ref);
		sqlite3_bind_int(tensor_symbol_insert_stmt, 9, symbol_info->peer_ref);
		sqlite3_bind_int(tensor_symbol_insert_stmt, 10, symbol_info->flags);
		sqlite3_bind_blob(tensor_symbol_insert_stmt, 11, symbol_info->ofs, sizeof(symbol_info->ofs), 0);
		sqlite3_bind_blob(tensor_symbol_insert_stmt, 12, symbol_info->inc, sizeof(symbol_info->inc), 0);
		if (symbol_info->s_ref)
			sqlite3_bind_blob(tensor_symbol_insert_stmt, 13, ccv_array_get(symbol_info->s_ref, 0), sizeof(int) * symbol_info->s_ref->rnum, 0);
		else
			sqlite3_bind_null(tensor_symbol_insert_stmt, 13);
		if (symbol_info->name)
			sqlite3_bind_text(tensor_symbol_insert_stmt, 14, symbol_info->name, -1, 0);
		else
			sqlite3_bind_null(tensor_symbol_insert_stmt, 14);
		sqlite3_bind_int(tensor_symbol_insert_stmt, 15, symbol_info->info.type);
		sqlite3_bind_int(tensor_symbol_insert_stmt, 16, symbol_info->info.format);
		sqlite3_bind_int(tensor_symbol_insert_stmt, 17, symbol_info->info.datatype);
		sqlite3_bind_blob(tensor_symbol_insert_stmt, 18, symbol_info->info.dim, sizeof(symbol_info->info.dim), 0);
		assert_sqlite(SQLITE_DONE == sqlite3_step(tensor_symbol_insert_stmt));
		sqlite3_reset(tensor_symbol_insert_stmt);
		sqlite3_clear_bindings(tensor_symbol_insert_stmt);
	}
	for (i = 0; i < graph->exec_symbol_info->rnum; i++)
	{
		const ccv_nnc_graph_exec_symbol_info_t* const symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, i);
		sqlite3_bind_int(exec_symbol_insert_stmt, 1, i);
		sqlite3_bind_int(exec_symbol_insert_stmt, 2, graph_idx);
		sqlite3_bind_int(exec_symbol_insert_stmt, 3, symbol_info->input_size);
		sqlite3_bind_int(exec_symbol_insert_stmt, 4, symbol_info->output_size);
		sqlite3_bind_int(exec_symbol_insert_stmt, 5, symbol_info->graph_ref_size);
		sqlite3_bind_int(exec_symbol_insert_stmt, 6, symbol_info->flags);
		sqlite3_bind_int(exec_symbol_insert_stmt, 7, symbol_info->peer_ref);
		if (symbol_info->input_size)
			sqlite3_bind_blob(exec_symbol_insert_stmt, 8, symbol_info->inputs, sizeof(int) * symbol_info->input_size, 0);
		if (symbol_info->output_size)
			sqlite3_bind_blob(exec_symbol_insert_stmt, 9, symbol_info->outputs, sizeof(int) * symbol_info->output_size, 0);
		if (symbol_info->outgoings && symbol_info->outgoings->rnum)
			sqlite3_bind_blob(exec_symbol_insert_stmt, 10, ccv_array_get(symbol_info->outgoings, 0), sizeof(int) * symbol_info->outgoings->rnum, 0);
		if (symbol_info->name)
			sqlite3_bind_text(exec_symbol_insert_stmt, 11, symbol_info->name, -1, 0);
		sqlite3_bind_int(exec_symbol_insert_stmt, 12, symbol_info->cmd.cmd);
		sqlite3_bind_int(exec_symbol_insert_stmt, 13, symbol_info->cmd.backend);
		sqlite3_bind_int(exec_symbol_insert_stmt, 14, symbol_info->cmd.algorithm);
		sqlite3_bind_blob(exec_symbol_insert_stmt, 15, &symbol_info->cmd.info, sizeof(symbol_info->cmd.info), 0);
		sqlite3_bind_blob(exec_symbol_insert_stmt, 16, &symbol_info->hint, sizeof(symbol_info->hint), 0);
		if (symbol_info->graph_ref_size)
			sqlite3_bind_blob(exec_symbol_insert_stmt, 17, CCV_NNC_GRAPH_REF(symbol_info), sizeof(int) * symbol_info->graph_ref_size, 0);
		if (symbol_info->flags & CCV_NNC_GRAPH_EXEC_CASE_OF)
		{
			sqlite3_bind_int(exec_symbol_insert_stmt, 18, 0);
			sqlite3_bind_int(exec_symbol_insert_stmt, 19, symbol_info->case_of.flags);
			sqlite3_bind_int(exec_symbol_insert_stmt, 20, symbol_info->case_of.argument.offset);
			sqlite3_bind_int(exec_symbol_insert_stmt, 21, symbol_info->case_of.argument.size);
		}
		if (symbol_info->flags & CCV_NNC_GRAPH_EXEC_P_WHILE)
		{
			sqlite3_bind_int(exec_symbol_insert_stmt, 22, 0);
			sqlite3_bind_int(exec_symbol_insert_stmt, 23, symbol_info->p_while.input_size);
			if (symbol_info->p_while.input_size)
				sqlite3_bind_blob(exec_symbol_insert_stmt, 24, symbol_info->p_while.inputs, sizeof(int) * symbol_info->p_while.input_size, 0);
		}
		assert_sqlite(SQLITE_DONE == sqlite3_step(exec_symbol_insert_stmt));
		sqlite3_reset(exec_symbol_insert_stmt);
		sqlite3_clear_bindings(exec_symbol_insert_stmt);
	}
	ccv_array_clear(ws);
	sqlite3_bind_int(graph_insert_stmt, 1, graph_idx);
	sqlite3_bind_int(graph_insert_stmt, 2, graph->tensor_symbol_info->rnum);
	sqlite3_bind_int(graph_insert_stmt, 3, graph->exec_symbol_info->rnum);
	if (graph->sources && graph->sources->rnum)
		for (i = 0; i < graph->sources->rnum; i++)
			ccv_array_push(ws, &((ccv_nnc_graph_exec_symbol_t*)ccv_array_get(graph->sources, i))->d);
	if (graph->destinations && graph->destinations->rnum)
		for (i = 0; i < graph->destinations->rnum; i++)
			ccv_array_push(ws, &((ccv_nnc_graph_exec_symbol_t*)ccv_array_get(graph->destinations, i))->d);
	if (graph->sub_graphs && graph->sub_graphs->rnum)
		for (i = 0; i < graph->sub_graphs->rnum; i++)
		{
			const int sub_graph_idx = _ccv_nnc_symbolic_graph_index_in_repo(*(ccv_nnc_symbolic_graph_t**)ccv_array_get(graph->sub_graphs, i), repo);
			ccv_array_push(ws, &sub_graph_idx);
		}
	if (graph->breakpoint_size && graph->breakpoints)
		for (i = 0; i < graph->breakpoint_size; i++)
			ccv_array_push(ws, &graph->breakpoints[i].d);
	const int* pos = (int*)ccv_array_get(ws, 0);
	if (graph->sources && graph->sources->rnum)
	{
		sqlite3_bind_blob(graph_insert_stmt, 4, pos, sizeof(int) * graph->sources->rnum, 0);
		pos += graph->sources->rnum;
	}
	if (graph->destinations && graph->destinations->rnum)
	{
		sqlite3_bind_blob(graph_insert_stmt, 5, pos, sizeof(int) * graph->destinations->rnum, 0);
		pos += graph->destinations->rnum;
	}
	if (graph->sub_graphs && graph->sub_graphs->rnum)
	{
		sqlite3_bind_blob(graph_insert_stmt, 6, pos, sizeof(int) * graph->sub_graphs->rnum, 0);
		pos += graph->sub_graphs->rnum;
	}
	sqlite3_bind_int(graph_insert_stmt, 7, _ccv_nnc_symbolic_graph_index_in_repo(graph->peer, repo));
	sqlite3_bind_int(graph_insert_stmt, 8, _ccv_nnc_symbolic_graph_index_in_repo(graph->p, repo));
	sqlite3_bind_int(graph_insert_stmt, 9, graph->p_idx);
	sqlite3_bind_int(graph_insert_stmt, 10, graph->exec_idx);
	sqlite3_bind_int(graph_insert_stmt, 11, graph->breakpoint_size);
	if (graph->breakpoint_size && graph->breakpoints)
		sqlite3_bind_blob(graph_insert_stmt, 12, pos, sizeof(int) * graph->breakpoint_size, 0);
	sqlite3_bind_int(graph_insert_stmt, 13, graph->backward_tensor_symbol_size);
	if (graph->backward_tensor_symbol_size)
		sqlite3_bind_blob(graph_insert_stmt, 14, graph->backward_tensor_symbols, sizeof(int) * graph->backward_tensor_symbol_size, 0);
	sqlite3_bind_int(graph_insert_stmt, 15, graph->backward_exec_symbol_size);
	if (graph->backward_exec_symbol_size)
		sqlite3_bind_blob(graph_insert_stmt, 16, graph->backward_exec_symbols, sizeof(int) * graph->backward_exec_symbol_size, 0);
	assert_sqlite(SQLITE_DONE == sqlite3_step(graph_insert_stmt));
	sqlite3_reset(graph_insert_stmt);
	sqlite3_clear_bindings(graph_insert_stmt);
}

static void _ccv_nnc_symbolic_graph_push_repo(const ccv_nnc_symbolic_graph_t* const graph, ccv_array_t* const repo)
{
	ccv_array_push(repo, &graph);
	int i;
	if (graph->sub_graphs && graph->sub_graphs->rnum)
		for (i = 0; i < graph->sub_graphs->rnum; i++)
		{
			const ccv_nnc_symbolic_graph_t* const sub_graph = *(ccv_nnc_symbolic_graph_t**)ccv_array_get(graph->sub_graphs, i);
			if (sub_graph)
				_ccv_nnc_symbolic_graph_push_repo(sub_graph, repo);
		}
}

void ccv_nnc_symbolic_graph_write(const ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_bind_t* const tensor_binds, const int tensor_bind_size, const char* const fn)
{
	sqlite3* conn = 0;
	if (SQLITE_OK == sqlite3_open(fn, &conn))
	{
		const char tensor_symbol_create_table_qs[] = "CREATE TABLE IF NOT EXISTS tensor_symbol "
			"(id INTEGER, graph INTEGER, assign_ref INTEGER, r_assign_ref INTEGER, "
			"bypass_ref INTEGER, r_bypass_ref INTEGER, p_ref INTEGER, alias_ref INTEGER, peer_ref INTEGER, "
			"flags INTEGER, ofs BLOB, inc BLOB, s_ref BLOB, name TEXT, type INTEGER, format INTEGER, "
			"datatype INTEGER, dim BLOB, PRIMARY KEY (id, graph))";
		assert_sqlite(SQLITE_OK == sqlite3_exec(conn, tensor_symbol_create_table_qs, 0, 0, 0));
		const char tensor_symbol_insert_qs[] = 
			"REPLACE INTO tensor_symbol "
			"(id, graph, assign_ref, r_assign_ref, bypass_ref, r_bypass_ref, p_ref, alias_ref, peer_ref, flags, "
			"ofs, inc, s_ref, name, type, format, datatype, dim) VALUES "
			"($id, $graph, $assign_ref, $r_assign_ref, $bypass_ref, $r_bypass_ref, $p_ref, $alias_ref, $peer_ref, "
			"$flags, $ofs, $inc, $s_ref, $name, $type, $format, $datatype, $dim)";
		sqlite3_stmt* tensor_symbol_insert_stmt = 0;
		assert_sqlite(SQLITE_OK == sqlite3_prepare_v2(conn, tensor_symbol_insert_qs, sizeof(tensor_symbol_insert_qs), &tensor_symbol_insert_stmt, 0));

		const char exec_symbol_create_table_qs[] = "CREATE TABLE IF NOT EXISTS graph_exec_symbol "
			"(id INTEGER, graph INTEGER, input_size INTEGER, output_size INTEGER, graph_ref_size INTEGER, "
			"flags INTEGER, peer_ref INTEGER, inputs BLOB, outputs BLOB, outgoings BLOB, name TEXT, "
			"cmd_cmd INTEGER, cmd_backend INTEGER, cmd_algorithm INTEGER, cmd_info BLOB, hint BLOB, graph_ref BLOB, "
			"case_of_expr INTEGER, case_of_flags INTEGER, case_of_argument_offset INTEGER, case_of_argument_size INTEGER, "
			"p_while_expr INTEGER, p_while_input_size INTEGER, p_while_inputs BLOB, PRIMARY KEY (id, graph))";
		assert_sqlite(SQLITE_OK == sqlite3_exec(conn, exec_symbol_create_table_qs, 0, 0, 0));
		const char exec_symbol_insert_qs[] = 
			"REPLACE INTO graph_exec_symbol "
			"(id, graph, input_size, output_size, graph_ref_size, flags, peer_ref, inputs, outputs, outgoings, "
			"name, cmd_cmd, cmd_backend, cmd_algorithm, cmd_info, hint, graph_ref, case_of_expr, case_of_flags, "
			"case_of_argument_offset, case_of_argument_size, p_while_expr, p_while_input_size, p_while_inputs) "
			"VALUES ($id, $graph, $input_size, $output_size, $graph_ref_size, $flags, $peer_ref, $inputs, $outputs, "
			"$outgoings, $name, $cmd_cmd, $cmd_backend, $cmd_algorithm, $cmd_info, $hint, $graph_ref, $case_of_expr, "
			"$case_of_flags, $case_of_argument_offset, $case_of_argument_size, $p_while_expr, $p_while_input_size, "
			"$p_while_inputs)";
		sqlite3_stmt* exec_symbol_insert_stmt = 0;
		assert_sqlite(SQLITE_OK == sqlite3_prepare_v2(conn, exec_symbol_insert_qs, sizeof(exec_symbol_insert_qs), &exec_symbol_insert_stmt, 0));

		const char graph_create_table_qs[] = "CREATE TABLE IF NOT EXISTS graph "
			"(graph INTEGER PRIMARY KEY, tensor_symbol_size INTEGER, exec_symbol_size INTEGER, sources BLOB, "
			"destinations BLOB, sub_graphs BLOB, peer INTEGER, p INTEGER, p_idx INTEGER, exec_idx INTEGER, "
			"breakpoint_size INTEGER, breakpoints BLOB, backward_tensor_symbol_size INTEGER, "
			"backward_tensor_symbols BLOB, backward_exec_symbol_size INTEGER, backward_exec_symbols BLOB)";
		assert_sqlite(SQLITE_OK == sqlite3_exec(conn, graph_create_table_qs, 0, 0, 0));
		const char graph_insert_qs[] = 
			"REPLACE INTO graph "
			"(graph, tensor_symbol_size, exec_symbol_size, sources, destinations, sub_graphs, peer, p, p_idx, "
			"exec_idx, breakpoint_size, breakpoints, backward_tensor_symbol_size, "
			"backward_tensor_symbols, backward_exec_symbol_size, backward_exec_symbols) VALUES "
			"($graph, $tensor_symbol_size, $exec_symbol_size, $sources, $destinations, $sub_graphs, $peer, $p, $p_idx, "
			"$exec_idx, $breakpoint_size, $breakpoints, $backward_tensor_symbol_size, "
			"$backward_tensor_symbols, $backward_exec_symbol_size, $backward_exec_symbols)";
		sqlite3_stmt* graph_insert_stmt = 0;
		assert_sqlite(SQLITE_OK == sqlite3_prepare_v2(conn, graph_insert_qs, sizeof(graph_insert_qs), &graph_insert_stmt, 0));
		ccv_array_t* const repo = ccv_array_new(sizeof(ccv_nnc_symbolic_graph_t*), 1, 0);
		_ccv_nnc_symbolic_graph_push_repo(graph, repo);
		ccv_array_t* const ws = ccv_array_new(sizeof(int), 1, 0);
		int i;
		for (i = 0; i < repo->rnum; i++)
			_ccv_nnc_symbolic_graph_write(*(ccv_nnc_symbolic_graph_t**)ccv_array_get(repo, i), repo, i,
				tensor_symbol_insert_stmt, exec_symbol_insert_stmt, graph_insert_stmt, ws);
		ccv_array_free(ws);
		sqlite3_finalize(tensor_symbol_insert_stmt);
		sqlite3_finalize(exec_symbol_insert_stmt);
		sqlite3_finalize(graph_insert_stmt);
		// Write tensor binds.
		const char tensor_bind_create_table_qs[] = "CREATE TABLE IF NOT EXISTS tensor_bind "
			"(id INTEGER, graph INTEGER, type INTEGER, format INTEGER, datatype INTEGER, "
			"dim BLOB, data BLOB, PRIMARY KEY (id, graph))";
		assert_sqlite(SQLITE_OK == sqlite3_exec(conn, tensor_bind_create_table_qs, 0, 0, 0));
		// Remove everything in that table.
		assert_sqlite(SQLITE_OK == sqlite3_exec(conn, "DELETE FROM tensor_bind", 0, 0, 0));
		const char tensor_bind_insert_qs[] =
			"REPLACE INTO tensor_bind "
			"(id, graph, type, format, datatype, dim, data) VALUES ("
			"$id, $graph, $type, $format, $datatype, $dim, $data)";
		sqlite3_stmt* tensor_bind_insert_stmt = 0;
		assert_sqlite(SQLITE_OK == sqlite3_prepare_v2(conn, tensor_bind_insert_qs, sizeof(tensor_bind_insert_qs), &tensor_bind_insert_stmt, 0));
		for (i = 0; i < tensor_bind_size; i++)
		{
			const int graph_idx = _ccv_nnc_symbolic_graph_index_in_repo(tensor_binds[i].symbol.graph, repo);
			if (graph_idx < 0)
				continue;
			sqlite3_bind_int(tensor_bind_insert_stmt, 1, tensor_binds[i].symbol.d);
			sqlite3_bind_int(tensor_bind_insert_stmt, 2, graph_idx);
			if (tensor_binds[i].tensor)
			{
				const ccv_nnc_tensor_t* const tensor = tensor_binds[i].tensor;
				assert(!CCV_IS_TENSOR_VIEW(tensor));
				sqlite3_bind_int(tensor_bind_insert_stmt, 3, tensor->info.type);
				sqlite3_bind_int(tensor_bind_insert_stmt, 4, tensor->info.format);
				sqlite3_bind_int(tensor_bind_insert_stmt, 5, tensor->info.datatype);
				sqlite3_bind_blob(tensor_bind_insert_stmt, 6, tensor->info.dim, sizeof(tensor->info.dim), 0);
				sqlite3_bind_blob(tensor_bind_insert_stmt, 7, tensor->data.u8, ccv_nnc_tensor_data_size(tensor->info), 0);
			} else {
				assert(tensor_binds[i].symbol.d >= 0);
				const ccv_nnc_tensor_symbol_info_t* const symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, tensor_binds[i].symbol.d);
				sqlite3_bind_int(tensor_bind_insert_stmt, 3, symbol_info->info.type);
				sqlite3_bind_int(tensor_bind_insert_stmt, 4, symbol_info->info.format);
				sqlite3_bind_int(tensor_bind_insert_stmt, 5, symbol_info->info.datatype);
				sqlite3_bind_blob(tensor_bind_insert_stmt, 6, symbol_info->info.dim, sizeof(symbol_info->info.dim), 0);
			}
			sqlite3_step(tensor_bind_insert_stmt);
			sqlite3_reset(tensor_bind_insert_stmt);
			sqlite3_clear_bindings(tensor_bind_insert_stmt);
		}
		sqlite3_finalize(tensor_bind_insert_stmt);
		ccv_array_free(repo);
		sqlite3_close(conn);
	}
}

static ccv_nnc_symbolic_graph_t* _ccv_nnc_symbolic_graph_get(const ccv_array_t* const repo, const ccv_nnc_symbolic_graph_t* const pos)
{
	const int idx = (uintptr_t)pos >> 1;
	assert(idx < repo->rnum);
	return *(ccv_nnc_symbolic_graph_t**)ccv_array_get(repo, idx);
}

#define CCV_NNC_IS_SYMBOLIC_GRAPH_POS(ptr) ((uintptr_t)(ptr) & 1)

static ccv_nnc_symbolic_graph_t* _ccv_nnc_symbolic_graph_pos(const int idx)
{
	if (idx < 0)
		return 0; // This is nil.
	return (ccv_nnc_symbolic_graph_t*)(((uintptr_t)idx << 1) + 1);
}

static void _ccv_nnc_symbolic_graph_read(const int graph_idx, sqlite3_stmt* const graph_select_stmt, sqlite3_stmt* const tensor_symbol_select_stmt, sqlite3_stmt* const exec_symbol_select_stmt, ccv_nnc_symbolic_graph_t* const graph)
{
	int i, j;
	ccv_array_resize(graph->tensor_symbol_info, sqlite3_column_int(graph_select_stmt, 1));
	ccv_array_resize(graph->exec_symbol_info, sqlite3_column_int(graph_select_stmt, 2));
	if (sqlite3_column_blob(graph_select_stmt, 3))
	{
		const int* const sources = sqlite3_column_blob(graph_select_stmt, 3);
		const int count = sqlite3_column_bytes(graph_select_stmt, 3) / sizeof(int);
		graph->sources = ccv_array_new(sizeof(ccv_nnc_graph_exec_symbol_t), count, 0);
		for (i = 0; i < count; i++)
		{
			const ccv_nnc_graph_exec_symbol_t symbol = {
				.graph = graph,
				.d = sources[i]
			};
			ccv_array_push(graph->sources, &symbol);
		}
	}
	if (sqlite3_column_blob(graph_select_stmt, 4))
	{
		const int* const destinations = sqlite3_column_blob(graph_select_stmt, 4);
		const int count = sqlite3_column_bytes(graph_select_stmt, 4) / sizeof(int);
		graph->destinations = ccv_array_new(sizeof(ccv_nnc_graph_exec_symbol_t), count, 0);
		for (i = 0; i < count; i++)
		{
			const ccv_nnc_graph_exec_symbol_t symbol = {
				.graph = graph,
				.d = destinations[i]
			};
			ccv_array_push(graph->destinations, &symbol);
		}
	}
	if (sqlite3_column_blob(graph_select_stmt, 5))
	{
		const int* const sub_graphs = sqlite3_column_blob(graph_select_stmt, 5);
		const int count = sqlite3_column_bytes(graph_select_stmt, 5) / sizeof(int);
		graph->sub_graphs = ccv_array_new(sizeof(ccv_nnc_symbolic_graph_t*), count, 0);
		for (i = 0; i < count; i++)
		{
			const ccv_nnc_symbolic_graph_t* const sub_graph = _ccv_nnc_symbolic_graph_pos(sub_graphs[i]);
			ccv_array_push(graph->sub_graphs, &sub_graph);
		}
	}
	graph->peer = _ccv_nnc_symbolic_graph_pos(sqlite3_column_int(graph_select_stmt, 6));
	graph->p = _ccv_nnc_symbolic_graph_pos(sqlite3_column_int(graph_select_stmt, 7));
	graph->p_idx = sqlite3_column_int(graph_select_stmt, 8);
	graph->exec_idx = sqlite3_column_int(graph_select_stmt, 9);
	graph->breakpoint_size = sqlite3_column_int(graph_select_stmt, 10);
	if (graph->breakpoint_size)
	{
		graph->breakpoints = (ccv_nnc_graph_exec_symbol_t*)ccmalloc(sizeof(ccv_nnc_graph_exec_symbol_t) * graph->breakpoint_size);
		assert(sizeof(int) * graph->breakpoint_size == sqlite3_column_bytes(graph_select_stmt, 11));
		const int* const breakpoints = sqlite3_column_blob(graph_select_stmt, 11);
		for (i = 0; i < graph->breakpoint_size; i++)
			graph->breakpoints[i] = (ccv_nnc_graph_exec_symbol_t){
				.d = breakpoints[i],
				.graph = graph
			};
	}
	graph->backward_tensor_symbol_size = sqlite3_column_int(graph_select_stmt, 12);
	if (graph->backward_tensor_symbol_size)
	{
		graph->backward_tensor_symbols = (int*)ccmalloc(sizeof(int) * graph->backward_tensor_symbol_size);
		assert(sizeof(int) * graph->backward_tensor_symbol_size == sqlite3_column_bytes(graph_select_stmt, 13));
		const int* const backward_tensor_symbols = sqlite3_column_blob(graph_select_stmt, 13);
		memcpy(graph->backward_tensor_symbols, backward_tensor_symbols, sizeof(int) * graph->backward_tensor_symbol_size);
	}
	graph->backward_exec_symbol_size = sqlite3_column_int(graph_select_stmt, 14);
	if (graph->backward_exec_symbol_size)
	{
		graph->backward_exec_symbols = (int*)ccmalloc(sizeof(int) * graph->backward_exec_symbol_size);
		assert(sizeof(int) * graph->backward_exec_symbol_size == sqlite3_column_bytes(graph_select_stmt, 15));
		const int* const backward_exec_symbols = sqlite3_column_blob(graph_select_stmt, 15);
		memcpy(graph->backward_exec_symbols, backward_exec_symbols, sizeof(int) * graph->backward_exec_symbol_size);
	}
	sqlite3_bind_int(tensor_symbol_select_stmt, 1, graph_idx);
	for (i = 0; SQLITE_ROW == sqlite3_step(tensor_symbol_select_stmt); i++)
	{
		assert(sqlite3_column_int(tensor_symbol_select_stmt, 0) == i); // id should match.
		assert(i < graph->tensor_symbol_info->rnum);
		assert(sqlite3_column_int(tensor_symbol_select_stmt, 0) == i);
		ccv_nnc_tensor_symbol_info_t* const symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, i);
		symbol_info->assign_ref = sqlite3_column_int(tensor_symbol_select_stmt, 1);
		symbol_info->r_assign_ref = sqlite3_column_int(tensor_symbol_select_stmt, 2);
		symbol_info->bypass_ref = sqlite3_column_int(tensor_symbol_select_stmt, 3);
		symbol_info->r_bypass_ref = sqlite3_column_int(tensor_symbol_select_stmt, 4);
		symbol_info->p_ref = sqlite3_column_int(tensor_symbol_select_stmt, 5);
		symbol_info->alias_ref = sqlite3_column_int(tensor_symbol_select_stmt, 6);
		symbol_info->peer_ref = sqlite3_column_int(tensor_symbol_select_stmt, 7);
		symbol_info->flags = sqlite3_column_int(tensor_symbol_select_stmt, 8);
		memset(symbol_info->ofs, 0, sizeof(symbol_info->ofs));
		const int* const ofs = sqlite3_column_blob(tensor_symbol_select_stmt, 9);
		if (ofs)
			memcpy(symbol_info->ofs, ofs, ccv_min(sqlite3_column_bytes(tensor_symbol_select_stmt, 8), sizeof(symbol_info->ofs)));
		memset(symbol_info->inc, 0, sizeof(symbol_info->inc));
		const int* const inc = sqlite3_column_blob(tensor_symbol_select_stmt, 10);
		if (inc)
			memcpy(symbol_info->inc, inc, ccv_min(sqlite3_column_bytes(tensor_symbol_select_stmt, 9), sizeof(symbol_info->inc)));
		const int* const s_ref = sqlite3_column_blob(tensor_symbol_select_stmt, 11);
		if (s_ref)
		{
			const int count = sqlite3_column_bytes(tensor_symbol_select_stmt, 11) / sizeof(int);
			symbol_info->s_ref = ccv_array_new(sizeof(int), count, 0);
			ccv_array_resize(symbol_info->s_ref, count);
			memcpy(ccv_array_get(symbol_info->s_ref, 0), s_ref, sizeof(int) * count);
		} else
			symbol_info->s_ref = 0;
		const char* const name = (char*)sqlite3_column_text(tensor_symbol_select_stmt, 12);
		if (name)
		{
			const int count = sqlite3_column_bytes(tensor_symbol_select_stmt, 12);
			symbol_info->name = (char*)ccmalloc(sizeof(char) * (count + 1));
			memcpy(symbol_info->name, name, count);
			symbol_info->name[count] = 0; // null terminator
		} else
			symbol_info->name = 0;
		symbol_info->info.type = sqlite3_column_int(tensor_symbol_select_stmt, 13);
		symbol_info->info.format = sqlite3_column_int(tensor_symbol_select_stmt, 14);
		symbol_info->info.datatype = sqlite3_column_int(tensor_symbol_select_stmt, 15);
		memset(symbol_info->info.dim, 0, sizeof(symbol_info->info.dim));
		const int* const dim = sqlite3_column_blob(tensor_symbol_select_stmt, 16);
		if (dim)
			memcpy(symbol_info->info.dim, dim, ccv_min(sqlite3_column_bytes(tensor_symbol_select_stmt, 16), sizeof(symbol_info->info.dim)));
		if (CCV_NNC_TENSOR_SYMBOL_IS_DEAD(symbol_info->flags) && graph->reuse.tensor < 0)
			graph->reuse.tensor = i;
	}
	sqlite3_reset(tensor_symbol_select_stmt);
	sqlite3_clear_bindings(tensor_symbol_select_stmt);
	sqlite3_bind_int(exec_symbol_select_stmt, 1, graph_idx);
	for (i = 0; SQLITE_ROW == sqlite3_step(exec_symbol_select_stmt); i++)
	{
		assert(sqlite3_column_int(exec_symbol_select_stmt, 0) == i); // id should match.
		assert(i < graph->exec_symbol_info->rnum);
		assert(sqlite3_column_int(exec_symbol_select_stmt, 0) == i);
		ccv_nnc_graph_exec_symbol_info_t* const symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, i);
		memset(symbol_info, 0, sizeof(ccv_nnc_graph_exec_symbol_info_t));
		symbol_info->input_size = sqlite3_column_int(exec_symbol_select_stmt, 1);
		symbol_info->output_size = sqlite3_column_int(exec_symbol_select_stmt, 2);
		symbol_info->graph_ref_size = sqlite3_column_int(exec_symbol_select_stmt, 3);
		symbol_info->flags = sqlite3_column_int(exec_symbol_select_stmt, 4);
		symbol_info->peer_ref = sqlite3_column_int(exec_symbol_select_stmt, 5);
		if (symbol_info->input_size > 0 || symbol_info->output_size > 0)
		{
			symbol_info->inputs = (int*)ccmalloc(sizeof(int) * (symbol_info->input_size + symbol_info->output_size));
			for (j = 0; j < symbol_info->input_size; j++)
				symbol_info->inputs[j] = CCV_NNC_NO_TENSOR_SYMBOL;
			symbol_info->outputs = symbol_info->inputs + symbol_info->input_size;
			for (j = 0; j < symbol_info->output_size; j++)
				symbol_info->outputs[j] = CCV_NNC_NO_TENSOR_SYMBOL;
		}
		if (symbol_info->input_size)
		{
			const int* const inputs = sqlite3_column_blob(exec_symbol_select_stmt, 6);
			if (inputs)
				memcpy(symbol_info->inputs, inputs, ccv_min(sizeof(int) * symbol_info->input_size, sqlite3_column_bytes(exec_symbol_select_stmt, 6)));
		}
		if (symbol_info->output_size)
		{
			const int* const outputs = sqlite3_column_blob(exec_symbol_select_stmt, 7);
			if (outputs)
				memcpy(symbol_info->outputs, outputs, ccv_min(sizeof(int) * symbol_info->output_size, sqlite3_column_bytes(exec_symbol_select_stmt, 7)));
		}
		const int* const outgoings = sqlite3_column_blob(exec_symbol_select_stmt, 8);
		if (outgoings)
		{
			const int count = sqlite3_column_bytes(exec_symbol_select_stmt, 8) / sizeof(int);
			symbol_info->outgoings = ccv_array_new(sizeof(int), count, 0);
			ccv_array_resize(symbol_info->outgoings, count);
			memcpy(ccv_array_get(symbol_info->outgoings, 0), outgoings, sizeof(int) * count);
		}
		const char* const name = (char*)sqlite3_column_text(exec_symbol_select_stmt, 9);
		if (name)
		{
			const int count = sqlite3_column_bytes(exec_symbol_select_stmt, 9);
			symbol_info->name = (char*)ccmalloc(sizeof(char) * (count + 1));
			memcpy(symbol_info->name, name, count);
			symbol_info->name[count] = 0; // null terminator
		}
		symbol_info->cmd.cmd = sqlite3_column_int(exec_symbol_select_stmt, 10);
		symbol_info->cmd.backend = sqlite3_column_int(exec_symbol_select_stmt, 11);
		symbol_info->cmd.algorithm = sqlite3_column_int(exec_symbol_select_stmt, 12);
		const void* const cmd_info = sqlite3_column_blob(exec_symbol_select_stmt, 13);
		if (cmd_info)
			memcpy(&symbol_info->cmd.info, cmd_info, ccv_min(sizeof(symbol_info->cmd.info), sqlite3_column_bytes(exec_symbol_select_stmt, 13)));
		const void* const hint = sqlite3_column_blob(exec_symbol_select_stmt, 14);
		if (hint)
			memcpy(&symbol_info->hint, hint, ccv_min(sizeof(symbol_info->hint), sqlite3_column_bytes(exec_symbol_select_stmt, 14)));
		if (symbol_info->graph_ref_size)
		{
			const int* const graph_ref = sqlite3_column_blob(exec_symbol_select_stmt, 15);
			if (symbol_info->graph_ref_size > sizeof(symbol_info->_inline_graph_ref) / sizeof(symbol_info->_inline_graph_ref[0]))
				symbol_info->_heap_graph_ref = (int*)cccalloc(symbol_info->graph_ref_size, sizeof(int));
			if (graph_ref)
				memcpy(CCV_NNC_GRAPH_REF(symbol_info), graph_ref, ccv_min(sizeof(int) * symbol_info->graph_ref_size, sqlite3_column_bytes(exec_symbol_select_stmt, 15)));
		}
		if (symbol_info->flags & CCV_NNC_GRAPH_EXEC_CASE_OF)
		{
			symbol_info->case_of.flags = sqlite3_column_int(exec_symbol_select_stmt, 17);
			symbol_info->case_of.argument.offset = sqlite3_column_int(exec_symbol_select_stmt, 18);
			symbol_info->case_of.argument.size = sqlite3_column_int(exec_symbol_select_stmt, 19);
		} else if (symbol_info->flags & CCV_NNC_GRAPH_EXEC_P_WHILE) {
			symbol_info->p_while.input_size = sqlite3_column_int(exec_symbol_select_stmt, 21);
			if (symbol_info->p_while.input_size)
			{
				symbol_info->p_while.inputs = (int*)cccalloc(symbol_info->p_while.input_size, sizeof(int));
				const int* const inputs = sqlite3_column_blob(exec_symbol_select_stmt, 22);
				if (inputs)
					memcpy(symbol_info->p_while.inputs, inputs, ccv_min(sizeof(int) * symbol_info->p_while.input_size, sqlite3_column_bytes(exec_symbol_select_stmt, 22)));
			}
		}
		if (CCV_NNC_GRAPH_EXEC_IS_DEAD(symbol_info->flags) && graph->reuse.exec < 0)
			graph->reuse.exec = i;
	}
	sqlite3_reset(exec_symbol_select_stmt);
	sqlite3_clear_bindings(exec_symbol_select_stmt);
}

static void _ccv_nnc_symbolic_graph_rewire(const ccv_array_t* const repo, ccv_nnc_symbolic_graph_t* const graph)
{
	if (graph->p && CCV_NNC_IS_SYMBOLIC_GRAPH_POS(graph->p))
		graph->p = _ccv_nnc_symbolic_graph_get(repo, graph->p);
	if (graph->peer && CCV_NNC_IS_SYMBOLIC_GRAPH_POS(graph->peer))
		graph->peer = _ccv_nnc_symbolic_graph_get(repo, graph->peer);
	int i;
	if (graph->sub_graphs)
		for (i = 0; i < graph->sub_graphs->rnum; i++)
		{
			ccv_nnc_symbolic_graph_t* const sub_graph = *(ccv_nnc_symbolic_graph_t**)ccv_array_get(graph->sub_graphs, i);
			if (sub_graph && CCV_NNC_IS_SYMBOLIC_GRAPH_POS(sub_graph))
				*(ccv_nnc_symbolic_graph_t**)ccv_array_get(graph->sub_graphs, i) = _ccv_nnc_symbolic_graph_get(repo, sub_graph);
		}
}

void ccv_nnc_symbolic_graph_read(const char* const fn, ccv_nnc_symbolic_graph_t** const graph_ref, ccv_nnc_tensor_bind_t** const tensor_binds_ref, int* const tensor_bind_size_ref)
{
	sqlite3* conn = 0;
	if (SQLITE_OK == sqlite3_open(fn, &conn))
	{
		ccv_array_t* const repo = ccv_array_new(sizeof(ccv_nnc_symbolic_graph_t*), 1, 0);
		const char graph_select_qs[] =
			"SELECT graph, tensor_symbol_size, exec_symbol_size, sources, destinations, sub_graphs, peer, p, p_idx, "
			"exec_idx, breakpoint_size, breakpoints, backward_tensor_symbol_size, "
			"backward_tensor_symbols, backward_exec_symbol_size, backward_exec_symbols FROM graph ORDER BY graph";
		sqlite3_stmt* graph_select_stmt = 0;
		assert_sqlite(SQLITE_OK == sqlite3_prepare_v2(conn, graph_select_qs, sizeof(graph_select_qs), &graph_select_stmt, 0));
		sqlite3_stmt* tensor_symbol_select_stmt = 0;
		const char tensor_symbol_select_qs[] =
			"SELECT id, assign_ref, r_assign_ref, bypass_ref, r_bypass_ref, p_ref, alias_ref, peer_ref, flags, ofs, inc, "
			"s_ref, name, type, format, datatype, dim FROM tensor_symbol WHERE graph=$graph ORDER BY id";
		assert_sqlite(SQLITE_OK == sqlite3_prepare_v2(conn, tensor_symbol_select_qs, sizeof(tensor_symbol_select_qs), &tensor_symbol_select_stmt, 0));
		const char exec_symbol_select_qs[] =
			"SELECT id, input_size, output_size, graph_ref_size, flags, peer_ref, inputs, outputs, outgoings, "
			"name, cmd_cmd, cmd_backend, cmd_algorithm, cmd_info, hint, graph_ref, case_of_expr, case_of_flags, "
			"case_of_argument_offset, case_of_argument_size, p_while_expr, p_while_input_size, p_while_inputs "
			"FROM graph_exec_symbol WHERE graph=$graph ORDER BY id";
		sqlite3_stmt* exec_symbol_select_stmt = 0;
		assert_sqlite(SQLITE_OK == sqlite3_prepare_v2(conn, exec_symbol_select_qs, sizeof(exec_symbol_select_qs), &exec_symbol_select_stmt, 0));
		while (SQLITE_ROW == sqlite3_step(graph_select_stmt))
		{
			ccv_nnc_symbolic_graph_t* const graph = ccv_nnc_symbolic_graph_new();
			const int graph_idx = sqlite3_column_int(graph_select_stmt, 0);
			assert(graph_idx == repo->rnum);
			ccv_array_push(repo, &graph);
			_ccv_nnc_symbolic_graph_read(graph_idx, graph_select_stmt, tensor_symbol_select_stmt, exec_symbol_select_stmt, graph);
		}
		int i;
		for (i = 0; i < repo->rnum; i++)
			_ccv_nnc_symbolic_graph_rewire(repo, *(ccv_nnc_symbolic_graph_t**)ccv_array_get(repo, i));
		*graph_ref = (repo->rnum > 0) ? *(ccv_nnc_symbolic_graph_t**)ccv_array_get(repo, 0) : 0;
		assert((tensor_bind_size_ref && tensor_binds_ref) || (!tensor_bind_size_ref && !tensor_binds_ref));
		if (tensor_bind_size_ref && tensor_binds_ref)
		{
			const char tensor_bind_count_qs[] =
				"SELECT COUNT(*) FROM tensor_bind";
			sqlite3_stmt* tensor_bind_count_stmt = 0;
			assert_sqlite(SQLITE_OK == sqlite3_prepare_v2(conn, tensor_bind_count_qs, sizeof(tensor_bind_count_qs), &tensor_bind_count_stmt, 0));
			sqlite3_step(tensor_bind_count_stmt);
			const int tensor_bind_size = *tensor_bind_size_ref = sqlite3_column_int(tensor_bind_count_stmt, 0);
			sqlite3_finalize(tensor_bind_count_stmt);
			// Respect the insert order (rowid).
			if (!tensor_bind_size)
				*tensor_binds_ref = 0;
			else {
				const char tensor_bind_select_qs[] =
					"SELECT id, graph, type, format, datatype, dim, data FROM tensor_bind";
				sqlite3_stmt* tensor_bind_select_stmt = 0;
				ccv_nnc_tensor_bind_t* const tensor_binds = *tensor_binds_ref = (ccv_nnc_tensor_bind_t*)ccmalloc(sizeof(ccv_nnc_tensor_bind_t) * tensor_bind_size);
				assert_sqlite(SQLITE_OK == sqlite3_prepare_v2(conn, tensor_bind_select_qs, sizeof(tensor_bind_select_qs), &tensor_bind_select_stmt, 0));
				for (i = 0; SQLITE_ROW == sqlite3_step(tensor_bind_select_stmt); i++)
				{
					assert(i < tensor_bind_size);
					tensor_binds[i].symbol.d = sqlite3_column_int(tensor_bind_select_stmt, 0);
					const int graph_idx = sqlite3_column_int(tensor_bind_select_stmt, 1);
					assert(graph_idx < repo->rnum);
					tensor_binds[i].symbol.graph = (graph_idx >= 0) ? *(ccv_nnc_symbolic_graph_t**)ccv_array_get(repo, graph_idx) : 0;
					ccv_nnc_tensor_param_t info;
					info.type = sqlite3_column_int(tensor_bind_select_stmt, 2);
					info.format = sqlite3_column_int(tensor_bind_select_stmt, 3);
					info.datatype = sqlite3_column_int(tensor_bind_select_stmt, 4);
					const int* const dim = sqlite3_column_blob(tensor_bind_select_stmt, 5);
					memset(info.dim, 0, sizeof(info.dim));
					if (dim)
						memcpy(info.dim, dim, ccv_min(sizeof(info.dim), sqlite3_column_bytes(tensor_bind_select_stmt, 5)));
					const void* const data = sqlite3_column_blob(tensor_bind_select_stmt, 7);
					if (!data)
						tensor_binds[i].tensor = 0;
					else {
						tensor_binds[i].tensor = ccv_nnc_tensor_new(0, info, 0);
						size_t data_size = ccv_nnc_tensor_data_size(info);
						memcpy(tensor_binds[i].tensor->data.u8, data, ccv_min(data_size, sqlite3_column_bytes(tensor_bind_select_stmt, 6)));
					}
				}
				for (; i < tensor_bind_size; i++)
				{
					tensor_binds[i].symbol.d = CCV_NNC_NO_TENSOR_SYMBOL;
					tensor_binds[i].symbol.graph = 0;
					tensor_binds[i].tensor = 0;
				}
				sqlite3_finalize(tensor_bind_select_stmt);
			}
		}
		ccv_array_free(repo);
		sqlite3_finalize(graph_select_stmt);
		sqlite3_finalize(tensor_symbol_select_stmt);
		sqlite3_finalize(exec_symbol_select_stmt);
		sqlite3_close(conn);
	}
}
