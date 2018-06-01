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
		sqlite3_bind_int(tensor_symbol_insert_stmt, 6, symbol_info->p_ref);
		sqlite3_bind_int(tensor_symbol_insert_stmt, 7, symbol_info->alias_ref);
		sqlite3_bind_int(tensor_symbol_insert_stmt, 8, symbol_info->peer_ref);
		sqlite3_bind_int(tensor_symbol_insert_stmt, 9, symbol_info->flags);
		sqlite3_bind_blob(tensor_symbol_insert_stmt, 10, symbol_info->ofs, sizeof(symbol_info->ofs), 0);
		sqlite3_bind_blob(tensor_symbol_insert_stmt, 11, symbol_info->inc, sizeof(symbol_info->inc), 0);
		if (symbol_info->s_ref)
			sqlite3_bind_blob(tensor_symbol_insert_stmt, 12, ccv_array_get(symbol_info->s_ref, 0), sizeof(int) * symbol_info->s_ref->rnum, 0);
		else
			sqlite3_bind_null(tensor_symbol_insert_stmt, 12);
		if (symbol_info->name)
			sqlite3_bind_text(tensor_symbol_insert_stmt, 13, symbol_info->name, 0, 0);
		else
			sqlite3_bind_null(tensor_symbol_insert_stmt, 13);
		sqlite3_bind_int(tensor_symbol_insert_stmt, 14, symbol_info->info.type);
		sqlite3_bind_int(tensor_symbol_insert_stmt, 15, symbol_info->info.format);
		sqlite3_bind_int(tensor_symbol_insert_stmt, 16, symbol_info->info.datatype);
		sqlite3_bind_blob(tensor_symbol_insert_stmt, 17, symbol_info->info.dim, sizeof(symbol_info->info.dim), 0);
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
			sqlite3_bind_text(exec_symbol_insert_stmt, 11, symbol_info->name, 0, 0);
		sqlite3_bind_int(exec_symbol_insert_stmt, 12, symbol_info->cmd.cmd);
		sqlite3_bind_int(exec_symbol_insert_stmt, 13, symbol_info->cmd.backend);
		sqlite3_bind_int(exec_symbol_insert_stmt, 14, symbol_info->cmd.algorithm);
		sqlite3_bind_blob(exec_symbol_insert_stmt, 15, &symbol_info->cmd.info, sizeof(symbol_info->cmd.info), 0);
		sqlite3_bind_blob(exec_symbol_insert_stmt, 16, &symbol_info->hint, sizeof(symbol_info->hint), 0);
		if (symbol_info->graph_ref_size)
			sqlite3_bind_blob(exec_symbol_insert_stmt, 17, CCV_NNC_GRAPH_REF(symbol_info), sizeof(int) * symbol_info->graph_ref_size, 0);
		if (symbol_info->flags & CCV_NNC_GRAPH_EXEC_CASE_OF)
		{
			sqlite3_bind_int(exec_symbol_insert_stmt, 18, symbol_info->case_of.flags);
			sqlite3_bind_int(exec_symbol_insert_stmt, 19, symbol_info->case_of.argument.offset);
			sqlite3_bind_int(exec_symbol_insert_stmt, 20, symbol_info->case_of.argument.size);
		}
		if (symbol_info->flags & CCV_NNC_GRAPH_EXEC_P_WHILE)
		{
			sqlite3_bind_int(exec_symbol_insert_stmt, 21, symbol_info->p_while.input_size);
			if (symbol_info->p_while.input_size)
				sqlite3_bind_blob(exec_symbol_insert_stmt, 22, symbol_info->p_while.inputs, sizeof(int) * symbol_info->p_while.input_size, 0);
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
	sqlite3_bind_int(graph_insert_stmt, 13, graph->forward_symbol_size);
	sqlite3_bind_int(graph_insert_stmt, 14, graph->backward_tensor_symbol_size);
	if (graph->backward_tensor_symbol_size)
		sqlite3_bind_blob(graph_insert_stmt, 15, graph->backward_tensor_symbols, sizeof(int) * graph->backward_tensor_symbol_size, 0);
	sqlite3_bind_int(graph_insert_stmt, 16, graph->backward_symbol_size);
	if (graph->backward_symbol_size)
		sqlite3_bind_blob(graph_insert_stmt, 17, graph->backward_exec_symbols, sizeof(int) * graph->backward_symbol_size, 0);
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
			"bypass_ref INTEGER, p_ref INTEGER, alias_ref INTEGER, peer_ref INTEGER, flags INTEGER, "
			"ofs BLOB, inc BLOB, s_ref BLOB, name TEXT, type INTEGER, format INTEGER, datatype INTEGER, "
			"dim BLOB, PRIMARY KEY (id, graph))";
		assert_sqlite(SQLITE_OK == sqlite3_exec(conn, tensor_symbol_create_table_qs, 0, 0, 0));
		const char tensor_symbol_insert_qs[] = 
			"REPLACE INTO tensor_symbol "
			"(id, graph, assign_ref, r_assign_ref, bypass_ref, p_ref, alias_ref, peer_ref, flags, ofs, "
			"inc, s_ref, name, type, format, datatype, dim) VALUES "
			"($id, $graph, $assign_ref, $r_assign_ref, $bypass_ref, $p_ref, $alias_ref, $peer_ref, $flags, "
			"$ofs, $inc, $s_ref, $name, $type, $format, $datatype, $dim)";
		sqlite3_stmt* tensor_symbol_insert_stmt = 0;
		assert_sqlite(SQLITE_OK == sqlite3_prepare_v2(conn, tensor_symbol_insert_qs, sizeof(tensor_symbol_insert_qs), &tensor_symbol_insert_stmt, 0));

		const char exec_symbol_create_table_qs[] = "CREATE TABLE IF NOT EXISTS graph_exec_symbol "
			"(id INTEGER, graph INTEGER, input_size INTEGER, output_size INTEGER, graph_ref_size INTEGER, "
			"flags INTEGER, peer_ref INTEGER, inputs BLOB, outputs BLOB, outgoings BLOB, name TEXT, "
			"cmd_cmd INTEGER, cmd_backend INTEGER, cmd_algorithm INTEGER, cmd_info BLOB, hint BLOB, graph_ref BLOB, "
			"case_of_flags INTEGER, case_of_argument_offset INTEGER, case_of_argument_size INTEGER, "
			"p_while_input_size INTEGER, p_while_inputs BLOB, PRIMARY KEY (id, graph))";
		assert_sqlite(SQLITE_OK == sqlite3_exec(conn, exec_symbol_create_table_qs, 0, 0, 0));
		const char exec_symbol_insert_qs[] = 
			"REPLACE INTO graph_exec_symbol "
			"(id, graph, input_size, output_size, graph_ref_size, flags, peer_ref, inputs, outputs, outgoings, "
			"name, cmd_cmd, cmd_backend, cmd_algorithm, cmd_info, hint, graph_ref, case_of_flags, "
			"case_of_argument_offset, case_of_argument_size, p_while_input_size, p_while_inputs) VALUES "
			"($id, $graph, $input_size, $output_size, $graph_ref_size, $flags, $peer_ref, $inputs, $outputs, "
			"$outgoings, $name, $cmd_cmd, $cmd_backend, $cmd_algorithm, $cmd_info, $hint, $graph_ref, $case_of_flags, "
			"$case_of_argument_offset, $case_of_argument_size, $p_while_input_size, $p_while_inputs)";
		sqlite3_stmt* exec_symbol_insert_stmt = 0;
		assert_sqlite(SQLITE_OK == sqlite3_prepare_v2(conn, exec_symbol_insert_qs, sizeof(exec_symbol_insert_qs), &exec_symbol_insert_stmt, 0));

		const char graph_create_table_qs[] = "CREATE TABLE IF NOT EXISTS graph "
			"(graph INTEGER PRIMARY KEY, tensor_symbol_size INTEGER, exec_symbol_size INTEGER, sources BLOB, "
			"destinations BLOB, sub_graphs BLOB, peer INTEGER, p INTEGER, p_idx INTEGER, exec_idx INTEGER, "
			"breakpoint_size INTEGER, breakpoints BLOB, forward_symbol_size INTEGER, backward_tensor_symbol_size INTEGER, "
			"backward_tensor_symbols BLOB, backward_symbol_size INTEGER, backward_exec_symbols BLOB)";
		assert_sqlite(SQLITE_OK == sqlite3_exec(conn, graph_create_table_qs, 0, 0, 0));
		const char graph_insert_qs[] = 
			"REPLACE INTO graph "
			"(graph, tensor_symbol_size, exec_symbol_size, sources, destinations, sub_graphs, peer, p, p_idx, "
			"exec_idx, breakpoint_size, breakpoints, forward_symbol_size, backward_tensor_symbol_size, "
			"backward_tensor_symbols, backward_symbol_size, backward_exec_symbols) VALUES "
			"($graph, $tensor_symbol_size, $exec_symbol_size, $sources, $destinations, $sub_graphs, $peer, $p, $p_idx, "
			"$exec_idx, $breakpoint_size, $breakpoints, $forward_symbol_size, $backward_tensor_symbol_size, "
			"$backward_tensor_symbols, $backward_symbol_size, $backward_exec_symbols)";
		sqlite3_stmt* graph_insert_stmt = 0;
		assert_sqlite(SQLITE_OK == sqlite3_prepare_v2(conn, graph_insert_qs, sizeof(graph_insert_qs), &graph_insert_stmt, 0));
		ccv_array_t* const repo = ccv_array_new(sizeof(ccv_nnc_symbolic_graph_t*), 1, 0);
		_ccv_nnc_symbolic_graph_push_repo(graph, repo);
		ccv_array_t* const ws = ccv_array_new(sizeof(int), 1, 0);
		int i;
		for (i = 0; i < repo->rnum; i++)
			_ccv_nnc_symbolic_graph_write(*(ccv_nnc_symbolic_graph_t**)ccv_array_get(repo, i),
				repo, i,
				tensor_symbol_insert_stmt, exec_symbol_insert_stmt, graph_insert_stmt,
				ws);
		ccv_array_free(repo);
		ccv_array_free(ws);
		sqlite3_finalize(tensor_symbol_insert_stmt);
		sqlite3_finalize(exec_symbol_insert_stmt);
		sqlite3_finalize(graph_insert_stmt);
		// Write tensor binds.
		const char tensor_bind_create_table_qs[] = "CREATE TABLE IF NOT EXISTS tensor_bind "
			"(id INTEGER, graph INTEGER, ofs BLOB, inc BLOB, type INTEGER, format INTEGER, datatype INTEGER, "
			"dim BLOB, data BLOB, PRIMARY KEY (id, graph))";
		assert_sqlite(SQLITE_OK == sqlite3_exec(conn, tensor_bind_create_table_qs, 0, 0, 0));
		sqlite3_close(conn);
	}
}

void ccv_nnc_symbolic_graph_read(const char* const fn, ccv_nnc_symbolic_graph_t** const graph_ref, ccv_nnc_tensor_bind_t** const tensor_binds_ref, int* const tensor_bind_size_ref)
{
	sqlite3* conn = 0;
	if (SQLITE_OK == sqlite3_open(fn, &conn))
	{
	}
}
