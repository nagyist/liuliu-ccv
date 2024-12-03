#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_nnc_symbolic_graph.h"

// MARK - Level-3.5 API

static void _ccv_nnc_remove_unused_from_marked(const uint32_t* const tensor_used, const int size, uint32_t* const tensor_marked)
{
	int i;
	for (i = 0; i < size; i++)
		tensor_marked[i] &= tensor_used[i];
}

typedef struct {
	int* chain_ids;
	int* chain_pos;
	ccv_sparse_matrix_t* deps;
} ccv_nnc_exec_dep_t;

// Implement the new method for exec_dep. We use chain decomposition such that each node only needs to log which chain and at which node to be dependent on.
static ccv_nnc_exec_dep_t _ccv_nnc_exec_dep_new(const ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_visit_t* const visit, const ccv_nnc_graph_visit_t* const reversed_visit)
{
	const int exec_symbol_info_size = graph->exec_symbol_info->rnum;
	int* chain_ids = ccmalloc(sizeof(int) * exec_symbol_info_size * 2);
	int* chain_pos = chain_ids + exec_symbol_info_size;
	int* buf = (int*)ccmalloc(sizeof(int) * exec_symbol_info_size * 3);
	int* reversed_depth = buf;
	const ccv_nnc_graph_exec_symbol_info_t* const exec_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, 0);
	int i, j;
	// Go reverse order to generate the distance from sink.
	ccv_nnc_graph_visit_for(reversed_visit, exec_symbol_info, node, idx, term) {
		chain_ids[idx] = -1;
		if (!node->outgoings || node->outgoings->rnum == 0)
		{
			reversed_depth[idx] = 0;
			continue;
		}
		const int outgoing = *(int*)ccv_array_get(node->outgoings, 0);
		int depth = reversed_depth[outgoing];
		for (i = 1; i < node->outgoings->rnum; i++)
		{
			const int outgoing = *(int*)ccv_array_get(node->outgoings, i);
			depth = ccv_max(depth, reversed_depth[outgoing]);
		}
		reversed_depth[idx] = depth + 1;
	} ccv_nnc_graph_visit_endfor
	// Go in order to generate chain ids (if there are multiple exits, we use the reverse depth to break the tie).
	// Note that we cannot use depth so-far because then multiple exit nodes are equally good to "inherit" the chain selection.
	int chain_count = 0;
	ccv_nnc_graph_visit_for(visit, exec_symbol_info, node, idx, term) {
		int chain_id = chain_ids[idx];
		if (chain_ids[idx] < 0)
		{
			chain_id = chain_count;
			chain_ids[idx] = chain_id;
			chain_pos[idx] = 1; // The first one in this chain. 1-based index because in sparse matrix, 0 is the default value.
			chain_count += 1;
		}
		if (!node->outgoings || node->outgoings->rnum == 0)
			continue;
		int depth = 0;
		int next_idx = -1;
		for (i = 0; i < node->outgoings->rnum; i++)
		{
			const int outgoing = *(int*)ccv_array_get(node->outgoings, i);
			if (chain_ids[outgoing] < 0 && reversed_depth[outgoing] > depth)
				depth = reversed_depth[outgoing], next_idx = outgoing;
		}
		if (next_idx >= 0)
		{
			chain_ids[next_idx] = chain_id;
			chain_pos[next_idx] = chain_pos[idx] + 1;
		}
	} ccv_nnc_graph_visit_endfor
	ccv_sparse_matrix_t* deps = ccv_sparse_matrix_new(graph->exec_symbol_info->rnum, chain_count, CCV_32S | CCV_C2, CCV_SPARSE_ROW_MAJOR, 0);
	// It logs which pos on that chain we depend on. We can simply compare that with the chain_pos for a node to know if they are ancestors.
#define for_block(x, val) \
	do { \
		if (((int32_t*)val)[0] > 0) \
		{ \
			buf[buf_size * 3] = x; \
			buf[buf_size * 3 + 1] = ((int32_t*)val)[0]; \
			buf[buf_size * 3 + 2] = ((int32_t*)val)[1] + 1; \
			++buf_size; \
		} \
	} while (0)
	int buf_size;
	ccv_nnc_graph_visit_for(visit, exec_symbol_info, node, idx, term) {
		buf_size = 0; /* save all its parent deps to this buffer */
		ccv_sparse_matrix_vector_t* vector = ccv_get_sparse_matrix_vector(deps, idx);
		if (vector)
			CCV_SPARSE_VECTOR_FOREACH(deps, vector, for_block);
		if (!node->outgoings)
			continue;
		const int chain_id = chain_ids[idx];
		const int pos = chain_pos[idx];
		for (i = 0; i < node->outgoings->rnum; i++)
		{
			const int outgoing = *(int*)ccv_array_get(node->outgoings, i);
			const int outgoing_chain_id = chain_ids[outgoing];
			if (outgoing_chain_id != chain_id)
			{
				ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(deps, outgoing, chain_id);
				/* If not found, set, if the current node is the destination node, no need 
				 * set itself as parent of subsequent nodes because its terminal nature. */
				if (!cell.i32 || cell.i32[0] == 0 || cell.i32[0] < pos)
				{
					int p[2] = { pos, 1 };
					ccv_set_sparse_matrix_cell(deps, outgoing, chain_id, &p);
				}
			}
			if (buf_size > 0)
			{
				ccv_sparse_matrix_vector_t* vector = ccv_get_sparse_matrix_vector(deps, outgoing);
				for (j = 0; j < buf_size; j++) /* set with all idx's dependencies as well */
				{
					if (outgoing_chain_id == buf[j * 3]) // We don't need to add as dependency for the same chain.
						continue;
					if (!vector)
					{
						ccv_set_sparse_matrix_cell(deps, outgoing, buf[j * 3], &buf[j * 3 + 1]);
						vector = ccv_get_sparse_matrix_vector(deps, outgoing);
						continue;
					}
					ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell_from_vector(deps, vector, buf[j * 3]);
					/* If not found, set. Otherwise, set to the latest one only if it is later. */
					if (!cell.i32)
						ccv_set_sparse_matrix_cell_from_vector(deps, vector, buf[j * 3], &buf[j * 3 + 1]);
					else if (cell.i32[0] == 0 || cell.i32[0] < buf[j * 3 + 1])
						ccv_set_sparse_matrix_cell_from_vector(deps, vector, buf[j * 3], &buf[j * 3 + 1]);
					else if (cell.i32[0] == buf[j * 3 + 1]) { // If we point to the same one, use the longest.
						int p[2] = { cell.i32[0], ccv_max(buf[j * 3 + 2], cell.i32[1]) };
						ccv_set_sparse_matrix_cell_from_vector(deps, vector, buf[j * 3], &p);
					}
				}
			}
		}
	} ccv_nnc_graph_visit_endfor
#undef for_block
	ccfree(buf);
	ccv_nnc_exec_dep_t exec_dep = {
		.chain_ids = chain_ids,
		.chain_pos = chain_pos,
		.deps = deps
	};
	return exec_dep;
}

static int _ccv_nnc_exec_dep_dist(const ccv_nnc_exec_dep_t exec_dep, const int d, ccv_sparse_matrix_vector_t* const vector, const int dd)
{
	// Check if dd is d's ancestor.
	const int dd_chain_id = exec_dep.chain_ids[dd];
	const int dd_chain_pos = exec_dep.chain_pos[dd];
	if (exec_dep.chain_ids[d] == dd_chain_id)
		return exec_dep.chain_pos[d] - dd_chain_pos;
	const ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell_from_vector(exec_dep.deps, vector, dd_chain_id);
	if (cell.i32 && cell.i32[0] > 0 && cell.i32[0] >= dd_chain_pos)
	{
		// Check if the chain pos is greater than or equal to dd_chain_pos. If it is, it is an ancestor.
		return cell.i32[0] - dd_chain_pos + cell.i32[1];
	}
	return -1;
}

static int _ccv_nnc_exec_dep_check(const ccv_nnc_exec_dep_t exec_dep, const int d, const int dd)
{
	// Check if dd is d's ancestor.
	const int dd_chain_id = exec_dep.chain_ids[dd];
	const int dd_chain_pos = exec_dep.chain_pos[dd];
	if (exec_dep.chain_ids[d] == dd_chain_id)
		return exec_dep.chain_pos[d] > dd_chain_pos;
	const ccv_numeric_data_t cell = ccv_get_sparse_matrix_cell(exec_dep.deps, d, dd_chain_id);
	if (cell.i32 && cell.i32[0] > 0)
	{
		// Check if the chain pos is greater than or equal to dd_chain_pos. If it is, it is an ancestor.
		return cell.i32[0] >= dd_chain_pos;
	}
	return 0;
}

static void _ccv_nnc_exec_dep_free(const ccv_nnc_exec_dep_t exec_dep)
{
	ccfree(exec_dep.chain_ids);
	ccv_matrix_free(exec_dep.deps);
}

typedef struct {
	int okay;
	int original;
	ccv_nnc_tensor_param_t info;
	ccv_array_t* old_conversion_nodes;
	struct {
		ccv_array_t* sources;
		ccv_array_t* nodes;
	} reconversion;
} ccv_nnc_conversion_info_t;

typedef struct {
	ccv_array_t* outgoings;
} ccv_nnc_graph_exec_symbol_reverse_t;

void ccv_nnc_symbolic_graph_memory_reduction(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t* const sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* const destinations, const int destination_size)
{
	// Note all these exec_symbol_info and tensor_symbol_info cannot be accessed once I start to mutate the graph. Therefore, I will do the
	// mutation at the last step, to carefully step away from that possibility.
	ccv_nnc_graph_exec_symbol_info_t* const exec_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, 0);
	ccv_nnc_tensor_symbol_info_t* const tensor_symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccv_array_get(graph->tensor_symbol_info, 0);
	ccv_nnc_graph_visit_t* const visit = ccv_nnc_graph_visit_new(graph, exec_symbol_info, graph->exec_symbol_info->rnum, sources, source_size, destinations, destination_size, 0);
	ccv_nnc_symbolic_graph_symbol_infer(graph, visit, sources, source_size, destinations, destination_size, 0, 0, tensor_symbol_info, exec_symbol_info);
	const int tensor_symbol_info_size = graph->tensor_symbol_info->rnum;
	const int exec_symbol_info_size = graph->exec_symbol_info->rnum;
	uint32_t* const tensor_marked = (uint32_t*)cccalloc(((tensor_symbol_info_size + 31) >> 5) * 2, sizeof(uint32_t));
	uint32_t* const tensor_used = tensor_marked + ((tensor_symbol_info_size + 31) >> 5);
	ccv_nnc_graph_exec_symbol_reverse_t* const reversed_nodes = cccalloc(exec_symbol_info_size, sizeof(ccv_nnc_graph_exec_symbol_reverse_t));
	int i, j, k;
	ccv_nnc_graph_visit_for(visit, exec_symbol_info, node, idx) {
		if (node->flags & CCV_NNC_GRAPH_EXEC_DEAD)
			continue;
		if (node->outgoings)
			for (i = 0; i < node->outgoings->rnum; i++)
			{
				const int d = *(int*)ccv_array_get(node->outgoings, i);
				if (!reversed_nodes[d].outgoings)
					reversed_nodes[d].outgoings = ccv_array_new(sizeof(int), 1, 0);
				ccv_array_add_unique_int(reversed_nodes[d].outgoings, idx);
			}
		if (node->cmd.cmd == CCV_NNC_DATATYPE_CONVERSION_FORWARD && node->output_size >= 1 && node->outputs[0] >= 0)
		{
			const int d = node->outputs[0];
			// If this tensor is alias, or assigned (while loop), or bypassed (case..of), skip.
			if (tensor_symbol_info[d].alias_ref || tensor_symbol_info[d].assign_ref || tensor_symbol_info[d].bypass_ref ||
					tensor_symbol_info[d].r_assign_ref || tensor_symbol_info[d].r_bypass_ref)
				continue;
			tensor_marked[d >> 5] |= (1u << (d & 0x1f));
		} else if (ccv_nnc_cmd_is_backward(node->cmd))
			for (i = 0; i < node->input_size; i++)
			{
				const int d = node->inputs[i];
				if (d >= 0)
					tensor_used[d >> 5] |= (1u << (d & 0x1f));
			}
	} ccv_nnc_graph_visit_endfor
	// If a tensor is marked but never used in backward pass, no need to reduce it.
	_ccv_nnc_remove_unused_from_marked(tensor_used, (tensor_symbol_info_size + 31) >> 5, tensor_marked);
	// If this tensor is pointed to by an alias, we don't want to reconversion.
	for (i = 0; i < tensor_symbol_info_size; i++)
		if (tensor_symbol_info[i].alias_ref)
		{
			const int d = tensor_symbol_info[i].alias_ref - 1;
			// unmark.
			if ((tensor_marked[d >> 5] & (1u << (d & 0x1f))))
				tensor_marked[d >> 5] &= ~(1u << (d & 0x1f));
		}
	ccv_nnc_graph_visit_t* const reversed_visit = ccv_nnc_graph_visit_new(graph, reversed_nodes, exec_symbol_info_size, destinations, destination_size, sources, source_size, 0);
	ccv_nnc_exec_dep_t exec_deps = _ccv_nnc_exec_dep_new(graph, visit, reversed_visit);
	ccv_nnc_graph_visit_free(reversed_visit);
	// Now tensor_marked only contains the tensors that we think beneficial to reconvert. Find the best place to insert conversion.
	ccv_nnc_conversion_info_t* const conversion_info = cccalloc(tensor_symbol_info_size, sizeof(ccv_nnc_conversion_info_t));
	ccv_nnc_graph_visit_for(visit, exec_symbol_info, node, idx) {
		if (node->cmd.cmd == CCV_NNC_DATATYPE_CONVERSION_FORWARD && node->output_size >= 1 && node->outputs[0] >= 0)
		{
			const int d = node->outputs[0];
			if (d >= 0 && (tensor_marked[d >> 5] & (1u << (d & 0x1f))))
			{
				conversion_info[d].original = node->inputs[0];
				if (!conversion_info[d].old_conversion_nodes)
					conversion_info[d].old_conversion_nodes = ccv_array_new(sizeof(int), 0, 0);
				ccv_array_add_unique_int(conversion_info[d].old_conversion_nodes, idx);
			}
		} else if (ccv_nnc_cmd_is_backward(node->cmd))
			for (i = 0; i < node->input_size; i++)
			{
				const int d = node->inputs[i];
				if (d >= 0 && (tensor_marked[d >> 5] & (1u << (d & 0x1f))))
				{
					if (!conversion_info[d].reconversion.nodes)
						conversion_info[d].reconversion.nodes = ccv_array_new(sizeof(int), 0, 0);
					ccv_array_add_unique_int(conversion_info[d].reconversion.nodes, idx);
				}
			}
	} ccv_nnc_graph_visit_endfor
	for (i = 0; i < tensor_symbol_info_size; i++)
	{
		if (!conversion_info[i].reconversion.nodes)
			continue;
		// Check to see if it is beneficial for reconversion (i.e. the output is larger than the input).
		const int original_datatype = tensor_symbol_info[conversion_info[i].original].info.datatype;
		const int converted_datatype = tensor_symbol_info[i].info.datatype;
		if (CCV_GET_DATA_TYPE_SIZE(original_datatype) >= CCV_GET_DATA_TYPE_SIZE(converted_datatype))
			continue;
		// If we have more than one destination, need to find the common ancestor.
		ccv_array_t* const nodes = conversion_info[i].reconversion.nodes;
		ccv_array_t* const old_conversion_nodes = conversion_info[i].old_conversion_nodes;
		assert(nodes->rnum > 0);
		assert(old_conversion_nodes && old_conversion_nodes->rnum > 0);
		int flag = 0;
		for (j = 0; j < nodes->rnum; j++)
		{
			const int d = *(int*)ccv_array_get(nodes, j);
			ccv_sparse_matrix_vector_t* const vector = ccv_get_sparse_matrix_vector(exec_deps.deps, d);
			assert(vector);
			for (k = 0; k < old_conversion_nodes->rnum; k++)
			{
				const int dd = *(int*)ccv_array_get(old_conversion_nodes, k);
				const int dist = _ccv_nnc_exec_dep_dist(exec_deps, d, vector, dd);
				if (dist >= 0 && dist <= 3)
					flag = 1;
			}
			if (flag)
				break;
		}
		// If there is no need to reconvert. Abort the whole thing.
		if (flag)
			continue;
		ccv_array_t* const reconversion_sources = ccv_array_new(sizeof(int), 0, 0);
		for (j = 0; j < nodes->rnum; j++)
		{
			const int d = *(int*)ccv_array_get(nodes, j);
			ccv_array_t* const outgoings = reversed_nodes[d].outgoings;
			if (!outgoings)
				continue;
			int x, y;
			for (x = 0; x < outgoings->rnum; x++)
			{
				const int dd = *(int*)ccv_array_get(outgoings, x);
				int flag = 0;
				for (y = 0; !flag && y < nodes->rnum; y++)
				{
					if (j == y)
						continue;
					const int ddd = *(int*)ccv_array_get(nodes, y);
					// If the outgoing is one of the nodes, we cannot add it as source.
					if (ddd == dd)
					{
						flag = 1;
						continue;
					}
					// Check dependencies, if there is a dependency from y node to dd, dd cannot be source.
					const int checked = _ccv_nnc_exec_dep_check(exec_deps, dd, ddd);
					if (checked)
						flag = 1;
				}
				if (!flag)
					ccv_array_add_unique_int(reconversion_sources, dd);
			}
		}
		// If there is no sources. Abort the whole thing.
		if (reconversion_sources->rnum == 0)
		{
			ccv_array_free(reconversion_sources);
			continue;
		}
		// Mark it as ready to be compressed.
		conversion_info[i].reconversion.sources = reconversion_sources;
		conversion_info[i].info = tensor_symbol_info[i].info;
		conversion_info[i].okay = 1;
	}
	// Do the graph mutation now based on the conversion info.
	for (i = 0; i < tensor_symbol_info_size; i++)
		if (conversion_info[i].okay)
		{
			const ccv_nnc_tensor_symbol_t reconverted = ccv_nnc_tensor_symbol_new(graph, conversion_info[i].info, 0);
			const ccv_nnc_tensor_symbol_t original = (ccv_nnc_tensor_symbol_t){
				.graph = graph,
				.d = conversion_info[i].original
			};
			const ccv_nnc_graph_exec_symbol_t reconversion_node = ccv_nnc_graph_exec_symbol_new(graph, CMD_DATATYPE_CONVERSION_FORWARD(), TENSOR_SYMBOL_LIST(original), TENSOR_SYMBOL_LIST(reconverted), 0);
			ccv_array_t* const nodes = conversion_info[i].reconversion.nodes;
			assert(nodes && nodes->rnum > 0);
			ccv_array_t* const sources = conversion_info[i].reconversion.sources;
			assert(sources && sources->rnum > 0);
			for (j = 0; j < sources->rnum; j++)
			{
				const int d = *(int*)ccv_array_get(sources, j);
				ccv_nnc_graph_exec_symbol_concat(graph, (ccv_nnc_graph_exec_symbol_t){
					.graph = graph,
					.d = d,
				}, reconversion_node);
			}
			for (j = 0; j < nodes->rnum; j++)
			{
				const int d = *(int*)ccv_array_get(nodes, j);
				ccv_nnc_graph_exec_symbol_concat(graph, reconversion_node, (ccv_nnc_graph_exec_symbol_t){
					.graph = graph,
					.d = d
				});
				ccv_nnc_graph_exec_symbol_info_t* const destination_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, d);
				for (k = 0; k < destination_info->input_size; k++)
					if (destination_info->inputs[k] == i)
						destination_info->inputs[k] = reconverted.d;
			}
		}
	ccv_nnc_graph_visit_free(visit);
	_ccv_nnc_exec_dep_free(exec_deps);
	ccfree(tensor_marked);
	for (i = 0; i < tensor_symbol_info_size; i++)
	{
		if (conversion_info[i].old_conversion_nodes)
			ccv_array_free(conversion_info[i].old_conversion_nodes);
		if (conversion_info[i].reconversion.nodes)
			ccv_array_free(conversion_info[i].reconversion.nodes);
		if (conversion_info[i].reconversion.sources)
			ccv_array_free(conversion_info[i].reconversion.sources);
	}
	for (i = 0; i < exec_symbol_info_size; i++)
		if (reversed_nodes[i].outgoings)
			ccv_array_free(reversed_nodes[i].outgoings);
	ccfree(reversed_nodes);
	ccfree(conversion_info);
}
