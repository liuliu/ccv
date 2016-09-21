/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

/**********************************************************
 * CCV - Neural Network Collection
 **********************************************************/

#ifndef GUARD_ccv_nnc_internal_h
#define GUARD_ccv_nnc_internal_h

// Defines common graph visit macro
// The visitor function / macro takes parameter visitor(node_type* node, int index, int level, int term);
#define CCV_NNC_GRAPH_VISIT(_graph, nodes, node_size, sources, source_size, destinations, destination_size, visitor) \
	do { \
		/* Use the same data structure to do topological ordering. */ \
		typedef struct { \
			int8_t d; /* tag if this is the destination node. */ \
			int8_t r; /* tag if this is reached as destination node. */ \
			int32_t c; /* number of incoming edges. */ \
		} ccv_nnc_incoming_t; \
		/* Statistics of how many incoming edges for all nodes of a graph. */ \
		ccv_nnc_incoming_t* _incomings_ = (ccv_nnc_incoming_t*)ccmalloc(sizeof(ccv_nnc_incoming_t) * (node_size) + sizeof(int32_t) * (node_size) * 2); \
		memset(_incomings_, 0, sizeof(ccv_nnc_incoming_t) * (node_size)); \
		int _i_, _j_; \
		for (_i_ = 0; _i_ < (node_size); _i_++) \
		{ \
			if ((nodes)[_i_].outgoings) \
				for (_j_ = 0; _j_ < (nodes)[_i_].outgoings->rnum; _j_++) \
					++_incomings_[*(int*)ccv_array_get((nodes)[_i_].outgoings, _j_)].c; \
		} \
		/* After we have that statistics, we can do topsort and run the command. */ \
		int32_t* _exists_[2] = { \
			(int32_t*)(_incomings_ + (node_size)), \
			(int32_t*)(_incomings_ + (node_size)) + (node_size), \
		}; \
		for (_i_ = 0; _i_ < (destination_size); _i_++) \
		{ \
			assert((destinations)[_i_].graph == _graph); \
			/* tagging destination nodes. */ \
			_incomings_[(destinations)[_i_].d].d = 1; \
		} \
		for (_i_ = 0; _i_ < (source_size); _i_++) \
		{ \
			assert((sources)[_i_].graph == _graph); \
			_exists_[0][_i_] = (sources)[_i_].d; \
		} \
		int _exist_size_[2] = { \
			(source_size), \
			0, \
		}; \
		int _p_ = 0, _q_ = 1, _k_ = 0, _d_ = 0; /* ping, pong swap. */ \
		while (_exist_size_[_p_] > 0) \
		{ \
			_exist_size_[_q_] = 0; \
			for (_i_ = 0; _i_ < _exist_size_[_p_]; _i_++) \
			{ \
				visitor(((nodes) + _exists_[_p_][_i_]), (_exists_[_p_][_i_]), _k_, (_incomings_[_exists_[_p_][_i_]].d)); \
				/* mark as reached */ \
				if (_incomings_[_exists_[_p_][_i_]].d) \
				{ \
					++_d_; \
					_incomings_[_exists_[_p_][_i_]].r = 1; \
				} \
				if ((nodes)[_exists_[_p_][_i_]].outgoings) \
					for (_j_ = 0; _j_ < (nodes)[_exists_[_p_][_i_]].outgoings->rnum; _j_++) \
					{ \
						int d = *(int*)ccv_array_get((nodes)[_exists_[_p_][_i_]].outgoings, _j_); \
						--_incomings_[d].c; \
						/* If all incoming edges are consumed, and not all destination node are computed, push it into next round */ \
						if (_incomings_[d].c == 0 && _d_ < (destination_size)) \
						{ \
							_exists_[_q_][_exist_size_[_q_]] = d; \
							++_exist_size_[_q_]; \
						} \
					} \
			} \
			/* swap p and q. */ \
			CCV_SWAP(_p_, _q_, _i_ /* using i as temp holder */); \
			++_k_; \
		} \
		for (_i_ = 0; _i_ < (destination_size); _i_++) \
		{ \
			assert((destinations)[_i_].graph == _graph); \
			/* skip if this is already reached. */ \
			if (_incomings_[(destinations)[_i_].d].r) \
				continue; \
			/* this destination node should have every incoming nodes consumed. */ \
			assert(_incomings_[(destinations)[_i_].d].c == 0); \
			/* fetch the info for destination node and exec current node. */ \
			visitor(((nodes) + (destinations)[_i_].d), ((destinations)[_i_].d), _k_, (_incomings_[(destinations)[_i_].d].d)); \
		} \
		ccfree(_incomings_); \
	} while (0)

#endif
