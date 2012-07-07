#include "ccv.h"
#include "ccv_internal.h"

#define CHI_TABLE_SIZE (400)

static double chitab3[] = { 0,  0.0150057,  0.0239478,  0.0315227,
							0.0383427,  0.0446605,  0.0506115,  0.0562786,
							0.0617174,  0.0669672,  0.0720573,  0.0770099,
							0.081843,  0.0865705,  0.0912043,  0.0957541,
							0.100228,  0.104633,  0.108976,  0.113261,
							0.117493,  0.121676,  0.125814,  0.12991,
							0.133967,  0.137987,  0.141974,  0.145929,
							0.149853,  0.15375,  0.15762,  0.161466,
							0.165287,  0.169087,  0.172866,  0.176625,
							0.180365,  0.184088,  0.187794,  0.191483,
							0.195158,  0.198819,  0.202466,  0.2061,
							0.209722,  0.213332,  0.216932,  0.220521,
							0.2241,  0.22767,  0.231231,  0.234783,
							0.238328,  0.241865,  0.245395,  0.248918,
							0.252435,  0.255947,  0.259452,  0.262952,
							0.266448,  0.269939,  0.273425,  0.276908,
							0.280386,  0.283862,  0.287334,  0.290803,
							0.29427,  0.297734,  0.301197,  0.304657,
							0.308115,  0.311573,  0.315028,  0.318483,
							0.321937,  0.32539,  0.328843,  0.332296,
							0.335749,  0.339201,  0.342654,  0.346108,
							0.349562,  0.353017,  0.356473,  0.35993,
							0.363389,  0.366849,  0.37031,  0.373774,
							0.377239,  0.380706,  0.384176,  0.387648,
							0.391123,  0.3946,  0.39808,  0.401563,
							0.405049,  0.408539,  0.412032,  0.415528,
							0.419028,  0.422531,  0.426039,  0.429551,
							0.433066,  0.436586,  0.440111,  0.44364,
							0.447173,  0.450712,  0.454255,  0.457803,
							0.461356,  0.464915,  0.468479,  0.472049,
							0.475624,  0.479205,  0.482792,  0.486384,
							0.489983,  0.493588,  0.4972,  0.500818,
							0.504442,  0.508073,  0.511711,  0.515356,
							0.519008,  0.522667,  0.526334,  0.530008,
							0.533689,  0.537378,  0.541075,  0.54478,
							0.548492,  0.552213,  0.555942,  0.55968,
							0.563425,  0.56718,  0.570943,  0.574715,
							0.578497,  0.582287,  0.586086,  0.589895,
							0.593713,  0.597541,  0.601379,  0.605227,
							0.609084,  0.612952,  0.61683,  0.620718,
							0.624617,  0.628526,  0.632447,  0.636378,
							0.64032,  0.644274,  0.648239,  0.652215,
							0.656203,  0.660203,  0.664215,  0.668238,
							0.672274,  0.676323,  0.680384,  0.684457,
							0.688543,  0.692643,  0.696755,  0.700881,
							0.70502,  0.709172,  0.713339,  0.717519,
							0.721714,  0.725922,  0.730145,  0.734383,
							0.738636,  0.742903,  0.747185,  0.751483,
							0.755796,  0.760125,  0.76447,  0.768831,
							0.773208,  0.777601,  0.782011,  0.786438,
							0.790882,  0.795343,  0.799821,  0.804318,
							0.808831,  0.813363,  0.817913,  0.822482,
							0.827069,  0.831676,  0.836301,  0.840946,
							0.84561,  0.850295,  0.854999,  0.859724,
							0.864469,  0.869235,  0.874022,  0.878831,
							0.883661,  0.888513,  0.893387,  0.898284,
							0.903204,  0.908146,  0.913112,  0.918101,
							0.923114,  0.928152,  0.933214,  0.938301,
							0.943413,  0.94855,  0.953713,  0.958903,
							0.964119,  0.969361,  0.974631,  0.979929,
							0.985254,  0.990608,  0.99599,  1.0014,
							1.00684,  1.01231,  1.01781,  1.02335,
							1.02891,  1.0345,  1.04013,  1.04579,
							1.05148,  1.05721,  1.06296,  1.06876,
							1.07459,  1.08045,  1.08635,  1.09228,
							1.09826,  1.10427,  1.11032,  1.1164,
							1.12253,  1.1287,  1.1349,  1.14115,
							1.14744,  1.15377,  1.16015,  1.16656,
							1.17303,  1.17954,  1.18609,  1.19269,
							1.19934,  1.20603,  1.21278,  1.21958,
							1.22642,  1.23332,  1.24027,  1.24727,
							1.25433,  1.26144,  1.26861,  1.27584,
							1.28312,  1.29047,  1.29787,  1.30534,
							1.31287,  1.32046,  1.32812,  1.33585,
							1.34364,  1.3515,  1.35943,  1.36744,
							1.37551,  1.38367,  1.39189,  1.4002,
							1.40859,  1.41705,  1.42561,  1.43424,
							1.44296,  1.45177,  1.46068,  1.46967,
							1.47876,  1.48795,  1.49723,  1.50662,
							1.51611,  1.52571,  1.53541,  1.54523,
							1.55517,  1.56522,  1.57539,  1.58568,
							1.59611,  1.60666,  1.61735,  1.62817,
							1.63914,  1.65025,  1.66152,  1.67293,
							1.68451,  1.69625,  1.70815,  1.72023,
							1.73249,  1.74494,  1.75757,  1.77041,
							1.78344,  1.79669,  1.81016,  1.82385,
							1.83777,  1.85194,  1.86635,  1.88103,
							1.89598,  1.91121,  1.92674,  1.94257,
							1.95871,  1.97519,  1.99201,  2.0092,
							2.02676,  2.04471,  2.06309,  2.08189,
							2.10115,  2.12089,  2.14114,  2.16192,
							2.18326,  2.2052,  2.22777,  2.25101,
							2.27496,  2.29966,  2.32518,  2.35156,
							2.37886,  2.40717,  2.43655,  2.46709,
							2.49889,  2.53206,  2.56673,  2.60305,
							2.64117,  2.6813,  2.72367,  2.76854,
							2.81623,  2.86714,  2.92173,  2.98059,
							3.04446,  3.1143,  3.19135,  3.27731,
							3.37455,  3.48653,  3.61862,  3.77982,
							3.98692,  4.2776,  4.77167,  133.333 };

static void _ccv_mscr_chi(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, int dx, int dy)
{
	assert((dx == 1 && dy == 0) || (dx == 0 && dy == 1));
	ccv_declare_derived_signature(sig, a->sig != 0, ccv_sign_with_format(64, "ccv_mscr_chi(%d,%d)", dx, dy), a->sig, 0);
	type = (CCV_GET_DATA_TYPE(type) == CCV_64F) ? CCV_64F | CCV_C1 : CCV_32F | CCV_C1;
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, a->rows - dy, a->cols - dx, CCV_C1 | CCV_32F | CCV_64F, type, sig);
	ccv_object_return_if_cached(, db);
	int i, j, k, ch = CCV_GET_CHANNEL(a->type);
	unsigned char *aptr = a->data.u8;
	unsigned char *bptr = db->data.u8;
	if (dx == 1)
	{
#define for_block(_for_set_b, _for_get_a) \
		for (i = 0; i < db->rows; i++) \
		{ \
			for (j = 0; j < db->cols; j++) \
			{ \
				double v = (double)((_for_get_a(aptr, j * ch + ch, 0) - _for_get_a(aptr, j * ch, 0)) * (_for_get_a(aptr, j * ch + ch, 0) - _for_get_a(aptr, j * ch, 0))) / (double)(_for_get_a(aptr, j * ch, 0) + _for_get_a(aptr, j * ch + ch, 0) + 1e-10); \
				for (k = 1; k < ch; k++) \
					v += (double)((_for_get_a(aptr, j * ch + ch + k, 0) - _for_get_a(aptr, j * ch + k, 0)) * (_for_get_a(aptr, j * ch + ch + k, 0) - _for_get_a(aptr, j * ch + k, 0))) / (double)(_for_get_a(aptr, j * ch + k, 0) + _for_get_a(aptr, j * ch + ch + k, 0) + 1e-10); \
				_for_set_b(bptr, j, v, 0); \
			} \
			aptr += a->step; \
			bptr += db->step; \
		}
		ccv_matrix_setter(db->type, ccv_matrix_getter, a->type, for_block);
#undef for_block
	} else if (dy == 1) {
#define for_block(_for_set_b, _for_get_a) \
		for (i = 0; i < db->rows; i++) \
		{ \
			for (j = 0; j < db->cols; j++) \
			{ \
				double v = (double)((_for_get_a(aptr, j * ch + a->step, 0) - _for_get_a(aptr, j * ch, 0)) * (_for_get_a(aptr, j * ch + a->step, 0) - _for_get_a(aptr, j * ch, 0))) / (double)(_for_get_a(aptr, j * ch, 0) + _for_get_a(aptr, j * ch + a->step, 0) + 1e-10); \
				for (k = 1; k < ch; k++) \
					v += (double)((_for_get_a(aptr, j * ch + a->step + k, 0) - _for_get_a(aptr, j * ch + k, 0)) * (_for_get_a(aptr, j * ch + a->step + k, 0) - _for_get_a(aptr, j * ch + k, 0))) / (double)(_for_get_a(aptr, j * ch + k, 0) + _for_get_a(aptr, j * ch + a->step + k, 0) + 1e-10); \
				_for_set_b(bptr, j, v, 0); \
			} \
			aptr += a->step; \
			bptr += db->step; \
		}
		ccv_matrix_setter(db->type, ccv_matrix_getter, a->type, for_block);
#undef for_block
	}
}

struct ccv_mscr_area;

typedef struct ccv_mscr_node {
	struct ccv_mscr_node* shortcut;
	// double link list
	struct ccv_mscr_node* prev;
	struct ccv_mscr_node* next;
	int rank;
	int reinit;
	int step_now;
	int last_size;
	int size;
	int prev_size;
	double chi;
	double prev_chi;
	double min_slope;
	ccv_point_t point;
	ccv_point_t min_point;
	ccv_point_t max_point;
	int last_mscr_area;
	int mscr_area;
} ccv_mscr_node_t;

typedef struct {
	double chi;
	ccv_mscr_node_t* node[2];
} ccv_mscr_edge_t;

typedef struct ccv_mscr_area {
	ccv_mscr_node_t* head;
	ccv_mscr_node_t* tail;
	double margin;
	int size;
	int seq_no;
} ccv_mscr_area_t;

#define less_than(e1, e2, aux) ((e1).chi < (e2).chi)
static CCV_IMPLEMENT_QSORT(_ccv_mscr_edge_qsort, ccv_mscr_edge_t, less_than)
#undef less_than

static void _ccv_mscr_init_node(ccv_mscr_node_t* node, int x, int y)
{
	node->rank = 0;
	node->reinit = 0x7FFFFFFF;
	node->point.x = x;
	node->point.y = y;
	node->min_point = node->max_point = node->point;
	node->step_now = 0;
	node->chi = node->prev_chi = 0;
	node->last_size = node->size = node->prev_size = 1;
	node->prev = node->next = node->shortcut = node; // endless double link list
	node->last_mscr_area = node->mscr_area = -1;
}

static ccv_mscr_node_t* _ccv_mscr_find_root(ccv_mscr_node_t* node)
{
	ccv_mscr_node_t* prev = node;
	ccv_mscr_node_t* root;
	for (;;)
	{
		root = node->shortcut;
		// use the shortcut as a temporary variable to record previous node,
		// we will update it again shortly with the real shortcut to root
		node->shortcut = prev;
		if (root == node)
			break;
		prev = node;
		node = root;
	}
	for (;;)
	{
		prev = node->shortcut;
		node->shortcut = root;
		if (prev == node)
			break;
		node = prev;
	}
	return root;
}

static void _ccv_mscr(ccv_dense_matrix_t* a, ccv_dense_matrix_t* h, ccv_dense_matrix_t* b, ccv_array_t* seq, ccv_mser_param_t params)
{
	ccv_dense_matrix_t* dx = 0;
	_ccv_mscr_chi(a, &dx, CCV_32F, 1, 0);
	ccv_dense_matrix_t* bdx = 0;
	ccv_blur(dx, &bdx, 0, params.edge_blur_sigma);
	ccv_matrix_free(dx);
	ccv_dense_matrix_t* dy = 0;
	_ccv_mscr_chi(a, &dy, CCV_32F, 0, 1);
	ccv_dense_matrix_t* bdy = 0;
	ccv_blur(dy, &bdy, 0, params.edge_blur_sigma);
	ccv_matrix_free(dy);
	int i, j;
	ccv_mscr_node_t* node = (ccv_mscr_node_t*)ccmalloc(sizeof(ccv_mscr_node_t) * a->rows * a->cols);
	ccv_mscr_node_t* pnode = node;
	for (i = 0; i < a->rows; i++)
		for (j = 0; j < a->cols; j++)
		{
			_ccv_mscr_init_node(pnode, j, i);
			++pnode;
		}
	ccv_mscr_edge_t* edge = (ccv_mscr_edge_t*)ccmalloc(sizeof(ccv_mscr_edge_t) * (bdx->rows * bdx->cols + bdy->rows * bdy->cols));
	ccv_mscr_edge_t* pedge = edge;
	/* generate edge graph and sort them */
	double mean = 0;
	pnode = node;
	float* bdx_ptr = bdx->data.f32;
	assert(bdx->cols == a->cols - 1);
	for (i = 0; i < bdx->rows; i++)
	{
		for (j = 0; j < bdx->cols; j++)
		{
			mean += pedge->chi = bdx_ptr[j];
			pedge->node[0] = pnode + j;
			pedge->node[1] = pnode + j + 1;
			++pedge;
		}
		pnode += a->cols;
		bdx_ptr += bdx->cols;
	}
	pnode = node;
	float* bdy_ptr = bdy->data.f32;
	assert(bdy->rows == a->rows - 1);
	for (i = 0; i < bdy->rows; i++)
	{
		for (j = 0; j < bdy->cols; j++)
		{
			mean += pedge->chi = bdy_ptr[j];
			pedge->node[0] = pnode + j;
			pedge->node[1] = pnode + a->cols + j;
			++pedge;
		}
		pnode += a->cols;
		bdy_ptr += bdy->cols;
	}
	_ccv_mscr_edge_qsort(edge, bdx->rows * bdx->cols + bdy->rows * bdy->cols, 0);
	ccv_matrix_free(bdx);
	ccv_matrix_free(bdy);
	/* evolute on the edge graph */
	int seq_no = 0;
	pedge = edge;
	ccv_mscr_edge_t* edge_end = edge + bdx->rows * bdx->cols + bdy->rows * bdy->cols;
	mean /= (double)(bdx->rows * bdx->cols + bdy->rows * bdy->cols);
	ccv_array_t* mscr_area_list = ccv_array_new(sizeof(ccv_mscr_area_t), 64, 0);
	for (i = 0; (i < params.max_evolution) && (pedge < edge_end); i++)
	{
		double dk = (double)i / (double)params.max_evolution * (CHI_TABLE_SIZE - 1);
		int k = (int)dk;
		double rk = dk - k;
		double thres = mean * (chitab3[k] * (1.0 - rk) + chitab3[k + 1] * rk);
		// to process all the edges in the list that chi < thres
		while (pedge < edge_end && pedge->chi < thres)
		{
			ccv_mscr_node_t* root0 = _ccv_mscr_find_root(pedge->node[0]);
			ccv_mscr_node_t* root1 = _ccv_mscr_find_root(pedge->node[1]);
			if (root0 != root1)
			{
				if (root1->rank > root0->rank)
				{
					ccv_mscr_node_t* root = root0;
					root0 = root1;
					root1 = root;
				} else if (root1->rank == root0->rank) {
					if (root1->size > root0->size)
					{
						ccv_mscr_node_t* root = root0;
						root0 = root1;
						root1 = root;
					}
					++root0->rank;
				}
				if (root0->step_now < i)
				/* faithful record the last size for area threshold check */
				{
					root0->last_size = root0->size;
					root0->step_now = i;
				}
				root1->shortcut = root0;
				root0->size += root1->size;
				root0->min_point.x = ccv_min(root0->min_point.x, root1->min_point.x);
				root0->min_point.y = ccv_min(root0->min_point.y, root1->min_point.y);
				root0->max_point.x = ccv_max(root0->max_point.x, root1->max_point.x);
				root0->max_point.y = ccv_max(root0->max_point.y, root1->max_point.y);
				root0->prev->next = root1;
				ccv_mscr_node_t* prev = root0->prev;
				root0->prev = root1->prev;
				root1->prev->next = root0; // consider self-referencing
				root1->prev = prev;
				/* insert one endless double link list to another, see illustration:
				 * 0->1->2->3->4->5->0
				 * a->b->c->d->a
				 * set 5.next (0.prev.next) point to a
				 * set 0.prev point to d
				 * set d.next (a.prev.next) point to 0
				 * set a.prev point to 5
				 * the result endless double link list will be:
				 * 0->1->2->3->4->5->a->b->c->d->0 */
				if (root0->size > root0->last_size * params.area_threshold)
				// this is one condition check for Equation (10) */
				{
					if (root0->mscr_area >= 0)
					{
						ccv_mscr_area_t* mscr_area = (ccv_mscr_area_t*)ccv_array_get(mscr_area_list, root0->mscr_area);
						/* Page (4), compute the margin between the reinit point and before the next reinit point */
						mscr_area->margin = root0->chi - root0->prev_chi;
						if (mscr_area->margin > params.min_margin &&
							root0->max_point.y - root0->min_point.y > 1 && // extreme rectangle rule
							root0->max_point.x - root0->min_point.x > 1)
							mscr_area->seq_no = ++seq_no;
						root0->mscr_area = -1;
					}
					root0->prev_size = root0->size;
					root0->prev_chi = pedge->chi;
					root0->reinit = i;
					root0->min_slope = DBL_MAX;
				}
				root0->chi = pedge->chi;
				if (i > root0->reinit)
				{
					double slope = (double)(root0->size - root0->prev_size) / (root0->chi - root0->prev_chi);
					if (slope < root0->min_slope)
					{
						if (i > root0->reinit + 1 && root0->size >= params.min_area && root0->size <= params.max_area)
						{
							if (root0->last_mscr_area < 0 || /* I added the diversity check for MSCR, as most MSER algorithm does */
								(double)(root0->size - ((ccv_mscr_area_t*)ccv_array_get(mscr_area_list, root0->last_mscr_area))->size) / (double)((ccv_mscr_area_t*)ccv_array_get(mscr_area_list, root0->last_mscr_area))->size > params.min_diversity)
							{
								if (root0->mscr_area >= 0)
								{
									ccv_mscr_area_t* mscr_area = (ccv_mscr_area_t*)ccv_array_get(mscr_area_list, root0->mscr_area);
									mscr_area->head = root0;
									mscr_area->tail = root0->prev;
									mscr_area->margin = 0;
									mscr_area->size = root0->size;
									mscr_area->seq_no = 0;
								} else {
									ccv_mscr_area_t mscr_area = {
										.head = root0,
										.tail = root0->prev,
										.margin = 0,
										.size = root0->size,
										.seq_no = 0,
									};
									root0->mscr_area = root0->last_mscr_area = mscr_area_list->rnum;
									ccv_array_push(mscr_area_list, &mscr_area);
								}
							}
						}
						root0->min_slope = slope;
					}
				}
			}
			++pedge;
		}
	}
	assert(seq->rsize == sizeof(ccv_mser_keypoint_t));
	ccv_zero(b);
	unsigned char* b_ptr = b->data.u8;
#define for_block(_, _for_set, _for_get) \
	for (i = 0; i < mscr_area_list->rnum; i++) \
	{ \
		ccv_mscr_area_t* mscr_area = (ccv_mscr_area_t*)ccv_array_get(mscr_area_list, i); \
		if (mscr_area->seq_no > 0) \
		{ \
			ccv_mscr_node_t* node = mscr_area->head; \
			ccv_mser_keypoint_t mser_keypoint = { \
				.size = mscr_area->size, \
				.keypoint = node->point, \
				.m10 = 0, .m01 = 0, .m11 = 0, \
				.m20 = 0, .m02 = 0, \
			}; \
			ccv_point_t min_point = node->point, \
						max_point = node->point; \
			for (j = 0; j < mscr_area->size; j++) \
			{ \
				if (_for_get(b_ptr + node->point.y * b->step, node->point.x, 0) == 0) \
					_for_set(b_ptr + node->point.y * b->step, node->point.x, mscr_area->seq_no, 0); \
				min_point.x = ccv_min(min_point.x, node->point.x); \
				min_point.y = ccv_min(min_point.y, node->point.y); \
				max_point.x = ccv_max(max_point.x, node->point.x); \
				max_point.y = ccv_max(max_point.y, node->point.y); \
				node = node->next; \
			} \
			assert(node->prev == mscr_area->tail); /* endless double link list */ \
			mser_keypoint.rect = ccv_rect(min_point.x, min_point.y, max_point.x - min_point.x + 1, max_point.y - min_point.y + 1); \
			ccv_array_push(seq, &mser_keypoint); \
		} \
	}
	ccv_matrix_setter_getter(b->type, for_block);
#undef for_block
	ccv_array_free(mscr_area_list);
	ccfree(edge);
	ccfree(node);
}

static void _ccv_linear_mser(ccv_dense_matrix_t* a, ccv_dense_matrix_t* h, ccv_dense_matrix_t* b, ccv_array_t* seq, ccv_mser_param_t params)
{
}

ccv_array_t* ccv_mser(ccv_dense_matrix_t* a, ccv_dense_matrix_t* h, ccv_dense_matrix_t** b, int type, ccv_mser_param_t params)
{
	uint64_t psig = ccv_cache_generate_signature((const char*)&params, sizeof(params), 0);
	ccv_declare_derived_signature_case(bsig, ccv_sign_with_literal("ccv_mser(matrix)"), ccv_sign_if(h == 0 && a->sig != 0, psig, a->sig, 0), ccv_sign_if(h != 0 && a->sig != 0 && h->sig != 0, psig, a->sig, h->sig, 0));
	ccv_declare_derived_signature_case(rsig, ccv_sign_with_literal("ccv_mser(array)"), ccv_sign_if(h == 0 && a->sig != 0, psig, a->sig, 0), ccv_sign_if(h != 0 && a->sig != 0 && h->sig != 0, psig, a->sig, h->sig, 0));
	type = (type == 0) ? CCV_32S | CCV_C1 : CCV_GET_DATA_TYPE(type) | CCV_C1;
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, a->rows, a->cols, CCV_C1 | CCV_ALL_DATA_TYPE, type, bsig);
	ccv_array_t* seq = ccv_array_new(sizeof(ccv_mser_keypoint_t), 64, rsig);
	ccv_object_return_if_cached(seq, db, seq);
	ccv_revive_object_if_cached(db, seq);
	if (CCV_GET_CHANNEL(a->type) > 1)
		_ccv_mscr(a, h, db, seq, params);
	else
		_ccv_linear_mser(a, h, db, seq, params);
	return seq;
}
