#include "ccv.h"
#include "ccv_internal.h"
#include "3rdparty/dsfmt/dSFMT.h"

ccv_ferns_t* ccv_ferns_new(int structs, int features, int scales, ccv_size_t* sizes)
{
	assert(structs > 0 && features > 0 && scales > 0);
	int posteriors = 1 << features;
	ccv_ferns_t* ferns = (ccv_ferns_t*)ccmalloc(sizeof(ccv_ferns_t) + sizeof(ccv_point_t) * (structs * features * scales * 2 - 1) + sizeof(float) * structs * posteriors * 2 + sizeof(int) * structs * posteriors * 2);
	ferns->structs = structs;
	ferns->features = features;
	ferns->scales = scales;
	ferns->posteriors = posteriors;
	ferns->cnum[0] = ferns->cnum[1] = 0;
	ferns->posterior = (float*)((uint8_t*)(ferns + 1) + sizeof(ccv_point_t) * (structs * features * scales * 2 - 1));
	// now only for 2 classes
	ferns->rnum = (int*)(ferns->posterior + structs * posteriors * 2);
	memset(ferns->rnum, 0, sizeof(int) * structs * posteriors * 2);
	int i, j, k;
	float log5 = logf(0.5);
	for (i = 0; i < structs * posteriors * 2; i++)
		ferns->posterior[i] = log5; // initialize to 0.5
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, (uint32_t)ferns);
	for (i = 0; i < structs; i++)
	{
		for (k = 0; k < features; k++)
		{
			double x1f, y1f, x2f, y2f;
			// to restrict the space of ferns feature
			if (dsfmt_genrand_uint32(&dsfmt) & 0x01)
			{
				do {
					x1f = dsfmt_genrand_close_open(&dsfmt);
					x2f = dsfmt_genrand_close_open(&dsfmt);
					y1f = y2f = dsfmt_genrand_close_open(&dsfmt);
				} while (fabs(x1f - x2f) >= 0.2);
			} else {
				do {
					x1f = x2f = dsfmt_genrand_close_open(&dsfmt);
					y1f = dsfmt_genrand_close_open(&dsfmt);
					y2f = dsfmt_genrand_close_open(&dsfmt);
				} while (fabs(y1f - y2f) >= 0.2);
			}
			for (j = 0; j < scales; j++)
			{
				ferns->fern[(j * structs * features + i * features + k) * 2] = ccv_point((int)(x1f * sizes[j].width), (int)(y1f * sizes[j].height));
				ferns->fern[(j * structs * features + i * features + k) * 2 + 1] = ccv_point((int)(x2f * sizes[j].width), (int)(y2f * sizes[j].height));
			}
		}
	}
	ferns->threshold = 0;
	return ferns;
}

void ccv_ferns_feature(ccv_ferns_t* ferns, ccv_dense_matrix_t* a, int scale, uint32_t* fern)
{
	ccv_point_t* fern_feature = ferns->fern + scale * ferns->structs * ferns->features * 2;
	int i, j;
	unsigned char* a_ptr = a->data.u8;
	assert(CCV_GET_CHANNEL(a->type) == CCV_C1);
#define for_block(_, _for_get) \
	for (i = 0; i < ferns->structs; i++) \
	{ \
		uint32_t leaf = 0; \
		for (j = 0; j < ferns->features; j++) \
		{ \
			if (_for_get(a_ptr + fern_feature[0].y * a->step, fern_feature[0].x, 0) > _for_get(a_ptr + fern_feature[1].y * a->step, fern_feature[1].x, 0)) \
				leaf = (leaf << 1) | 1; \
			else \
				leaf = leaf << 1; \
			fern_feature += 2; \
		} \
		fern[i] = leaf; \
	}
	ccv_matrix_getter(a->type, for_block);
#undef for_block
}

void ccv_ferns_correct(ccv_ferns_t* ferns, uint32_t* fern, int c, int repeat)
{
	assert(c == 0 || c == 1);
	assert(repeat >= 0);
	repeat += 1;
	int i;
	int* cnum = ferns->cnum;
	int* rnum = ferns->rnum;
	float* post = ferns->posterior;
	cnum[c] += repeat;
	float cw[] = {
		1.0 / (cnum[0] + 1),
		1.0 / (cnum[1] + 1),
	};
	for (i = 0; i < ferns->structs; i++)
	{
		uint32_t k = fern[i];
		rnum[k * 2 + c] += repeat;
		// needs to compute the log of it, otherwise, this is not a "real" fern implementation
		float rcw[] = {
			rnum[k * 2] * cw[0] + 1e-5,
			rnum[k * 2 + 1] * cw[1] + 1e-5,
		};
		post[k * 2] = logf(rcw[0] / (rcw[0] + rcw[1]));
		post[k * 2 + 1] = logf(rcw[1] / (rcw[0] + rcw[1]));
		rnum += ferns->posteriors * 2;
		post += ferns->posteriors * 2;
	}
}

float ccv_ferns_predict(ccv_ferns_t* ferns, uint32_t* fern)
{
	float votes[] = {0, 0};
	int i;
	float* post = ferns->posterior;
	for (i = 0; i < ferns->structs; i++)
	{
		votes[0] += post[fern[i] * 2];
		votes[1] += post[fern[i] * 2 + 1];
		post += ferns->posteriors * 2;
	}
	return votes[1] - votes[0];
}

void ccv_ferns_free(ccv_ferns_t* ferns)
{
	ccfree(ferns);
}
