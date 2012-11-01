#include "ccv.h"
#include "ccv_internal.h"
#include "3rdparty/dsfmt/dSFMT.h"

ccv_ferns_t* ccv_ferns_new(int structs, int features, int scales, ccv_size_t* sizes)
{
	assert(structs > 0 && features > 0 && scales > 0);
	int posteriors = (int)(powf(2.0, features) + 0.5);
	ccv_ferns_t* ferns = (ccv_ferns_t*)ccmalloc(sizeof(ccv_ferns_t) + sizeof(ccv_point_t) * (structs * features * scales * 2 - 1) + sizeof(float) * structs * posteriors * 2 + sizeof(int) * structs * posteriors * 2);
	ferns->structs = structs;
	ferns->features = features;
	ferns->scales = scales;
	ferns->posteriors = posteriors;
	ferns->posterior = (float*)((uint8_t*)(ferns + 1) + sizeof(ccv_point_t) * (structs * features * scales * 2 - 1));
	// now only for 2 classes
	ferns->cnum = (int*)(ferns->posterior + structs * posteriors * 2);
	memset(ferns->posterior, 0, sizeof(float) * structs * posteriors * 2 + sizeof(int) * structs * posteriors * 2);
	int i, j, k;
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, (uint32_t)ferns);
	for (i = 0; i < structs; i++)
	{
		for (k = 0; k < features; k++)
		{
			double x1f = dsfmt_genrand_close_open(&dsfmt);
			double y1f = dsfmt_genrand_close_open(&dsfmt);
			double x2f = dsfmt_genrand_close_open(&dsfmt);
			double y2f = dsfmt_genrand_close_open(&dsfmt);
			for (j = 0; j < scales; j++)
			{
				ferns->fern[(j * structs * features + i * features + k) * 2] = ccv_point((int)(x1f * sizes[j].width), (int)(y1f * sizes[j].height));
				ferns->fern[(j * structs * features + i * features + k) * 2 + 1] = ccv_point((int)(x2f * sizes[j].width), (int)(y2f * sizes[j].height));
			}
		}
	}
	ferns->threshold = 0.5 * structs;
	return ferns;
}

void ccv_ferns_feature(ccv_ferns_t* ferns, ccv_dense_matrix_t* a, int scale, uint32_t* fern)
{
	ccv_point_t* fern_feature = ferns->fern + scale * ferns->structs * ferns->features;
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
	float* post = ferns->posterior;
	for (i = 0; i < ferns->structs; i++)
	{
		uint32_t k = fern[i];
		cnum[k * 2 + c] += repeat;
			// needs to compute the log of it
		post[k * 2] = (float)(cnum[k * 2] + 1) / (cnum[k * 2] + cnum[k * 2 + 1] + 2);
		post[k * 2 + 1] = (float)(cnum[k * 2 + 1] + 1) / (cnum[k * 2] + cnum[k * 2 + 1] + 2);
		cnum += ferns->posteriors * 2;
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
	return votes[1];
}

void ccv_ferns_free(ccv_ferns_t* ferns)
{
	ccfree(ferns);
}
