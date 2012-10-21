#include "ccv.h"
#include "ccv_internal.h"
#include "3rdparty/dsfmt/dSFMT.h"

ccv_ferns_t* ccv_ferns_new(int nferns, int features, int scale, ccv_size_t* sizes)
{
	ccv_ferns_t* ferns = (ccv_ferns_t*)ccmalloc(sizeof(ccv_ferns_t) + sizeof(ccv_point_t) * (nferns * features * scale * 2 - 1));
	ferns->ferns = nferns;
	ferns->features = features;
	ferns->scales = scale;
	int i, j, k;
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, (uint32_t)ferns);
	for (i = 0; i < nferns; i++)
	{
		for (k = 0; k < features; k++)
		{
			double x1f = dsfmt_genrand_close_open(&dsfmt);
			double y1f = dsfmt_genrand_close_open(&dsfmt);
			double x2f = dsfmt_genrand_close_open(&dsfmt);
			double y2f = dsfmt_genrand_close_open(&dsfmt);
			for (j = 0; j < scale; j++)
			{
				ferns->fern[(j * nferns * features + i * features + k) * 2] = ccv_point((int)(x1f * sizes[j].width), (int)(y1f * sizes[j].height));
				ferns->fern[(j * nferns * features + i * features + k) * 2 + 1] = ccv_point((int)(x2f * sizes[j].width), (int)(y2f * sizes[j].height));
			}
		}
	}
	return ferns;
}

void ccv_ferns_free(ccv_ferns_t* ferns)
{
	ccfree(ferns);
}
