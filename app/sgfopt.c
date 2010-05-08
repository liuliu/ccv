#include "ccv.h"
#include <time.h>

typedef struct {
	double x[100];
	double m0x[100];
	double m1x[100];
	double n0x[100];
	double n1x[100];
	int y;
} ccv_sample_t;

ccv_sample_t samples[100];
double w[100];

double cont_f(const ccv_dense_matrix_t* x)
{
	int i, j;
	double f = 0;
	double eps = 1;
	for (i = 0; i < 100; i++)
	{
		double m0 = 0, m1 = 0;
		double n0 = 0, n1 = 0;
		for (j = 0; j < 100; j++)
		{
			m0 += samples[i].m0x[j] * ccv_max(x->data.db[j], 0);
			m1 += samples[i].m1x[j] * ccv_max(x->data.db[j], 0);
			n0 += samples[i].n0x[j] * ccv_max(x->data.db[j + 100], 0);
			n1 += samples[i].n1x[j] * ccv_max(x->data.db[j + 100], 0);
		}
		if (samples[i].y)
			f += w[i] / (1.0 + exp(-1 * (n0 / (n1 + eps) - m0 / (m1 + eps) + 0.1)));
		else
			f += w[i] * (1.0 - 1.0 / (1.0 + exp(-1 * (n0 / (n1 + eps) - m0 / (m1 + eps) + 0.1))));
	}
	for (i = 0; i < 200; i++)
		f += 0.01 * (1 - exp(-x->data.db[i] * x->data.db[i]));
	/*
		if (x->data.db[i] > 0)
			f += 0.1 * x->data.db[i];
		else
			f += -1.0 * x->data.db[i];
	*/
	//f += fabs(ccv_sum(x) - 3);
	return f;
}

int cont_sgf_feature(const ccv_dense_matrix_t* x, double* f, ccv_dense_matrix_t* df, void* data)
{
	*f = cont_f(x);
	printf("%f\n", *f);
	int i, j, k;
	ccv_zero(df);
	/*for (k = 0; k < 200; k++)
	{
		x->data.db[k] += 0.0001;
		df->data.db[k] = (cont_f(x) - *f) / 0.0001;
		x->data.db[k] -= 0.0001;
	}*/
	double eps = 1;
	for (k = 0; k < 100; k++)
	{
		df->data.db[k] = 0.02 * exp(-x->data.db[k] * x->data.db[k]) * x->data.db[k];
		for (i = 0; i < 100; i++)
		{
			double m0 = 0, m1 = 0;
			double n0 = 0, n1 = 0;
			for (j = 0; j < 100; j++)
			{
				m0 += samples[i].m0x[j] * ccv_max(x->data.db[j], 0);
				m1 += samples[i].m1x[j] * ccv_max(x->data.db[j], 0);
				n0 += samples[i].n0x[j] * ccv_max(x->data.db[j + 100], 0);
				n1 += samples[i].n1x[j] * ccv_max(x->data.db[j + 100], 0);
			}
			if (samples[i].y)
			{
				double e = exp(-1 * (n0 / (n1 + eps) - m0 / (m1 + eps) + 0.1));
				df->data.db[k] += -w[i] * 1 * e / ((1 + e) * (1 + e)) * (samples[i].m0x[k] / (m1 + eps) - samples[i].m1x[k] * m0 / ((m1 + eps) * (m1 + eps)));
			} else {
				double e = exp(-1 * (n0 / (n1 + eps) - m0 / (m1 + eps) + 0.1));
				df->data.db[k] += w[i] * 1 * e / ((1 + e) * (1 + e)) * (samples[i].m0x[k] / (m1 + eps) - samples[i].m1x[k] * m0 / ((m1 + eps) * (m1 + eps)));
			}
		}
	}
	for (k = 0; k < 100; k++)
	{
		df->data.db[k + 100] = 0.02 * exp(-x->data.db[k + 100] * x->data.db[k + 100]) * x->data.db[k + 100];
		for (i = 0; i < 100; i++)
		{
			double m0 = 0, m1 = 0;
			double n0 = 0, n1 = 0;
			for (j = 0; j < 100; j++)
			{
				m0 += samples[i].m0x[j] * ccv_max(x->data.db[j], 0);
				m1 += samples[i].m1x[j] * ccv_max(x->data.db[j], 0);
				n0 += samples[i].n0x[j] * ccv_max(x->data.db[j + 100], 0);
				n1 += samples[i].n1x[j] * ccv_max(x->data.db[j + 100], 0);
			}
			if (samples[i].y)
			{
				double e = exp(-1 * (n0 / (n1 + eps) - m0 / (m1 + eps) + 0.1));
				df->data.db[k + 100] += w[i] * 1 * e / ((1 + e) * (1 + e)) * (samples[i].n0x[k] / (n1 + eps) - samples[i].n1x[k] * n0 / ((n1 + eps) * (n1 + eps)));
			} else {
				double e = exp(-1 * (n0 / (n1 + eps) - m0 / (m1 + eps) + 0.1));
				df->data.db[k + 100] += -w[i] * 1 * e / ((1 + e) * (1 + e)) * (samples[i].n0x[k] / (n1 + eps) - samples[i].n1x[k] * n0 / ((n1 + eps) * (n1 + eps)));
			}
		}
	}
	return 0;
}

int main(int argc, char** argv)
{
	ccv_dense_matrix_t* x = ccv_dense_matrix_new(1, 200, CCV_64F | CCV_C1, NULL, NULL);
	ccv_zero(x);
	int i, j;
//	for (i = 0; i < 200; i++)
//		x->data.db[i] = 1;
//	x->data.db[4] = 1; x->data.db[12] = 1;
	for (i = 0; i < 100; i++)
		w[i] = 0.01;
	srand(time(NULL));
	for (i = 0; i < 100; i++)
	{
		for (j = 0; j < 100; j++)
		{
			if ((j == 2 || j == 3) && i < 50)
				samples[i].x[j] = (rand() % 10000) / 10000.0 + 0.5;
			else if (j == 4 && i >= 50)
				samples[i].x[j] = (rand() % 10000) / 10000.0 + 0.5;
			else
				samples[i].x[j] = (rand() % 10000) / 10000.0;
		}
		samples[i].y = i >= 50;
	}
	for (i = 0; i < 100; i++)
		for (j = 0; j < 100; j++)
		{
			samples[i].m0x[j] = samples[i].x[j] * exp(samples[i].x[j]);
			samples[i].m1x[j] = exp(samples[i].x[j]);
			samples[i].n0x[j] = samples[i].x[j] * exp(-samples[i].x[j]);
			samples[i].n1x[j] = exp(-samples[i].x[j]);
		}
	ccv_minimize_param_t params;
	params.interp = 0.1;
	params.extrap = 3.0;
	params.max_iter = 20;
	params.ratio = 10.0;
	params.sig = 0.1;
	params.rho = 0.05;
	ccv_minimize(x, 25, 1.0, cont_sgf_feature, params, 0);
	printf("x :");
	double maxx = x->data.db[0];
	int maxxi = 0;
	for (i = 0; i < 100; i++)
	{
		printf(" %f", x->data.db[i]);
		if (x->data.db[i] > maxx)
		{
			maxxi = i;
			maxx = x->data.db[i];
		}
	}
	printf("\ny :");
	double maxy = x->data.db[100];
	int maxyi = 0;
	for (i = 0; i < 100; i++)
	{
		printf(" %f", x->data.db[i + 100]);
		if (x->data.db[i + 100] > maxy)
		{
			maxyi = i;
			maxy = x->data.db[i + 100];
		}
	}
	printf("\nmaxxi : %d, maxyi : %d\n", maxxi, maxyi);
	return 0;
}
