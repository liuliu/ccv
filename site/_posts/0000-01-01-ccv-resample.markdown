---
layout: page
lib: ccv
slug: ccv-resample
status: publish
title: lib/ccv_resample.c
desc: image resampling utilities
categories:
- lib
---

ccv_resample
------------

	void ccv_resample(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int btype, int rows, int cols, int type)

Resample a given matrix to different size, as for now, ccv only supports either downsampling (with CCV\_INTER\_AREA) or upsampling (with CCV\_INTER\_CUBIC).

 * **a**: the input matrix.
 * **b**: the output matrix.
 * **btype**: the type of output matrix, if 0, ccv will try to match the input matrix for appropriate type.
 * **rows**: the new row.
 * **cols**: the new column.
 * **type**: for now, ccv supports CCV\_INTER\_AREA, which is an extension to [bilinear resampling](https://en.wikipedia.org/wiki/Bilinear_filtering) for downsampling and CCV\_INTER\_CUBIC [bicubic resampling](https://en.wikipedia.org/wiki/Bicubic_interpolation) for upsampling.

ccv_sample_down
---------------

	void ccv_sample_down(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, int src_x, int src_y)

Downsample a given matrix to exactly half size with a [Gaussian filter](https://en.wikipedia.org/wiki/Gaussian_filter). The half size is approximated by floor(rows * 0.5) x floor(cols * 0.5).

 * **a**: the input matrix.
 * **b**: the output matrix.
 * **type**: the type of output matrix, if 0, ccv will try to match the input matrix for appropriate type.
 * **src\_x**: shift the start point by src\_x.
 * **src\_y**: shift the start point by src\_y.

ccv_sample_up
-------------

	void ccv_sample_up(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, int src_x, int src_y)

Upsample a given matrix to exactly double size with a [Gaussian filter](https://en.wikipedia.org/wiki/Gaussian_filter).

 * **a**: the input matrix.
 * **b**: the output matrix.
 * **type**: the type of output matrix, if 0, ccv will try to match the input matrix for appropriate type.
 * **src\_x**: shift the start point by src\_x.
 * **src\_y**: shift the start point by src\_y.