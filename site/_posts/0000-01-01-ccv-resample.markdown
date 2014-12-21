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

ccv\_resample
-------------

	void ccv_resample(ccv_dense_matrix_t *a, ccv_dense_matrix_t **b, int btype, int rows, int cols, int type)

Resample a given matrix to different size, as for now, ccv only supports either downsampling (with CCV\_INTER\_AREA) or upsampling (with CCV\_INTER\_CUBIC).

 * **a**: The input matrix.
 * **b**: The output matrix.
 * **btype**: The type of output matrix, if 0, ccv will try to match the input matrix for appropriate type.
 * **rows**: The new row.
 * **cols**: The new column.
 * **type**: For now, ccv supports CCV\_INTER\_AREA, which is an extension to [bilinear resampling](https://en.wikipedia.org/wiki/Bilinear\_filtering) for downsampling and CCV\_INTER\_CUBIC [bicubic resampling](https://en.wikipedia.org/wiki/Bicubic\_interpolation) for upsampling.

ccv\_sample\_down
-----------------

	void ccv_sample_down(ccv_dense_matrix_t *a, ccv_dense_matrix_t **b, int type, int src_x, int src_y)

Downsample a given matrix to exactly half size with a [Gaussian filter](https://en.wikipedia.org/wiki/Gaussian\_filter). The half size is approximated by floor(rows * 0.5) x floor(cols * 0.5).

 * **a**: The input matrix.
 * **b**: The output matrix.
 * **type**: The type of output matrix, if 0, ccv will try to match the input matrix for appropriate type.
 * **src\_x**: Shift the start point by src\_x.
 * **src\_y**: Shift the start point by src\_y.

ccv\_sample\_up
---------------

	void ccv_sample_up(ccv_dense_matrix_t *a, ccv_dense_matrix_t **b, int type, int src_x, int src_y)

Upsample a given matrix to exactly double size with a [Gaussian filter](https://en.wikipedia.org/wiki/Gaussian\_filter).

 * **a**: The input matrix.
 * **b**: The output matrix.
 * **type**: The type of output matrix, if 0, ccv will try to match the input matrix for appropriate type.
 * **src\_x**: Shift the start point by src\_x.
 * **src\_y**: Shift the start point by src\_y.
