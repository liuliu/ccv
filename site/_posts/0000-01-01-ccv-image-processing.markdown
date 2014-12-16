---
layout: page
lib: ccv
slug: ccv-image-processing
status: publish
title: lib/ccv_image_processing.c
desc: basic image processing utilities
categories:
- lib
---

The utilities in this file provides image processing methods that are widely used for image enhancements.

ccv_color_transform
-------------------

Convert matrix from one color space representation to another.

	void ccv_color_transform(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, int flag)

 * **a**: the input matrix.
 * **b**: the output matrix.
 * **btype**: the type of output matrix, if 0, ccv will use the sample type as the input matrix.
 * **flag**: **CCV\_RGB\_TO\_YUV** to convert from RGB color space to YUV color space.

ccv_saturation
--------------

Manipulate image's saturation.

	void ccv_saturation(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, double ds)

 * **a**: the input matrix.
 * **b**: the output matrix (it is in-place safe).
 * **btype**: the type of output matrix, if 0, ccv will use the sample type as the input matrix.
 * **ds**: the coefficient (0: grayscale, 1: original).

ccv_contrast
------------

Manipulate image's contrast.

	void ccv_contrast(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, double ds)

 * **a**: the input matrix.
 * **b**: the output matrix (it is in-place safe).
 * **btype**: the type of output matrix, if 0, ccv will use the sample type as the input matrix.
 * **ds**: the coefficient (0: mean image, 1: original).
