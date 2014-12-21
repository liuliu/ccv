---
layout: page
lib: ccv
slug: ccv-transform
status: publish
title: lib/ccv_transform.c
desc: image transform utilities
categories:
- lib
---

ccv\_decimal\_slice
-------------------

	void ccv_decimal_slice(ccv_dense_matrix_t *a, ccv_dense_matrix_t **b, int type, float y, float x, int rows, int cols)

Similar to ccv\_slice, it will slice a given matrix into required rows / cols, but it will interpolate the value with bilinear filter if x and y is non-integer.

 * **a**: The given matrix that will be sliced
 * **b**: The output matrix
 * **type**: The type of output matrix
 * **y**: The top point to slice
 * **x**: The left point to slice
 * **rows**: The number of rows for destination matrix
 * **cols**: The number of cols for destination matrix

ccv\_perspective\_transform\_apply
----------------------------------

	ccv_decimal_point_t ccv_perspective_transform_apply(ccv_decimal_point_t point, ccv_size_t size, float m00, float m01, float m02, float m10, float m11, float m12, float m20, float m21, float m22)

Apply a [3D transform](https://en.wikipedia.org/wiki/Perspective\_transform#Perspective\_projection) against the given point in a given image size, assuming field of view is 60 (in degree).

 * **point**: The point to be transformed in decimal
 * **size**: The image size
 * **m00, m01, m02, m10, m11, m12, m20, m21, m22**: The transformation matrix

ccv\_perspective\_transform
---------------------------

	void ccv_perspective_transform(ccv_dense_matrix_t *a, ccv_dense_matrix_t **b, int type, float m00, float m01, float m02, float m10, float m11, float m12, float m20, float m21, float m22)

Apply a [3D transform](https://en.wikipedia.org/wiki/Perspective\_transform#Perspective\_projection) on a given matrix, assuming field of view is 60 (in degree).

 * **a**: The given matrix to be transformed
 * **b**: The output matrix
 * **type**: The type of output matrix
 * **m00, m01, m02, m10, m11, m12, m20, m21, m22**: The transformation matrix
