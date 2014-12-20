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

ccv_decimal_slice
-----------------

	void ccv_decimal_slice(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, float y, float x, int rows, int cols)

Similar to ccv\_slice, it will slice a given matrix into required rows / cols, but it will interpolate the value with bilinear filter if x and y is non-integer.

 * **a**: the given matrix that will be sliced
 * **b**: the output matrix
 * **type**: the type of output matrix
 * **y**: the top point to slice
 * **x**: the left point to slice
 * **rows**: the number of rows for destination matrix
 * **cols**: the number of cols for destination matrix

ccv_perspective_transform_apply
-------------------------------

	ccv_decimal_point_t ccv_perspective_transform_apply(ccv_decimal_point_t point, ccv_size_t size, float m00, float m01, float m02, float m10, float m11, float m12, float m20, float m21, float m22)

Apply a [3D transform](https://en.wikipedia.org/wiki/Perspective_transform#Perspective_projection) against the given point in a given image size, assuming field of view is 60 (in degree).

 * **point**: the point to be transformed in decimal
 * **size**: the image size
 * **m00, m01, m02, m10, m11, m12, m20, m21, m22**: the transformation matrix

ccv_perspective_transform
-------------------------

	void ccv_perspective_transform(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, float m00, float m01, float m02, float m10, float m11, float m12, float m20, float m21, float m22)

Apply a [3D transform](https://en.wikipedia.org/wiki/Perspective_transform#Perspective_projection) on a given matrix, assuming field of view is 60 (in degree).

 * **a**: the given matrix to be transformed
 * **b**: the output matrix
 * **type**: the type of output matrix
 * **m00, m01, m02, m10, m11, m12, m20, m21, m22**: the transformation matrix
