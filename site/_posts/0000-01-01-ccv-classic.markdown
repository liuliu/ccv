---
layout: page
lib: ccv
slug: ccv-classic
status: publish
title: lib/ccv_classic.c
desc: classic computer vision algorithms
categories:
- lib
---

ccv\_hog
--------

	void ccv_hog(ccv_dense_matrix_t *a, ccv_dense_matrix_t **b, int b_type, int sbin, int size)

[Histogram-of-Oriented-Gradients](https://en.wikipedia.org/wiki/Histogram\_of\_oriented\_gradients) implementation, specifically, it implements the HOG described in *Object Detection with Discriminatively Trained Part-Based Models, Pedro F. Felzenszwalb, Ross B. Girshick, David McAllester and Deva Ramanan*.

 * **a**: The input matrix.
 * **b**: The output matrix.
 * **b\_type**: The type of output matrix, if 0, ccv will try to match the input matrix for appropriate type.
 * **sbin**: The number of bins for orientation (default to 9, thus, for **b**, it will have 9 * 2 + 9 + 4 = 31 channels).
 * **size**: The window size for HOG (default to 8)

ccv\_canny
----------

	void ccv_canny(ccv_dense_matrix_t *a, ccv_dense_matrix_t **b, int type, int size, double low_thresh, double high_thresh)

[Canny edge detector](https://en.wikipedia.org/wiki/Canny\_edge\_detector) implementation. For performance reason, this is a clean-up reimplementation of OpenCV's Canny edge detector, it has very similar performance characteristic as the OpenCV one. As of today, ccv's Canny edge detector only works with CCV\_8U or CCV\_32S dense matrix type.

 * **a**: The input matrix.
 * **b**: The output matrix.
 * **type**: The type of output matrix, if 0, ccv will create a CCV\_8U \| CCV\_C1 matrix.
 * **size**: The underlying Sobel filter size.
 * **low\_thresh**: The low threshold that makes the point interesting.
 * **high\_thresh**: The high threshold that makes the point acceptable.

ccv\_otsu
---------

	int ccv_otsu(ccv_dense_matrix_t *a, double *outvar, int range)

[OTSU](https://en.wikipedia.org/wiki/Otsu%27s\_method) implementation.

 * **a**: The input matrix.
 * **outvar**: The inter-class variance.
 * **range**: The maximum range of data in the input matrix.

**return**: The threshold, inclusively. e.g. 5 means 0~5 is in the background, and 6~255 is in the foreground.

ccv\_optical\_flow\_lucas\_kanade
---------------------------------

	void ccv_optical_flow_lucas_kanade(ccv_dense_matrix_t *a, ccv_dense_matrix_t *b, ccv_array_t *point_a, ccv_array_t **point_b, ccv_size_t win_size, int level, double min_eigen)

[Lucas Kanade](https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade\_Optical\_Flow\_Method) optical flow implementation with image pyramid extension.

 * **a**: The first frame
 * **b**: The next frame
 * **point\_a**: The points in first frame, of **ccv\_decimal\_point\_t** type
 * **point\_b**: The output points in the next frame, of **ccv\_decimal\_point\_with\_status\_t** type
 * **win\_size**: The window size to compute each optical flow, it must be a odd number
 * **level**: How many image pyramids to be used for the computation
 * **min\_eigen**: The minimal eigen-value to pass optical flow computation
