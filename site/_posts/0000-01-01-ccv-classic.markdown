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

ccv_hog
-------

	void ccv_hog(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int b_type, int sbin, int size)

[Histogram-of-Oriented-Gradients](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) implementation, specifically, it implements the HOG described in *Object Detection with Discriminatively Trained Part-Based Models, Pedro F. Felzenszwalb, Ross B. Girshick, David McAllester and Deva Ramanan*.

 * **a**: the input matrix.
 * **b**: the output matrix.
 * **b_type**: the type of output matrix, if 0, ccv will try to match the input matrix for appropriate type.
 * **sbin**: the number of bins for orientation (default to 9, thus, for **b**, it will have 9 * 2 + 9 + 4 = 31 channels).
 * **size**: the window size for HOG (default to 8)

ccv_canny
---------

	void ccv_canny(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, int size, double low_thresh, double high_thresh)

[Canny edge detector](https://en.wikipedia.org/wiki/Canny_edge_detector) implementation. For performance size, this is a clean-up reimplementation of OpenCV's Canny edge detector, it has very similar performance characteristic as the OpenCV one. As of today, ccv's Canny edge detector only works with CCV\_8U or CCV\_32S dense matrix type.

 * **a**: the input matrix.
 * **b**: the output matrix.
 * **type**:the type of output matrix, if 0, ccv will create a CCV\_8U \| CCV\_C1 matrix.
 * **size**: the underlying Sobel filter size.
 * **low_thresh**: the low threshold that makes the point interesting.
 * **high_thresh**: the high threshold that makes the point acceptable.

ccv_otsu
--------

	int ccv_otsu(ccv_dense_matrix_t* a, double* outvar, int range)

[OTSU](https://en.wikipedia.org/wiki/Otsu%27s_method) implementation.

 * **a**: the input matrix.
 * **outvar**: the inter-class variance.
 * **range**: the maximum range of data in the input matrix.

It returns the threshold, inclusively. e.g. 5 means 0~5 is in the background, and 6~255 is in the foreground.

ccv_optical_flow_lucas_kanade
-----------------------------

	void ccv_optical_flow_lucas_kanade(ccv_dense_matrix_t* a, ccv_dense_matrix_t* b, ccv_array_t* point_a, ccv_array_t** point_b, ccv_size_t win_size, int level, double min_eigen)

[Lucas Kanade](https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_Optical_Flow_Method) optical flow implementation with image pyramid extension.

 * **a**: the first frame
 * **b**: the next frame
 * **point_a**: the points in first frame, of **ccv\_decimal\_point\_t** type
 * **point_b**: the output points in the next frame, of **ccv\_decimal\_point\_with\_status\_t** type
 * **win_size**: the window size to compute each optical flow, it must be a odd number
 * **level**: how many image pyramids to be used for the computation
 * **min_eigen**: the minimal eigen-value to pass optical flow computation
