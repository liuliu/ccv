---
layout: page
lib: ccv
slug: ccv-numeric
status: publish
title: lib/ccv_numeric.c
desc: numerical algorithms
categories:
- lib
---

ccv\_minimize
-------------

	void ccv_minimize(ccv_dense_matrix_t *x, int length, double red, ccv_minimize_f func, ccv_minimize_param_t params, void *data)

Linear-search to minimize function with partial derivatives. It is formed after [minimize.m](http://www.gatsby.ucl.ac.uk/~edward/code/minimize/example.html).

 * **x**: The input vector.
 * **length**: The length of line.
 * **red**: The step size.
 * **func**: (int ccv\_minimize\_f)(const ccv\_dense\_matrix\_t* x, double* f, ccv\_dense\_matrix\_t* df, void* data). Compute the function value, and its partial derivatives.
 * **params**: A **ccv\_minimize\_param\_t** structure that defines various aspect of the minimize function.
 * **data**: Any extra user data.

ccv\_minimize\_param\_t
-----------------------

 * **extrap**: Extrapolate value.
 * **interp**: Interpolate value.
 * **max\_iter**: Maximum iterations.
 * **ratio**: Increase ratio.
 * **rho**: Decrease ratio.
 * **sig**: Sigma.

ccv\_filter
-----------

	void ccv_filter(ccv_dense_matrix_t *a, ccv_dense_matrix_t *b, ccv_dense_matrix_t **d, int type, int padding_pattern)

Convolve on dense matrix a with dense matrix b. This function has a soft dependency on [FFTW3](http://fftw.org/). If no FFTW3 exists, ccv will use [KissFFT](http://sourceforge.net/projects/kissfft/) shipped with it. FFTW3 is about 35% faster than KissFFT.

 * **a**: Dense matrix a.
 * **b**: Dense matrix b.
 * **d**: The output matrix.
 * **type**: The type of output matrix, if 0, ccv will try to match the input matrix for appropriate type.
 * **padding\_pattern**: ccv doesn't support padding pattern for now.

ccv\_filter\_kernel
-------------------

	void ccv_filter_kernel(ccv_dense_matrix_t *x, ccv_filter_kernel_f func, void *data)

Fill a given dense matrix with a kernel function.

 * **x**: The matrix to be filled with.
 * **func**: (double ccv\_filter\_kernel\_f(double x, double y, void* data), compute the value with given x, y.
 * **data**: Any extra user data.

ccv\_distance\_transform
------------------------

	void ccv_distance_transform(ccv_dense_matrix_t *a, ccv_dense_matrix_t **b, int type, ccv_dense_matrix_t **x, int x_type, ccv_dense_matrix_t **y, int y_type, double dx, double dy, double dxx, double dyy, int flag)

[Distance transform](https://en.wikipedia.org/wiki/Distance\_transform). The current implementation follows [Distance Transforms of Sampled Functions](http://www.cs.cornell.edu/~dph/papers/dt.pdf). The dynamic programming technique has O(n) time complexity.

 * **a**: The input matrix.
 * **b**: The output matrix.
 * **type**: The type of output matrix, if 0, ccv will try to match the input matrix for appropriate type.
 * **x**: The x coordinate offset.
 * **x\_type**: The type of output x coordinate offset, if 0, ccv will default to CCV\_32S \| CCV\_C1.
 * **y**: The y coordinate offset.
 * **y\_type**: The type of output x coordinate offset, if 0, ccv will default to CCV\_32S \| CCV\_C1.
 * **dx**: The x coefficient.
 * **dy**: The y coefficient.
 * **dxx**: The x^2 coefficient.
 * **dyy**: The y^2 coefficient.
 * **flag**: CCV\_GSEDT, generalized squared Euclidean distance transform. CCV\_NEGATIVE, negate value in input matrix for computation; effectively, this enables us to compute the maximum distance transform rather than minimum (default one).
