---
layout: page
lib: ccv
slug: ccv-convnet
status: publish
title: lib/ccv_convnet.c
desc: Deep Convolutional Networks
categories:
- lib
---

ccv_convnet_new
---------------

	ccv_convnet_t* ccv_convnet_new(int use_cwc_accel, ccv_size_t input, ccv_convnet_layer_param_t params[], int count);

Create a new (deep) convolutional network with specified parameters. ccv only supports convolutional layer (shared weights), max pooling layer, average pooling layer, full connect layer and local response normalization layer.


 * **use_cwc_accel**: whether use CUDA-enabled GPU to accelerate various computations for convolutional network.
 * **input**: the input size of the image, it is not necessarily the input size of the first convolutional layer.
 * **params[]**: the C-array of **ccv\_convnet\_layer\_param\_t** that specifies the parameters for each layer.
 * **count** the size of params[] C-array.

ccv_convnet_layer_param_t
-------------------------

 * **type**: one of following value to specify the network layer type, **CCV\_CONVNET\_CONVOLUTIONAL**, **CCV\_CONVNET\_FULL\_CONNECT**, **CCV\_CONVNET\_MAX\_POOL**, **CCV\_CONVNET\_AVERAGE\_POOL**, **CCV\_CONVNET\_LOCAL\_RESPONSE\_NORM**.
 * **bias**: the initialization value for bias if applicable (for convolutional layer and full connect layer).
 * **glorot**: the truncated uniform distribution coefficients for weights if applicable (for convolutional layer and full connect layer, glorot / sqrt(in + out)).
 * **input**: a **ccv\_convnet\_input\_t** specifies the input structure.
 * **output**: a **ccv\_convnet\_type\_t** specifies the output parameters and structure.

ccv_convnet_input_t
-------------------

 * **matrix.rows**: the number of rows of the input matrix.
 * **matrix.cols**: the number of columns of the input matrix.
 * **matrix.channels**: the number of channels of the input matrix.
 * **matrix.partition**: the number of partitions of the input matrix, it must be dividable by the number of channels (it is partitioned by channels).
 * **node.count**: the number of nodes. You should either use **node** or **matrix** to specify the input structure.

ccv_convnet_type_t
------------------

 * **convolutional.count**: the number of filters for convolutional layer.
 * **convolutional.strides**: the strides for convolutional filter.
 * **convolutional.border**: the padding border size for the input matrix.
 * **convolutional.rows**: the number of rows for convolutional filter.
 * **convolutional.cols**: the number of columns for convolutional filter.
 * **convolutional.channels**: the number of channels for convolutional filter.
 * **convolutional.partition**: the number of partitions for convolutional filter.
 * **pool.strides**: the strides for pooling layer.
 * **pool.size**: the size for pooling layer.
 * **pool.border**: the padding border size for the input matrix.
 * **rnorm.size**: the size of local response normalization layer.
 * **rnorm.kappa**: as of b[i] = a[i] / (rnorm.kappa + rnorm.alpha * sum(a, i - rnorm.size / 2, i + rnorm.size / 2)) ^ rnorm.beta
 * **rnorm.alpha**: see **rnorm.kappa**.
 * **rnorm.beta**: see **rnorm.kappa**.
 * **full\_connect.count**: the number of output nodes for full connect layer.

ccv_convnet_verify
------------------

	int ccv_convnet_verify(ccv_convnet_t* convnet, int output);

Verify the specified parameters make sense as a deep convolutional network. Return 0 if the given deep convolutional network making sense.

 * **convnet**: A deep convolutional network to verify.
 * **output**: The output number of nodes (for the last full connect layer).

ccv_convnet_supervised_train
----------------------------

	void ccv_convnet_supervised_train(ccv_convnet_t* convnet, ccv_array_t* categorizeds, ccv_array_t* tests, const char* filename, ccv_convnet_train_param_t params);

Start to train a deep convolutional network with given parameters and data.

 * **convnet**: A deep convolutional network that is initialized.
 * **categorizeds**: An array of images with its category information for training.
 * **tests**: An array of images with its category information for validating.
 * **filename**: The working file to save progress and the trained convolutional network.
 * **params**: The training parameters.

ccv_convnet_train_param_t
-------------------------

 * **max\_epoch**: The number of epoch (an epoch sweeps through all the examples) to go through before end the training.
 * **mini\_batch**: The number of examples for a batch in stochastic gradient descent.
 * **iterations**: The number of iterations (an iteration is for one batch) before save the progress.
 * **sgd\_frequency**: After how many batches when we do a SGD update.
 * **symmetric**: Whether to exploit the symmetric property of the provided examples.
 * **device\_count**: Use how many GPU devices, this is capped by available CUDA devices on your system.
 * **peer\_access**: Enable peer access for cross device communications or not, this will enable faster multiple device training.
 * **image\_manipulation**: Coefficient for random contrast, brightness, and saturation manipulations for training images.
 * **color\_gain**: The color variance for data augmentation (0 means no such augmentation).
 * **input.min\_dim**: The minimum dimensions for random resize of training images.
 * **input.max\_dim**: The maximum dimensions for random resize of training images.
 * **layer\_params**: An C-array of **ccv\_convnet\_layer\_train\_param\_t** training parameters for each layer.

ccv_convnet_layer_train_param_t
-------------------------------

 * **dor**: The dropout rate for this layer, it is only applicable for full connect layer.
 * **w**: A **ccv\_convnet\_layer\_sgd\_param\_t** specifies the stochastic gradient descent update rule for weight, it is only applicable for full connect layer and convolutional layer.
 * **bias**: A **ccv\_convnet\_layer\_sgd\_param\_t** specifies the stochastic gradient descent update rule for bias, it is only applicable for full connect layer and convolutional layerweight.

ccv_convnet_layer_sgd_param_t
-----------------------------

 * **learn\_rate**: new velocity = **momentum** * old velocity - **decay** * **learn\_rate** * old value + **learn\_rate** * delta, new value = old value + new velocity
 * **decay**: see **learn\_rate**.
 * **momentum**: see **learn\_rate**.

ccv_convnet_encode
------------------

	void ccv_convnet_encode(ccv_convnet_t* convnet, ccv_dense_matrix_t** a, ccv_dense_matrix_t** b, int batch);

Use a convolutional network to encode an image into a compact representation.

 * **convnet**: The given convolutional network.
 * **a**: A C-array of input images.
 * **b**: A C-array of output matrix of compact representation.
 * **batch**: The number of input images.

ccv_convnet_classify
--------------------

	void ccv_convnet_classify(ccv_convnet_t* convnet, ccv_dense_matrix_t** a, int symmetric, ccv_array_t** ranks, int tops, int batch);

Use a convolutional network to classify an image into categories.

 * **convnet**: The given convolutional network.
 * **a**: A C-array of input images.
 * **symmetric**: Whether the input is symmetric.
 * **ranks**: A C-array of **ccv\_array\_t** contains top categories by the convolutional network.
 * **tops**: The number of top categories return for each image.
 * **batch**: The number of input images.

ccv_convnet_read
----------------

	ccv_convnet_t* ccv_convnet_read(int use_cwc_accel, const char* filename);

Read a convolutional network that persisted on the disk.

 * **use_cwc_accel**: Use CUDA-enabled GPU acceleration.
 * **filename**: The file on the disk.

ccv_convnet_write
-----------------

	void ccv_convnet_write(ccv_convnet_t* convnet, const char* filename, ccv_convnet_write_param_t params);

Write a convolutional network to a disk.

 * **convnet**: A given convolutional network.
 * **filename**: The file on the disk.
 * **params**: a **ccv\_convnet\_write\_param\_t** to specify the write parameters.

ccv_convnet_write_param_t
-------------------------

 * **half\_precision**: Use half precision float point to represent network parameters.

ccv_convnet_compact
-------------------

	void ccv_convnet_compact(ccv_convnet_t* convnet);

Free up temporary resources of a given convolutional network.

 * **convnet**: A convolutional network.

ccv_convnet_free
----------------

	void ccv_convnet_free(ccv_convnet_t* convnet);

Free up the memory of a given convolutional network.

 * **convnet**: A convolutional network.
