// VGG Configuration A Model
ccv_convnet_layer_param_t vgg_a_params[16] = {
	// first layer (convolutional 64 => max pool)
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 225,
				.cols = 225,
				.channels = 3,
				.partition = 1,
			},
		},
		.output = {
			.convolutional = {
				.count = 64,
				.strides = 1,
				.border = 0,
				.rows = 3,
				.cols = 3,
				.channels = 3,
				.partition = 1,
			},
		},
	},
	{
		.type = CCV_CONVNET_MAX_POOL,
		.input = {
			.matrix = {
				.rows = 223,
				.cols = 223,
				.channels = 64,
				.partition = 1,
			},
		},
		.output = {
			.pool = {
				.strides = 2,
				.size = 3,
				.border = 0,
			},
		},
	},
	// second layer (convolutional 128 => max pool)
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 111,
				.cols = 111,
				.channels = 64,
				.partition = 1,
			},
		},
		.output = {
			.convolutional = {
				.count = 128,
				.strides = 1,
				.border = 1,
				.rows = 3,
				.cols = 3,
				.channels = 64,
				.partition = 1,
			},
		},
	},
	{
		.type = CCV_CONVNET_MAX_POOL,
		.input = {
			.matrix = {
				.rows = 111,
				.cols = 111,
				.channels = 128,
				.partition = 1,
			},
		},
		.output = {
			.pool = {
				.strides = 2,
				.size = 3,
				.border = 0,
			},
		},
	},
	// third layer (convolutional 256x2 => max pool)
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 55,
				.cols = 55,
				.channels = 128,
				.partition = 1,
			},
		},
		.output = {
			.convolutional = {
				.count = 256,
				.strides = 1,
				.border = 1,
				.rows = 3,
				.cols = 3,
				.channels = 128,
				.partition = 1,
			},
		},
	},
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 55,
				.cols = 55,
				.channels = 256,
				.partition = 1,
			},
		},
		.output = {
			.convolutional = {
				.count = 256,
				.strides = 1,
				.border = 1,
				.rows = 3,
				.cols = 3,
				.channels = 256,
				.partition = 1,
			},
		},
	},
	{
		.type = CCV_CONVNET_MAX_POOL,
		.input = {
			.matrix = {
				.rows = 55,
				.cols = 55,
				.channels = 256,
				.partition = 1,
			},
		},
		.output = {
			.pool = {
				.strides = 2,
				.size = 3,
				.border = 0,
			},
		},
	},
	// forth layer (convolutional 512x2 => max pool)
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 27,
				.cols = 27,
				.channels = 256,
				.partition = 1,
			},
		},
		.output = {
			.convolutional = {
				.count = 512,
				.strides = 1,
				.border = 1,
				.rows = 3,
				.cols = 3,
				.channels = 256,
				.partition = 1,
			},
		},
	},
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 27,
				.cols = 27,
				.channels = 512,
				.partition = 1,
			},
		},
		.output = {
			.convolutional = {
				.count = 512,
				.strides = 1,
				.border = 1,
				.rows = 3,
				.cols = 3,
				.channels = 512,
				.partition = 1,
			},
		},
	},
	{
		.type = CCV_CONVNET_MAX_POOL,
		.input = {
			.matrix = {
				.rows = 27,
				.cols = 27,
				.channels = 512,
				.partition = 1,
			},
		},
		.output = {
			.pool = {
				.strides = 2,
				.size = 3,
				.border = 0,
			},
		},
	},
	// fifth layer (convolutional 512x2 => max pool)
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 13,
				.cols = 13,
				.channels = 512,
				.partition = 1,
			},
		},
		.output = {
			.convolutional = {
				.count = 512,
				.strides = 1,
				.border = 1,
				.rows = 3,
				.cols = 3,
				.channels = 512,
				.partition = 1,
			},
		},
	},
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 13,
				.cols = 13,
				.channels = 512,
				.partition = 1,
			},
		},
		.output = {
			.convolutional = {
				.count = 512,
				.strides = 1,
				.border = 1,
				.rows = 3,
				.cols = 3,
				.channels = 512,
				.partition = 1,
			},
		},
	},
	{
		.type = CCV_CONVNET_MAX_POOL,
		.input = {
			.matrix = {
				.rows = 13,
				.cols = 13,
				.channels = 512,
				.partition = 1,
			},
		},
		.output = {
			.pool = {
				.strides = 2,
				.size = 3,
				.border = 0,
			},
		},
	},
	// sixth layer (full connect)
	{
		.type = CCV_CONVNET_FULL_CONNECT,
		.bias = 0,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 6,
				.cols = 6,
				.channels = 512,
				.partition = 1,
			},
			.node = {
				.count = 6 * 6 * 512,
			},
		},
		.output = {
			.full_connect = {
				.relu = 1,
				.count = 4096,
			},
		},
	},
	// seventh layer (full connect)
	{
		.type = CCV_CONVNET_FULL_CONNECT,
		.bias = 0,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 4096,
				.cols = 1,
				.channels = 1,
				.partition = 1,
			},
			.node = {
				.count = 4096,
			},
		},
		.output = {
			.full_connect = {
				.relu = 1,
				.count = 4096,
			},
		},
	},
	// eighth layer (full connect)
	{
		.type = CCV_CONVNET_FULL_CONNECT,
		.bias = 0,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 4096,
				.cols = 1,
				.channels = 1,
				.partition = 1,
			},
			.node = {
				.count = 4096,
			},
		},
		.output = {
			.full_connect = {
				.relu = 0,
				.count = 1000,
			},
		},
	},
};
// VGG Configuration D Model
ccv_convnet_layer_param_t vgg_d_params[21] = {
	// first layer (convolutional 64x2 => max pool)
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 225,
				.cols = 225,
				.channels = 3,
				.partition = 1,
			},
		},
		.output = {
			.convolutional = {
				.count = 64,
				.strides = 1,
				.border = 0,
				.rows = 3,
				.cols = 3,
				.channels = 3,
				.partition = 1,
			},
		},
	},
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 223,
				.cols = 223,
				.channels = 64,
				.partition = 1,
			},
		},
		.output = {
			.convolutional = {
				.count = 64,
				.strides = 1,
				.border = 1,
				.rows = 3,
				.cols = 3,
				.channels = 64,
				.partition = 1,
			},
		},
	},
	{
		.type = CCV_CONVNET_MAX_POOL,
		.input = {
			.matrix = {
				.rows = 223,
				.cols = 223,
				.channels = 64,
				.partition = 1,
			},
		},
		.output = {
			.pool = {
				.strides = 2,
				.size = 3,
				.border = 0,
			},
		},
	},
	// second layer (convolutional 128x2 => max pool)
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 111,
				.cols = 111,
				.channels = 64,
				.partition = 1,
			},
		},
		.output = {
			.convolutional = {
				.count = 128,
				.strides = 1,
				.border = 1,
				.rows = 3,
				.cols = 3,
				.channels = 64,
				.partition = 1,
			},
		},
	},
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 111,
				.cols = 111,
				.channels = 128,
				.partition = 1,
			},
		},
		.output = {
			.convolutional = {
				.count = 128,
				.strides = 1,
				.border = 1,
				.rows = 3,
				.cols = 3,
				.channels = 128,
				.partition = 1,
			},
		},
	},
	{
		.type = CCV_CONVNET_MAX_POOL,
		.input = {
			.matrix = {
				.rows = 111,
				.cols = 111,
				.channels = 128,
				.partition = 1,
			},
		},
		.output = {
			.pool = {
				.strides = 2,
				.size = 3,
				.border = 0,
			},
		},
	},
	// third layer (convolutional 256x3 => max pool)
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 55,
				.cols = 55,
				.channels = 128,
				.partition = 1,
			},
		},
		.output = {
			.convolutional = {
				.count = 256,
				.strides = 1,
				.border = 1,
				.rows = 3,
				.cols = 3,
				.channels = 128,
				.partition = 1,
			},
		},
	},
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 55,
				.cols = 55,
				.channels = 256,
				.partition = 1,
			},
		},
		.output = {
			.convolutional = {
				.count = 256,
				.strides = 1,
				.border = 1,
				.rows = 3,
				.cols = 3,
				.channels = 256,
				.partition = 1,
			},
		},
	},
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 55,
				.cols = 55,
				.channels = 256,
				.partition = 1,
			},
		},
		.output = {
			.convolutional = {
				.count = 256,
				.strides = 1,
				.border = 1,
				.rows = 3,
				.cols = 3,
				.channels = 256,
				.partition = 1,
			},
		},
	},
	{
		.type = CCV_CONVNET_MAX_POOL,
		.input = {
			.matrix = {
				.rows = 55,
				.cols = 55,
				.channels = 256,
				.partition = 1,
			},
		},
		.output = {
			.pool = {
				.strides = 2,
				.size = 3,
				.border = 0,
			},
		},
	},
	// forth layer (convolutional 512x3 => max pool)
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 27,
				.cols = 27,
				.channels = 256,
				.partition = 1,
			},
		},
		.output = {
			.convolutional = {
				.count = 512,
				.strides = 1,
				.border = 1,
				.rows = 3,
				.cols = 3,
				.channels = 256,
				.partition = 1,
			},
		},
	},
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 27,
				.cols = 27,
				.channels = 512,
				.partition = 1,
			},
		},
		.output = {
			.convolutional = {
				.count = 512,
				.strides = 1,
				.border = 1,
				.rows = 3,
				.cols = 3,
				.channels = 512,
				.partition = 1,
			},
		},
	},
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 27,
				.cols = 27,
				.channels = 512,
				.partition = 1,
			},
		},
		.output = {
			.convolutional = {
				.count = 512,
				.strides = 1,
				.border = 1,
				.rows = 3,
				.cols = 3,
				.channels = 512,
				.partition = 1,
			},
		},
	},
	{
		.type = CCV_CONVNET_MAX_POOL,
		.input = {
			.matrix = {
				.rows = 27,
				.cols = 27,
				.channels = 512,
				.partition = 1,
			},
		},
		.output = {
			.pool = {
				.strides = 2,
				.size = 3,
				.border = 0,
			},
		},
	},
	// fifth layer (convolutional 512x3 => max pool)
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 13,
				.cols = 13,
				.channels = 512,
				.partition = 1,
			},
		},
		.output = {
			.convolutional = {
				.count = 512,
				.strides = 1,
				.border = 1,
				.rows = 3,
				.cols = 3,
				.channels = 512,
				.partition = 1,
			},
		},
	},
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 13,
				.cols = 13,
				.channels = 512,
				.partition = 1,
			},
		},
		.output = {
			.convolutional = {
				.count = 512,
				.strides = 1,
				.border = 1,
				.rows = 3,
				.cols = 3,
				.channels = 512,
				.partition = 1,
			},
		},
	},
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 13,
				.cols = 13,
				.channels = 512,
				.partition = 1,
			},
		},
		.output = {
			.convolutional = {
				.count = 512,
				.strides = 1,
				.border = 1,
				.rows = 3,
				.cols = 3,
				.channels = 512,
				.partition = 1,
			},
		},
	},
	{
		.type = CCV_CONVNET_MAX_POOL,
		.input = {
			.matrix = {
				.rows = 13,
				.cols = 13,
				.channels = 512,
				.partition = 1,
			},
		},
		.output = {
			.pool = {
				.strides = 2,
				.size = 3,
				.border = 0,
			},
		},
	},
	// sixth layer (full connect)
	{
		.type = CCV_CONVNET_FULL_CONNECT,
		.bias = 0,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 6,
				.cols = 6,
				.channels = 512,
				.partition = 1,
			},
			.node = {
				.count = 6 * 6 * 512,
			},
		},
		.output = {
			.full_connect = {
				.relu = 1,
				.count = 4096,
			},
		},
	},
	// seventh layer (full connect)
	{
		.type = CCV_CONVNET_FULL_CONNECT,
		.bias = 0,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 4096,
				.cols = 1,
				.channels = 1,
				.partition = 1,
			},
			.node = {
				.count = 4096,
			},
		},
		.output = {
			.full_connect = {
				.relu = 1,
				.count = 4096,
			},
		},
	},
	// eighth layer (full connect)
	{
		.type = CCV_CONVNET_FULL_CONNECT,
		.bias = 0,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 4096,
				.cols = 1,
				.channels = 1,
				.partition = 1,
			},
			.node = {
				.count = 4096,
			},
		},
		.output = {
			.full_connect = {
				.relu = 0,
				.count = 1000,
			},
		},
	},
};
