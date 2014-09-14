#ifndef GUARD_ccv_convnet_inl_h
#define GUARD_ccv_convnet_inl_h

inline static void ccv_convnet_make_output(ccv_convnet_layer_t* layer, int input_rows, int input_cols, int* rows, int* cols, int* partition)
{
	assert(rows != 0 && cols != 0);
	switch(layer->type)
	{
		case CCV_CONVNET_CONVOLUTIONAL:
			assert(layer->net.convolutional.rows % 2); // as of now, don't support even number of kernel size
			assert(layer->net.convolutional.cols % 2);
			assert((input_rows + layer->net.convolutional.border * 2 - layer->net.convolutional.rows) % layer->net.convolutional.strides == 0);
			assert((input_cols + layer->net.convolutional.border * 2 - layer->net.convolutional.cols) % layer->net.convolutional.strides == 0);
			*rows = (input_rows + layer->net.convolutional.border * 2 - layer->net.convolutional.rows + layer->net.convolutional.strides - 1) / layer->net.convolutional.strides + 1;
			*cols = (input_cols + layer->net.convolutional.border * 2 - layer->net.convolutional.cols + layer->net.convolutional.strides - 1) / layer->net.convolutional.strides + 1;
			*partition = layer->input.matrix.partition;
			break;
		case CCV_CONVNET_FULL_CONNECT:
			*rows = layer->net.full_connect.count;
			*cols = 1;
			*partition = 1;
			break;
		case CCV_CONVNET_LOCAL_RESPONSE_NORM:
			*rows = input_rows;
			*cols = input_cols;
			*partition = layer->input.matrix.partition;
			break;
		case CCV_CONVNET_MAX_POOL:
		case CCV_CONVNET_AVERAGE_POOL:
			assert((input_rows + layer->net.pool.border * 2 - layer->net.pool.size) % layer->net.pool.strides == 0);
			assert((input_cols + layer->net.pool.border * 2 - layer->net.pool.size) % layer->net.pool.strides == 0);
			*rows = (input_rows + layer->net.pool.border * 2 - layer->net.pool.size + layer->net.pool.strides - 1) / layer->net.pool.strides + 1;
			*cols = (input_cols + layer->net.pool.border * 2 - layer->net.pool.size + layer->net.pool.strides - 1) / layer->net.pool.strides + 1;
			*partition = layer->input.matrix.partition;
			break;
		default:
			assert(0);
			break;
	}
}

#endif
