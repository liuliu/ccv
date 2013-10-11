#include "ccv.h"
#include <sys/time.h>
#include <ctype.h>

unsigned int get_current_time()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

int main(int argc, char** argv)
{
	ccv_enable_default_cache();
	ccv_convnet_param_t params[] = {
		{
			.type = CCV_CONVNET_CONVOLUTIONAL,
			.bias = 1,
			.sigma = 0.0001,
			.input = {
				.matrix = {
					.rows = 31,
					.cols = 31,
					.channels = 3,
				},
			},
			.output = {
				.convolutional = {
					.rows = 5,
					.cols = 5,
					.channels = 3,
					.border = 2,
					.strides = 1,
					.count = 32,
				},
			},
		},
		{
			.type = CCV_CONVNET_MAX_POOL,
			.input = {
				.matrix = {
					.rows = 31,
					.cols = 31,
					.channels = 32,
				},
			},
			.output = {
				.pool = {
					.size = 3,
					.strides = 2,
				},
			},
		},
		{
			.type = CCV_CONVNET_CONVOLUTIONAL,
			.bias = 1,
			.sigma = 0.01,
			.input = {
				.matrix = {
					.rows = 15,
					.cols = 15,
					.channels = 32,
				},
			},
			.output = {
				.convolutional = {
					.rows = 5,
					.cols = 5,
					.channels = 32,
					.border = 2,
					.strides = 1,
					.count = 32,
				},
			},
		},
		{
			.type = CCV_CONVNET_AVERAGE_POOL,
			.input = {
				.matrix = {
					.rows = 15,
					.cols = 15,
					.channels = 32,
				},
			},
			.output = {
				.pool = {
					.size = 3,
					.strides = 2,
				},
			},
		},
		{
			.type = CCV_CONVNET_CONVOLUTIONAL,
			.bias = 1,
			.sigma = 0.01,
			.input = {
				.matrix = {
					.rows = 7,
					.cols = 7,
					.channels = 32,
				},
			},
			.output = {
				.convolutional = {
					.rows = 5,
					.cols = 5,
					.channels = 32,
					.border = 2,
					.strides = 1,
					.count = 64,
				},
			},
		},
		{
			.type = CCV_CONVNET_AVERAGE_POOL,
			.input = {
				.matrix = {
					.rows = 7,
					.cols = 7,
					.channels = 64,
				},
			},
			.output = {
				.pool = {
					.size = 3,
					.strides = 2,
				},
			},
		},
		{
			.type = CCV_CONVNET_FULL_CONNECT,
			.bias = 1,
			.sigma = 0.01,
			.input = {
				.matrix = {
					.rows = 3,
					.cols = 3,
					.channels = 64,
				},
			},
			.output = {
				.full_connect = {
					.count = 10,
				},
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(params, 7);
	ccv_convnet_free(convnet);
	ccv_disable_cache();
	return 0;
}
