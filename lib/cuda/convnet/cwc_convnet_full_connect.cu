#include <cuda.h>
#include <cublas_v2.h>
extern "C" {
#include "../cwc.h"
#include "../cwc_internal.h"
}
#include "../../inl/ccv_convnet_inl.h"

__global__ static void _cwc_kern_relu_forward_propagate(float* a)
{
	a += blockIdx.x * blockDim.x;
	const int thidx = threadIdx.x;
	a[thidx] = max(0.0, a[thidx]);
}

void cwc_convnet_full_connect_forward_propagate(ccv_convnet_layer_t* layer, int batch, float* a, float* b, float* batch_unit /* this is just 1's in device */, const cudaStream_t& stream, const cublasHandle_t& handle)
{
	int rows, out_rows, out_cols, out_partition;
	ccv_convnet_make_output(layer, layer->input.matrix.rows, layer->input.matrix.cols, &out_rows, &out_cols, &out_partition);
	out_cols = batch;
	rows = layer->input.matrix.rows * layer->input.matrix.cols * layer->input.matrix.channels;
	// make copies of bias into db's columns, note that for cublas, it is row-major matrix
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, batch, out_rows, 1, &one, batch_unit, batch, layer->bias, 1, &zero, b, batch);
	// and then do the GEMM by adding bias
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, batch, out_rows, rows, &one, a, batch, layer->w, rows, &one, b, batch);
	if (layer->net.full_connect.relu)
		_cwc_kern_relu_forward_propagate
		<<<layer->net.full_connect.count, batch, 0, stream>>>
		(b);

}

void cwc_convnet_full_connect_backward_propagate(ccv_convnet_layer_t* layer, int batch, float* a, float* n, float* m, float* b, float* batch_unit, float* w, float* bias, const cudaStream_t& stream, const cublasHandle_t& handle)
{
	int rows, out_rows, out_cols, out_partition;
	ccv_convnet_make_output(layer, layer->input.matrix.rows, layer->input.matrix.cols, &out_rows, &out_cols, &out_partition);
	out_cols = batch;
	rows = layer->input.matrix.rows * layer->input.matrix.cols * layer->input.matrix.channels;
	// apply relu for full connect layer, not that this requires both n and a, and for the last full connect layer, we re-used the forwards, thus, it required the last full connect layer to not have relu enabled
	if (layer->net.full_connect.relu)
		cwc_kern_relu_backward_propagate
		<<<dim3(1, out_rows, 1), batch, 0, stream>>>
		(batch, n, a, out_rows, 1, 1);
	// propagate bias
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, out_rows, batch, &one, batch_unit, 1, a, batch, &one, bias, 1);
	// propagate error
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, batch, rows, out_rows, &one, a, batch, layer->w, rows, &zero, b, batch);
	// propagate weights
	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, out_rows, batch, &one, m, batch, a, batch, &one, w, rows);
}
