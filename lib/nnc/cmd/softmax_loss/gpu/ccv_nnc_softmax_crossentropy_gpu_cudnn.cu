extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

#ifdef HAVE_CUDNN

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_softmax_crossentropy_forw_kernel(const int batch_size, const int count, const NUM1* const label, const NUM2* const a, NUM2* const c)
{
	CUDA_1D_KERNEL_LOOP(i, batch_size) {
		const int idx = (int)((float)label[i] + 0.5);
		c[i] -= a[i * count + idx];
	}
}

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_softmax_crossentropy_forw_kernel_trim(const int batch_size, const int count, const float trim0, const float trim1, const NUM1* const label, const NUM2* const a, NUM2* const c)
{
	CUDA_1D_KERNEL_LOOP(i, batch_size) {
		const int idx = (int)((float)label[i] + 0.5);
		const NUM2 maxval = c[i];
		NUM2 p = (NUM2)trim1 * (maxval - a[i * count + idx]);
		for (int j = 0; j < idx; j++)
			p += (NUM2)trim0 * (maxval - a[i * count + j]);
		for (int j = idx + 1; j < count; j++)
			p += (NUM2)trim0 * (maxval - a[i * count + j]);
		c[i] = p;
	}
}

template<typename NUM>
__global__ void _ccv_nnc_softmax_crossentropy_one_hot_forw_kernel(const int batch_size, const int count, const NUM* const label, const NUM* const a, NUM* const c)
{
	CUDA_1D_KERNEL_LOOP(i, batch_size) {
		const NUM maxval = c[i];
		NUM p = label[i * count] * (maxval - a[i * count]);
		for (int j = 1; j < count; j++)
			p += label[i * count + j] * (maxval - a[i * count + j]);
		c[i] = p;
	}
}

template<typename NUM>
__global__ void _ccv_nnc_softmax_crossentropy_forw_kernel(const int batch_size, const int count, const int* const label, const NUM* const a, NUM* const c)
{
	CUDA_1D_KERNEL_LOOP(i, batch_size) {
		c[i] -= a[i * count + label[i]];
	}
}

template<typename NUM>
__global__ void _ccv_nnc_softmax_crossentropy_forw_kernel_trim(const int batch_size, const int count, const float trim0, const float trim1, const int* const label, const NUM* const a, NUM* const c)
{
	CUDA_1D_KERNEL_LOOP(i, batch_size) {
		const int idx = label[i];
		const NUM maxval = c[i];
		NUM p = (NUM)trim1 * (maxval - a[i * count + idx]);
		for (int j = 0; j < idx; j++)
			p += (NUM)trim0 * (maxval - a[i * count + j]);
		for (int j = idx + 1; j < count; j++)
			p += (NUM)trim0 * (maxval - a[i * count + j]);
		c[i] = p;
	}
}

static int _ccv_nnc_softmax_crossentropy_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 2);
	assert(output_size == 2);
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	const ccv_nnc_cudnn_tensor_view_descriptor_t a = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[0]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t b = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)outputs[1]);
	static const float one = 1, zero = 0;
	CUDNN_ENFORCE(cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &one, a.descriptor, a.data.u8, &zero, b.descriptor, b.data.u8));
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(b);
	if (outputs[0])
	{
		const int axis_count = ccv_nnc_tensor_nd(inputs[0]->info.dim);
		const int batch_size = axis_count < 2 ? 1 : inputs[0]->info.dim[0];
		const int count = ccv_nnc_tensor_count(inputs[0]->info) / batch_size;
		assert(CCV_IS_TENSOR_CONTIGUOUS(outputs[0]));
		// Be explicit the c tensor size.
		ccv_nnc_tensor_param_t c_info = outputs[0]->info;
		c_info.dim[0] = batch_size;
		c_info.dim[1] = 1;
		ccv_nnc_tensor_t c_tensor = ccv_nnc_tensor(outputs[0]->data.f32, c_info, 0);
		const ccv_nnc_cudnn_tensor_view_descriptor_t c = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)&c_tensor);
		cudnnReduceTensorDescriptor_t reduce_max = ccv_nnc_stream_context_get_reduce_tensor_descriptor(stream_context);
		cudnnSetReduceTensorDescriptor(reduce_max, CUDNN_REDUCE_TENSOR_MAX, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES);
		size_t workspace_size = 0;
		CUDNN_ENFORCE(cudnnGetReductionWorkspaceSize(cudnn, reduce_max, a.descriptor, c.descriptor, &workspace_size));
		void* workspace = 0;
		if (workspace_size)
			workspace = ccv_nnc_stream_context_get_workspace(stream_context, workspace_size, CCV_TENSOR_GPU_MEMORY);
		CUDNN_ENFORCE(cudnnReduceTensor(cudnn, reduce_max, 0, 0, workspace, workspace_size, &one, a.descriptor, a.data.u8, &zero, c.descriptor, c.data.u8));
		ccv_nnc_stream_context_return_reduce_tensor_descriptor(stream_context, reduce_max);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(c);
		assert(inputs[0]->info.datatype == outputs[0]->info.datatype);
		const int datatype = inputs[0]->info.datatype;
		if (inputs[1]->info.datatype == CCV_32F || inputs[1]->info.datatype == CCV_16F)
		{
			// If has more than 1 axis, then the range is the channel count. Otherwise, if our batch size is 1, then the range is
			// the channel count. Otherwise, the range is 1 (and the only axis is the batch size).
			const int range = ccv_nnc_tensor_nd(inputs[1]->info.dim) > 1 ? ccv_nnc_tensor_get_c(inputs[1]->info) : (batch_size == 1 ? inputs[1]->info.dim[0] : 1);
			if (range == 1)
			{
				const float trim0 = cmd.info.label_smoothing.trim0;
				const float trim1 = cmd.info.label_smoothing.trim1;
				if (trim0 == 0 && trim1 == 1)
				{
					if (inputs[1]->info.datatype == CCV_32F)
					{
						if (datatype == CCV_16F)
							_ccv_nnc_softmax_crossentropy_forw_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, inputs[1]->data.f32, (__half*)inputs[0]->data.f16, (__half*)outputs[0]->data.f16);
						else
							_ccv_nnc_softmax_crossentropy_forw_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, inputs[1]->data.f32, inputs[0]->data.f32, outputs[0]->data.f32);
					} else {
						assert(inputs[1]->info.datatype == CCV_16F);
						assert(datatype == CCV_16F);
						_ccv_nnc_softmax_crossentropy_forw_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)inputs[1]->data.f16, (__half*)inputs[0]->data.f16, (__half*)outputs[0]->data.f16);
					}
				} else {
					if (inputs[1]->info.datatype == CCV_32F)
					{
						if (datatype == CCV_16F)
							_ccv_nnc_softmax_crossentropy_forw_kernel_trim<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, trim0, trim1, inputs[1]->data.f32, (__half*)inputs[0]->data.f16, (__half*)outputs[0]->data.f16);
						else
							_ccv_nnc_softmax_crossentropy_forw_kernel_trim<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, trim0, trim1, inputs[1]->data.f32, inputs[0]->data.f32, outputs[0]->data.f32);
					} else {
						assert(inputs[1]->info.datatype == CCV_16F);
						assert(datatype == CCV_16F);
						_ccv_nnc_softmax_crossentropy_forw_kernel_trim<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, trim0, trim1, (__half*)inputs[1]->data.f16, (__half*)inputs[0]->data.f16, (__half*)outputs[0]->data.f16);
					}
				}
			} else {
				assert(inputs[1]->info.datatype == datatype);
				if (datatype == CCV_16F)
					_ccv_nnc_softmax_crossentropy_one_hot_forw_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)inputs[1]->data.f16, (__half*)inputs[0]->data.f16, (__half*)outputs[0]->data.f16);
				else
					_ccv_nnc_softmax_crossentropy_one_hot_forw_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, inputs[1]->data.f32, inputs[0]->data.f32, outputs[0]->data.f32);
			}
		} else if (inputs[1]->info.datatype == CCV_32S) {
			const float trim0 = cmd.info.label_smoothing.trim0;
			const float trim1 = cmd.info.label_smoothing.trim1;
			if (trim0 == 0 && trim1 == 1)
			{
				if (datatype == CCV_16F)
					_ccv_nnc_softmax_crossentropy_forw_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, inputs[1]->data.i32, (__half*)inputs[0]->data.f16, (__half*)outputs[0]->data.f16);
				else
					_ccv_nnc_softmax_crossentropy_forw_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, inputs[1]->data.i32, inputs[0]->data.f32, outputs[0]->data.f32);
			} else {
				if (datatype == CCV_16F)
					_ccv_nnc_softmax_crossentropy_forw_kernel_trim<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, trim0, trim1, inputs[1]->data.i32, (__half*)inputs[0]->data.f16, (__half*)outputs[0]->data.f16);
				else
					_ccv_nnc_softmax_crossentropy_forw_kernel_trim<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, trim0, trim1, inputs[1]->data.i32, inputs[0]->data.f32, outputs[0]->data.f32);
			}
		}
	}
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(a);
	return CCV_NNC_EXEC_SUCCESS;
}

template<typename NUM>
__global__ void _ccv_nnc_copy_kernel(const int n, const NUM* const a, NUM* const b)
{
	CUDA_1D_KERNEL_LOOP(i, n) {
		b[i] = a[i];
	}
}

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_softmax_crossentropy_back_kernel(const int batch_size, const int count, const NUM1* const label, NUM2* const h)
{
	CUDA_1D_KERNEL_LOOP(i, batch_size) {
		const int idx = (int)((float)label[i] + 0.5);
		h[i * count + idx] -= 1;
	}
}

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_softmax_crossentropy_back_kernel_trim(const int batch_size_count, const int count, const float trim0, const float trim1, const NUM1* const label, NUM2* const h)
{
	CUDA_1D_KERNEL_LOOP(i, batch_size_count) {
		const int idx = i / count;
		const int batch_idx = i % count;
		const int lbl = (int)((float)label[idx] + 0.5);
		if (batch_idx == lbl)
			h[i] -= (NUM2)trim1;
		else
			h[i] -= (NUM2)trim0;
	}
}

template<typename NUM>
__global__ void _ccv_nnc_softmax_crossentropy_one_hot_back_kernel(const int batch_size_count, const int count, const NUM* const label, NUM* const h)
{
	CUDA_1D_KERNEL_LOOP(i, batch_size_count) {
		h[i] -= label[i];
	}
}

template<typename NUM>
__global__ void _ccv_nnc_softmax_crossentropy_back_kernel(const int batch_size, const int count, const int* const label, NUM* const h)
{
	CUDA_1D_KERNEL_LOOP(i, batch_size) {
		h[i * count + label[i]] -= 1;
	}
}

template<typename NUM>
__global__ void _ccv_nnc_softmax_crossentropy_back_kernel_trim(const int batch_size_count, const int count, const float trim0, const float trim1, const int* const label, NUM* const h)
{
	CUDA_1D_KERNEL_LOOP(i, batch_size_count) {
		const int idx = i / count;
		const int batch_idx = i % count;
		const int lbl = label[idx];
		if (batch_idx == lbl)
			h[i] -= (NUM)trim1;
		else
			h[i] -= (NUM)trim0;
	}
}

static int _ccv_nnc_softmax_crossentropy_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 6);
	assert(output_size >= 1);
	const ccv_nnc_tensor_t* b = inputs[3];
	assert(CCV_IS_TENSOR_CONTIGUOUS(b));
	const ccv_nnc_tensor_t* d = inputs[5];
	assert(CCV_IS_TENSOR_CONTIGUOUS(d));
	ccv_nnc_tensor_t* h = outputs[0];
	assert(CCV_IS_TENSOR_CONTIGUOUS(h));
	const int axis_count = ccv_nnc_tensor_nd(d->info.dim);
	const int batch_size = axis_count < 2 ? 1 : d->info.dim[0];
	const int bcount = ccv_nnc_tensor_count(d->info);
	const int count = bcount / batch_size;
	int i;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && d->info.dim[i] > 0; i++)
		{ assert(d->info.dim[i] == h->info.dim[i]); }
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	assert(h->info.datatype == d->info.datatype);
	if (h->info.datatype == CCV_16F)
		_ccv_nnc_copy_kernel<<<CUDA_GET_BLOCKS(bcount), CUDA_NUM_THREADS, 0, stream>>>(bcount, (__half*)d->data.f16, (__half*)h->data.f16);
	else
		_ccv_nnc_copy_kernel<<<CUDA_GET_BLOCKS(bcount), CUDA_NUM_THREADS, 0, stream>>>(bcount, d->data.f32, h->data.f32);
	if (b->info.datatype == CCV_32F || b->info.datatype == CCV_16F)
	{
		// If has more than 1 axis, then the range is the channel count. Otherwise, if our batch size is 1, then the range is
		// the channel count. Otherwise, the range is 1 (and the only axis is the batch size).
		const int range = ccv_nnc_tensor_nd(b->info.dim) > 1 ? ccv_nnc_tensor_get_c(b->info) : (batch_size == 1 ? b->info.dim[0] : 1);
		if (range == 1)
		{
			const float trim0 = cmd.info.label_smoothing.trim0;
			const float trim1 = cmd.info.label_smoothing.trim1;
			if (trim0 == 0 && trim1 == 1)
			{
				if (b->info.datatype == CCV_32F)
				{
					if (h->info.datatype == CCV_16F)
						_ccv_nnc_softmax_crossentropy_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, b->data.f32, (__half*)h->data.f16);
					else
						_ccv_nnc_softmax_crossentropy_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, b->data.f32, h->data.f32);
				} else {
					assert(b->info.datatype == CCV_16F);
					assert(h->info.datatype == CCV_16F);
					_ccv_nnc_softmax_crossentropy_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)b->data.f16, (__half*)h->data.f16);
				}
			} else {
				if (b->info.datatype == CCV_32F)
				{
					if (h->info.datatype == CCV_16F)
						_ccv_nnc_softmax_crossentropy_back_kernel_trim<<<CUDA_GET_BLOCKS(bcount), CUDA_NUM_THREADS, 0, stream>>>(bcount, count, trim0, trim1, b->data.f32, (__half*)h->data.f16);
					else
						_ccv_nnc_softmax_crossentropy_back_kernel_trim<<<CUDA_GET_BLOCKS(bcount), CUDA_NUM_THREADS, 0, stream>>>(bcount, count, trim0, trim1, b->data.f32, h->data.f32);
				} else {
					assert(b->info.datatype == CCV_16F);
					assert(h->info.datatype == CCV_16F);
					_ccv_nnc_softmax_crossentropy_back_kernel_trim<<<CUDA_GET_BLOCKS(bcount), CUDA_NUM_THREADS, 0, stream>>>(bcount, count, trim0, trim1, (__half*)b->data.f16, (__half*)h->data.f16);
				}
			}
		} else {
			assert(b->info.datatype == h->info.datatype);
			if (h->info.datatype == CCV_16F)
				_ccv_nnc_softmax_crossentropy_one_hot_back_kernel<<<CUDA_GET_BLOCKS(bcount), CUDA_NUM_THREADS, 0, stream>>>(bcount, count, (__half*)b->data.f16, (__half*)h->data.f16);
			else
				_ccv_nnc_softmax_crossentropy_one_hot_back_kernel<<<CUDA_GET_BLOCKS(bcount), CUDA_NUM_THREADS, 0, stream>>>(bcount, count, b->data.f32, h->data.f32);
		}
	} else if (b->info.datatype == CCV_32S) {
		const float trim0 = cmd.info.label_smoothing.trim0;
		const float trim1 = cmd.info.label_smoothing.trim1;
		if (trim0 == 0 && trim1 == 1)
		{
			if (h->info.datatype == CCV_16F)
				_ccv_nnc_softmax_crossentropy_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, b->data.i32, (__half*)h->data.f16);
			else
				_ccv_nnc_softmax_crossentropy_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, b->data.i32, h->data.f32);
		} else {
			if (h->info.datatype == CCV_16F)
				_ccv_nnc_softmax_crossentropy_back_kernel_trim<<<CUDA_GET_BLOCKS(bcount), CUDA_NUM_THREADS, 0, stream>>>(bcount, count, trim0, trim1, b->data.i32, (__half*)h->data.f16);
			else
				_ccv_nnc_softmax_crossentropy_back_kernel_trim<<<CUDA_GET_BLOCKS(bcount), CUDA_NUM_THREADS, 0, stream>>>(bcount, count, trim0, trim1, b->data.i32, h->data.f32);
		}
	}
	if (inputs[0])
	{
		cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
		static const float one = 1, zero = 0;
		assert(CCV_IS_TENSOR_CONTIGUOUS(inputs[0]));
		ccv_nnc_tensor_param_t g_info = inputs[0]->info;
		g_info.dim[0] = batch_size;
		g_info.dim[1] = 1;
		ccv_nnc_tensor_t g_tensor = ccv_nnc_tensor(inputs[0]->data.f32, g_info, 0);
		const ccv_nnc_cudnn_tensor_view_descriptor_t g = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)&g_tensor);
		const ccv_nnc_cudnn_tensor_view_descriptor_t hcu = ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(stream_context, (const ccv_nnc_tensor_view_t*)h);
		cudnnOpTensorDescriptor_t mul = ccv_nnc_stream_context_get_op_tensor_descriptor(stream_context);
		cudnnSetOpTensorDescriptor(mul, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);
		CUDNN_ENFORCE(cudnnOpTensor(cudnn, mul, &one, hcu.descriptor, hcu.data.u8, &one, g.descriptor, g.data.u8, &zero, hcu.descriptor, hcu.data.u8));
		ccv_nnc_stream_context_return_op_tensor_descriptor(stream_context, mul);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(g);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(hcu);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_SOFTMAX_CROSSENTROPY_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_32S | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_softmax_crossentropy_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SOFTMAX_CROSSENTROPY_BACKWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_32S | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_softmax_crossentropy_back;
#endif
}
