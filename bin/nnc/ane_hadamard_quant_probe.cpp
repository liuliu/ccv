#include "nnc/mfa/ccv_nnc_mfa.hpp"
#include "nnc/mfa/ccv_nnc_mfa_ane_rowwise_internal.hpp"
#include "nnc/mfa/kernels/ANERowwiseTransformDescriptor.hpp"
#include "nnc/mfa/kernels/ANERowwiseTransformKernel.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

extern "C" ccv_nnc_mfa_context_t* ccv_nnc_default_mfa_context(void);

static float store_value(void* data, uint32_t type, size_t i, float value)
{
	if (type == MTL::DataTypeFloat) {
		((float*)data)[i] = value;
	} else if (type == MTL::DataTypeHalf) {
		ccv_float_to_half_precision(&value, (uint16_t*)data + i, 1);
		ccv_half_precision_to_float((uint16_t*)data + i, &value, 1);
	} else {
		ccv_float_to_bfloat(&value, (uint16_t*)data + i, 1);
		ccv_bfloat_to_float((uint16_t*)data + i, &value, 1);
	}
	return value;
}

static void encode(MTL::CommandBuffer* cb, PipelineValue<ANERowwiseTransformKernel>* p,
	const ANERowwiseTransformDescriptor& d, MTL::Buffer* src, MTL::Buffer* scales, MTL::Buffer* output)
{
	auto e = cb->computeCommandEncoder();
	e->setComputePipelineState(p->pipeline.get());
	e->setBuffer(src, 0, 0);
	e->setBuffer(scales, 0, 1);
	const bool fused = d.K <= 8192;
	const bool row_prepare = d.activationHadamard256 || fused;
	if (fused)
		e->setBuffer(output, 0, 2);
	e->dispatchThreadgroups(row_prepare ? p->kernel->activationPrepareGridSize(d.paddedM, d.K) : MTL::Size(d.paddedM, 1, 1),
		row_prepare ? p->kernel->activationPrepareThreadgroupSize(d.K) : MTL::Size(256, 1, 1));
	e->endEncoding();
	if (!fused) {
		e = cb->computeCommandEncoder();
		e->setComputePipelineState(p->second.get());
		e->setBuffer(src, 0, 0);
		e->setBuffer(scales, 0, 1);
		e->setBuffer(output, 0, 2);
		e->dispatchThreadgroups(p->kernel->activationQuantizeGridSize(d.paddedM, d.K), p->kernel->activationQuantizeThreadgroupSize());
		e->endEncoding();
	}
}

static int validate(ccv_nnc_mfa_context_t* context, uint32_t type, uint32_t m, uint32_t k, bool hadamard)
{
	ANERowwiseTransformDescriptor d = {};
	d.memoryPrecision = type == MTL::DataTypeFloat ? GEMMOperandPrecision::FP32 :
		type == MTL::DataTypeHalf ? GEMMOperandPrecision::FP16 : GEMMOperandPrecision::BF16;
	d.activationHadamard256 = hadamard;
	d.M = m; d.batchDimension = 2; d.paddedM = ((m * 2 + 127) / 128) * 128;
	d.N = 256; d.K = k;
	d.sourceRowOffset = 1;
	d.batchStrideA = (m + 3) * k;
	auto p = ccv_nnc_mfa_prepare_ane_rowwise_transform(context, d);
	auto device = ccv_nnc_mfa_context_device(context);
	auto queue = NS::TransferPtr(device->newCommandQueue());
	auto src = NS::TransferPtr(device->newBuffer((size_t)d.batchStrideA * 2 * 4, MTL::ResourceStorageModeShared));
	auto scales = NS::TransferPtr(device->newBuffer(d.paddedM * 4, MTL::ResourceStorageModeShared));
	auto output = NS::TransferPtr(device->newBuffer((size_t)d.paddedM * k, MTL::ResourceStorageModeShared));
	std::vector<float> input((size_t)m * 2 * k);
	unsigned state = 42;
	for (uint32_t r = 0; r < m * 2; ++r)
		for (uint32_t j = 0; j < k; ++j) {
			state = state * 1664525u + 1013904223u;
			float value = r == 0 ? 0 : ((int)(state >> 16) - 32768) / 32768.f;
			if (r == 1 && j == 17) value = 8;
			// Exercise the same quantization at larger and smaller finite magnitudes.
			if (hadamard && r >= 2) value *= r % 3 == 0 ? 256.f : r % 3 == 1 ? 1.f / 256.f : 1.f;
			input[(size_t)r * k + j] = store_value(src->contents(), type,
				(r / m) * d.batchStrideA + ((r % m) + 1) * k + j, value);
		}
	auto cb = queue->commandBuffer();
	encode(cb, p, d, src.get(), scales.get(), output.get());
	cb->commit(); cb->waitUntilCompleted();
	if (cb->status() != MTL::CommandBufferStatusCompleted) return 1;
	int errors = 0;
	std::vector<float> actual_scales(d.paddedM);
	if (hadamard && k > 8192) {
		for (uint32_t r = 0; r < d.paddedM; ++r) {
			const float maximum = ((float*)scales->contents())[r];
			float scale = maximum > 0 ? maximum / 2032.f : 1.f / 127.f;
			if (type == MTL::DataTypeHalf)
				scale = (float)(_Float16)scale;
			else if (type == MTL::DataTypeBFloat) {
				uint32_t bits;
				memcpy(&bits, &scale, sizeof(bits));
				bits = (bits + 0x7fffu + ((bits >> 16) & 1)) & 0xffff0000u;
				memcpy(&scale, &bits, sizeof(bits));
			}
			actual_scales[r] = scale;
		}
	} else if (type == MTL::DataTypeHalf)
		ccv_half_precision_to_float((uint16_t*)scales->contents(), actual_scales.data(), d.paddedM);
	else if (type == MTL::DataTypeBFloat)
		ccv_bfloat_to_float((uint16_t*)scales->contents(), actual_scales.data(), d.paddedM);
	else
		memcpy(actual_scales.data(), scales->contents(), d.paddedM * sizeof(float));
	for (uint32_t r = 0; r < m * 2; ++r) {
		std::vector<double> rotated(k);
		for (uint32_t j = 0; j < k; ++j) rotated[j] = input[(size_t)r * k + j];
		// Independent scalar H4 Kronecker product in double precision.
		if (hadamard)
			for (uint32_t base = 0; base < k; base += 256) {
				for (uint32_t stride = 1; stride <= 64; stride *= 4)
					for (uint32_t b = base; b < base + 256; b += 4 * stride)
						for (uint32_t j = 0; j < stride; ++j) {
							double x[4];
							for (int l = 0; l < 4; ++l) x[l] = rotated[b + j + l * stride];
							const double sum = x[0] + x[1] + x[2] + x[3];
							for (int l = 0; l < 4; ++l) rotated[b + j + l * stride] = sum - 2 * x[3 - l];
						}
				for (uint32_t j = base; j < base + 256; ++j) rotated[j] /= 16;
			}
		float max_abs = 0;
		for (double x : rotated) max_abs = fmaxf(max_abs, fabs(x));
		float expected_scale = max_abs > 0 ? max_abs / 127.f : 1.f / 127.f;
		// Metal casts round to nearest; ccv's host storage converters truncate.
		if (type == MTL::DataTypeHalf)
			expected_scale = (float)(_Float16)expected_scale;
		else if (type == MTL::DataTypeBFloat) {
			uint32_t bits;
			memcpy(&bits, &expected_scale, sizeof(bits));
			bits = (bits + 0x7fffu + ((bits >> 16) & 1)) & 0xffff0000u;
			memcpy(&expected_scale, &bits, sizeof(bits));
		}
		if (fabsf(actual_scales[r] - expected_scale) > expected_scale * 1e-5f) ++errors;
		const float inv = hadamard ? (max_abs > 0 ? 127.f / max_abs : 127.f) : 1.f / expected_scale;
		for (uint32_t j = 0; j < k; ++j) {
			const float value = (float)rotated[j] * inv;
			const int expected = std::max(-127, std::min(127, (int)lrintf(value)));
			const int actual = ((int8_t*)output->contents())[(size_t)j * d.paddedM + r];
			// Fast reciprocal may land on either side of an exact half-integer tie.
			if (actual != expected && !(abs(actual - expected) == 1 && fabsf(fabsf(value - floorf(value)) - 0.5f) < 3e-4f)) ++errors;
		}
	}
	for (uint32_t r = m * 2; r < d.paddedM; ++r)
		for (uint32_t j = 0; j < k; ++j)
			if (((int8_t*)output->contents())[(size_t)j * d.paddedM + r] != ((int8_t*)output->contents())[(size_t)j * d.paddedM + m * 2 - 1]) ++errors;
	printf("type=%u M=%u K=%u H256=%d errors=%d\n", type, m, k, hadamard, errors);
	return errors != 0;
}

int main()
{
	auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
	ccv_nnc_init();
	auto context = ccv_nnc_default_mfa_context();
	printf("device=%s\n", ccv_nnc_mfa_context_device(context)->name()->utf8String());
	int failed = 0;
	for (uint32_t type : {uint32_t(MTL::DataTypeFloat), uint32_t(MTL::DataTypeHalf), uint32_t(MTL::DataTypeBFloat)})
		for (uint32_t k : {256u, 512u, 768u, 1024u, 1536u, 2304u, 3072u, 3840u, 4608u, 4096u, 5376u, 6144u, 7168u, 9216u, 12288u, 14336u, 16384u, 65536u}) {
			failed += validate(context, type, k == 65536 ? 3 : (k == 1024 ? 129 : 17), k, true);
			failed += validate(context, type, 3, k, false);
			failed += validate(context, type, 3, k, true);
		}
	for (uint32_t type : {uint32_t(MTL::DataTypeFloat), uint32_t(MTL::DataTypeHalf), uint32_t(MTL::DataTypeBFloat)})
		for (uint32_t k : {1u, 127u, 130u, 513u, 8192u, 8193u, 16384u, 16385u, 65537u})
			failed += validate(context, type, 129, k, false);
	return failed ? 1 : 0;
}
