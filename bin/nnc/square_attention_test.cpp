extern "C" {
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <sys/time.h>
#include <ctype.h>
}
#include "nnc/mfa/v2/ShaderCache.hpp"
#include "nnc/mfa/v2/AttentionDescriptor.hpp"
#include "nnc/mfa/v2/AttentionKernelDescriptor.hpp"
#include "nnc/mfa/v2/AttentionKernel.hpp"
#include "3rdparty/dsfmt/dSFMT.h"
#include <iostream>

#include <vector>
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>

struct NetworkDescriptor {
	int rowDimension;
	int columnDimension;
	int headDimension;
	float scale;
};

class Network {
private:
	int rowDimension;
	int columnDimension;
	int headDimension;
	float scale;

	static std::pair<float, float> boxMullerTransform() {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> dis(0.0, 1.0);

		float u1 = dis(gen);
		float u2 = dis(gen);

		float magnitudePart = std::sqrt(-2.0f * std::log(u1));
		float anglePart = 2.0f * M_PI * u2;

		return {
			magnitudePart * std::cos(anglePart),
			magnitudePart * std::sin(anglePart)
		};
	}

public:
	std::vector<float> Q;
	std::vector<float> K;
	std::vector<float> V;
	std::vector<float> dO;

	Network(const NetworkDescriptor& descriptor)
		: rowDimension(descriptor.rowDimension),
		  columnDimension(descriptor.columnDimension),
		  headDimension(descriptor.headDimension),
		  scale(descriptor.scale),
		  Q(rowDimension * headDimension),
		  K(columnDimension * headDimension),
		  V(columnDimension * headDimension),
		  dO(rowDimension * headDimension)
	{
		if (rowDimension <= 0 || columnDimension <= 0 || headDimension <= 0) {
			throw std::runtime_error("Descriptor was incomplete.");
		}

		for (int rowID = 0; rowID < rowDimension; ++rowID) {
			for (int d = 0; d < headDimension; ++d) {
				int matrixAddress = rowID * headDimension + d;
				auto [r1, r2] = boxMullerTransform();
				Q[matrixAddress] = r1;
				dO[matrixAddress] = r2;
			}
		}

		for (int columnID = 0; columnID < columnDimension; ++columnID) {
			for (int d = 0; d < headDimension; ++d) {
				int matrixAddress = columnID * headDimension + d;
				auto [r1, r2] = boxMullerTransform();
				K[matrixAddress] = r1;
				V[matrixAddress] = r2;
			}
		}
	}

	std::vector<float> createMatrixSRow(int rowID) const {
		std::vector<float> output(columnDimension, 0.0f);

		for (int columnID = 0; columnID < columnDimension; ++columnID) {
			float dotProduct = 0.0f;
			for (int d = 0; d < headDimension; ++d) {
				int addressQ = rowID * headDimension + d;
				int addressK = columnID * headDimension + d;
				dotProduct += Q[addressQ] * K[addressK];
			}
			output[columnID] = dotProduct;
		}

		return output;
	}

	std::vector<float> createMatrixPRow(int rowID) const {
		std::vector<float> output = createMatrixSRow(rowID);
		float scaleFactor = scale;

		float maximum = *std::max_element(output.begin(), output.end()) * scaleFactor;

		float sum = 0.0f;
		for (float& value : output) {
			value *= scaleFactor;
			float expTerm = std::exp(value - maximum);
			sum += expTerm;
		}

		float lse = maximum + std::log(sum);
		for (float& value : output) {
			value = std::exp(value - lse);
		}

		return output;
	}

	float createLTerm(int rowID) const {
		std::vector<float> matrixSRow = createMatrixSRow(rowID);
		float scaleFactor = 1.0f / std::sqrt(static_cast<float>(headDimension));

		float maximum = *std::max_element(matrixSRow.begin(), matrixSRow.end()) * scaleFactor;

		float sum = 0.0f;
		for (float value : matrixSRow) {
			value *= scaleFactor;
			float expTerm = std::exp(value - maximum);
			sum += expTerm;
		}

		return maximum + std::log(sum);
	}

	std::vector<float> createDerivativePRow(int rowID) const {
		std::vector<float> output(columnDimension, 0.0f);
		for (int columnID = 0; columnID < columnDimension; ++columnID) {
			float dotProduct = 0.0f;
			for (int d = 0; d < headDimension; ++d) {
				int addressO = rowID * headDimension + d;
				int addressV = columnID * headDimension + d;
				dotProduct += dO[addressO] * V[addressV];
			}
			output[columnID] = dotProduct;
		}
		return output;
	}

	std::vector<float> createDerivativeSRow(int rowID) const {
		std::vector<float> matrixPRow = createMatrixPRow(rowID);
		std::vector<float> matrixORow(headDimension, 0.0f);

		for (int d = 0; d < headDimension; ++d) {
			float dotProduct = 0.0f;
			for (int columnID = 0; columnID < columnDimension; ++columnID) {
				float valueP = matrixPRow[columnID];
				int addressV = columnID * headDimension + d;
				dotProduct += valueP * V[addressV];
			}
			matrixORow[d] = dotProduct;
		}

		float termD = 0.0f;
		for (int d = 0; d < headDimension; ++d) {
			int addressDerivativeO = rowID * headDimension + d;
			termD += matrixORow[d] * dO[addressDerivativeO];
		}

		std::vector<float> derivativeSRow(columnDimension, 0.0f);
		std::vector<float> derivativePRow = createDerivativePRow(rowID);
		float scaleFactor = 1.0f / std::sqrt(static_cast<float>(headDimension));

		for (int columnID = 0; columnID < columnDimension; ++columnID) {
			float valueP = matrixPRow[columnID];
			float valueDerivativeP = derivativePRow[columnID];
			float valueS = valueP * (valueDerivativeP - termD);
			valueS *= scaleFactor;
			derivativeSRow[columnID] = valueS;
		}

		return derivativeSRow;
	}

	float createDTerm(int rowID) const {
		std::vector<float> matrixPRow = createMatrixPRow(rowID);
		std::vector<float> matrixORow(headDimension, 0.0f);

		for (int d = 0; d < headDimension; ++d) {
			float dotProduct = 0.0f;
			for (int columnID = 0; columnID < columnDimension; ++columnID) {
				float valueP = matrixPRow[columnID];
				int addressV = columnID * headDimension + d;
				dotProduct += valueP * V[addressV];
			}
			matrixORow[d] = dotProduct;
		}

		float termD = 0.0f;
		for (int d = 0; d < headDimension; ++d) {
			int addressDerivativeO = rowID * headDimension + d;
			termD += matrixORow[d] * dO[addressDerivativeO];
		}
		return termD;
	}

	std::vector<float> inferenceAttention() const {
		std::vector<float> output(rowDimension * headDimension, 0.0f);
		for (int rowID = 0; rowID < rowDimension; ++rowID) {
			std::vector<float> matrixPRow = createMatrixPRow(rowID);
			std::vector<float> matrixORow(headDimension, 0.0f);

			for (int d = 0; d < headDimension; ++d) {
				float dotProduct = 0.0f;
				for (int columnID = 0; columnID < columnDimension; ++columnID) {
					float valueP = matrixPRow[columnID];
					int addressV = columnID * headDimension + d;
					dotProduct += valueP * V[addressV];
				}
				matrixORow[d] = dotProduct;
			}

			for (int d = 0; d < headDimension; ++d) {
				float valueO = matrixORow[d];
				int addressO = rowID * headDimension + d;
				output[addressO] = valueO;
			}
		}

		return output;
	}

	float loss() const {
		std::vector<float> O = inferenceAttention();
		float output = 0.0f;

		for (int rowID = 0; rowID < rowDimension; ++rowID) {
			for (int d = 0; d < headDimension; ++d) {
				int address = rowID * headDimension + d;
				output += dO[address] * O[address];
			}
		}
		return output;
	}

	std::vector<float> derivativeV() const {
		std::vector<float> output(columnDimension * headDimension, 0.0f);

		for (int rowID = 0; rowID < rowDimension; ++rowID) {
			std::vector<float> matrixPRow = createMatrixPRow(rowID);

			for (int columnID = 0; columnID < columnDimension; ++columnID) {
				for (int d = 0; d < headDimension; ++d) {
					int addressV = columnID * headDimension + d;
					int addressDerivativeO = rowID * headDimension + d;

					output[addressV] += matrixPRow[columnID] * dO[addressDerivativeO];
				}
			}
		}
		return output;
	}

	std::vector<float> derivativeK() const {
		std::vector<float> output(columnDimension * headDimension, 0.0f);

		for (int rowID = 0; rowID < rowDimension; ++rowID) {
			std::vector<float> derivativeSRow = createDerivativeSRow(rowID);

			for (int columnID = 0; columnID < columnDimension; ++columnID) {
				for (int d = 0; d < headDimension; ++d) {
					int addressK = columnID * headDimension + d;
					int addressQ = rowID * headDimension + d;

					output[addressK] += derivativeSRow[columnID] * Q[addressQ];
				}
			}
		}
		return output;
	}

	std::vector<float> derivativeQ() const {
		std::vector<float> output(rowDimension * headDimension, 0.0f);

		for (int rowID = 0; rowID < rowDimension; ++rowID) {
			std::vector<float> derivativeSRow = createDerivativeSRow(rowID);
			std::vector<float> derivativeQRow(headDimension, 0.0f);

			for (int d = 0; d < headDimension; ++d) {
				float dotProduct = 0.0f;
				for (int columnID = 0; columnID < columnDimension; ++columnID) {
					float derivativeSValue = derivativeSRow[columnID];
					int addressK = columnID * headDimension + d;
					dotProduct += derivativeSValue * K[addressK];
				}
				derivativeQRow[d] = dotProduct;
			}

			for (int d = 0; d < headDimension; ++d) {
				float derivativeQValue = derivativeQRow[d];
				int addressQ = rowID * headDimension + d;
				output[addressQ] = derivativeQValue;
			}
		}

		return output;
	}
};

ShaderCache shaderCache;

void validateProblemSize(int sequenceDimension, int headDimension)
{
	NetworkDescriptor networkDesc;
	networkDesc.rowDimension = sequenceDimension;
	networkDesc.columnDimension = sequenceDimension;
	networkDesc.headDimension = headDimension;
	networkDesc.scale = 1.0 / sqrtf((float)headDimension);
	Network network(networkDesc);
	AttentionDescriptor attentionDesc;
	attentionDesc.lowPrecisionInputs = false;
	attentionDesc.lowPrecisionIntermediates = false;
	attentionDesc.matrixDimensions[0] = sequenceDimension;
	attentionDesc.matrixDimensions[1] = sequenceDimension;
	attentionDesc.matrixDimensions[2] = headDimension;
	attentionDesc.transposeState[0] = false;
	attentionDesc.transposeState[1] = false;
	attentionDesc.transposeState[2] = false;
	attentionDesc.transposeState[3] = false;
	attentionDesc.type = AttentionKernelType::forward;
	attentionDesc.scale = 1.0 / sqrtf((float)headDimension);

	DeviceProperties dprops;
	dprops.coreCount = 18;
	NS::SharedPtr<MTL::Device> device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
	NS::SharedPtr<MTL::CommandQueue> queue = NS::TransferPtr(device->newCommandQueue());
	{
		// Generate the kernel.
		auto pipelineValue = shaderCache.findKernel<AttentionKernel, AttentionDescriptor, AttentionKernelDescriptor>(attentionDesc, device.get(), dprops);
		NS::SharedPtr<MTL::Buffer> bufferQ = NS::TransferPtr(device->newBuffer(network.Q.data(), sizeof(float) * sequenceDimension * headDimension, MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));
		NS::SharedPtr<MTL::Buffer> bufferK = NS::TransferPtr(device->newBuffer(network.K.data(), sizeof(float) * sequenceDimension * headDimension, MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));
		NS::SharedPtr<MTL::Buffer> bufferV = NS::TransferPtr(device->newBuffer(network.V.data(), sizeof(float) * sequenceDimension * headDimension, MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));
		float* resultO = (float*)ccmalloc(sizeof(float) * sequenceDimension * headDimension);
		resultO[0] = NAN;
		NS::SharedPtr<MTL::Buffer> bufferO = NS::TransferPtr(device->newBuffer(resultO, sizeof(float) * sequenceDimension * headDimension, MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));
		NS::SharedPtr<MTL::CommandBuffer> commandBuffer = NS::TransferPtr(queue->commandBuffer());
		NS::SharedPtr<MTL::ComputeCommandEncoder> encoder = NS::TransferPtr(commandBuffer->computeCommandEncoder());
		encoder->setComputePipelineState(pipelineValue->pipeline.get());
		encoder->setThreadgroupMemoryLength(pipelineValue->kernel->threadgroupMemoryAllocation, 0);
		encoder->setBuffer(bufferQ.get(), 0, 0);
		encoder->setBuffer(bufferK.get(), 0, 1);
		encoder->setBuffer(bufferV.get(), 0, 2);
		encoder->setBuffer(bufferO.get(), 0, 3);
		encoder->useResource(bufferQ.get(), MTL::ResourceUsageRead);
		encoder->useResource(bufferK.get(), MTL::ResourceUsageRead);
		encoder->useResource(bufferV.get(), MTL::ResourceUsageRead);
		encoder->useResource(bufferO.get(), MTL::ResourceUsageWrite);
		auto ceilDivide =
			[=](int64_t target, uint16_t granularity) -> int64_t {
				return (target + int64_t(granularity) - 1) / int64_t(granularity);
			};
		MTL::Size gridSize = MTL::Size(ceilDivide(sequenceDimension, pipelineValue->kernel->blockDimensions[0]), 1, 1);
		MTL::Size groupSize = MTL::Size(pipelineValue->kernel->threadgroupSize, 1, 1);
		encoder->dispatchThreadgroups(gridSize, groupSize);
		encoder->endEncoding();
		commandBuffer->commit();
		commandBuffer->waitUntilCompleted();
		auto start = commandBuffer->GPUStartTime();
		auto end = commandBuffer->GPUEndTime();
		auto latency = end - start;
		auto O = network.inferenceAttention();
		auto raw = bufferO->contents();
		for (int rowID = 0; rowID < sequenceDimension; rowID++)
		{
			for (int columnID = 0; columnID < headDimension; columnID++)
			{
				const int address = rowID * headDimension + columnID;
				float entry32;
				entry32 = ((float*)raw)[address];
				resultO[address] = entry32;
			}
		}
		auto check = [=](std::vector<float> expected, float* actual, float tolerance) {
			int errorCount = 0;
			for (int i = 0; i < expected.size(); i++) {
				auto error = fabs(expected[i] - actual[i]);
				if (error > tolerance || isnan(error)) {
					// Don't report errors in this case.
					if ((isnan(expected[i]) || isinf(expected[i])) && (isnan(actual[i]) || isinf(actual[i]))) {
						continue;
					}

					// Update the error count in the outer scope.
					if (errorCount < 10) {
						errorCount += 1;
						std::cerr << "error: "<< error << " / ~1.000" << std::endl;
						std::cerr << "- expected[" << i << "] =" << expected[i] << std::endl;
						std::cerr << "-   actual[" << i << "] =" << actual[i] << std::endl;
					}
				}
			}
		};
		if (attentionDesc.lowPrecisionInputs || attentionDesc.lowPrecisionIntermediates) {
			check(O, resultO, 5e-2);
		} else {
			check(O, resultO, 2e-5);
		}
		ccfree(resultO);
	}
}

int main(int argc, char** argv)
{
	ccv_nnc_init();
	{
		validateProblemSize(10, 3);
		validateProblemSize(10, 80);
		validateProblemSize(8, 2);
		validateProblemSize(9, 2);
		validateProblemSize(23, 2);
		validateProblemSize(24, 2);
		validateProblemSize(25, 2);
		validateProblemSize(192, 77);
		validateProblemSize(192, 80);
		validateProblemSize(93, 32);
		validateProblemSize(99, 35);
		validateProblemSize(64, 32);
		validateProblemSize(64, 34);
		validateProblemSize(64, 36);
		validateProblemSize(64, 40);
		validateProblemSize(32, 64);
		validateProblemSize(4, 1);
		validateProblemSize(4, 2);
		validateProblemSize(384, 95);
		validateProblemSize(777, 199);
	}
	return 0;
}
