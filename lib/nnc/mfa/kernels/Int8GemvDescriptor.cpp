#include "Int8GemvDescriptor.hpp"
#include "TMCQDecodeSource.hpp"
#include "Int8GemvKernel.hpp"
#include "../ccv_nnc_mfa.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"
#include "../../ccv_nnc_ftq_schedule.inc"

namespace {

bool is_tmcq(const uint32_t format)
{
  const uint32_t codec = format & CCV_NNC_QX_8I_ROWWISE_FORMAT_MASK;
  return codec >= CCV_NNC_QX_8I_ROWWISE_TMCQ_N16 && codec <= CCV_NNC_QX_8I_ROWWISE_TMCQ_N32;
}

// FTQ (CCVFTQSR blobs, see CCV_NNC_QX_8I_ROWWISE_FTQ_N16).
bool is_ftq(const uint32_t format)
{
  const uint32_t codec = format & CCV_NNC_QX_8I_ROWWISE_FORMAT_MASK;
  return codec >= CCV_NNC_QX_8I_ROWWISE_FTQ_N16 && codec <= CCV_NNC_QX_8I_ROWWISE_FTQ_N88;
}

uint32_t ftq_rate(const uint32_t format)
{
  return 8 + 2 * ((format & CCV_NNC_QX_8I_ROWWISE_FORMAT_MASK) - CCV_NNC_QX_8I_ROWWISE_FTQ_BLOCK_FIRST);
}

// Variable-rate FTQ (CCVFTQVR blobs, see CCV_NNC_QX_8I_ROWWISE_FTQ_VR_N16); its mean rate.
bool is_ftq_vr(const uint32_t format)
{
  const uint32_t codec = format & CCV_NNC_QX_8I_ROWWISE_FORMAT_MASK;
  return codec >= CCV_NNC_QX_8I_ROWWISE_FTQ_VR_N16 && codec <= CCV_NNC_QX_8I_ROWWISE_FTQ_VR_N88;
}

uint32_t ftq_vr_rate(const uint32_t format)
{
  return 8 + 2 * ((format & CCV_NNC_QX_8I_ROWWISE_FORMAT_MASK) - CCV_NNC_QX_8I_ROWWISE_FTQ_VR_BLOCK_FIRST);
}

bool is_ftq_any(const uint32_t format)
{
  return is_ftq(format) || is_ftq_vr(format);
}

}

bool Int8GemvDescriptor::operator==(const Int8GemvDescriptor& rhs) const {
  return
  fusedBias == rhs.fusedBias &&
  mrows == rhs.mrows &&
  format == rhs.format &&
  levelMask == rhs.levelMask &&
  memoryPrecision == rhs.memoryPrecision &&
  nrows == rhs.nrows &&
  ncols == rhs.ncols &&
  vectorRowStride == rhs.vectorRowStride &&
  outputRowStride == rhs.outputRowStride &&
  batchStrides.has_value() == rhs.batchStrides.has_value() &&
  simd_all(batchStrides.value_or(simd::uint4(UINT32_MAX)) == rhs.batchStrides.value_or(simd::uint4(UINT32_MAX)));
}

std::size_t std::hash<Int8GemvDescriptor>::operator()(const Int8GemvDescriptor& hash) const noexcept {
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  seed = combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.memoryPrecision.value, (unsigned int)hash.fusedBias | ((unsigned int)hash.mrows << 8) }));
  seed = combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.format, (unsigned int)hash.nrows }));
  seed = combine_64(seed, pack_64(simd::uint2 { hash.ncols, (uint32_t)hash.levelMask }));
  seed = combine_32(seed, (uint32_t)(hash.levelMask >> 32));
  seed = combine_64(seed, pack_64(simd::uint2 { hash.vectorRowStride, hash.outputRowStride }));
  if (hash.batchStrides.has_value()) {
    seed = combine_32(seed, hash.batchStrides.value()[0]);
    seed = combine_32(seed, hash.batchStrides.value()[1]);
    seed = combine_32(seed, hash.batchStrides.value()[2]);
    seed = combine_32(seed, hash.batchStrides.value()[3]);
  }
  return seed;
}

uint32_t Int8GemvDescriptor::groupSize() const noexcept {
  // A TMCQ group is one 16 x 16 tile; it spans 16 rows, so inputScaleOffset() counts the tiles of the grid
  // (tmcq_tile_count) instead of nrows * groupsPerRow().
  if (is_tmcq(format) || is_ftq_any(format))
    return 256;
  switch (format) {
    case 0:
      return 0;
    case CCV_NNC_QX_8I_ROWWISE_Q5_K:
    case CCV_NNC_QX_8I_ROWWISE_Q4_K:
    case CCV_NNC_QX_8I_ROWWISE_Q3_K:
    case CCV_NNC_QX_8I_ROWWISE_Q2_K:
    case CCV_NNC_QX_8I_ROWWISE_IQ2_S:
    case CCV_NNC_QX_8I_ROWWISE_IQ3_S:
    case CCV_NNC_QX_8I_ROWWISE_IQ1_M:
      return 16;
    case CCV_NNC_QX_8I_ROWWISE_IQ2_XXS:
    case CCV_NNC_QX_8I_ROWWISE_IQ1_S:
      return 32;
    case CCV_NNC_QX_8I_ROWWISE_Q6_K:
    case CCV_NNC_QX_8I_ROWWISE_IQ2_XS:
    case CCV_NNC_QX_8I_ROWWISE_IQ3_XXS:
      return 8;
    default:
      CCV_NNC_MFA_PRECONDITION(false);
      return 0;
  }
}

uint32_t Int8GemvDescriptor::groupsPerRow() const noexcept {
  const uint32_t size = groupSize();
  return size > 0 ? (ncols + size - 1) / size : 0;
}

uint32_t Int8GemvDescriptor::groupBits() const noexcept {
  if (is_tmcq(format))
    return 16 * (16 + 2 * ((format & CCV_NNC_QX_8I_ROWWISE_FORMAT_MASK) - CCV_NNC_QX_8I_ROWWISE_TMCQ_N16));
  // 16 n bits per tile, after the 32-byte FTQ header (inputScaleOffset).
  if (is_ftq(format))
    return 16 * ftq_rate(format);
  // The variable-rate FTQ budget: 16 nm bits per tile on average.
  if (is_ftq_vr(format))
    return 16 * ftq_vr_rate(format);
  switch (format) {
    case 0:
      return 0;
    case CCV_NNC_QX_8I_ROWWISE_Q5_K:
      return 88;
    case CCV_NNC_QX_8I_ROWWISE_Q4_K:
      return 72;
    case CCV_NNC_QX_8I_ROWWISE_Q3_K:
    case CCV_NNC_QX_8I_ROWWISE_IQ3_S:
      return 56;
    case CCV_NNC_QX_8I_ROWWISE_Q2_K:
    case CCV_NNC_QX_8I_ROWWISE_IQ2_S:
      return 42;
    case CCV_NNC_QX_8I_ROWWISE_IQ2_XS:
      return 21;
    case CCV_NNC_QX_8I_ROWWISE_IQ3_XXS:
    case CCV_NNC_QX_8I_ROWWISE_IQ1_M:
      return 28;
    case CCV_NNC_QX_8I_ROWWISE_IQ1_S:
      return 50;
    case CCV_NNC_QX_8I_ROWWISE_IQ2_XXS:
      return 64;
    case CCV_NNC_QX_8I_ROWWISE_Q6_K:
      return 52;
    default:
      CCV_NNC_MFA_PRECONDITION(false);
      return 0;
  }
}

uint32_t Int8GemvDescriptor::rowsPerThreadgroup() const noexcept {
  if (is_tmcq(format) || is_ftq_any(format))
    return kInt8GemvTMCQRowsPerThreadgroup;
  // The 4 to 8 row packed kernel gives each half SIMD group a row of its own.
  if (format != 0 && mrows > 3)
    return kInt8GemvSIMDGroupsPerThreadgroup * 2;
  return kInt8GemvRowsPerThreadgroup;
}

uint32_t Int8GemvDescriptor::threadsPerThreadgroup() const noexcept {
  if (is_tmcq(format) || is_ftq_any(format))
    return int8GemvTMCQSIMDGroups(mrows) * 32;
  return kInt8GemvSIMDGroupsPerThreadgroup * 32;
}

uint32_t int8GemvTMCQSIMDGroups(const uint32_t mrows)
{
  return 16;
}

uint64_t Int8GemvDescriptor::inputScaleOffset() const noexcept {
  // TMCQ / FTQ: (nrows / 16) (ncols / 16) tiles of groupBits() bits for any ncols % 16 == 0.
  const uint64_t payloadBits = (is_tmcq(format) || is_ftq_any(format)) ?
    tmcq_tile_count(nrows, ncols) * groupBits() :
    (uint64_t)nrows * groupsPerRow() * groupBits();
  // The 32-byte header, and for variable-rate FTQ the R + 1 word tile row offsets after it.
  const uint64_t payloadBytes = (payloadBits + 7) / 8 + (is_ftq(format) ? 32 : 0) + (is_ftq_vr(format) ? 32 + 4 * ((uint64_t)nrows / 16 + 1) : 0);
  return (payloadBytes + 127) & ~UINT64_C(127);
}

std::pair<Int8GemvKernelDescriptor, PipelineValue<Int8GemvKernel>*> Int8GemvDescriptor::findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<Int8GemvKernelDescriptor, std::unique_ptr<Int8GemvKernel>> *const libraryCache) const noexcept {
  if (is_tmcq(format) || is_ftq_any(format)) {
    // Bits 16 - 23: log2 of an external codebook's entry count, 0 for the built-in table (TMCQDecodeSource.hpp).
    // Bit 24 (FTQ only): a quadrant-folded codebook.
    CCV_NNC_MFA_PRECONDITION(tmcq_codebook_format_ok(format, is_ftq_any(format)));
    CCV_NNC_MFA_PRECONDITION(mrows >= 1 && mrows <= 3);
    // The 16 x 16 tile grid: one threadgroup per tile row, the tile loop runs over any ncols / 16 tiles.
    CCV_NNC_MFA_PRECONDITION((nrows % 16) == 0 && (ncols % 16) == 0 && ncols > 0);
  }
  // FTQ: FP16 scales in the blob, no batched weights (one blob is one 2-D tensor).
  if (is_ftq_any(format))
    CCV_NNC_MFA_PRECONDITION(memoryPrecision == GEMMOperandPrecision::FP16 && !batchStrides.has_value());
  // Variable-rate FTQ: a non-empty level mask of rates 8 ... 88, at most 6 of them; nothing else carries one.
  CCV_NNC_MFA_PRECONDITION(is_ftq_vr(format) ? (levelMask != 0 && (levelMask >> 41) == 0 && __builtin_popcountll(levelMask) <= 6) : levelMask == 0);
  auto createKernel =
  [=](Int8GemvKernelDescriptor descriptor) -> Int8GemvKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end()) {
      return iterator->second.get();
    } else {
      Int8GemvKernel* kernel = new Int8GemvKernel(descriptor, device);
      (*libraryCache)[descriptor] = std::unique_ptr<Int8GemvKernel>(kernel);
      return kernel;
    }
  };

  Int8GemvKernelDescriptor kernelDesc;
  kernelDesc.fusedBias = fusedBias;
  kernelDesc.mrows = mrows;
  kernelDesc.batched = batchStrides.has_value();
  kernelDesc.format = format;
  kernelDesc.memoryPrecision = memoryPrecision;
  kernelDesc.levelMask = levelMask;

  auto createPipeline =
  [=](MTL::Library* library) -> MTL::ComputePipelineState* {
    auto constants = NS::TransferPtr
    (MTL::FunctionConstantValues::alloc()->init());
    const uint32_t groupSize = this->groupSize();
    const uint32_t groupsPerRow = this->groupsPerRow();
    constants->setConstantValue(&ncols, MTL::DataTypeUInt, NS::UInteger(0));
    constants->setConstantValue(&nrows, MTL::DataTypeUInt, NS::UInteger(1));
    constants->setConstantValue(&groupSize, MTL::DataTypeUInt, NS::UInteger(3));
    constants->setConstantValue(&groupsPerRow, MTL::DataTypeUInt, NS::UInteger(4));
    constants->setConstantValue(&vectorRowStride, MTL::DataTypeUInt, NS::UInteger(9));
    constants->setConstantValue(&outputRowStride, MTL::DataTypeUInt, NS::UInteger(10));
    if (is_ftq(format)) {
      // Every tile has 16 n bits (the kernel takes n as a source constant and needs no schedule constants).
      const ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule = ccv_nnc_8i_rowwise_x_ftq_schedule_of_rate(ftq_rate(format), nrows, ncols);
      CCV_NNC_MFA_PRECONDITION(schedule.tile_count <= UINT32_MAX && schedule.scale_offset == inputScaleOffset());
    }
    if (batchStrides.has_value()) {
      const uint32_t batchStrideWeights = batchStrides.value()[0];
      const uint32_t batchStrideVector = batchStrides.value()[1];
      const uint32_t batchStrideOutput = batchStrides.value()[2];
      const uint32_t batchStrideScale = batchStrides.value()[3];
      constants->setConstantValue(&batchStrideWeights, MTL::DataTypeUInt, NS::UInteger(5));
      constants->setConstantValue(&batchStrideVector, MTL::DataTypeUInt, NS::UInteger(6));
      constants->setConstantValue(&batchStrideOutput, MTL::DataTypeUInt, NS::UInteger(7));
      constants->setConstantValue(&batchStrideScale, MTL::DataTypeUInt, NS::UInteger(8));
    }

    NS::String* swiftName = NS::String::string("int8_gemv", NS::UTF8StringEncoding);
    NS::Error* error = nil;

    auto function = NS::TransferPtr
    (library->newFunction(swiftName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);

    auto pipeline = device->newComputePipelineState(function.get(), &error);
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };
  Int8GemvKernel* kernel = createKernel(kernelDesc);
  auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get()));

  PipelineValue<Int8GemvKernel>* output = new PipelineValue<Int8GemvKernel> { kernel, pipeline };
  return std::make_pair(kernelDesc, output);
}
