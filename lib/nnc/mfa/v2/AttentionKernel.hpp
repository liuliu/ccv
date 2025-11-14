#ifndef AttentionKernel_hpp
#define AttentionKernel_hpp

#include "AttentionKernelDescriptor.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>

class CodeWriter;

struct AttentionAccumulateDescriptor;
struct AttentionOuterProductDescriptor;

struct AttentionKernel {
  NS::SharedPtr<MTL::Library> library;
  
  std::string source;

  AttentionKernelType type;

  AttentionOperands<bool> cacheState;

  AttentionOperands<GEMMOperandPrecision> memoryPrecisions;

  bool preferAsyncCache;

  bool preferAsyncLoad;

  AttentionOperands<GEMMOperandPrecision> registerPrecisions;

  AttentionOperands<bool> transposeState;

  /// The leading dimensions after transposed if applied.
  AttentionOperands<bool> leadingDimensions;

  /// parallelization, traversal, head
  simd::ushort3 blockDimensions;

  unsigned short headDimension;

  bool disableAsyncCopy;

  unsigned short threadgroupMemoryAllocation;

  /// The number of threads per group.
  uint16_t threadgroupSize;

  AttentionKernel(AttentionKernelDescriptor descriptor, MTL::Device *const device);

private:
  /// AttentionKernel.
  std::string memoryName(AttentionOperand operand) const noexcept;
  std::string registerName(AttentionOperand operand) const noexcept;
  std::string loadFunction(AttentionOperand operand) const noexcept;
  std::string storeFunction(AttentionOperand operand) const noexcept;
  bool cached(AttentionOperand operand) const noexcept;
  bool transposed(AttentionOperand operand) const noexcept;
  std::string sequenceLength(AttentionOperand operand) const noexcept;
  unsigned short blockSequenceLength(AttentionOperand operand) const noexcept;
  std::string leadingDimension(AttentionOperand operand) const noexcept;
  unsigned short leadingBlockDimension(AttentionOperand operand) const noexcept;

  std::string parallelizationDimensionValue() const noexcept;
  std::string parallelizationGroupOffsetValue() const noexcept;
  std::string unsafeParallelizationThreadOffsetValue() const noexcept;
  std::string clampedParallelizationThreadOffsetValue() const noexcept;
  std::string traversalDimensionValue() const noexcept;
  std::string traversalOffsetValue() const noexcept;
  std::string paddedTraversalEdgeValue() const noexcept;
  unsigned short paddedHeadDimensionValue() const noexcept;
  unsigned short paddedHeadEdgeValue() const noexcept;
  unsigned short threadgroupSizeValue() const noexcept;
  unsigned short createThreadgroupMemoryAllocation() const noexcept;
  std::string operandLocationValue(AttentionOperand operand) const noexcept;
  std::string operandLocationWithHeadOffsetValue(AttentionOperand operand) const noexcept;

  /// AttentionKernel+Source
  std::string createSource() const noexcept;
  std::string createConstants() const noexcept;
  std::string createAdjustOffsets() const noexcept;
  std::string createBufferBindings() const noexcept;
  std::string loopForward() const noexcept;
  std::string loopBackwardQuery() const noexcept;
  std::string loopBackwardKeyValue() const noexcept;

  /// AttentionKernel+Accumulate
  std::string accumulate(const AttentionAccumulateDescriptor& descriptor) const noexcept;

  /// AttentionKernel+Caching
  class CachingOperationType {
    // Hijack some C++ syntax, making it look like Swift's enumerations with
    // member functions.
    //
    // Source: https://stackoverflow.com/a/53284026
  public:
    enum Value: uint16_t {
      load = 0,
      store = 1,
    };
  
    CachingOperationType() = default;
    constexpr CachingOperationType(Value aValue) : value(aValue) { }
  
    explicit operator bool() const = delete;
  
    constexpr bool operator==(const CachingOperationType &rhs) const { return value == rhs.value; }
    constexpr bool operator!=(const CachingOperationType &rhs) const { return value != rhs.value; }

    Value value;
  };
  std::string cache(AttentionOperand operand, CachingOperationType type) const noexcept;
  std::string createSetup() const noexcept;
  std::string createCleanup(const AttentionKernelType type) const noexcept;

  /// AttentionKernel+OuterProduct
  std::string outerProduct(const AttentionOuterProductDescriptor& descriptor) const noexcept;

  /// AttentionKernel+Softmax
  std::string computeD() const noexcept;
  std::string maskAttentionMatrixEdge() const noexcept;
  std::string onlineReduceMaximum() const noexcept;
  std::string onlineCorrectO() const noexcept;
  std::string onlineReduceSum() const noexcept;
  std::string softmax(bool derivative) const noexcept;
  MTL::Library* findPrecompiledLibrary(AttentionKernelDescriptor descriptor, MTL::Device *const device, NS::Error **error) const noexcept;
};

#endif /* AttentionKernel_hpp */

