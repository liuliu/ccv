#ifndef GEMMOperandPrecision_hpp
#define GEMMOperandPrecision_hpp

#include <stdint.h>
#include <string>

/// An enumeration of the precisions supported by the kernel.
///
/// If you wish to support quantized precisions, copy/translate the source code
/// and integrate a modified version into your app. Something similar to a Swift
/// `enum` (e.g. C++ `enum class`) could enumerate the quantization formats
/// used by application code. An exemplary set could be:
/// - FP32
/// - FP16
/// - BF16
/// - signed 8-bit integer
/// - s1ezm7
/// - FP8
/// - palletized
///
/// If you support non-floating-point formats, you have the responsibility of
/// authoring correct and performant GPU code for them. A general rule of thumb,
/// is keep the data compressed in `device` or `threadgroup` memory. Transform
/// into a floating point type while loading into the registers. Keep the
/// accumulator in floating point until the output needs to be written.
/// If the output is quantized, it will be compressed when writing back to
/// `device` memory (or `threadgroup` before the async copy in edge cases).
///
/// For example, the reference implementation treats BF16 like a quantized
/// integer type on Apple7 and Apple8 GPUs. It is decompressed to FP32 in
/// registers.
class GEMMOperandPrecision {
  // Hijack some C++ syntax, making it look like Swift's enumerations with
  // member functions.
  //
  // Source: https://stackoverflow.com/a/53284026
public:
  enum Value: uint16_t {
    FP32 = 0,
    FP16 = 1,
    BF16 = 2,
  };
  
  GEMMOperandPrecision() = default;
  constexpr GEMMOperandPrecision(Value aPrecision) : value(aPrecision) { }

  // Prevent usage: if(precision)
  explicit operator bool() const = delete;

  constexpr bool operator==(GEMMOperandPrecision a) const { return value == a.value; }
  constexpr bool operator!=(GEMMOperandPrecision a) const { return value != a.value; }

  // The MSL keyword corresponding to the precision.
  std::string name() {
    switch (value) {
      case FP32:
        return "float";
      case FP16:
        return "half";
      case BF16:
        return "bfloat";
    }
  }
  
  // The size of the scalar, in bytes.
  int64_t size() {
    switch (value) {
      case FP32:
        return 4;
      case FP16:
        return 2;
      case BF16:
        return 2;
    }
  }
  
  Value value;
};

/// A way to emulate the API of the Swift tuple with labeled members.
struct GEMMOperandPrecisions {
  GEMMOperandPrecision A;
  GEMMOperandPrecision B;
  GEMMOperandPrecision C;
};

#endif /* GEMMOperandPrecision_hpp */
