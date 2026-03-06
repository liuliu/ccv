#ifndef AttentionKernelType_hpp
#define AttentionKernelType_hpp

#include <stdint.h>
#include <string>

class AttentionKernelType {
  // Hijack some C++ syntax, making it look like Swift's enumerations with
  // member functions.
  //
  // Source: https://stackoverflow.com/a/53284026
public:
  enum Value: uint16_t {
    forward = 0,
    backwardQuery = 1,
    backwardKeyValue = 2,
  };

  AttentionKernelType() = default;
  constexpr AttentionKernelType(Value aKernelType) : value(aKernelType) { }

  explicit operator bool() const = delete;

  constexpr bool operator==(const AttentionKernelType &rhs) const { return value == rhs.value; }
  constexpr bool operator!=(const AttentionKernelType &rhs) const { return value != rhs.value; }

  std::string name() const noexcept {
    switch (value) {
      case forward:
        return "forward";
      case backwardQuery:
        return "backwardQuery";
      case backwardKeyValue:
        return "backwardKeyValue";
    }
  }

  Value value;
};

#endif
