#ifndef AttentionOperand_hpp
#define AttentionOperand_hpp

#include <stdint.h>
#include <optional>
#include <string>

class AttentionOperand {
  // Hijack some C++ syntax, making it look like Swift's enumerations with
  // member functions.
  //
  // Source: https://stackoverflow.com/a/53284026
public:
  enum Value: uint16_t {
    Q = 0,
    K = 1,
    S = 2,
    P = 3,
    V = 4,
    O = 5,

    L = 6,
    D = 7,

    dO = 8,
    dV = 9,
    dP = 10,
    dS = 11,
    dK = 12,
    dQ = 13,
  };

  constexpr static int size() noexcept {
	  return 14;
  }
  
  AttentionOperand() = default;
  constexpr AttentionOperand(Value aOperand) : value(aOperand) { }

  explicit operator bool() const = delete;
  operator Value&() { return value; }
  operator const Value&() const { return value; }

  constexpr bool operator==(const AttentionOperand &rhs) const { return value == rhs.value; }
  constexpr bool operator!=(const AttentionOperand &rhs) const { return value != rhs.value; }

  std::string name() const noexcept {
    switch (value) {
      case Q:
        return "Q";
      case K:
        return "K";
      case S:
        return "S";
      case P:
        return "P";
      case V:
        return "V";
      case O:
        return "O";

      case L:
        return "L";
      case D:
        return "D";

      case dO:
        return "dO";
      case dV:
        return "dV";
      case dP:
        return "dP";
      case dS:
        return "dS";
      case dK:
        return "dK";
      case dQ:
        return "dQ";
    }
  }

  int bufferIndex() const noexcept {
    switch (value) {
      case AttentionOperand::Q:
        return 0;
      case AttentionOperand::K:
        return 1;
      case AttentionOperand::S:
      case AttentionOperand::P:
        return -1;
      case AttentionOperand::V:
        return 2;
      case AttentionOperand::O:
        return 3;
  
      case AttentionOperand::L:
        return 4;
      case AttentionOperand::D:
        return 5;
  
      case AttentionOperand::dO:
        return 6;
      case AttentionOperand::dV:
        return 7;
      case AttentionOperand::dP:
      case AttentionOperand::dS:
        return -1;
      case AttentionOperand::dK:
        return 8;
      case AttentionOperand::dQ:
        return 9;
    }
    return -1;
  }

  Value value;
};

template<typename Value>
struct AttentionOperands {
  Value Q;
  Value K;
  Value S;
  Value P;
  Value V;
  Value O;

  Value L;
  Value D;

  Value dO;
  Value dV;
  Value dP;
  Value dS;
  Value dK;
  Value dQ;

  constexpr AttentionOperands() : bitmap(0) {}

  constexpr bool operator==(const AttentionOperands<Value>& rhs) const {
    if (bitmap != rhs.bitmap) {
      return false;
    }
    if (bitmap & (1 << (AttentionOperand::Q)) && Q != rhs.Q) {
      return false;
    }
    if (bitmap & (1 << (AttentionOperand::K)) && K != rhs.K) {
      return false;
    }
    if (bitmap & (1 << (AttentionOperand::S)) && S != rhs.S) {
      return false;
    }
    if (bitmap & (1 << (AttentionOperand::P)) && P != rhs.P) {
      return false;
    }
    if (bitmap & (1 << (AttentionOperand::V)) && V != rhs.V) {
      return false;
    }
    if (bitmap & (1 << (AttentionOperand::O)) && O != rhs.O) {
      return false;
    }
    if (bitmap & (1 << (AttentionOperand::L)) && L != rhs.L) {
      return false;
    }
    if (bitmap & (1 << (AttentionOperand::D)) && D != rhs.D) {
      return false;
    }
    if (bitmap & (1 << (AttentionOperand::dO)) && dO != rhs.dO) {
      return false;
    }
    if (bitmap & (1 << (AttentionOperand::dV)) && dV != rhs.dV) {
      return false;
    }
    if (bitmap & (1 << (AttentionOperand::dP)) && dP != rhs.dP) {
      return false;
    }
    if (bitmap & (1 << (AttentionOperand::dS)) && dS != rhs.dS) {
      return false;
    }
    if (bitmap & (1 << (AttentionOperand::dK)) && dK != rhs.dK) {
      return false;
    }
    if (bitmap & (1 << (AttentionOperand::dQ)) && dQ != rhs.dQ) {
      return false;
    }
    return true;
  }

  class Reference {
  private:
    AttentionOperands& operands;
    Value& value;
    unsigned short offset;
    unsigned short& bitmap;

  public:
    Reference(AttentionOperands& o, Value& v, unsigned short& b, const unsigned short& of) : operands(o), value(v), offset(of), bitmap(b) {}

    // Implicit conversion to Value
    operator std::optional<Value>() {
      if (bitmap & (1 << offset))
        return std::make_optional(value);
      return std::nullopt;
    }
    operator const std::optional<Value>() const {
      if (bitmap & (1 << offset))
        return std::make_optional(value);
      return std::nullopt;
    }

    // Assignment operator
    Reference& operator=(const Value& newValue) {
      bitmap |= (1 << offset);
      value = newValue;
      return *this;
    }
  };

  Reference operator[](const AttentionOperand& operand) {
    switch (operand) {
    case AttentionOperand::Q:
      return Reference(*this, this->Q, this->bitmap, operand.value);
    case AttentionOperand::K:
      return Reference(*this, this->K, this->bitmap, operand.value);
    case AttentionOperand::S:
      return Reference(*this, this->S, this->bitmap, operand.value);
    case AttentionOperand::P:
      return Reference(*this, this->P, this->bitmap, operand.value);
    case AttentionOperand::V:
      return Reference(*this, this->V, this->bitmap, operand.value);
    case AttentionOperand::O:
      return Reference(*this, this->O, this->bitmap, operand.value);

    case AttentionOperand::L:
      return Reference(*this, this->L, this->bitmap, operand.value);
    case AttentionOperand::D:
      return Reference(*this, this->D, this->bitmap, operand.value);

    case AttentionOperand::dO:
      return Reference(*this, this->dO, this->bitmap, operand.value);
    case AttentionOperand::dV:
      return Reference(*this, this->dV, this->bitmap, operand.value);
    case AttentionOperand::dP:
      return Reference(*this, this->dP, this->bitmap, operand.value);
    case AttentionOperand::dS:
      return Reference(*this, this->dS, this->bitmap, operand.value);
    case AttentionOperand::dK:
      return Reference(*this, this->dK, this->bitmap, operand.value);
    case AttentionOperand::dQ:
      return Reference(*this, this->dQ, this->bitmap, operand.value);
    }
  }

  const std::optional<Value> operator[](const AttentionOperand& operand) const {
    if (bitmap & (1 << operand.value)) {
      switch (operand) {
      case AttentionOperand::Q:
        return std::make_optional(this->Q);
      case AttentionOperand::K:
        return std::make_optional(this->K);
      case AttentionOperand::S:
        return std::make_optional(this->S);
      case AttentionOperand::P:
        return std::make_optional(this->P);
      case AttentionOperand::V:
        return std::make_optional(this->V);
      case AttentionOperand::O:
        return std::make_optional(this->O);

      case AttentionOperand::L:
        return std::make_optional(this->L);
      case AttentionOperand::D:
        return std::make_optional(this->D);

      case AttentionOperand::dO:
        return std::make_optional(this->dO);
      case AttentionOperand::dV:
        return std::make_optional(this->dV);
      case AttentionOperand::dP:
        return std::make_optional(this->dP);
      case AttentionOperand::dS:
        return std::make_optional(this->dS);
      case AttentionOperand::dK:
        return std::make_optional(this->dK);
      case AttentionOperand::dQ:
        return std::make_optional(this->dQ);
      }
    }
    return std::nullopt;
  }

private:
  unsigned short bitmap;
};

#endif
