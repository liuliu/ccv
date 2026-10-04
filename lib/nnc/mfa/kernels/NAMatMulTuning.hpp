#ifndef MFA_NAMATMUL_TUNING_HPP_
#define MFA_NAMATMUL_TUNING_HPP_

#include "../3rdparty/metal-cpp/Metal.hpp"

// These profiles use Apple10 MPP register fragments and bounded partial buffers.
// Their layout, occupancy and reduction-work limits live in the descriptors.
// Select by the same feature family as MFA's NA detection, so all M5 variants
// can use them. Crossover bounds were measured on M5 Ultra; the permanent sweep
// must validate throughput on other devices (no physical cache size is assumed).
inline bool useNeuralAcceleratorMatMulTuning(MTL::Device* device) noexcept {
  return device && device->supportsFamily(MTL::GPUFamily(1010));
}

#endif
