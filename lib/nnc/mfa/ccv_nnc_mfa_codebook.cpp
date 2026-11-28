#include "ccv_nnc_mfa.hpp"
extern "C" {
#include "../ccv_nnc_codebook.h"
}
#include <cstring>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

mtl_buffer_t* ccv_nnc_mfa_codebook_buffer(ccv_nnc_mfa_context_t* context, int slot, int* entries)
{
  int count = 0;
  const int8_t* const pairs = ccv_nnc_codebook_get(slot, &count, nullptr);
  if (!pairs)
    return nullptr;
  if (entries)
    *entries = count;
  static std::mutex mutex;
  static std::map<std::pair<ccv_nnc_mfa_context_t*, int>, MTL::Buffer*> buffers;
  std::lock_guard<std::mutex> lock(mutex);
  const auto key = std::make_pair(context, slot);
  const auto it = buffers.find(key);
  if (it != buffers.end())
    return it->second;
  // Pair k as half2 ((half)q8[2 k], (half)q8[2 k + 1]): int8 values are exact in half; then the int8 pairs.
  std::vector<uint8_t> bytes((size_t)count * 6);
  __fp16* const halves = reinterpret_cast<__fp16*>(bytes.data());
  for (size_t i = 0; i < (size_t)count * 2; i++)
    halves[i] = (__fp16)pairs[i];
  memcpy(bytes.data() + (size_t)count * 4, pairs, (size_t)count * 2);
  MTL::Buffer* const buffer = context->device->newBuffer(bytes.data(), (NS::UInteger)bytes.size(), MTL::ResourceStorageModeShared);
  buffers[key] = buffer;
  return buffer;
}

int ccv_nnc_mfa_codebook_format(ccv_nnc_mfa_context_t* context, int slot, int allow_fold, uint32_t* format, mtl_buffer_t** codebook)
{
  *codebook = nullptr;
  if (!slot)
    return 1;
  int entries = 0;
  *codebook = ccv_nnc_mfa_codebook_buffer(context, slot, &entries);
  if (!*codebook)
    return 0;
  *format |= (uint32_t)ccv_nnc_codebook_log2_entries(entries) << 16;
  // A quadrant-folded table (FTQ only) is bit 24 of the MFA format (CCV_NNC_MFA_CODEBOOK_FOLD).
  if (ccv_nnc_codebook_flags(slot) & CCV_NNC_CODEBOOK_FOLD)
  {
    if (!allow_fold)
      return 0;
    *format |= 1u << 24;
  }
  return 1;
}
