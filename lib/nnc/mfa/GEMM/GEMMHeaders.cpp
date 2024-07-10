#include "../ccv_nnc_mfa.hpp"

#include <optional>
#include <vector>

std::string createMetalSimdgroupEvent() {
  // Return the source string.
  return R"(
// -*- Metal -*-
//===-- metal_simdgroup_event ---------------------------------------------===//
// Copyright (c) 2024 Philip Turner. See MIT LICENSE
//===----------------------------------------------------------------------===//

#ifndef __METAL_SIMDGROUP_EVENT
#define __METAL_SIMDGROUP_EVENT

// Invoking the generation of LLVM bitcode for async copies.
//
//   %struct._simdgroup_event_t = type opaque
//
struct _simdgroup_event_t;

// Invoking the generation of LLVM bitcode for async copies.
//
//   Bitcode: TBD
//
thread _simdgroup_event_t*
__metal_simdgroup_async_copy_1d(
  ulong, ulong, threadgroup void *, const device void *, ulong)
  __asm("air.simdgroup_async_copy_1d.p3i8.p1i8");

// Invoking the generation of LLVM bitcode for async copies.
//
//   Bitcode: TBD
//
thread _simdgroup_event_t*
__metal_simdgroup_async_copy_1d(
  ulong, ulong, device void *, const threadgroup void *, ulong)
  __asm("air.simdgroup_async_copy_1d.p1i8.p3i8");

// Invoking the generation of LLVM bitcode for async copies.
//
//   ; Function Attrs: argmemonly convergent nounwind
//   declare %struct._simdgroup_event_t*
//     @air.simdgroup_async_copy_2d.p3i8.p1i8(
//       i64, i64,
//       i8 addrspace(3)* nocapture writeonly, i64, i64, <2 x i64>,
//       i8 addrspace(1)* nocapture readonly,  i64, i64, <2 x i64>,
//       <2 x i64>, i32)
//     local_unnamed_addr #4
//
thread _simdgroup_event_t*
__metal_simdgroup_async_copy_2d(
  ulong, ulong,
  threadgroup void *, ulong, ulong, ulong2,
  const device void *, ulong, ulong, ulong2,
  long2, int)
  __asm("air.simdgroup_async_copy_2d.p3i8.p1i8");

// Invoking the generation of LLVM bitcode for async copies.
//
//   ; Function Attrs: argmemonly convergent nounwind
//   declare %struct._simdgroup_event_t*
//     @air.simdgroup_async_copy_2d.p1i8.p3i8(
//       i64, i64,
//       i8 addrspace(1)* nocapture writeonly, i64, i64, <2 x i64>,
//       i8 addrspace(3)* nocapture readonly,  i64, i64, <2 x i64>,
//       <2 x i64>, i32)
//     local_unnamed_addr #4
//
thread _simdgroup_event_t*
__metal_simdgroup_async_copy_2d(
  ulong, ulong,
  device void *, ulong, ulong, ulong2,
  const threadgroup void *, ulong, ulong, ulong2,
  long2, int)
  __asm("air.simdgroup_async_copy_2d.p1i8.p3i8");

// Invoking the generation of LLVM bitcode for async copies.
//
//   ; Function Attrs: convergent nounwind
//   declare void
//     @air.wait_simdgroup_events(i32, %struct._simdgroup_event_t** nocapture)
//     local_unnamed_addr #3
//
void __metal_wait_simdgroup_events(
  int, thread _simdgroup_event_t**)
  __asm("air.wait_simdgroup_events");

#pragma METAL internals : enable
namespace metal
{
  enum class simdgroup_async_copy_clamp_mode {
    clamp_to_zero = 0,
    clamp_to_edge = 1
  };
  
  struct simdgroup_event {
    METAL_FUNC simdgroup_event() thread {}

    template <typename T>
    METAL_FUNC void async_copy(
      threadgroup T *dst,
      const device T *src,
      ulong n_elements
    ) thread {
      event = __metal_simdgroup_async_copy_1d(
        // Description of the data type.
        sizeof(T),
        alignof(T),
        
        // Description of the arguments.
        reinterpret_cast<threadgroup void *>(dst),
        reinterpret_cast<const device void *>(src),
        n_elements);
    }
    
    template <typename T>
    METAL_FUNC void async_copy(
      device T *dst,
      const threadgroup T *src,
      ulong n_elements
    ) thread {
      event = __metal_simdgroup_async_copy_1d(
        // Description of the data type.
        sizeof(T),
        alignof(T),
        
        // Description of the arguments.
        reinterpret_cast<device void *>(dst),
        reinterpret_cast<const threadgroup void *>(src),
        n_elements);
    }
    
    template <typename T>
    METAL_FUNC void async_copy(
      // Description of the destination.
      threadgroup T *dst,
      ushort dst_elements_per_row,
      ushort2 dst_tile_dimensions,

      // Description of the source.
      const device T *src,
      uint src_elements_per_row,
      ushort2 src_tile_dimensions,

      // Other arguments.
      bool transpose_matrix = false,
      simdgroup_async_copy_clamp_mode clamp_mode =
        simdgroup_async_copy_clamp_mode::clamp_to_zero
    ) thread {
      if (transpose_matrix) {
        src_tile_dimensions = src_tile_dimensions.yx;
        dst_tile_dimensions = dst_tile_dimensions.yx;
      }
      event = __metal_simdgroup_async_copy_2d(
        // Description of the data type.
        sizeof(T),
        alignof(T),

        // Description of the destination.
        reinterpret_cast<threadgroup void *>(dst),
        ushort(dst_elements_per_row),
        1,
        ulong2(dst_tile_dimensions),

        // Description of the source.
        reinterpret_cast<const device void *>(src),
        uint(src_elements_per_row),
        1,
        ulong2(src_tile_dimensions),

        // Other arguments.
        long2(0),
        static_cast<int>(clamp_mode));
    }
    
    template <typename T>
    METAL_FUNC void async_copy(
      // Description of the destination.
      device T *dst,
      uint dst_elements_per_row,
      ushort2 dst_tile_dimensions,

      // Description of the source.
      const threadgroup T *src,
      ushort src_elements_per_row,
      ushort2 src_tile_dimensions,

      // Other arguments.
      bool transpose_matrix = false
    ) thread {
      if (transpose_matrix) {
        src_tile_dimensions = src_tile_dimensions.yx;
        dst_tile_dimensions = dst_tile_dimensions.yx;
      }
      event = __metal_simdgroup_async_copy_2d(
        // Description of the data type.
        sizeof(T),
        alignof(T),

        // Description of the destination.
        reinterpret_cast<device void *>(dst),
        uint(dst_elements_per_row),
        1,
        ulong2(dst_tile_dimensions),

        // Description of the source.
        reinterpret_cast<const threadgroup void *>(src),
        ushort(src_elements_per_row),
        1,
        ulong2(src_tile_dimensions),

        // Other arguments.
        long2(0),
        0);
    }
    
    METAL_FUNC static void wait(int count, thread simdgroup_event *events) {
      __metal_wait_simdgroup_events(
        count, reinterpret_cast<thread _simdgroup_event_t**>(events));
    }
    
  private:
    // Invoking the generation of LLVM bitcode for async copies.
    //
    //   %"struct.metal::simdgroup_event" = type { %struct._simdgroup_event_t* }
    //
    thread _simdgroup_event_t* event;
  };
} // namespace metal
#pragma METAL internals : disable

#endif // __METAL_SIMDGROUP_EVENT
)";
}

std::string createMetalSimdgroupMatrixStorage() {
  // How this header spawning code was designed.
  //
  // Find the patterns between the load/store functions:
  // - device has 'uint' elements_per_row
  // - threadgroup has 'ushort' elements_per_row
  // - both have 'ushort2' matrix_origin
  //
  // The origin is 'ushort2' because the 32-bit part of the address should have
  // been applied previously during 'apply_offset'. The 16-bit part should be
  // hard-coded into the assembly when the GEMM loop is unrolled.
  //
  // Transpose path:
  // - load: reads two values; should split each one onto a separate line.
  //   - overwrites the value of *thread_elements() with a new vec<T, 2>
  // - store: the two instructions are on two separate lines.
  //   - fetches from lane 0 or 1 of thread_elements()[0]
  // - adds 0 or 1 to the hard-coded matrix_origin.x
  //
  // Address generation:
  // - casts some intermediate address fragments to 'ulong' for 'device'
  // - keeps all address fragments in 'ushort' for 'threadgroup'
  
  enum class AddressSpace {
    device,
    threadgroup,
  };
  
  auto keyword =
  [=](AddressSpace value) -> std::string {
    switch (value) {
      case AddressSpace::device:
        return "device";
      case AddressSpace::threadgroup:
        return "threadgroup";
    }
  };
  
  auto offsetType =
  [=](AddressSpace value) -> std::string {
    switch (value) {
      case AddressSpace::device:
        return "uint";
      case AddressSpace::threadgroup:
        return "ushort";
    }
  };
  
  enum class Action {
    load,
    store,
  };
  
  struct MemoryAccessDescriptor {
    std::optional<Action> action;
    std::optional<AddressSpace> addressSpace;
    std::optional<bool> decodingBF16;
    int64_t indentationSpaceCount = 0;
  };
  
  auto createMemoryAccess =
  [=](MemoryAccessDescriptor descriptor) -> std::string {
    CCV_NNC_MFA_PRECONDITION(descriptor.action.has_value());
    CCV_NNC_MFA_PRECONDITION(descriptor.addressSpace.has_value());
    CCV_NNC_MFA_PRECONDITION(descriptor.decodingBF16.has_value());
    auto action = descriptor.action.value();
    auto addressSpace = descriptor.addressSpace.value();
    auto decodingBF16 = descriptor.decodingBF16.value();
    std::string indentation(descriptor.indentationSpaceCount, ' ');
    
    // Determine the arguments.
    std::vector<std::string> arguments;
    auto pointerArgument = [=](std::string dataType) {
      if (action == Action::load) {
        return "const " + keyword(addressSpace) + " " + dataType + " *src";
      } else {
        return keyword(addressSpace) + " " + dataType + " *dst";
      }
    };
    if (decodingBF16) {
      arguments.push_back(pointerArgument("bfloat"));
    } else {
      arguments.push_back(pointerArgument("U"));
    }
    arguments.push_back(offsetType(addressSpace) + " elements_per_row");
    arguments.push_back("ushort2 matrix_origin");
    arguments.push_back("bool transpose_matrix = false");
    
    // Create the warning comment.
    std::string output = "";
    if (decodingBF16) {
      output += indentation + "// WARNING: 'T' must be 'float'.\n";
    } else {
      output += indentation + "template <typename U>\n";
    }
    
    // Create the function signature.
    output += indentation + "METAL_FUNC void";
    if (action == Action::load) {
      output += " load";
    } else {
      output += " store";
    }
    if (decodingBF16) {
      output += "_bfloat";
    }
    
    output += "(";
    for (auto it = 0; it < arguments.size(); ++it) {
      int64_t argumentID = it;
      std::string argument = arguments[argumentID];
      
      output += argument;
      if (argumentID < arguments.size() - 1) {
        output += ", ";
      }
    }
    output += ") {\n";
    
    auto createAddress =
    [=](bool transposed, int64_t offset) -> std::string {
      auto lineY = offsetType(addressSpace) + "(matrix_origin.y)";
      auto lineX = "matrix_origin.x + " + std::to_string(offset);
      lineX = offsetType(addressSpace) + "(" + lineX + ")";
      
      if (transposed) {
        return lineX + " * elements_per_row + " + lineY;
      } else {
        return lineY + " * elements_per_row + " + lineX;
      }
    };
    
    auto createTwoPartAccess =
    [=](bool transposed) -> std::vector<std::string> {
      // Generate the addresses.
      std::vector<std::string> lines;
      for (int64_t laneID = 0; laneID < 2; ++laneID) {
        lines.push_back
        (offsetType(addressSpace) + " address" + std::to_string(laneID) +
         " = " + createAddress(transposed, laneID));
      }
      
      if (action == Action::load) {
        if (decodingBF16) {
          lines.push_back("bfloat memoryForm0 = src[address0]");
          lines.push_back("bfloat memoryForm1 = src[address1]");
        } else {
          lines.push_back("U memoryForm0 = src[address0]");
          lines.push_back("U memoryForm1 = src[address1]");
        }
      }
      
      if (action == Action::load) {
        if (decodingBF16) {
          // Separate the loading logic from the decoding logic for clarity.
          lines.push_back
          ("");
          
          // BF16 decoding logic.
          lines.push_back
          ("bfloat4 registerForm = *(thread bfloat4*)(thread_elements())");
          lines.push_back
          ("registerForm[1] = memoryForm0");
          lines.push_back
          ("registerForm[3] = memoryForm1");
          lines.push_back
          ("((thread bfloat4*)thread_elements())[0] = registerForm");
        } else {
          // Perform a type cast natively supported by the hardware.
          lines.push_back
          ("((thread T*)thread_elements())[0] = T(memoryForm0)");
          lines.push_back
          ("((thread T*)thread_elements())[1] = T(memoryForm1)");
        }
      } else {
        if (decodingBF16) {
          // BF16 encoding logic.
          lines.push_back
          ("bfloat4 registerForm = *(thread bfloat4*)(thread_elements())");
          lines.push_back
          ("registerForm[2] = registerForm[1]");
        } else {
          // Type casts supported natively by the hardware.
          lines.push_back
          ("T registerForm0 = ((thread T*)thread_elements())[0]");
          lines.push_back
          ("T registerForm1 = ((thread T*)thread_elements())[1]");
        }
      }
      
      if (action == Action::store) {
        if (decodingBF16) {
          lines.push_back("dst[address0] = registerForm[2]");
          lines.push_back("dst[address1] = registerForm[3]");
        } else {
          lines.push_back("dst[address0] = U(registerForm0)");
          lines.push_back("dst[address1] = U(registerForm1)");
        }
      }
      return lines;
    };
    
    auto createOnePartAccess =
    [=]() -> std::vector<std::string> {
      std::vector<std::string> lines;
      {
        auto address = createAddress(false, 0);
        lines.push_back("auto combinedAddress = " + address);
      }
      if (action == Action::load) {
        if (decodingBF16) {
          lines.push_back
          ("bfloat2 memoryForm = *(const " +
           keyword(addressSpace) + " packed_bfloat2*)(src + combinedAddress)");
          
          // Separate the loading logic from the decoding logic for clarity.
          lines.push_back
          ("");
          
          // BF16 decoding logic.
          lines.push_back
          ("bfloat4 registerForm = *(thread bfloat4*)(thread_elements())");
          lines.push_back
          ("((thread float*)&registerForm)[1] = *(thread float*)(&memoryForm)");
          lines.push_back
          ("((thread bfloat*)&registerForm)[1] = memoryForm[0]");
          lines.push_back
          ("((thread bfloat4*)thread_elements())[0] = registerForm");
        } else {
          lines.push_back
          ("vec<U, 2> memoryForm = *(const " +
           keyword(addressSpace) + " vec<U, 2>*)(src + combinedAddress)");
          lines.push_back
          ("*(thread_elements()) = vec<T, 2>(memoryForm)");
        }
      } else {
        if (decodingBF16) {
          // BF16 encoding logic.
          lines.push_back
          ("bfloat4 registerForm = *(thread bfloat4*)(thread_elements())");
          lines.push_back
          ("registerForm[2] = registerForm[1]");
          lines.push_back
          ("float memoryForm = ((thread float*)&registerForm)[1]");
          lines.push_back
          ("*(" + keyword(addressSpace) + " float*)" +
           "(dst + combinedAddress) = memoryForm");
        } else {
          lines.push_back
          ("vec<T, 2> registerForm = *(thread_elements())");
          lines.push_back
          ("*(" + keyword(addressSpace) + " vec<U, 2>*)" +
           "(dst + combinedAddress) = vec<U, 2>(registerForm)");
        }
      }
      return lines;
    };
    
    return output;
  };
  
  MemoryAccessDescriptor desc;
  desc.indentationSpaceCount = 4;
  desc.action = Action::load;
  desc.addressSpace = AddressSpace::device;
  desc.decodingBF16 = false;
  return createMemoryAccess(desc);
}
