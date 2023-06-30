//
//  Attention.metal
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/26/23.
//

#include <metal_stdlib>
using namespace metal;

// Dimensions of each matrix.
constant uint M [[function_constant(0)]];
constant uint N [[function_constant(1)]];
constant uint K [[function_constant(2)]];

// Supporting transposes would require supporting strided reads.
// QK is always scaled by rsqrt(D), so you cannot set any scale constants.
constant bool batched [[function_constant(100)]];

constant ushort R_simd [[function_constant(200)]];
constant ushort C_simd [[function_constant(201)]];

constant ushort R_splits [[function_constant(210)]];
constant ushort C_splits [[function_constant(211)]]; /* 1, 2, 3, 4, 6, 8 */

//constant ushort R_group = R_simd * R_splits;
//constant ushort C_group = C_simd * C_splits;

kernel void attention()
{
  
}
