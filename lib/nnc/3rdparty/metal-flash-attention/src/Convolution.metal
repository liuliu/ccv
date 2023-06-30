//
//  Convolution.metal
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/26/23.
//

#include <metal_stdlib>
using namespace metal;

// Only supports NHWC layout.

// Whether this is the pre- or post-transform of convolution.
// - boolean constant
// Whether to transform the data or weights.
// - boolean constant
// Does not fuse the transform with the GEMM yet; use batched GEMM instead.

// Numerical function constant for dimensionality.
// - only 2 supported

// Function constants for window size.
// - only 1x1 and 3x3 supported

// Function constants for Winograd tile size.
// - only 3x3 supported for now
// - only runs when window size is 3x3
// - must always be specified, even if not doing Winograd

// Numerical function constant for bits of palletization.
// - only 0 and 6 supported
// - only supports one-way quantized -> dequantized for now
// constant bool is_palletized = (palletization_bits > 0);

kernel void convolution(/*reserve buffers 0-9*/
                        /*data input*/
                        /*data output*/
                        /*weights input or palletization indices*/
                        /*weights output or palletization indices*/
                        /*actual weights input for palletized*/
                        /*actual weights output for palletized*/
                        /*more buffers might be needed for Winograd temporaries*/
                        
                        /*reserve buffers 10-19*/)
{
  
}
