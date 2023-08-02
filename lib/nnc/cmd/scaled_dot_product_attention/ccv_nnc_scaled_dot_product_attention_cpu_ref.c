#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#ifdef USE_OPENMP
#include <omp.h>
#endif
#ifdef USE_DISPATCH
#include <dispatch/dispatch.h>
#endif

#if __APPLE__
#include <Accelerate/Accelerate.h>
#endif

// Shared methods.
#include "../_ccv_nnc_cpu_ref.h"

static int _ccv_nnc_scaled_dot_product_attention_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
  // attention layer:
  //
  // B = batch size
  // R = attn matrix rows
  // C = attn matrix cols
  // H = head count
  // D = head size
  //
  // Q shape = [B, R, H, D]
  // K shape = [B, C, H, D]
  // V shape = [B, C, H, D]
  // O shape = [B, R, H, D]
  // mask shape = [B, R, C]
  //
  // O = sm(QK^T) * V
  
  // feedforward layer:
  //
  // M = R
  // N = output channel length
  // K = input channel length (H * D)
  //
  // reshape O from [B, R, H, D] to [B, R, H * D]
  // reshape O from [B, R, H * D] to [B, M, N]
  // weights shape = [N, K]
  // bias shape = [N]
  // feedforward output shape = [B, M, N]
  //
  // feedforward_output = O * weights^T + bias
  
  assert(input_size == 6);
  assert(output_size == 2);
  ccv_nnc_tensor_view_t* const Q = (ccv_nnc_tensor_view_t*)inputs[0];
  ccv_nnc_tensor_view_t* const K = (ccv_nnc_tensor_view_t*)inputs[1];
  ccv_nnc_tensor_view_t* const V = (ccv_nnc_tensor_view_t*)inputs[2];
  ccv_nnc_tensor_view_t* const O = (ccv_nnc_tensor_view_t*)outputs[0];
  
  ccv_nnc_tensor_view_t* const mask = (inputs[3] != NULL) ? (ccv_nnc_tensor_view_t*)inputs[3] : NULL;
  ccv_nnc_tensor_view_t* const weights = (inputs[4] != NULL) ? (ccv_nnc_tensor_view_t*)inputs[4] : NULL;
  ccv_nnc_tensor_view_t* const bias = (inputs[5] != NULL) ? (ccv_nnc_tensor_view_t*)inputs[5] : NULL;
  
  // bias always requires a weight matrix.
  if (bias) {
    assert(weights);
  }
  ccv_nnc_tensor_view_t* const feedforward_output = (outputs[1] != NULL) ? (ccv_nnc_tensor_view_t*)outputs[1] : NULL;
  
  assert(ccv_nnc_tensor_nd(Q->info.dim) == 4);
  assert(ccv_nnc_tensor_nd(K->info.dim) == 4);
  assert(ccv_nnc_tensor_nd(V->info.dim) == 4);
  assert(ccv_nnc_tensor_nd(O->info.dim) == 4);
  
  int Q_dim[CCV_NNC_MAX_DIM_ALLOC];
  int K_dim[CCV_NNC_MAX_DIM_ALLOC];
  int V_dim[CCV_NNC_MAX_DIM_ALLOC];
  int O_dim[CCV_NNC_MAX_DIM_ALLOC];
  ccv_nnc_tensor_view_get_dim(Q, Q_dim);
  ccv_nnc_tensor_view_get_dim(K, K_dim);
  ccv_nnc_tensor_view_get_dim(V, V_dim);
  ccv_nnc_tensor_view_get_dim(O, O_dim);
  
  int B = Q_dim[0];
  int R = Q_dim[1];
  int C = K_dim[1];
  int H = Q_dim[2];
  int D = Q_dim[3];
  assert(B == Q_dim[0] && B == K_dim[0] && B == V_dim[0] && B == O_dim[0]);
  assert(R == Q_dim[1] && R == O_dim[1]);
  assert(C == K_dim[1] && C == V_dim[1]);
  assert(H == Q_dim[2] && H == K_dim[2] && H == V_dim[2] && H == O_dim[2]);
  assert(D == Q_dim[3] && D == K_dim[3] && D == V_dim[3] && D == O_dim[3]);
  
  int mask_dim[CCV_NNC_MAX_DIM_ALLOC];
  if (mask)
  {
    assert(ccv_nnc_tensor_nd(mask->info.dim) == 3);
    
    ccv_nnc_tensor_view_get_dim(mask, mask_dim);
    assert(mask_dim[0] == B);
    assert(mask_dim[1] == R);
    assert(mask_dim[2] == C);
  }
  
  const float* Q_pointer = Q->data.f32;
  const float* K_pointer = K->data.f32;
  const float* V_pointer = V->data.f32;
  const float* mask_pointer = mask ? mask->data.f32 : NULL;
  float* O_pointer = O->data.f32;
  
  float scale = cmd.info.scaled_dot_product_attention.scale;
  
  // If it is causal, the mask will be triangular. Therefore we can just to
  // dense attention and explicitly apply the triangular mask. It will be slower
  // than implicit masks (sparse attention), but it works.
  //
  // In addition, we cannot a causal-ish mask is perfectly triangular. There are
  // cases where NNC makes a triangular mask, changes some columns on the edge
  // in a way that's unpredictable at compile-time, and feeds them in as a mask.
  //
  // If you can ensure those are simply an addition of a matrix of [0, -INF] to
  // an upper triangular matrix filled with [0 \ -INF], the unpredictable extra
  // entries will just eliminate zeroes in the lower triangle. Therefore you can
  // use a slightly more efficient CPU implementation that always skips the
  // upper triangle.
  //	const int is_causal = cmd.info.scaled_dot_product_attention.is_causal;
  
  for (int b = 0; b < B; ++b)
  {
    parallel_for(h, H) {
      const int Q_trans = false;
      const int K_trans = true;
      const int V_trans = false;
      const int O_trans = false;
      
      const int Q_leading_dim = Q_trans ? R : H * D;
      const int K_leading_dim = K_trans ? H * D : C;
      const int V_leading_dim = V_trans ? C : H * D;
      const int O_leading_dim = O_trans ? R : H * D;
      
      const float* Q = Q_pointer + h * D;
      float* O = O_pointer + h * D;
      const float* mask = mask_pointer;
      
      for (int r = 0; r < R; ++r)
      {
        const float* K = K_pointer + h * D;
        const float* V = V_pointer + h * D;
        
        // Multiply Q * K.
        float QK[C];
        for (int c = 0; c < C; ++c) {
          float attention_matrix_element = 0;
          
          const int D_floor = (D / 16) * 16;
          for (int d = 0; d < D_floor; d += 16) {
#pragma clang loop unroll_count(16)
            for (int d_offset = 0; d_offset < 16; ++d_offset) {
              attention_matrix_element += Q[d + d_offset] * K[d + d_offset];
            }
          }
          for (int d = D_floor; d < D; ++d) {
            attention_matrix_element += Q[d] * K[d];
          }
          
          attention_matrix_element *= scale;
          QK[c] = attention_matrix_element;
          K += H * D;
        }
        
        // Apply explicit mask.
        if (mask != NULL) {
          for (int c = 0; c < C; ++c) {
            QK[c] += mask[c];
          }
        }
        
        // Compute softmax.
        float maximum_value = -1e38;
        for (int c = 0; c < C; ++c) {
          float attention_matrix_element = QK[c];
          if (attention_matrix_element > maximum_value) {
            maximum_value = attention_matrix_element;
          }
        }
        float denominator = 0;
        for (int c = 0; c < C; ++c) {
          float attention_matrix_element = expf(QK[c] - maximum_value);
          QK[c] = attention_matrix_element;
          denominator += attention_matrix_element;
        }
        float denominator_reciprocal = 1 / denominator;
        for (int c = 0; c < C; ++c) {
          QK[c] *= denominator_reciprocal;
        }
        
        // Multiply P * V.
        float O_temp[D];
        for (int d = 0; d < D; ++d) {
          O_temp[d] = 0;
        }
        for (int c = 0; c < C; ++c) {
          float attention_matrix_element = QK[c];
          
          const int D_floor = (D / 16) * 16;
          for (int d = 0; d < D_floor; d += 16) {
#pragma clang loop unroll_count(16)
            for (int d_offset = 0; d_offset < 16; ++d_offset) {
              O_temp[d + d_offset] += attention_matrix_element * V[d + d_offset];
            }
          }
          for (int d = D_floor; d < D; ++d) {
            O_temp[d] += attention_matrix_element * V[d];
          }
          
          V += H * D;
        }
        memcpy(O, O_temp, D * sizeof(float));
        
        Q += Q_leading_dim;
        O += O_leading_dim;
      }
    } parallel_endfor
    
    Q_pointer += R * H * D;
    K_pointer += C * H * D;
    V_pointer += C * H * D;
    if (mask != NULL) {
      mask_pointer += R * C;
    }
    O_pointer += R * H * D;
  }
  
  if (weights)
  {
    assert(feedforward_output);
    assert(ccv_nnc_tensor_nd(weights->info.dim) == 2);
    assert(ccv_nnc_tensor_nd(feedforward_output->info.dim) == 3);
    
    int weights_dim[CCV_NNC_MAX_DIM_ALLOC];
    int feedforward_output_dim[CCV_NNC_MAX_DIM_ALLOC];
    ccv_nnc_tensor_view_get_dim(weights, weights_dim);
    ccv_nnc_tensor_view_get_dim(feedforward_output, feedforward_output_dim);
    
    const int M = R;
    const int N = feedforward_output_dim[2];
    const int K = H * D;
    assert(B == feedforward_output_dim[0] && B == O_dim[0]);
    assert(M == feedforward_output_dim[1] && M == O_dim[1]);
    assert(N == feedforward_output_dim[1] && N == weights_dim[0]);
    assert(K == (O_dim[2] * O_dim[3]) && K == weights_dim[1]);
    
    parallel_for(b, B) {
      float* A = O->data.f32 + b * M * K;
      float* B = weights->data.f32;
      float* C = feedforward_output->data.f32 + b * M * N;
      
      // Multiply O * W.
      float alpha = 1.0;
      float beta = 0.0;
      
#if __APPLE__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wincompatible-pointer-types-discards-qualifiers"
      sgemm_(/*TRANSA*/"N",
             /*TRANSB*/"T",
             /*M*/&M,
             /*N*/&N,
             /*K*/&K,
             /*ALPHA*/&alpha,
             /*A*/A,
             /*LDA*/&K,
             /*B*/B,
             /*LDB*/&K,
             /*BETA*/&beta,
             /*C*/C,
             /*LDC*/&N);
#pragma clang diagnostic pop
#else
      for (int m = 0; m < M; ++m) {
        B = weights->data.f32;
        for (int n = 0; n < N; ++n) {
          float C_element = 0;
          
          const int K_floor = (K / 16) * 16;
          for (int k = 0; k < K_floor; k += 16) {
#pragma clang loop unroll_count(16)
            for (int k_offset = 0; k_offset < 16; ++k_offset) {
              C_element += A[k + k_offset] * B[k + k_offset];
            }
          }
          for (int k = K_floor; k < K; ++k) {
            C_element += A[k] * B[k];
          }
          
          B += K;
          C[n] = C_element;
        }
        A += K;
        C += N;
      }
      C = feedforward_output->data.f32 + b * M * N;
#endif
      
      // Apply bias.
      if (bias) {
        float *bias_pointer = bias->data.f32;
        
        for (int m = 0; m < M; ++m) {
          const int N_floor = (N / 16) * 16;
          for (int n = 0; n < N_floor; n += 16) {
#pragma clang loop unroll_count(16)
            for (int n_offset = 0; n_offset < 16; ++n_offset) {
              C[n + n_offset] += bias_pointer[n + n_offset];
            }
          }
          for (int n = N_floor; n < N; ++n) {
            C[n] += bias_pointer[n];
          }
          C += N;
        }
      }
    } parallel_endfor
  }
  return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_scaled_dot_product_attention_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_scaled_dot_product_attention_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_scaled_dot_product_attention_back;
}
