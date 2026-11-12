#ifndef MFA_TMCQ_DECODE_SOURCE_HPP_
#define MFA_TMCQ_DECODE_SOURCE_HPP_

#include <stdint.h>
#include <string>

// The frozen v1.2 pair codebook, ccv_nnc_tmcq_p2_table[1024] (512 pairs, interleaved int8). The Metal
// table below is generated from it, so the .inc stays the only copy.
#include "../../ccv_nnc_tmcq_table.inc"

// Metal helpers shared by every kernel that decodes CCV_NNC_QX_8I_ROWWISE_TMCQ_N* weights.
// Codec definition: see CCV_NNC_QX_8I_ROWWISE_TMCQ_N16 in ccv_nnc.h. A tile is 16 * n bits = n / 2
// little-endian 32-bit words (n is even), so every tile starts on a word boundary. Weight t's window
// is the 15-bit window ending at E_t = floor((t + 1) n / 16); weight 0's window wraps from the tile's
// end. v1.2 decodes weights in pairs: weights 2j and 2j + 1 both read the window of weight 2j + 1,
// cw, and take entry k = (cw * 0x83DCD12D) >> 23 of the 512-pair table as (q8[2j], q8[2j + 1]).
static const char* const ccv_nnc_mfa_tmcq_decode_helpers = R"(
// The 32 bits of the little-endian pair (lo, hi) starting at bit o (0 <= o < 32).
// (hi << 1) << (31 - o) is hi << (32 - o) without an undefined 32-bit shift when o == 0.
inline uint tmcq_funnel(uint lo, uint hi, uint o)
{
  return (lo >> o) | ((hi << 1) << (31u - o));
}

// Fills a threadgroup copy of the table, lane `index` of `count` threads. The caller adds the barrier.
inline void tmcq_load_table(threadgroup char2* table, uint index, uint count)
{
  for (uint i = index; i < 512u; i += count)
    table[i] = tmcq_table[i];
}

// int8 codes of row r (weights 16 r ... 16 r + 15) of a tile, as q[0 .. 3] = columns 0-3, ..., 12-15.
// `table` is tmcq_table (constant) or a threadgroup copy from tmcq_load_table.
// The row's windows start at bits first + k_j, k_j = floor((j + 1) n / 16) - floor(n / 16), with
// first = r n + floor(n / 16) - 15, and end by (r + 1) n: at most n + 14 <= 46 bits. Three words from
// the first window's word cover them; shifting that 96-bit value right by first % 32 leaves a 64-bit
// (a, b) where window j is bits [k_j, k_j + 15) with k_j <= 30 known at compile time. Only row 0
// starts before the tile (its first window is the tail-biting one); its word index wraps to the tile's
// end. No other window crosses the tile's end, so the wrapped third word is never used. Only the 8 odd
// windows (j = 2 i + 1) are read.
template <typename Table>
inline void tmcq_decode_row16(device const uint* tile, uint n, uint r, thread int4* q, Table table)
{
  const int words = int(n >> 1);
  const int first = int(r * n + (n >> 4)) - 15;
  const int w = first >> 5; // -1 only for row 0
  const uint s0 = uint(first) & 31u;
  const int i0 = w < 0 ? w + words : w;
  const int i1 = w + 1 >= words ? w + 1 - words : w + 1;
  const int i2 = w + 2 >= words ? w + 2 - words : w + 2;
  const uint w0 = tile[i0];
  const uint w1 = tile[i1];
  const uint w2 = tile[i2];
  const uint a = tmcq_funnel(w0, w1, s0);
  const uint b = tmcq_funnel(w1, w2, s0);
  int c[16];
  #pragma clang loop unroll(full)
  for (uint i = 0; i < 8u; ++i)
  {
    const uint k = (((2u * i + 2u) * n) >> 4) - (n >> 4);
    const uint window = k + 15u <= 32u ? (a >> k) : ((a >> k) | (b << (32u - k)));
    const char2 v = table[((window & 0x7fffu) * 0x83DCD12Du) >> 23];
    c[2u * i] = v.x;
    c[2u * i + 1u] = v.y;
  }
  q[0] = int4(c[0], c[1], c[2], c[3]);
  q[1] = int4(c[4], c[5], c[6], c[7]);
  q[2] = int4(c[8], c[9], c[10], c[11]);
  q[3] = int4(c[12], c[13], c[14], c[15]);
}
)";

// Helpers used only by the TMCQ GEMV kernels (dense Int8GemvKernel and SegmentedInt8GemvKernel).
// They take the three tile words a row needs and return the row's 16 weights as float4s, reading a
// half2 threadgroup copy of the pair table: every entry is one 32-bit bank word, and the weights need
// no int-to-float converts before the float dot products.
static const char* const ccv_nnc_mfa_tmcq_w1_helpers = R"(
// Word indices (within the n / 2-word tile) and bit offset of the first window of row r; see
// tmcq_decode_row16 for the layout. Only row 0 wraps, and only its first word.
struct tmcq_w1_row_words
{
  uint i0, i1, i2, s0;
};

inline tmcq_w1_row_words tmcq_w1_row_layout(uint n, uint r)
{
  const int words = int(n >> 1);
  const int first = int(r * n + (n >> 4)) - 15;
  const int w = first >> 5;
  tmcq_w1_row_words l;
  l.s0 = uint(first) & 31u;
  l.i0 = uint(w < 0 ? w + words : w);
  l.i1 = uint(w + 1 >= words ? w + 1 - words : w + 1);
  l.i2 = uint(w + 2 >= words ? w + 2 - words : w + 2);
  return l;
}

// Fills a half2 threadgroup copy of tmcq_table (int8 values are exact in half), lane `index` of
// `count` threads. The caller adds the barrier.
inline void tmcq_w1_load_table(threadgroup half2* table, uint index, uint count)
{
  for (uint i = index; i < 512u; i += count)
    table[i] = half2(float2(tmcq_table[i]));
}

// The 16 weights of a tile row from its three words w0, w1, w2 (tile words i0, i1, i2 of
// tmcq_w1_row_layout) and offset s0, as q[0 .. 3] = columns 0-3, ..., 12-15; same windows and
// codebook entries as tmcq_decode_row16.
inline void tmcq_w1_decode_words(uint w0, uint w1, uint w2, uint s0, uint n, thread float4* q, threadgroup const half2* table)
{
  const uint a = tmcq_funnel(w0, w1, s0);
  const uint b = tmcq_funnel(w1, w2, s0);
  float2 c[8];
  #pragma clang loop unroll(full)
  for (uint i = 0; i < 8u; ++i)
  {
    const uint k = (((2u * i + 2u) * n) >> 4) - (n >> 4);
    const uint window = k + 15u <= 32u ? (a >> k) : ((a >> k) | (b << (32u - k)));
    c[i] = float2(table[((window & 0x7fffu) * 0x83DCD12Du) >> 23]);
  }
  q[0] = float4(c[0], c[1]);
  q[1] = float4(c[2], c[3]);
  q[2] = float4(c[4], c[5]);
  q[3] = float4(c[6], c[7]);
}
)";

// MSL for the TMCQ GEMV main loop. In scope it needs M, STREAMS, stream, tc_end, tiles, layout and
// float sum[M]. Each lane walks tile columns tc = stream, stream + STREAMS, ... below tc_end, `unroll`
// tiles per iteration: all of their words are read before any is decoded, and each adds into its own
// accumulators acc<u>[M]; a one-tile loop takes the remainder, and the accumulators end in sum[M].
// `accumulate` is the per-tile statement, with '@' standing for the unroll index u: it reads tile
// column tc@ and weights q@[0 .. 3] and adds into acc@. `n` names the rate in MSL.
static inline std::string tmcq_w1_loop_source(const uint32_t unroll, const std::string& table, const std::string& accumulate, const std::string& n)
{
  auto at = [](const std::string& text, const uint32_t u) {
    std::string out;
    for (const char c : text)
      out += c == '@' ? std::to_string(u) : std::string(1, c);
    return out;
  };
  auto body = [&](const uint32_t count) {
    std::string b;
    for (uint32_t u = 0; u < count; u++) {
      const std::string i = std::to_string(u);
      b += "    const uint tc" + i + " = tc + " + i + "u * STREAMS;\n";
      b += "    device const uint* tile" + i + " = tiles + (ulong)tc" + i + " * (" + n + " / 2u);\n";
      b += "    const uint w0_" + i + " = tile" + i + "[layout.i0];\n";
      b += "    const uint w1_" + i + " = tile" + i + "[layout.i1];\n";
      b += "    const uint w2_" + i + " = tile" + i + "[layout.i2];\n";
    }
    for (uint32_t u = 0; u < count; u++) {
      const std::string i = std::to_string(u);
      b += "    float4 q" + i + "[4];\n";
      b += "    tmcq_w1_decode_words(w0_" + i + ", w1_" + i + ", w2_" + i + ", layout.s0, " + n + ", q" + i + ", " + table + ");\n";
      b += at(accumulate, u);
    }
    return b;
  };
  std::string s;
  for (uint32_t u = 0; u < unroll; u++)
    s += "  float acc" + std::to_string(u) + "[M];\n  for (uint m = 0; m < M; m++)\n    acc" + std::to_string(u) + "[m] = 0;\n";
  s += "  uint tc = stream;\n";
  s += "  for (; tc + " + std::to_string(unroll - 1) + "u * STREAMS < tc_end; tc += " + std::to_string(unroll) + "u * STREAMS) {\n" + body(unroll) + "  }\n";
  if (unroll > 1)
    s += "  for (; tc < tc_end; tc += STREAMS) {\n" + body(1) + "  }\n";
  s += "  for (uint m = 0; m < M; m++)\n    sum[m] = acc0[m]";
  for (uint32_t u = 1; u < unroll; u++)
    s += " + acc" + std::to_string(u) + "[m]";
  s += ";\n";
  return s;
}

// MSL for the table (`constant char2 tmcq_table[512]`) followed by the helpers.
static inline const char* ccv_nnc_mfa_tmcq_decode_source()
{
  static const std::string source = [] {
    std::string s = "constant char2 tmcq_table[512] = {\n";
    for (int i = 0; i < 512; i++)
    {
      s += "char2(" + std::to_string(ccv_nnc_tmcq_p2_table[2 * i]) + ", " + std::to_string(ccv_nnc_tmcq_p2_table[2 * i + 1]) + ")";
      s += (i % 8 == 7) ? ",\n" : ", ";
    }
    s += "};\n";
    s += ccv_nnc_mfa_tmcq_decode_helpers;
    return s;
  }();
  return source.c_str();
}

// Register-fragment helpers for the segmented scaled GEMM prefill kernel only (tmcq_w3_*). They
// decode exactly like tmcq_decode_row16 but return a row's 16 codes packed as int8 bytes, columns
// 4c ... 4c + 3 in word c (little-endian), which is the layout the register fragments take.
static const char* const tmcq_w3_decode_helpers = R"(
// tmcq_decode_row16's windows, with each table entry read as one ushort (q8[2j] in the low byte).
template <typename Table>
inline uint4 tmcq_w3_decode_row16_words(device const uint* tile, uint n, uint r, Table table)
{
  const int words = int(n >> 1);
  const int first = int(r * n + (n >> 4)) - 15;
  const int w = first >> 5; // -1 only for row 0
  const uint s0 = uint(first) & 31u;
  const int i0 = w < 0 ? w + words : w;
  const int i1 = w + 1 >= words ? w + 1 - words : w + 1;
  const int i2 = w + 2 >= words ? w + 2 - words : w + 2;
  const uint w0 = tile[i0];
  const uint w1 = tile[i1];
  const uint w2 = tile[i2];
  const uint a = tmcq_funnel(w0, w1, s0);
  const uint b = tmcq_funnel(w1, w2, s0);
  uint t[8];
  #pragma clang loop unroll(full)
  for (uint i = 0; i < 8u; ++i)
  {
    const uint k = (((2u * i + 2u) * n) >> 4) - (n >> 4);
    const uint window = k + 15u <= 32u ? (a >> k) : ((a >> k) | (b << (32u - k)));
    t[i] = table[((window & 0x7fffu) * 0x83DCD12Du) >> 23];
  }
  return uint4(t[0] | (t[1] << 16), t[2] | (t[3] << 16), t[4] | (t[5] << 16), t[6] | (t[7] << 16));
}
)";

// MSL for `constant ushort tmcq_w3_table[512]` (entry i is ccv_nnc_tmcq_p2_table[2 i] in the low byte
// and [2 i + 1] in the high byte) followed by the tmcq_w3_* helpers. Append after
// ccv_nnc_mfa_tmcq_decode_source(), which defines tmcq_funnel.
static inline const char* tmcq_w3_decode_source()
{
  static const std::string source = [] {
    std::string s = "constant ushort tmcq_w3_table[512] = {\n";
    for (int i = 0; i < 512; i++)
    {
      const unsigned value = (unsigned)(uint8_t)ccv_nnc_tmcq_p2_table[2 * i] | ((unsigned)(uint8_t)ccv_nnc_tmcq_p2_table[2 * i + 1] << 8);
      s += std::to_string(value) + "u";
      s += (i % 16 == 15) ? ",\n" : ", ";
    }
    s += "};\n";
    s += tmcq_w3_decode_helpers;
    return s;
  }();
  return source.c_str();
}

// Helpers for the segmented SwiGLU decode. They read a half2 copy of the pair table: every int8 code
// is exact in half, and a half feeds a float FMA without the sign extension and int -> float
// conversion that a char2 entry needs. The decoded codes are the same as tmcq_decode_row16's.
static const char* const ccv_nnc_mfa_tmcq_w2_helpers = R"(
// Fills a threadgroup copy of tmcq_w2_table, lane `index` of `count` threads. The caller adds the
// barrier.
inline void tmcq_w2_load_table(threadgroup half2* table, uint index, uint count)
{
  for (uint i = index; i < 512u; i += count)
    table[i] = tmcq_w2_table[i];
}

// Codes of row r of a tile as q[0 .. 3] = columns 0-3, ..., 12-15: the same windows and table
// entries as tmcq_decode_row16, from a half2 table (tmcq_w2_table or a tmcq_w2_load_table copy).
template <typename Table>
inline void tmcq_w2_decode_row16(device const uint* tile, uint n, uint r, thread half4* q, Table table)
{
  const int words = int(n >> 1);
  const int first = int(r * n + (n >> 4)) - 15;
  const int w = first >> 5; // -1 only for row 0
  const uint s0 = uint(first) & 31u;
  const int i0 = w < 0 ? w + words : w;
  const int i1 = w + 1 >= words ? w + 1 - words : w + 1;
  const int i2 = w + 2 >= words ? w + 2 - words : w + 2;
  const uint w0 = tile[i0];
  const uint w1 = tile[i1];
  const uint w2 = tile[i2];
  const uint a = tmcq_funnel(w0, w1, s0);
  const uint b = tmcq_funnel(w1, w2, s0);
  half2 c[8];
  #pragma clang loop unroll(full)
  for (uint i = 0; i < 8u; ++i)
  {
    const uint k = (((2u * i + 2u) * n) >> 4) - (n >> 4);
    const uint window = k + 15u <= 32u ? (a >> k) : ((a >> k) | (b << (32u - k)));
    c[i] = table[((window & 0x7fffu) * 0x83DCD12Du) >> 23];
  }
  q[0] = half4(c[0], c[1]);
  q[1] = half4(c[2], c[3]);
  q[2] = half4(c[4], c[5]);
  q[3] = half4(c[6], c[7]);
}
)";

// MSL for `constant half2 tmcq_w2_table[512]`, generated from ccv_nnc_tmcq_p2_table, followed by the
// W2 helpers. Goes after ccv_nnc_mfa_tmcq_decode_source(), whose tmcq_funnel the helpers use.
static inline const char* ccv_nnc_mfa_tmcq_w2_source()
{
  static const std::string source = [] {
    std::string s = "constant half2 tmcq_w2_table[512] = {\n";
    for (int i = 0; i < 512; i++)
    {
      s += "half2(" + std::to_string(ccv_nnc_tmcq_p2_table[2 * i]) + ", " + std::to_string(ccv_nnc_tmcq_p2_table[2 * i + 1]) + ")";
      s += (i % 8 == 7) ? ",\n" : ", ";
    }
    s += "};\n";
    s += ccv_nnc_mfa_tmcq_w2_helpers;
    return s;
  }();
  return source.c_str();
}

#endif
