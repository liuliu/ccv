#ifndef MFA_TMCQ_DECODE_SOURCE_HPP_
#define MFA_TMCQ_DECODE_SOURCE_HPP_

#include <stdint.h>
#include <string>
#include "../ccv_nnc_mfa_error.hpp"

// The built-in pair codebook, ccv_nnc_trellis_builtin_table[1024] (512 pairs, interleaved int8), for TMCQ and
// FTQ table id 0. The Metal tables below are generated from it, so the .inc file stays the only copy.
#include "../../ccv_nnc_trellis_builtin_table.inc"
#include "../../ccv_nnc_ftq_schedule.inc"

// Metal helpers shared by every kernel that decodes CCV_NNC_QX_8I_ROWWISE_TMCQ_N* weights.
// Codec definition: see CCV_NNC_QX_8I_ROWWISE_TMCQ_N16 in ccv_nnc.h. A tile is 16 * n bits = n / 2
// little-endian 32-bit words (n is even), so every tile starts on a word boundary. Weight t's window
// is the 15-bit window ending at E_t = floor((t + 1) n / 16); weight 0's window wraps from the tile's
// end. TMCQ decodes weights in pairs: weights 2j and 2j + 1 both read the window of weight 2j + 1,
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

// MSL for `constant char2 tmcq_table[512]` holding `table` followed by the helpers.
static inline std::string ccv_nnc_mfa_tmcq_decode_source_of(const int8_t* const table)
{
  std::string s = "constant char2 tmcq_table[512] = {\n";
  for (int i = 0; i < 512; i++)
  {
    s += "char2(" + std::to_string(table[2 * i]) + ", " + std::to_string(table[2 * i + 1]) + ")";
    s += (i % 8 == 7) ? ",\n" : ", ";
  }
  s += "};\n";
  s += ccv_nnc_mfa_tmcq_decode_helpers;
  return s;
}

// tmcq_table is the built-in table: TMCQ and FTQ with table id 0.
static inline const char* ccv_nnc_mfa_tmcq_decode_source()
{
  static const std::string source = ccv_nnc_mfa_tmcq_decode_source_of(ccv_nnc_trellis_builtin_table);
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

// MSL for `constant ushort tmcq_w3_table[512]` (entry i is ccv_nnc_trellis_builtin_table[2 i] in the low byte
// and [2 i + 1] in the high byte) followed by the tmcq_w3_* helpers. Append after
// ccv_nnc_mfa_tmcq_decode_source(), which defines tmcq_funnel.
static inline const char* tmcq_w3_decode_source()
{
  static const std::string source = [] {
    std::string s = "constant ushort tmcq_w3_table[512] = {\n";
    for (int i = 0; i < 512; i++)
    {
      const unsigned value = (unsigned)(uint8_t)ccv_nnc_trellis_builtin_table[2 * i] | ((unsigned)(uint8_t)ccv_nnc_trellis_builtin_table[2 * i + 1] << 8);
      s += std::to_string(value) + "u";
      s += (i % 16 == 15) ? ",\n" : ", ";
    }
    s += "};\n";
    s += tmcq_w3_decode_helpers;
    return s;
  }();
  return source.c_str();
}

// FTQ register-fragment helpers for the segmented scaled GEMM prefill kernel only (ftq_w3_*, CCV_NNC_QX_8I_ROWWISE_FTQ_N16):
// a tile row's 16 int8 codes packed like tmcq_w3_decode_row16_words, with the windows, pairs and fold of
// ftq_store_segment (Dequantize8iRowwiseXKernel), so the codes are the selected-expert decode's bit for bit. The table
// is a threadgroup ushort copy of the pairs (q8[2 j] in the low byte); log2Entries / fold as ccv_nnc_mfa_ftq_w1_source.
// Append after ccv_nnc_mfa_tmcq_decode_source(), which defines tmcq_funnel.
// ftq_w3_decode_row16_words holds up to N = 90, so for every open rate (n16 - n88): the last window ends at
// N - floor(N / 8) + 16 <= 95 bits past the row's start (three funnels, f[0 .. 2]), from sh <= 31 that is four words, and
// every window from offset 64 has (offset & 31) <= 16, so none reads past f[2].
static inline std::string ccv_nnc_mfa_ftq_w3_source(const uint32_t log2Entries, const bool fold)
{
  std::string shader = "constant uint ftq_w3_index_shift = " + std::to_string(32 - log2Entries) + "u;\n";
  shader += "#define FTQ_W3_ENTRIES " + std::to_string(1u << log2Entries) + "u\n";
  shader += std::string("#define FTQ_W3_FOLD ") + (fold ? "1" : "0") + "\n";
  shader += R"(
// The 32 bits of the little-endian pair (lo, hi) from bit o (0 <= o < 32) as one 64-bit shift. A row's bit offset mod
// 32 varies with the row unless 32 divides n. With such a varying offset above rate 32 this form took less time than
// tmcq_funnel's two 32-bit shifts on M5 Ultra (the fused GEMM, n34 - n56: 2 - 16%), but more at n24 (4%) and, with the
// constant offset, at n64 (1 - 3%), so ftq_w3_decode_row16_words picks it by that rule.
inline uint ftq_w3_funnel(uint lo, uint hi, uint o)
{
  return uint(as_type<ulong>(uint2(lo, hi)) >> o);
}

// The packed pair of a 16-bit state. FTQ_W3_FOLD: base pair (h >> 16) & 511, each int8 negated by bit 15 / 31 of h,
// the expression of the selected-expert decode's ftq_fold_lookup.
inline uint ftq_w3_pair(threadgroup const ushort* table, uint state)
{
#if FTQ_W3_FOLD
  const uint h = state * 0x83DCD12Du;
  const char2 v = as_type<char2>(table[(h >> 16) & 511u]);
  return as_type<ushort>(char2((h & 0x8000u) ? -v.x : v.x, (h & 0x80000000u) ? -v.y : v.y));
#else
  return table[(state * 0x83DCD12Du) >> ftq_w3_index_shift];
#endif
}

// Row r of a rate N tile whose n / 2 words start at `tile`: its windows start at bit r N + floor(N / 8) - 16, row 0's
// before the tile, wrapping to its end, and the words from there are taken mod N / 2, as ftq_store_segment reads them.
// Window i is at offset floor((i + 1) N / 8) - floor(N / 8) of the funnelled bits, so the windows span
// N - floor(N / 8) + 16 bits: from bit offset sh <= 31 that is within three words up to N = 56 (ftq_store_segment reads a
// fourth from N = 50, whose bits no window uses there) and four words from N = 58.
template <uint N>
inline uint4 ftq_w3_decode_row16_words(device const uint* tile, uint r, threadgroup const ushort* table)
{
  constexpr uint W = N / 2u;
  const uint lane_bit = (r == 0u ? 16u * N : r * N) + (N >> 3) - 16u;
  const uint o0 = lane_bit >> 5;
  const uint o1 = o0 + 1u == W ? 0u : o0 + 1u;
  const uint o2 = o1 + 1u == W ? 0u : o1 + 1u;
  const uint sh = lane_bit & 31u;
  const uint w0 = tile[o0], w1 = tile[o1], w2 = tile[o2];
  constexpr bool wide_funnel = N > 32u && (N % 32u) != 0u;
  uint f[3];
  f[0] = wide_funnel ? ftq_w3_funnel(w0, w1, sh) : tmcq_funnel(w0, w1, sh);
  f[1] = wide_funnel ? ftq_w3_funnel(w1, w2, sh) : tmcq_funnel(w1, w2, sh);
  if (N > 48u) {
    // Bits of f[2] past the windows' end are never read, so up to N = 56 the fourth word may as well be 0.
    const uint w3 = 31u + N - (N >> 3) + 16u > 96u ? tile[o2 + 1u == W ? 0u : o2 + 1u] : 0u;
    f[2] = wide_funnel ? ftq_w3_funnel(w2, w3, sh) : tmcq_funnel(w2, w3, sh);
  } else
    f[2] = 0;
  uint t[8];
  #pragma clang loop unroll(full)
  for (uint i = 0; i < 8u; ++i)
  {
    const uint k = (((i + 1u) * N) >> 3) - (N >> 3);
    const uint sub = k & 31u;
    const uint state = (sub <= 16u ? (f[k >> 5] >> sub) : ((f[k >> 5] >> sub) | (f[(k >> 5) + 1u] << (32u - sub)))) & 0xffffu;
    t[i] = ftq_w3_pair(table, state);
  }
  return uint4(t[0] | (t[1] << 16), t[2] | (t[3] << 16), t[4] | (t[5] << 16), t[6] | (t[7] << 16));
}
)";
  return shader;
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

// Codes of rows r0 .. r0 + R - 1 of an n = N tile: q[4 * rr .. 4 * rr + 3] are row r0 + rr's columns
// 0-3, ..., 12-15, with the same windows and table entries as tmcq_decode_row16, from a half2 table
// (tmcq_w2_table or a tmcq_w2_load_table copy). The R rows are one run of consecutive tile words
// (mod N / 2), loaded once for all of them: for N = 22 two rows take the same three words as one.
// N and R are template parameters so that the word and funnel arrays have fixed sizes and stay in
// registers.
template <uint N, uint R, typename Table>
inline void tmcq_w2_decode_rows(device const uint* tile, uint r0, thread half4* q, Table table)
{
  constexpr int words = int(N >> 1);
  // Bits from the first window of row r0 to the end of row r0 + R - 1, in 32-bit funnels.
  constexpr uint funnels = (R * N - (N >> 4) + 15u + 31u) >> 5;
  const int first = int(r0 * N + (N >> 4)) - 15;
  const int w = first >> 5; // -1 only for row 0
  const uint s0 = uint(first) & 31u;
  uint x[funnels + 1];
  #pragma clang loop unroll(full)
  for (uint m = 0; m <= funnels; ++m)
  {
    const int i = w + int(m);
    x[m] = tile[i < 0 ? i + words : (i >= words ? i - words : i)];
  }
  // One spare funnel so that the not taken arm of the window select below stays in bounds.
  uint f[funnels + 1];
  #pragma clang loop unroll(full)
  for (uint m = 0; m < funnels; ++m)
    f[m] = tmcq_funnel(x[m], x[m + 1], s0);
  f[funnels] = 0;
  #pragma clang loop unroll(full)
  for (uint rr = 0; rr < R; ++rr)
  {
    half2 c[8];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 8u; ++i)
    {
      const uint k = rr * N + (((2u * i + 2u) * N) >> 4) - (N >> 4);
      const uint fi = k >> 5;
      const uint sub = k & 31u;
      const uint window = sub + 15u <= 32u ? (f[fi] >> sub) : ((f[fi] >> sub) | (f[fi + 1] << (32u - sub)));
      c[i] = table[((window & 0x7fffu) * 0x83DCD12Du) >> 23];
    }
    q[4 * rr + 0] = half4(c[0], c[1]);
    q[4 * rr + 1] = half4(c[2], c[3]);
    q[4 * rr + 2] = half4(c[4], c[5]);
    q[4 * rr + 3] = half4(c[6], c[7]);
  }
}
)";

// MSL for `constant half2 tmcq_w2_table[512]`, generated from ccv_nnc_trellis_builtin_table, followed by the
// W2 helpers. Goes after ccv_nnc_mfa_tmcq_decode_source(), whose tmcq_funnel the helpers use.
static inline const char* ccv_nnc_mfa_tmcq_w2_source()
{
  static const std::string source = [] {
    std::string s = "constant half2 tmcq_w2_table[512] = {\n";
    for (int i = 0; i < 512; i++)
    {
      s += "half2(" + std::to_string(ccv_nnc_trellis_builtin_table[2 * i]) + ", " + std::to_string(ccv_nnc_trellis_builtin_table[2 * i + 1]) + ")";
      s += (i % 8 == 7) ? ",\n" : ", ";
    }
    s += "};\n";
    s += ccv_nnc_mfa_tmcq_w2_helpers;
    return s;
  }();
  return source.c_str();
}

// FTQ GEMV helpers (CCV_NNC_QX_8I_ROWWISE_FTQ_N16), shared by the dense Int8GemvKernel and SegmentedInt8GemvKernel: the
// pair lookup from a half2 threadgroup copy of the table and the 8 windows of one lane. log2Entries / fold: the pair
// table (9 for the built-in one); fold is a quadrant-folded external table (CCV_NNC_MFA_CODEBOOK_FOLD, log2Entries 9).
// Append after ccv_nnc_mfa_tmcq_decode_source(), whose tmcq_funnel the rate loop uses.
// ftq_decode_lane holds up to N = 90, so for every open rate (n16 - n88): window offsets up to 48 read lo, the others
// hi >> (offset - 32) with offset - 32 + 16 <= 64, and the four words of the rate loops (n > 48) cover the last window's
// end, N - floor(N / 8) + 16 <= 95 bits from sh <= 31.
static inline std::string ccv_nnc_mfa_ftq_w1_source(const uint32_t log2Entries, const bool fold)
{
  // 32 - log2 of the pair count.
  std::string shader = "constant uint ftq_index_shift = " + std::to_string(32 - log2Entries) + ";\n";
  shader += "constant uint ftq_entries = " + std::to_string(1u << log2Entries) + ";\n";
  shader += std::string("#define FTQ_FOLD ") + (fold ? "1" : "0") + "\n";
  shader += R"(
// The pair of a 16-bit state. FTQ_FOLD: quadrant fold, one and + one xor on the half2 bits (the fused GEMV is ALU
// bound, so the folded lookup must stay this short).
inline float2 ftq_lookup(threadgroup const half2* table, uint state)
{
#if FTQ_FOLD
  const uint h = state * 0x83DCD12Du;
  uint v = as_type<uint>(table[(h >> 16) & 511u]);
  v ^= h & 0x80008000u;
  return float2(as_type<half2>(v));
#else
  return float2(table[(state * 0x83DCD12Du) >> ftq_index_shift]);
#endif
}

// The 8 windows of a lane from its funnelled bits at rate N: window i at offset floor((i + 1) N / 8) - floor(N / 8) of the
// lane's first window start, from the low 64 bits or (N > 48 only, offsets past 48) from bits 32 .. 95.
template <uint N>
inline void ftq_decode_lane(ulong lo, ulong hi, thread float4* q, threadgroup const half2* table)
{
  float2 c[8];
  #pragma clang loop unroll(full)
  for (uint i = 0; i < 8u; i++) {
    const uint k = (((i + 1u) * N) >> 3) - (N >> 3);
    const uint state = (k <= 48u ? uint(lo >> k) : uint(hi >> (k - 32u))) & 0xffffu;
    c[i] = ftq_lookup(table, state);
  }
  q[0] = float4(c[0], c[1]);
  q[1] = float4(c[2], c[3]);
  q[2] = float4(c[4], c[5]);
  q[3] = float4(c[6], c[7]);
}
)";
  return shader;
}

// The tile loop of one threadgroup's 16 rows at the compile-time rate n, over row_code (the tile row's first word):
// the lane's three (n <= 48) or four words from its first window's start and its 8 windows at compile-time offsets.
// Below n = 16 the first windows of rows 0 and 1 start before the tile and wrap to its end. In scope it needs r, stream,
// STREAMS, tiles_per_row, row_code, tmcq_tg (the half2 table); `accumulate` is the per-tile statement, which reads
// tile column tc and its weights qa[0 .. 3].
static inline std::string ccv_nnc_mfa_ftq_w1_rate_source(const uint32_t n, const std::string& accumulate)
{
  std::string body = "  {\n  constexpr uint N = " + std::to_string(n) + "u;\n  constexpr uint W = N / 2u;\n";
  if (n >= 16)
    body += "  const uint lane_bit = (r == 0u ? 16u * N : r * N) + (N >> 3) - 16u;\n";
  else
    body += "  const uint lane_raw = r * N + (N >> 3) + 16u * N - 16u;\n"
      "  const uint lane_bit = lane_raw >= 16u * N ? lane_raw - 16u * N : lane_raw;\n";
  body += "  const uint o0 = lane_bit >> 5;\n"
    "  const uint o1 = o0 + 1u == W ? 0u : o0 + 1u;\n"
    "  const uint o2 = o1 + 1u == W ? 0u : o1 + 1u;\n";
  if (n > 48)
    body += "  const uint o3 = o2 + 1u == W ? 0u : o2 + 1u;\n";
  body += "  const uint sh = lane_bit & 31u;\n"
    "  // One tile per iteration.\n"
    "  for (uint tc = stream; tc < tiles_per_row; tc += STREAMS) {\n"
    "    device const uint* t = row_code + tc * W;\n";
  if (n > 48)
    body += "    const uint w0 = t[o0], w1 = t[o1], w2 = t[o2], w3 = t[o3];\n"
      "    const uint f1 = tmcq_funnel(w1, w2, sh);\n"
      "    const ulong lo = ((ulong)f1 << 32) | tmcq_funnel(w0, w1, sh);\n"
      "    const ulong hi = ((ulong)tmcq_funnel(w2, w3, sh) << 32) | f1;\n";
  else
    body += "    const uint w0 = t[o0], w1 = t[o1], w2 = t[o2];\n"
      "    const ulong lo = ((ulong)tmcq_funnel(w1, w2, sh) << 32) | tmcq_funnel(w0, w1, sh);\n"
      "    const ulong hi = 0;\n";
  body += "    float4 qa[4];\n"
    "    ftq_decode_lane<N>(lo, hi, qa, (threadgroup const half2*)tmcq_tg);\n";
  body += accumulate;
  body += "  }\n"
    "  }\n";
  return body;
}

// ccv_nnc_mfa_ftq_w1_rate_source over two weights of the same shape and rate at once (the segmented SwiGLU's gate and
// up): the same lane setup, then for every tile the windows of `codes[0]` decoded into `q[0]`[0 .. 3] and those of
// `codes[1]` into `q[1]`[0 .. 3], with the same words, funnels and lookups as the one-weight loop, so each decodes
// exactly as there. `accumulate` reads tile column tc and both weights.
static inline std::string ccv_nnc_mfa_ftq_w1_rate_pair_source(const uint32_t n, const std::string codes[2], const std::string q[2], const std::string& accumulate)
{
  std::string body = "  {\n  constexpr uint N = " + std::to_string(n) + "u;\n  constexpr uint W = N / 2u;\n";
  if (n >= 16)
    body += "  const uint lane_bit = (r == 0u ? 16u * N : r * N) + (N >> 3) - 16u;\n";
  else
    body += "  const uint lane_raw = r * N + (N >> 3) + 16u * N - 16u;\n"
      "  const uint lane_bit = lane_raw >= 16u * N ? lane_raw - 16u * N : lane_raw;\n";
  body += "  const uint o0 = lane_bit >> 5;\n"
    "  const uint o1 = o0 + 1u == W ? 0u : o0 + 1u;\n"
    "  const uint o2 = o1 + 1u == W ? 0u : o1 + 1u;\n";
  if (n > 48)
    body += "  const uint o3 = o2 + 1u == W ? 0u : o2 + 1u;\n";
  body += "  const uint sh = lane_bit & 31u;\n"
    "  // One tile per iteration.\n"
    "  for (uint tc = stream; tc < tiles_per_row; tc += STREAMS) {\n";
  for (int i = 0; i < 2; i++) {
    body += "    float4 " + q[i] + "[4];\n    {\n"
      "    device const uint* t = " + codes[i] + " + tc * W;\n";
    if (n > 48)
      body += "    const uint w0 = t[o0], w1 = t[o1], w2 = t[o2], w3 = t[o3];\n"
        "    const uint f1 = tmcq_funnel(w1, w2, sh);\n"
        "    const ulong lo = ((ulong)f1 << 32) | tmcq_funnel(w0, w1, sh);\n"
        "    const ulong hi = ((ulong)tmcq_funnel(w2, w3, sh) << 32) | f1;\n";
    else
      body += "    const uint w0 = t[o0], w1 = t[o1], w2 = t[o2];\n"
        "    const ulong lo = ((ulong)tmcq_funnel(w1, w2, sh) << 32) | tmcq_funnel(w0, w1, sh);\n"
        "    const ulong hi = 0;\n";
    body += "    ftq_decode_lane<N>(lo, hi, " + q[i] + ", (threadgroup const half2*)tmcq_tg);\n    }\n";
  }
  body += accumulate;
  body += "  }\n"
    "  }\n";
  return body;
}


// External codebooks (ccv_nnc_codebook.h). In the format of an MFA params struct, bits 16 - 23 of a TMCQ / FTQ format
// hold log2 of the entry count of an external codebook (9, 10 or 11), 0 for the built-in table: the dispatch puts
// it there in place of the tensor's slot, so the pipeline key carries the table size and the table itself is a
// buffer. With 0 every kernel is generated exactly as before, with the compile-time constant table.
static inline uint32_t tmcq_codebook_log2(const uint32_t format)
{
  return (format >> 16) & 0xffu;
}

// Bit 24 of an FTQ format: the external codebook is quadrant folded (CCV_NNC_CODEBOOK_FOLD, log2 9): 512 base pairs,
// h = state * 0x83DCD12D takes base pair (h >> 16) & 511 with the first value negated by bit 15 of h and the second
// by bit 31. Folded and plain tables are different pipelines.
#define CCV_NNC_MFA_CODEBOOK_FOLD (1u << 24)

static inline bool tmcq_codebook_fold(const uint32_t format)
{
  return (format & CCV_NNC_MFA_CODEBOOK_FOLD) != 0;
}

// Variable-rate FTQ int8 decode: blocks of 2 tiles x 16 rows (one per 32 threads) each thread takes in its tile row
// out of `segments` row segments (16 weights of one row inside one tile). 2, except 1 for a folded table on fewer
// than 2^20 segments, whose longer lookups need the threads. The kernel computes the same value.
static inline uint32_t ftq_vr_blocks_per_thread(const uint64_t segments, const bool fold)
{
  return fold && segments < (UINT64_C(1) << 20) ? 1 : 2;
}

// The 16 x 16 tile grid of a rows x cols TMCQ / FTQ weight (rows % 16 == 0, cols % 16 == 0): rows / 16 tile rows of
// cols / 16 tiles, tile-row-major. Every size and offset is a whole number of tiles; with cols % 256 != 0 the row-wise
// "groups of 256 weights per row" count ceil(cols / 256) is not the tile count (cols = 1152 has 4.5 tiles' worth of
// bits per row, cols = 272 has 17 tiles per tile row).
static inline uint64_t tmcq_tile_count(const uint64_t rows, const uint64_t cols)
{
  // The CPU codec's count (ccv_nnc_ftq_schedule.inc), so sizes and offsets agree with ccv_nnc_tensor_data_size.
  return ccv_nnc_8i_rowwise_x_tile_count(rows, cols);
}

// Bytes of the tiles of `rows` rows (rows % 16 == 0) at tile_bits = 16 n bits per tile (n even, a whole number of
// 32-bit words per tile).
static inline uint64_t tmcq_tile_rows_bytes(const uint64_t rows, const uint64_t cols, const uint32_t tile_bits)
{
  return tmcq_tile_count(rows, cols) * tile_bits / 8;
}

// Variable-rate FTQ int8 decode threads per tile row of tiles_per_row tiles at blocks_per_thread = T: 32 threads per
// G = ceil(floor(tiles_per_row / 2) / T) strided blocks of 2 tiles x 16 rows, plus 16 (one per row) for the last tile
// of an odd tiles_per_row. With tiles_per_row % (2 T) == 0 this is 16 tiles_per_row / T, the original mapping
// (ftq_vr_generic_blocks false); the descriptor's dispatch count and the kernel use this same value.
static inline uint32_t ftq_vr_threads_per_tile_row(const uint32_t tiles_per_row, const uint32_t blocks_per_thread)
{
  const uint32_t pairs = tiles_per_row / 2;
  return 32 * ((pairs + blocks_per_thread - 1) / blocks_per_thread) + 16 * (tiles_per_row & 1);
}

static inline bool ftq_vr_generic_blocks(const uint32_t tiles_per_row, const uint32_t blocks_per_thread)
{
  return (tiles_per_row % (2 * blocks_per_thread)) != 0;
}

// The format bits an MFA trellis kernel accepts: the codec, the codebook's log2 (bits 16 - 23) and, for FTQ, the fold
// bit with log2 9.
static inline bool tmcq_codebook_format_ok(const uint32_t format, const bool ftq)
{
  const uint32_t log2 = tmcq_codebook_log2(format);
  if (log2 != 0 && (log2 < 9 || log2 > 11))
    return false;
  if (tmcq_codebook_fold(format))
    return ftq && log2 == 9 && (format & ~(0xffu | 0xff0000u | CCV_NNC_MFA_CODEBOOK_FOLD)) == 0;
  return (format & ~(0xffu | 0xff0000u)) == 0;
}

// MSL helpers for an external codebook of 1 << log2 pairs: the index shift and entry count, a copy of the table into
// threadgroup memory, and accessors that read that copy like the built-in tables. The buffer holds the pairs already
// converted to half2 (ccv_nnc_mfa_codebook_buffer; int8 values are exact in half; the int8 pairs that follow are for
// the FTQ int8 decode), so the copy is plain 16-byte moves
// and the GEMV lookups are the same float2(table[k]) as with the built-in half2 copy. Storage and lookups go through
// these helpers only, so a different threadgroup layout changes this function alone.
static inline std::string tmcq_codebook_source(const uint32_t log2)
{
  std::string s;
  s += "constant uint tmcq_codebook_shift = " + std::to_string(32 - log2) + "u;\n";
  s += "#define TMCQ_CODEBOOK_ENTRIES " + std::to_string(1u << log2) + "u\n";
  s += R"(
// Lane `index` of `count` threads copies the table, four pairs per 16-byte load (the buffer starts at offset 0). The
// caller adds the barrier.
inline void tmcq_codebook_load(threadgroup half2* table, device const uint4* codebook, uint index, uint count)
{
  for (uint i = index; i < TMCQ_CODEBOOK_ENTRIES / 4u; i += count)
    ((threadgroup uint4*)table)[i] = codebook[i];
}

// The copy read as char2 (tmcq_decode_row16) and as ushort pairs, q8[2 k] in the low byte (the
// tmcq_w3 helpers).
struct tmcq_codebook_char2
{
  threadgroup const half2* t;
  char2 operator[](uint i) const { return char2(t[i]); }
};

struct tmcq_codebook_ushort
{
  threadgroup const half2* t;
  ushort operator[](uint i) const { return as_type<ushort>(char2(t[i])); }
};
)";
  return s;
}

// Replaces every `from` in `text` with `to`; returns the number of replacements.
static inline size_t tmcq_codebook_replace(std::string& text, const std::string& from, const std::string& to)
{
  size_t count = 0;
  size_t at = 0;
  while ((at = text.find(from, at)) != std::string::npos) {
    text.replace(at, from.size(), to);
    at += to.size();
    count++;
  }
  return count;
}

// Rewrites a generated shader for an external codebook: the helpers go after the first `using namespace metal;`,
// every pair index takes the codebook's shift instead of 23.
static inline void tmcq_codebook_rewrite(std::string& shader, const uint32_t log2)
{
  const std::string anchor = "using namespace metal;\n";
  const size_t at = shader.find(anchor);
  CCV_NNC_MFA_PRECONDITION(at != std::string::npos);
  shader.insert(at + anchor.size(), tmcq_codebook_source(log2));
  CCV_NNC_MFA_PRECONDITION(tmcq_codebook_replace(shader, "0x83DCD12Du) >> 23", "0x83DCD12Du) >> tmcq_codebook_shift") > 0);
}

#endif
