#ifndef GUARD_ccv_nnc_mfa_codebook_hpp
#define GUARD_ccv_nnc_mfa_codebook_hpp

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * The GPU copy of a pair codebook (ccv_nnc_codebook.h): first converted to half2, 4 * entries bytes, pair k at half2 k
 * = ((half)q8[2 k], (half)q8[2 k + 1]) (int8 values are exact in half), which kernels copy into threadgroup half2 with
 * 16-byte moves; then the int8 pairs as they are, 2 * entries bytes at offset 4 * entries, which the FTQ int8
 * decode reads as constant char2. One buffer per context and slot, made on first use and kept for the life of the
 * process. Slot 0 is the built-in table, compiled into the kernels, which do not bind it.
 * @param context The MFA context.
 * @param slot The codebook slot.
 * @param entries The entry count of the table (512, 1024 or 2048), if not NULL.
 * @return The buffer, or NULL if the slot is not registered.
 */
mtl_buffer_t* ccv_nnc_mfa_codebook_buffer(ccv_nnc_mfa_context_t* context, int slot, int* entries);

/**
 * The MFA format and codebook buffer of a trellis weight (TMCQ / FTQ) that names codebook slot `slot` (bits 16 - 23
 * of its reserved field, see ccv_nnc_codebook.h). For slot 0 (the built-in table) *format stays as is and *codebook is
 * NULL. Otherwise *format gets log2 of the table's entry count in bits 16 - 23, and bit 24 for a quadrant-folded table
 * (accepted only when allow_fold, i.e. for FTQ), and *codebook the table's buffer (ccv_nnc_mfa_codebook_buffer).
 * @return 1, or 0 if the slot is not registered or is folded without allow_fold.
 */
int ccv_nnc_mfa_codebook_format(ccv_nnc_mfa_context_t* context, int slot, int allow_fold, uint32_t* format, mtl_buffer_t** codebook);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif
