/* Copyright (c) V-Nova International Limited 2026. All rights reserved.
 * This software is licensed under the BSD-3-Clause-Clear License by V-Nova Limited.
 * No patent licenses are granted under this license. For enquiries about patent licenses,
 * please contact legal@v-nova.com.
 * The LCEVCdec software is a stand-alone project and is NOT A CONTRIBUTION to any other project.
 * If the software is incorporated into another project, THE TERMS OF THE BSD-3-CLAUSE-CLEAR LICENSE
 * AND THE ADDITIONAL LICENSING INFORMATION CONTAINED IN THIS FILE MUST BE MAINTAINED, AND THE
 * SOFTWARE DOES NOT AND MUST NOT ADOPT THE LICENSE OF THE INCORPORATING PROJECT. However, the
 * software may be incorporated into a project under a compatible license provided the requirements
 * of the BSD-3-Clause-Clear license are respected, and V-Nova Limited remains
 * licensor of the software ONLY UNDER the BSD-3-Clause-Clear license (not the compatible license).
 * ANY ONWARD DISTRIBUTION, WHETHER STAND-ALONE OR AS PART OF ANY OTHER PROJECT, REMAINS SUBJECT TO
 * THE EXCLUSION OF PATENT LICENSES PROVISION OF THE BSD-3-CLAUSE-CLEAR LICENSE. */

//
// SSE NV12 upscaler kernel (8-wide)
//
// This header is included multiple times with different KERNEL_VARIANT definitions
// to generate start, body, and end variants of the 8-wide NV12 filter loop.
//

#if !defined(KERNEL_VARIANT)
#error "KERNEL_VARIANT must be defined before including this header"
#endif

#if IN_FIXED_POINT != FP_NV12 || OUT_FIXED_POINT != FP_NV12
#error "sse_kernel_nv12.h is for FP_NV12 -> FP_NV12 only"
#endif

#if !defined(DIMENSION)
#error "DIMENSION must be defined before including this header"
#endif

#if DIMENSION != DIM_1D && DIMENSION != DIM_2D
#error "DIMENSION must be DIM_1D or DIM_2D"
#endif

{
#if KERNEL_VARIANT == 1
    // Start: clamp x-2 and x-1 to x=0 for each channel (UV interleaved).
    const __m128i src0a =
        U8ToS16(_mm_shuffle_epi8(_mm_loadl_epi64((const __m128i*)srcRow0), kStartShuffle));
    const __m128i src1a =
        U8ToS16(_mm_shuffle_epi8(_mm_loadl_epi64((const __m128i*)srcRow1), kStartShuffle));
#if DIMENSION == DIM_2D
    const __m128i src2a =
        U8ToS16(_mm_shuffle_epi8(_mm_loadl_epi64((const __m128i*)srcRow2), kStartShuffle));
    const __m128i src3a =
        U8ToS16(_mm_shuffle_epi8(_mm_loadl_epi64((const __m128i*)srcRow3), kStartShuffle));
    const __m128i src4a =
        U8ToS16(_mm_shuffle_epi8(_mm_loadl_epi64((const __m128i*)srcRow4), kStartShuffle));
#endif

    const __m128i src0b = U8ToS16(_mm_loadl_epi64((const __m128i*)(srcRow0 + 4)));
    const __m128i src1b = U8ToS16(_mm_loadl_epi64((const __m128i*)(srcRow1 + 4)));
#if DIMENSION == DIM_2D
    const __m128i src2b = U8ToS16(_mm_loadl_epi64((const __m128i*)(srcRow2 + 4)));
    const __m128i src3b = U8ToS16(_mm_loadl_epi64((const __m128i*)(srcRow3 + 4)));
    const __m128i src4b = U8ToS16(_mm_loadl_epi64((const __m128i*)(srcRow4 + 4)));
#endif
    const int32_t dstX = 0;
#if APPLY_PA == 1 && DIMENSION == DIM_2D
    const __m128i base_raw = _mm_loadl_epi64((const __m128i*)srcRow2);
    const __m128i base_s16 = U8ToS16(base_raw);
    const __m128i base_u = _mm_shuffle_epi8(base_s16, kDeinterleaveU);
    const __m128i base_v = _mm_shuffle_epi8(base_s16, kDeinterleaveV);
    const __m128i base = _mm_unpacklo_epi64(base_u, base_v);
#elif APPLY_PA == 1 // && DIMENSION == DIM_1D
    const __m128i base0_raw = _mm_loadl_epi64((const __m128i*)srcRow0);
    const __m128i base0_s16 = U8ToS16(base0_raw);
    const __m128i base0_u = _mm_shuffle_epi8(base0_s16, kDeinterleaveU);
    const __m128i base0_v = _mm_shuffle_epi8(base0_s16, kDeinterleaveV);
    const __m128i base0 = _mm_unpacklo_epi64(base0_u, base0_v);

    const __m128i base1_raw = _mm_loadl_epi64((const __m128i*)srcRow1);
    const __m128i base1_s16 = U8ToS16(base1_raw);
    const __m128i base1_u = _mm_shuffle_epi8(base1_s16, kDeinterleaveU);
    const __m128i base1_v = _mm_shuffle_epi8(base1_s16, kDeinterleaveV);
    const __m128i base1 = _mm_unpacklo_epi64(base1_u, base1_v);
#endif
#elif KERNEL_VARIANT == 2
    // End: clamp x+2 and x+3 to the last pixel for each channel.
    const __m128i src0a = U8ToS16(_mm_loadl_epi64((const __m128i*)(srcRow0 + srcWidth - 12)));
    const __m128i src1a = U8ToS16(_mm_loadl_epi64((const __m128i*)(srcRow1 + srcWidth - 12)));
#if DIMENSION == DIM_2D
    const __m128i src2a = U8ToS16(_mm_loadl_epi64((const __m128i*)(srcRow2 + srcWidth - 12)));
    const __m128i src3a = U8ToS16(_mm_loadl_epi64((const __m128i*)(srcRow3 + srcWidth - 12)));
    const __m128i src4a = U8ToS16(_mm_loadl_epi64((const __m128i*)(srcRow4 + srcWidth - 12)));
#endif

    const __m128i src0b =
        U8ToS16(_mm_shuffle_epi8(_mm_loadl_epi64((const __m128i*)(srcRow0 + srcWidth - 8)), kEndShuffle));
    const __m128i src1b =
        U8ToS16(_mm_shuffle_epi8(_mm_loadl_epi64((const __m128i*)(srcRow1 + srcWidth - 8)), kEndShuffle));
#if DIMENSION == DIM_2D
    const __m128i src2b =
        U8ToS16(_mm_shuffle_epi8(_mm_loadl_epi64((const __m128i*)(srcRow2 + srcWidth - 8)), kEndShuffle));
    const __m128i src3b =
        U8ToS16(_mm_shuffle_epi8(_mm_loadl_epi64((const __m128i*)(srcRow3 + srcWidth - 8)), kEndShuffle));
    const __m128i src4b =
        U8ToS16(_mm_shuffle_epi8(_mm_loadl_epi64((const __m128i*)(srcRow4 + srcWidth - 8)), kEndShuffle));
#endif
    const int32_t dstX = ((int32_t)srcWidth * 2) - 16;
#if APPLY_PA == 1 && DIMENSION == DIM_2D
    const __m128i base_raw = _mm_loadl_epi64((const __m128i*)(srcRow2 + srcWidth - 8));
    const __m128i base_s16 = U8ToS16(base_raw);
    const __m128i base_u = _mm_shuffle_epi8(base_s16, kDeinterleaveU);
    const __m128i base_v = _mm_shuffle_epi8(base_s16, kDeinterleaveV);
    const __m128i base = _mm_unpacklo_epi64(base_u, base_v);
#elif APPLY_PA == 1 // && DIMENSION == DIM_1D
    const __m128i base0_raw = _mm_loadl_epi64((const __m128i*)(srcRow0 + srcWidth - 8));
    const __m128i base0_s16 = U8ToS16(base0_raw);
    const __m128i base0_u = _mm_shuffle_epi8(base0_s16, kDeinterleaveU);
    const __m128i base0_v = _mm_shuffle_epi8(base0_s16, kDeinterleaveV);
    const __m128i base0 = _mm_unpacklo_epi64(base0_u, base0_v);

    const __m128i base1_raw = _mm_loadl_epi64((const __m128i*)(srcRow1 + srcWidth - 8));
    const __m128i base1_s16 = U8ToS16(base1_raw);
    const __m128i base1_u = _mm_shuffle_epi8(base1_s16, kDeinterleaveU);
    const __m128i base1_v = _mm_shuffle_epi8(base1_s16, kDeinterleaveV);
    const __m128i base1 = _mm_unpacklo_epi64(base1_u, base1_v);
#endif
#else
    // Body: normal load from srcX-4 and srcX+4.
    const __m128i src0a = U8ToS16(_mm_loadl_epi64((const __m128i*)(srcRow0 + srcX - 4)));
    const __m128i src1a = U8ToS16(_mm_loadl_epi64((const __m128i*)(srcRow1 + srcX - 4)));
#if DIMENSION == DIM_2D
    const __m128i src2a = U8ToS16(_mm_loadl_epi64((const __m128i*)(srcRow2 + srcX - 4)));
    const __m128i src3a = U8ToS16(_mm_loadl_epi64((const __m128i*)(srcRow3 + srcX - 4)));
    const __m128i src4a = U8ToS16(_mm_loadl_epi64((const __m128i*)(srcRow4 + srcX - 4)));
#endif

    const __m128i src0b = U8ToS16(_mm_loadl_epi64((const __m128i*)(srcRow0 + srcX + 4)));
    const __m128i src1b = U8ToS16(_mm_loadl_epi64((const __m128i*)(srcRow1 + srcX + 4)));
#if DIMENSION == DIM_2D
    const __m128i src2b = U8ToS16(_mm_loadl_epi64((const __m128i*)(srcRow2 + srcX + 4)));
    const __m128i src3b = U8ToS16(_mm_loadl_epi64((const __m128i*)(srcRow3 + srcX + 4)));
    const __m128i src4b = U8ToS16(_mm_loadl_epi64((const __m128i*)(srcRow4 + srcX + 4)));
#endif
    const int32_t dstX = srcX * 2;
#if APPLY_PA == 1 && DIMENSION == DIM_2D
    const __m128i base_raw = _mm_loadl_epi64((const __m128i*)(srcRow2 + srcX));
    const __m128i base_s16 = U8ToS16(base_raw);
    const __m128i base_u = _mm_shuffle_epi8(base_s16, kDeinterleaveU);
    const __m128i base_v = _mm_shuffle_epi8(base_s16, kDeinterleaveV);
    const __m128i base = _mm_unpacklo_epi64(base_u, base_v);
#elif APPLY_PA == 1 // && DIMENSION == DIM_1D
    const __m128i base0_raw = _mm_loadl_epi64((const __m128i*)(srcRow0 + srcX));
    const __m128i base0_s16 = U8ToS16(base0_raw);
    const __m128i base0_u = _mm_shuffle_epi8(base0_s16, kDeinterleaveU);
    const __m128i base0_v = _mm_shuffle_epi8(base0_s16, kDeinterleaveV);
    const __m128i base0 = _mm_unpacklo_epi64(base0_u, base0_v);

    const __m128i base1_raw = _mm_loadl_epi64((const __m128i*)(srcRow1 + srcX));
    const __m128i base1_s16 = U8ToS16(base1_raw);
    const __m128i base1_u = _mm_shuffle_epi8(base1_s16, kDeinterleaveU);
    const __m128i base1_v = _mm_shuffle_epi8(base1_s16, kDeinterleaveV);
    const __m128i base1 = _mm_unpacklo_epi64(base1_u, base1_v);
#endif
#endif

    // === Row preparation ===
#if DIMENSION == DIM_1D
    // 1D mode: skip vertical convolution, feed loaded rows directly into horizontal stage.
    __m128i v0a = src0a;
    __m128i v0b = src0b;
    __m128i v1a = src1a;
    __m128i v1b = src1b;
#else // 2D
    // === Vertical convolution for row0 src0-src3 -> v0a, v0b ===
    const __m128i src01aLo = _mm_unpacklo_epi16(src0a, src1a);
    const __m128i src01aHi = _mm_unpackhi_epi16(src0a, src1a);
    const __m128i src23aLo = _mm_unpacklo_epi16(src2a, src3a);
    const __m128i src23aHi = _mm_unpackhi_epi16(src2a, src3a);

    const __m128i src12aLo = _mm_unpacklo_epi16(src1a, src2a);
    const __m128i src12aHi = _mm_unpackhi_epi16(src1a, src2a);
    const __m128i src34aLo = _mm_unpacklo_epi16(src3a, src4a);
    const __m128i src34aHi = _mm_unpackhi_epi16(src3a, src4a);

    const __m128i v0aLo = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(_mm_madd_epi16(src01aLo, k32), _mm_madd_epi16(src23aLo, k10)), kRound),
        14);
    const __m128i v0aHi = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(_mm_madd_epi16(src01aHi, k32), _mm_madd_epi16(src23aHi, k10)), kRound),
        14);

    const __m128i v1aLo = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(_mm_madd_epi16(src12aLo, k01), _mm_madd_epi16(src34aLo, k23)), kRound),
        14);
    const __m128i v1aHi = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(_mm_madd_epi16(src12aHi, k01), _mm_madd_epi16(src34aHi, k23)), kRound),
        14);

    __m128i v0a = _mm_packs_epi32(v0aLo, v0aHi);
    __m128i v1a = _mm_packs_epi32(v1aLo, v1aHi);
    v0a = _mm_max_epi16(_mm_min_epi16(v0a, kMax16), kMin16);
    v1a = _mm_max_epi16(_mm_min_epi16(v1a, kMax16), kMin16);

    // === Vertical convolution for row1 src1-src4 -> v1a, v1b ===
    const __m128i src01bLo = _mm_unpacklo_epi16(src0b, src1b);
    const __m128i src01bHi = _mm_unpackhi_epi16(src0b, src1b);
    const __m128i src23bLo = _mm_unpacklo_epi16(src2b, src3b);
    const __m128i src23bHi = _mm_unpackhi_epi16(src2b, src3b);

    const __m128i src12bLo = _mm_unpacklo_epi16(src1b, src2b);
    const __m128i src12bHi = _mm_unpackhi_epi16(src1b, src2b);
    const __m128i src34bLo = _mm_unpacklo_epi16(src3b, src4b);
    const __m128i src34bHi = _mm_unpackhi_epi16(src3b, src4b);

    const __m128i v0bLo = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(_mm_madd_epi16(src01bLo, k32), _mm_madd_epi16(src23bLo, k10)), kRound),
        14);
    const __m128i v0bHi = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(_mm_madd_epi16(src01bHi, k32), _mm_madd_epi16(src23bHi, k10)), kRound),
        14);

    const __m128i v1bLo = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(_mm_madd_epi16(src12bLo, k01), _mm_madd_epi16(src34bLo, k23)), kRound),
        14);
    const __m128i v1bHi = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(_mm_madd_epi16(src12bHi, k01), _mm_madd_epi16(src34bHi, k23)), kRound),
        14);

    __m128i v0b = _mm_packs_epi32(v0bLo, v0bHi);
    __m128i v1b = _mm_packs_epi32(v1bLo, v1bHi);
    v0b = _mm_max_epi16(_mm_min_epi16(v0b, kMax16), kMin16);
    v1b = _mm_max_epi16(_mm_min_epi16(v1b, kMax16), kMin16);
#endif

    {
        // Deinterleave U,V,U,V,... lanes so each register holds a single channel.
        const __m128i v0a_u = _mm_shuffle_epi8(v0a, kDeinterleaveU);
        const __m128i v0a_v = _mm_shuffle_epi8(v0a, kDeinterleaveV);
        const __m128i v0b_u = _mm_shuffle_epi8(v0b, kDeinterleaveU);
        const __m128i v0b_v = _mm_shuffle_epi8(v0b, kDeinterleaveV);
        v0a = _mm_unpacklo_epi64(v0a_u, v0b_u);
        v0b = _mm_unpacklo_epi64(v0a_v, v0b_v);

        const __m128i v1a_u = _mm_shuffle_epi8(v1a, kDeinterleaveU);
        const __m128i v1a_v = _mm_shuffle_epi8(v1a, kDeinterleaveV);
        const __m128i v1b_u = _mm_shuffle_epi8(v1b, kDeinterleaveU);
        const __m128i v1b_v = _mm_shuffle_epi8(v1b, kDeinterleaveV);
        v1a = _mm_unpacklo_epi64(v1a_u, v1b_u);
        v1b = _mm_unpacklo_epi64(v1a_v, v1b_v);
    }

    // === Horizontal convolution for row0 ===
    const __m128i v0a_s1 = _mm_srli_si128(v0a, 2);
    const __m128i v0a_s2 = _mm_srli_si128(v0a, 4);
    const __m128i v0a_s3 = _mm_srli_si128(v0a, 6);
    const __m128i v0a_s4 = _mm_srli_si128(v0a, 8);

    const __m128i w0a_01 = _mm_unpacklo_epi64(v0a, v0a_s1);
    const __m128i w0a_23 = _mm_unpacklo_epi64(v0a_s2, v0a_s3);
    const __m128i w0a_12 = _mm_alignr_epi8(v0a_s2, w0a_01, 8);
    const __m128i w0a_34 = _mm_alignr_epi8(v0a_s4, w0a_23, 8);

    const __m128i row0aOdd = _mm_srai_epi32(
        _mm_add_epi32(_mm_hadd_epi32(_mm_madd_epi16(w0a_01, kHOdd), _mm_madd_epi16(w0a_23, kHOdd)), kRound),
        14);
    const __m128i row0aEven = _mm_srai_epi32(
        _mm_add_epi32(_mm_hadd_epi32(_mm_madd_epi16(w0a_12, kHEven), _mm_madd_epi16(w0a_34, kHEven)), kRound),
        14);

    const __m128i v0b_s1 = _mm_srli_si128(v0b, 2);
    const __m128i v0b_s2 = _mm_srli_si128(v0b, 4);
    const __m128i v0b_s3 = _mm_srli_si128(v0b, 6);
    const __m128i v0b_s4 = _mm_srli_si128(v0b, 8);

    const __m128i w0b_01 = _mm_unpacklo_epi64(v0b, v0b_s1);
    const __m128i w0b_23 = _mm_unpacklo_epi64(v0b_s2, v0b_s3);
    const __m128i w0b_12 = _mm_alignr_epi8(v0b_s2, w0b_01, 8);
    const __m128i w0b_34 = _mm_alignr_epi8(v0b_s4, w0b_23, 8);

    const __m128i row0bOdd = _mm_srai_epi32(
        _mm_add_epi32(_mm_hadd_epi32(_mm_madd_epi16(w0b_01, kHOdd), _mm_madd_epi16(w0b_23, kHOdd)), kRound),
        14);
    const __m128i row0bEven = _mm_srai_epi32(
        _mm_add_epi32(_mm_hadd_epi32(_mm_madd_epi16(w0b_12, kHEven), _mm_madd_epi16(w0b_34, kHEven)), kRound),
        14);

    // === Horizontal convolution for row1 ===
    const __m128i v1a_s1 = _mm_srli_si128(v1a, 2);
    const __m128i v1a_s2 = _mm_srli_si128(v1a, 4);
    const __m128i v1a_s3 = _mm_srli_si128(v1a, 6);
    const __m128i v1a_s4 = _mm_srli_si128(v1a, 8);

    const __m128i w1a_01 = _mm_unpacklo_epi64(v1a, v1a_s1);
    const __m128i w1a_23 = _mm_unpacklo_epi64(v1a_s2, v1a_s3);
    const __m128i w1a_12 = _mm_alignr_epi8(v1a_s2, w1a_01, 8);
    const __m128i w1a_34 = _mm_alignr_epi8(v1a_s4, w1a_23, 8);

    const __m128i row1aOdd = _mm_srai_epi32(
        _mm_add_epi32(_mm_hadd_epi32(_mm_madd_epi16(w1a_01, kHOdd), _mm_madd_epi16(w1a_23, kHOdd)), kRound),
        14);
    const __m128i row1aEven = _mm_srai_epi32(
        _mm_add_epi32(_mm_hadd_epi32(_mm_madd_epi16(w1a_12, kHEven), _mm_madd_epi16(w1a_34, kHEven)), kRound),
        14);

    const __m128i v1b_s1 = _mm_srli_si128(v1b, 2);
    const __m128i v1b_s2 = _mm_srli_si128(v1b, 4);
    const __m128i v1b_s3 = _mm_srli_si128(v1b, 6);
    const __m128i v1b_s4 = _mm_srli_si128(v1b, 8);

    const __m128i w1b_01 = _mm_unpacklo_epi64(v1b, v1b_s1);
    const __m128i w1b_23 = _mm_unpacklo_epi64(v1b_s2, v1b_s3);
    const __m128i w1b_12 = _mm_alignr_epi8(v1b_s2, w1b_01, 8);
    const __m128i w1b_34 = _mm_alignr_epi8(v1b_s4, w1b_23, 8);

    const __m128i row1bOdd = _mm_srai_epi32(
        _mm_add_epi32(_mm_hadd_epi32(_mm_madd_epi16(w1b_01, kHOdd), _mm_madd_epi16(w1b_23, kHOdd)), kRound),
        14);
    const __m128i row1bEven = _mm_srai_epi32(
        _mm_add_epi32(_mm_hadd_epi32(_mm_madd_epi16(w1b_12, kHEven), _mm_madd_epi16(w1b_34, kHEven)), kRound),
        14);

    // === Pack and interleave results ===
    __m128i row0a_16 = _mm_packs_epi32(_mm_unpacklo_epi32(row0aOdd, row0aEven),
                                       _mm_unpackhi_epi32(row0aOdd, row0aEven));
    __m128i row0b_16 = _mm_packs_epi32(_mm_unpacklo_epi32(row0bOdd, row0bEven),
                                       _mm_unpackhi_epi32(row0bOdd, row0bEven));
    __m128i row1a_16 = _mm_packs_epi32(_mm_unpacklo_epi32(row1aOdd, row1aEven),
                                       _mm_unpackhi_epi32(row1aOdd, row1aEven));
    __m128i row1b_16 = _mm_packs_epi32(_mm_unpacklo_epi32(row1bOdd, row1bEven),
                                       _mm_unpackhi_epi32(row1bOdd, row1bEven));

    // === Limit to S15 range after horizontal convolution ===
    row0a_16 = _mm_max_epi16(_mm_min_epi16(row0a_16, kMax16), kMin16);
    row0b_16 = _mm_max_epi16(_mm_min_epi16(row0b_16, kMax16), kMin16);
    row1a_16 = _mm_max_epi16(_mm_min_epi16(row1a_16, kMax16), kMin16);
    row1b_16 = _mm_max_epi16(_mm_min_epi16(row1b_16, kMax16), kMin16);

    // === Apply PA ===
#if APPLY_PA == 1 && DIMENSION == DIM_2D
    const __m128i kPAOnes = _mm_set1_epi16(1);
    const __m128i kPATwos = _mm_set1_epi32(2);

    const __m128i sumU = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(_mm_madd_epi16(row0a_16, kPAOnes), _mm_madd_epi16(row1a_16, kPAOnes)),
                      kPATwos),
        2);
    const __m128i sumV = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(_mm_madd_epi16(row0b_16, kPAOnes), _mm_madd_epi16(row1b_16, kPAOnes)),
                      kPATwos),
        2);

    const __m128i mean = _mm_packs_epi32(sumU, sumV);
    const __m128i avg = _mm_sub_epi16(base, mean);
    const __m128i avgU = _mm_unpacklo_epi16(avg, avg);
    const __m128i avgV = _mm_unpackhi_epi16(avg, avg);

    row0a_16 = _mm_adds_epi16(row0a_16, avgU);
    row0b_16 = _mm_adds_epi16(row0b_16, avgV);
    row1a_16 = _mm_adds_epi16(row1a_16, avgU);
    row1b_16 = _mm_adds_epi16(row1b_16, avgV);
#elif APPLY_PA == 1 // && DIMENSION == DIM_1D
    const __m128i kPAOnes = _mm_set1_epi16(1);
    const __m128i kPAOne = _mm_set1_epi32(1);

    const __m128i mean0u = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(row0a_16, kPAOnes), kPAOne), 1);
    const __m128i mean0v = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(row0b_16, kPAOnes), kPAOne), 1);
    const __m128i avg0 = _mm_sub_epi16(base0, _mm_packs_epi32(mean0u, mean0v));
    row0a_16 = _mm_adds_epi16(row0a_16, _mm_unpacklo_epi16(avg0, avg0));
    row0b_16 = _mm_adds_epi16(row0b_16, _mm_unpackhi_epi16(avg0, avg0));

    const __m128i mean1u = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(row1a_16, kPAOnes), kPAOne), 1);
    const __m128i mean1v = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(row1b_16, kPAOnes), kPAOne), 1);
    const __m128i avg1 = _mm_sub_epi16(base1, _mm_packs_epi32(mean1u, mean1v));
    row1a_16 = _mm_adds_epi16(row1a_16, _mm_unpacklo_epi16(avg1, avg1));
    row1b_16 = _mm_adds_epi16(row1b_16, _mm_unpackhi_epi16(avg1, avg1));
#endif

    {
        // Re-interleave NV12
        const __m128i row0uv0 = _mm_unpacklo_epi16(row0a_16, row0b_16);
        const __m128i row0uv1 = _mm_unpackhi_epi16(row0a_16, row0b_16);
        row0a_16 = row0uv0;
        row0b_16 = row0uv1;

        const __m128i row1uv0 = _mm_unpacklo_epi16(row1a_16, row1b_16);
        const __m128i row1uv1 = _mm_unpackhi_epi16(row1a_16, row1b_16);
        row1a_16 = row1uv0;
        row1b_16 = row1uv1;
    }

    // === Store results ===
    const __m128i row0a_u16 = S16ToU8(row0a_16, kOffset, kMidpoint);
    const __m128i row0b_u16 = S16ToU8(row0b_16, kOffset, kMidpoint);
    const __m128i row1a_u16 = S16ToU8(row1a_16, kOffset, kMidpoint);
    const __m128i row1b_u16 = S16ToU8(row1b_16, kOffset, kMidpoint);

    const __m128i row0_u8 = _mm_packus_epi16(row0a_u16, row0b_u16);
    const __m128i row1_u8 = _mm_packus_epi16(row1a_u16, row1b_u16);

    _mm_storeu_si128((__m128i*)(dstRow0 + dstX), row0_u8);
    _mm_storeu_si128((__m128i*)(dstRow1 + dstX), row1_u8);
}

#undef KERNEL_VARIANT
