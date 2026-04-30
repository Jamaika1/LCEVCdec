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

#if defined(VN_UPSCALE_LOOP)
#define IN_FIXED_POINT VN_UPSCALE_LOOP_CONFIG_IN_FIXED_POINT(VN_UPSCALE_LOOP)
#define OUT_FIXED_POINT VN_UPSCALE_LOOP_CONFIG_OUT_FIXED_POINT(VN_UPSCALE_LOOP)
#define APPLY_PA VN_UPSCALE_LOOP_CONFIG_APPLY_PA(VN_UPSCALE_LOOP)
#define DITHER VN_UPSCALE_LOOP_CONFIG_DITHER(VN_UPSCALE_LOOP)
#endif

#if !defined(IN_FIXED_POINT) || !defined(OUT_FIXED_POINT) || !defined(APPLY_PA) || !defined(DITHER)
#error "Define VN_UPSCALE_LOOP before including this header"
#endif

#if !defined(DIMENSION_1D)
#define DIMENSION_1D 1
#endif

#if !defined(DIMENSION_2D)
#define DIMENSION_2D 2
#endif

#if !defined(DIMENSION)
#error "DIMENSION must be defined before including this header"
#endif

#if DIMENSION != DIMENSION_1D && DIMENSION != DIMENSION_2D
#error "DIMENSION must be DIMENSION_1D or DIMENSION_2D"
#endif

#if IN_FIXED_POINT != FP_NV12 || OUT_FIXED_POINT != FP_NV12
#error "neon_nv12_loop.h is for FP_NV12 -> FP_NV12 only"
#endif

{
    const int16x8_t kMin16 = vdupq_n_s16(-0x4000); /* see saturateS15 for explanation */
    const int16x8_t kMax16 = vdupq_n_s16(0x4000 - 1);
    const int16x8_t kMidpointS16 = vdupq_n_s16(0x4000);

    const int16x8_t kOffset = vdupq_n_s16(64);
    const int16x8_t kMidpoint = vdupq_n_s16(128);
    const int16x8_t kS8Min = vdupq_n_s16(INT8_MIN);
    const int16x8_t kU8Max = vdupq_n_s16(UINT8_MAX);

    const int16x4_t k0 = vdup_n_s16(params.kernel[0]);
    const int16x4_t k1 = vdup_n_s16(params.kernel[1]);
    const int16x4_t k2 = vdup_n_s16(params.kernel[2]);
    const int16x4_t k3 = vdup_n_s16(params.kernel[3]);

#if DIMENSION == DIMENSION_2D
    const int32_t dstHeight = srcHeight << 1;
    for (int32_t srcY = yStart; srcY <= yEnd - 1; ++srcY) {
#else
    for (int32_t srcY = yStart; srcY <= yEnd - 1; srcY += 2) {
#endif

#if DIMENSION == DIMENSION_2D
        const int32_t srcY0 = clampInt(srcY - 2, 0, srcHeight - 1);
        const int32_t srcY1 = clampInt(srcY - 1, 0, srcHeight - 1);
        const int32_t srcY2 = clampInt(srcY - 0, 0, srcHeight - 1);
        const int32_t srcY3 = clampInt(srcY + 1, 0, srcHeight - 1);
        const int32_t srcY4 = clampInt(srcY + 2, 0, srcHeight - 1);

        const uint8_t* const srcRow0 = getRowU8(src, srcStride, srcY0);
        const uint8_t* const srcRow1 = getRowU8(src, srcStride, srcY1);
        const uint8_t* const srcRow2 = getRowU8(src, srcStride, srcY2);
        const uint8_t* const srcRow3 = getRowU8(src, srcStride, srcY3);
        const uint8_t* const srcRow4 = getRowU8(src, srcStride, srcY4);

        const int32_t dstY0 = clampInt((srcY * 2) + 0, 0, dstHeight - 1);
        const int32_t dstY1 = clampInt((srcY * 2) + 1, 0, dstHeight - 1);
        uint8_t* const dstRow0 = getRowU8(dst, dstStride, dstY0);
        uint8_t* const dstRow1 = getRowU8(dst, dstStride, dstY1);
#else // Run two rows at a time for 1D
        const int32_t srcY0 = srcY;
        const int32_t srcY1 = clampInt(srcY + 1, 0, srcHeight - 1);

        const uint8_t* const srcRow0 = getRowU8(src, srcStride, srcY0);
        const uint8_t* const srcRow1 = getRowU8(src, srcStride, srcY1);

        uint8_t* const dstRow0 = getRowU8(dst, dstStride, srcY0);
        uint8_t* const dstRow1 = getRowU8(dst, dstStride, srcY1);
#endif

#define KERNEL_VARIANT 1
#include "neon_kernel_nv12.h"

        for (int32_t srcX = 8; srcX <= (int32_t)srcWidth - 8; srcX += 8) {
#define KERNEL_VARIANT 0
#include "neon_kernel_nv12.h"
        }

#define KERNEL_VARIANT 2
#include "neon_kernel_nv12.h"
    }
}

#undef DITHER
#undef APPLY_PA
#undef OUT_FIXED_POINT
#undef IN_FIXED_POINT
