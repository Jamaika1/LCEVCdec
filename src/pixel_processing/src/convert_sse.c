/* Copyright (c) V-Nova International Limited 2022-2026. All rights reserved.
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

#include "convert_common.h"
#include "fp_types.h"

#include <LCEVC/build_config.h>
#include <LCEVC/common/limit.h>
#include <LCEVC/common/platform.h>
#include <LCEVC/pipeline/types.h>

#if VN_SDK_FEATURE(SSE)

#include <LCEVC/common/sse.h>

/*------------------------------------------------------------------------------*/

static const uint32_t kStep = 16;

static inline uint32_t simdAlignment(const uint32_t width) { return alignTruncU32(width, kStep); }

/*------------------------------------------------------------------------------*/

/* Copy U8 to S16: (val << 7) - 0x4000 */
static void copyU8_S16_SSE(const LdppConvertArgs* args)
{
    static const int16_t kShift = 7;
    const __m128i offset = _mm_set1_epi16(0x4000);

    VN_BLIT_SIMD_BOILERPLATE(uint8_t, int16_t);

    for (uint32_t y = 0; y < count; y++) {
        const uint8_t* srcPixel = srcRow;
        int16_t* dstPixel0 = dstRow;
        int16_t* dstPixel1 = dstRow + 8;
        uint32_t x = 0;

        for (; x < simdWidth; x += kStep, srcPixel += kStep, dstPixel0 += kStep, dstPixel1 += kStep) {
            /* Load 16-pixels & split */
            __m128i left = _mm_loadu_si128((const __m128i*)srcPixel);
            __m128i right = _mm_srli_si128(left, 8);

            /* Convert to int16_t */
            left = _mm_cvtepu8_epi16(left);
            right = _mm_cvtepu8_epi16(right);

            /* val <<= shift */
            left = _mm_slli_epi16(left, kShift);
            right = _mm_slli_epi16(right, kShift);

            /* val -= sign_offset */
            left = _mm_sub_epi16(left, offset);
            right = _mm_sub_epi16(right, offset);

            /* Store 16-pixels */
            _mm_storeu_si128((__m128i*)dstPixel0, left);
            _mm_storeu_si128((__m128i*)dstPixel1, right);
        }

        for (; x < width; x++, srcPixel++, dstPixel0++) {
            *dstPixel0 = fpU16ToS16(*srcPixel, kShift);
        }

        srcRow += srcStride;
        dstRow += dstStride;
    }
}

/* Copy U16 to S16: (val << shift) - 0x4000 */
static void copyU16_S16_SSE(const LdppConvertArgs* args, const int16_t shift)
{
    const __m128i offset = _mm_set1_epi16(0x4000);

    VN_BLIT_SIMD_BOILERPLATE(uint16_t, int16_t);

    for (uint32_t y = 0; y < count; y++) {
        const uint16_t* srcPixel0 = srcRow;
        const uint16_t* srcPixel1 = srcRow + 8;
        int16_t* dstPixel0 = dstRow;
        int16_t* dstPixel1 = dstRow + 8;
        uint32_t x = 0;

        for (; x < simdWidth; x += kStep, srcPixel0 += kStep, srcPixel1 += kStep,
                              dstPixel0 += kStep, dstPixel1 += kStep) {
            /* Load 16-pixels */
            __m128i left = _mm_loadu_si128((const __m128i*)srcPixel0);
            __m128i right = _mm_loadu_si128((const __m128i*)srcPixel1);

            /* val <<= shift */
            left = _mm_slli_epi16(left, shift);
            right = _mm_slli_epi16(right, shift);

            /* val -= signOffset */
            left = _mm_sub_epi16(left, offset);
            right = _mm_sub_epi16(right, offset);

            /* Store 16-pixels */
            _mm_storeu_si128((__m128i*)dstPixel0, left);
            _mm_storeu_si128((__m128i*)dstPixel1, right);
        }

        for (; x < width; x++, srcPixel0++, dstPixel0++) {
            *dstPixel0 = fpU16ToS16(*srcPixel0, shift);
        }

        srcRow += srcStride;
        dstRow += dstStride;
    }
}

/* Copy S8.7 to U8: ((val + 64) >> 7) + 128 */
static void copyS8_7_U8_SSE(const LdppConvertArgs* args)
{
    const __m128i rounding = _mm_set1_epi16(0x40);
    const __m128i offset = _mm_set1_epi16(0x80);

    VN_BLIT_SIMD_BOILERPLATE(int16_t, uint8_t);

    for (uint32_t y = 0; y < count; y++) {
        const int16_t* srcPixel0 = srcRow;
        const int16_t* srcPixel1 = srcRow + 8;
        uint8_t* dstPixel = dstRow;
        uint32_t x = 0;

        for (; x < simdWidth; x += kStep, srcPixel0 += kStep, srcPixel1 += kStep, dstPixel += kStep) {
            /* Load 16-pixels */
            __m128i left = _mm_loadu_si128((const __m128i*)srcPixel0);
            __m128i right = _mm_loadu_si128((const __m128i*)srcPixel1);

            /* val += rounding */
            left = _mm_adds_epi16(left, rounding);
            right = _mm_adds_epi16(right, rounding);

            /* val >>= shift */
            left = _mm_srai_epi16(left, 7);
            right = _mm_srai_epi16(right, 7);

            /* val += signed_offset */
            left = _mm_add_epi16(left, offset);
            right = _mm_add_epi16(right, offset);

            /* clamp & store */
            _mm_storeu_si128((__m128i*)dstPixel, _mm_packus_epi16(left, right));
        }

        for (; x < width; x++, srcPixel0++, dstPixel++) {
            *dstPixel = fpS8ToU8(*srcPixel0);
        }

        srcRow += srcStride;
        dstRow += dstStride;
    }
}

/* Copy S16 to U16: clamped(0, maxValue, ((((val + rounding) >> shift) + signed_offset)) */
static void copyS16_U16_SSE(const LdppConvertArgs* args, const int16_t shift,
                            const int16_t signOffset, const uint16_t maxValue)
{
    const int16_t roundingValue = (int16_t)(1 << (shift - 1));
    const __m128i rounding = _mm_set1_epi16(roundingValue);
    const __m128i offset = _mm_set1_epi16(signOffset);
    const __m128i minV = _mm_set1_epi16(0);
    const __m128i maxV = _mm_set1_epi16((int16_t)maxValue);

    VN_BLIT_SIMD_BOILERPLATE(int16_t, uint16_t);

    for (uint32_t y = 0; y < count; y++) {
        const int16_t* srcPixel0 = srcRow;
        const int16_t* srcPixel1 = srcRow + 8;
        uint16_t* dstPixel0 = dstRow;
        uint16_t* dstPixel1 = dstRow + 8;
        uint32_t x = 0;

        for (; x < simdWidth; x += kStep, srcPixel0 += kStep, srcPixel1 += kStep,
                              dstPixel0 += kStep, dstPixel1 += kStep) {
            /* Load 16-pixels */
            __m128i left = _mm_loadu_si128((const __m128i*)srcPixel0);
            __m128i right = _mm_loadu_si128((const __m128i*)srcPixel1);

            /* val += rounding */
            left = _mm_adds_epi16(left, rounding);
            right = _mm_adds_epi16(right, rounding);

            /* val >>= shift */
            left = _mm_srai_epi16(left, shift);
            right = _mm_srai_epi16(right, shift);

            /* val += signed_offset */
            left = _mm_add_epi16(left, offset);
            right = _mm_add_epi16(right, offset);

            /* clamp */
            left = _mm_max_epi16(_mm_min_epi16(left, maxV), minV);
            right = _mm_max_epi16(_mm_min_epi16(right, maxV), minV);

            /* Store 16-pixels */
            _mm_storeu_si128((__m128i*)dstPixel0, left);
            _mm_storeu_si128((__m128i*)dstPixel1, right);
        }

        for (; x < width; x++, srcPixel0++, dstPixel0++) {
            *dstPixel0 = fpS16ToU16(*srcPixel0, shift, roundingValue, signOffset, maxValue);
        }

        srcRow += srcStride;
        dstRow += dstStride;
    }
}

static void copyU10_S16_SSE(const LdppConvertArgs* args) { copyU16_S16_SSE(args, 5); }

static void copyU12_S16_SSE(const LdppConvertArgs* args) { copyU16_S16_SSE(args, 3); }

static void copyU14_S16_SSE(const LdppConvertArgs* args) { copyU16_S16_SSE(args, 1); }

static void copyS16_U10_SSE(const LdppConvertArgs* args) { copyS16_U16_SSE(args, 5, 0x200, 1023); }

static void copyS16_U12_SSE(const LdppConvertArgs* args) { copyS16_U16_SSE(args, 3, 0x800, 4095); }

static void copyS16_U14_SSE(const LdppConvertArgs* args)
{
    copyS16_U16_SSE(args, 1, 0x2000, 16383);
}

/*------------------------------------------------------------------------------
 * Tables
 *------------------------------------------------------------------------------*/

/* clang-format off */

static const PlaneConvertFunction kCopyTable[LdpFPCount][LdpFPCount] = {
	/* src/dst   U8                U10               U12               U14               S8.7             S10.5             S12.3             S14.1*/
	/* U8    */ {NULL,             NULL,             NULL,             NULL,             &copyU8_S16_SSE, &copyU8_S16_SSE,  &copyU8_S16_SSE,  &copyU8_S16_SSE},
	/* U10   */ {NULL,             NULL,             NULL,             NULL,             NULL,            &copyU10_S16_SSE, &copyU10_S16_SSE, &copyU10_S16_SSE},
	/* U12   */ {NULL,             NULL,             NULL,             NULL,             NULL,            NULL,             &copyU12_S16_SSE, &copyU12_S16_SSE},
	/* U14   */ {NULL,             NULL,             NULL,             NULL,             NULL,            NULL,             NULL,             &copyU14_S16_SSE},
	/* S8.7  */ {&copyS8_7_U8_SSE, &copyS16_U10_SSE, &copyS16_U12_SSE, &copyS16_U14_SSE, NULL,            NULL,             NULL,             NULL},
	/* S10.5 */ {NULL,             &copyS16_U10_SSE, &copyS16_U12_SSE, &copyS16_U14_SSE, NULL,            NULL,             NULL,             NULL},
	/* S12.3 */ {NULL,             NULL,             &copyS16_U12_SSE, &copyS16_U14_SSE, NULL,            NULL,             NULL,             NULL},
	/* S14.1 */ {NULL,             NULL,             NULL,             &copyS16_U14_SSE, NULL,            NULL,             NULL,             NULL},
};

/* clang-format on */

/*------------------------------------------------------------------------------*/

PlaneConvertFunction planeConvertGetFunctionSSE(LdpFixedPoint srcFP, LdpFixedPoint dstFP, bool isNV12)
{
    if (isNV12) {
        return NULL;
    }
    return kCopyTable[srcFP][dstFP];
}

/*------------------------------------------------------------------------------*/

#else

PlaneConvertFunction planeConvertGetFunctionSSE(LdpFixedPoint srcFP, LdpFixedPoint dstFP, bool isNV12)
{
    VNUnused(srcFP);
    VNUnused(dstFP);
    VNUnused(isNV12);

    return NULL;
}

#endif
