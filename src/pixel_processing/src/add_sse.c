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

#include "add_common.h"
#include "fp_types.h"

#include <LCEVC/build_config.h>
#include <LCEVC/common/limit.h>
#include <LCEVC/pipeline/types.h>
#include <LCEVC/pixel_processing/add.h>

#if VN_SDK_FEATURE(SSE)

#include <assert.h>
#include <LCEVC/common/sse.h>

/*------------------------------------------------------------------------------*/

static const uint32_t kStep = 16;

static inline uint32_t simdAlignment(const uint32_t width) { return alignTruncU32(width, kStep); }

/*------------------------------------------------------------------------------*/

/*! \brief Performs an additive blit of an S16 input onto a U8 destination in SSE */
static void addU8_SSE(const LdppAddArgs* args)
{
    static const int32_t kShift = 7;
    const __m128i fractOffset = _mm_set1_epi16(0x40);
    const __m128i signOffset = _mm_set1_epi16(0x80);

    VN_ADD_SIMD_BOILERPLATE(uint8_t);

    for (uint32_t y = 0; y < count; y++) {
        const int16_t* restrict srcPixel1 = srcRow1;
        const int16_t* restrict srcPixel2 = srcRow2;
        uint8_t* restrict dstPixel = dstRow;
        uint32_t x = 0;

        /* SIMD loop */
        for (; x < simdWidth; x += kStep, dstPixel += kStep, srcPixel1 += kStep, srcPixel2 += kStep) {
            /* Load 16-pixels from each src */
            const __m128i src1left = _mm_loadu_si128((const __m128i*)srcPixel1);
            const __m128i src1right = _mm_loadu_si128((const __m128i*)(srcPixel1 + 8));
            const __m128i src2left = _mm_loadu_si128((const __m128i*)srcPixel2);
            const __m128i src2right = _mm_loadu_si128((const __m128i*)(srcPixel2 + 8));

            /* add sources */
            __m128i dstLeft = _mm_adds_epi16(src1left, src2left);
            __m128i dstRight = _mm_adds_epi16(src1right, src2right);

            /* val += 0x40*/
            dstLeft = _mm_adds_epi16(dstLeft, fractOffset);
            dstRight = _mm_adds_epi16(dstRight, fractOffset);

            /* val >>= 7 */
            dstLeft = _mm_srai_epi16(dstLeft, kShift);
            dstRight = _mm_srai_epi16(dstRight, kShift);

            /* val += 0x80 */
            dstLeft = _mm_add_epi16(dstLeft, signOffset);
            dstRight = _mm_add_epi16(dstRight, signOffset);

            /* Saturated cast back to u8 */
            const __m128i dstResult = _mm_packus_epi16(dstLeft, dstRight);

            /* Store 16-pixels */
            _mm_storeu_si128((__m128i*)dstPixel, dstResult);
        }

        /* Remainder */
        for (; x < width; x++, dstPixel++, srcPixel1++, srcPixel2++) {
            int32_t pel = *srcPixel1 + *srcPixel2;
            *dstPixel = fpS8ToU8(pel);
        }

        srcRow1 += src1Stride;
        srcRow2 += src2Stride;
        dstRow += dstStride;
    }
}

/*! \brief Performs an additive blit of an S16 input onto a U16 destination in SSE */
static void addUN_SSE(const LdppAddArgs* args, int32_t shift, int16_t roundingOffset,
                      int16_t signOffset, int16_t resultMax, LdpFixedPoint unsignedFP)
{
    FixedPointDemotionFunction sToU = fixedPointGetDemotionFunction(unsignedFP);

    const __m128i roundingOffsetV = _mm_set1_epi16(roundingOffset);
    const __m128i signOffsetV = _mm_set1_epi16(signOffset);
    const __m128i minV = _mm_set1_epi16(0);
    const __m128i maxV = _mm_set1_epi16(resultMax);

    VN_ADD_SIMD_BOILERPLATE(uint16_t);

    for (uint32_t y = 0; y < count; y++) {
        const int16_t* srcPixel1 = srcRow1;
        const int16_t* srcPixel2 = srcRow2;
        uint16_t* dstPixel = dstRow;
        uint32_t x = 0;

        /* SIMD loop*/
        for (; x < simdWidth; x += kStep, dstPixel += kStep, srcPixel1 += kStep, srcPixel2 += kStep) {
            /* Load 16-pixels from each src */
            const __m128i src1left = _mm_loadu_si128((const __m128i*)srcPixel1);
            const __m128i src1right = _mm_loadu_si128((const __m128i*)(srcPixel1 + 8));
            const __m128i src2left = _mm_loadu_si128((const __m128i*)srcPixel2);
            const __m128i src2right = _mm_loadu_si128((const __m128i*)(srcPixel2 + 8));

            /* add sources */
            __m128i dst0 = _mm_adds_epi16(src1left, src2left);
            __m128i dst1 = _mm_adds_epi16(src1right, src2right);

            /* val += fract_half_offset */
            dst0 = _mm_adds_epi16(dst0, roundingOffsetV);
            dst1 = _mm_adds_epi16(dst1, roundingOffsetV);

            /* val >>= shift */
            dst0 = _mm_srai_epi16(dst0, shift);
            dst1 = _mm_srai_epi16(dst1, shift);

            /* val += sign_offset */
            dst0 = _mm_add_epi16(dst0, signOffsetV);
            dst1 = _mm_add_epi16(dst1, signOffsetV);

            /* clamp to unsigned range */
            dst0 = _mm_max_epi16(_mm_min_epi16(dst0, maxV), minV);
            dst1 = _mm_max_epi16(_mm_min_epi16(dst1, maxV), minV);

            /* Store 16-pixels */
            _mm_storeu_si128((__m128i*)dstPixel, dst0);
            _mm_storeu_si128((__m128i*)(dstPixel + 8), dst1);
        }

        /* Remainder */
        for (; x < width; x++, dstPixel++, srcPixel1++, srcPixel2++) {
            int32_t pel = *srcPixel1 + *srcPixel2;
            *dstPixel = sToU(pel);
        }

        srcRow1 += src1Stride;
        srcRow2 += src2Stride;
        dstRow += dstStride;
    }
}

static void addU10_SSE(const LdppAddArgs* args) { addUN_SSE(args, 5, 16, 512, 1023, LdpFPU10); }

static void addU12_SSE(const LdppAddArgs* args) { addUN_SSE(args, 3, 4, 2048, 4095, LdpFPU12); }

static void addU14_SSE(const LdppAddArgs* args) { addUN_SSE(args, 1, 1, 8192, 16383, LdpFPU14); }

/*------------------------------------------------------------------------------*/

/* clang-format off */
static const PlaneAddFunction kAddTable[LdpFPCount] = {
	&addU8_SSE,  /* FP_U8 */
	&addU10_SSE, /* FP_U10 */
	&addU12_SSE, /* FP_U12 */
	&addU14_SSE, /* FP_U14 */
    NULL, /* FP_S8_7 */
    NULL, /* FP_S10_5 */
    NULL, /* FP_S12_3 */
    NULL, /* FP_S14_1 */
};
/* clang-format on */

/*------------------------------------------------------------------------------*/

PlaneAddFunction planeAddGetFunctionSSE(LdpFixedPoint dstFP)
{
    assert(!fixedPointIsSigned(dstFP));

    return kAddTable[dstFP];
}

/*------------------------------------------------------------------------------*/

#else

PlaneAddFunction planeAddGetFunctionSSE(LdpFixedPoint dstFP)
{
    VNUnused(dstFP);

    return NULL;
}

#endif
