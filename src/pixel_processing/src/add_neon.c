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

#if VN_SDK_FEATURE(NEON)

#include <assert.h>
#include <LCEVC/common/neon.h>

/*------------------------------------------------------------------------------*/

/* Macros used for fixed values. This is a requirement for building with Clang
 * as it is more strict about passing in constants to the shift intrinsics. It
 * does not interpret the value on a static const uint32_t variable unfortunately. */
#define VN_SHIFT_U8() 7
#define VN_SHIFT_S7() 8

/*------------------------------------------------------------------------------*/

static const uint32_t kStep = 16;

/*! \brief Rounds width down to SIMD alignment requirements. */
static inline uint32_t simdAlignment(const uint32_t width) { return alignTruncU32(width, kStep); }

/*------------------------------------------------------------------------------*/

/*! \brief Performs an addition of two S16 inputs onto a U8 destination in NEON */
void addU8_NEON(const LdppAddArgs* args)
{
    const int16x8_t fractOffset = vdupq_n_s16(64);
    const int16x8_t signOffset = vdupq_n_s16(128);

    VN_ADD_SIMD_BOILERPLATE(uint8_t)

    for (uint32_t y = 0; y < count; y++) {
        const int16_t* srcPixel1 = srcRow1;
        const int16_t* srcPixel2 = srcRow2;
        uint8_t* dstPixel = dstRow;
        uint32_t x = 0;

        /* SIMD loop */
        for (; x < simdWidth; x += kStep, dstPixel += kStep, srcPixel1 += kStep, srcPixel2 += kStep) {
            /* Load 16-pixels from each src */
            int16x8x2_t src1 = {{vld1q_s16(srcPixel1), vld1q_s16(srcPixel1 + 8)}};
            int16x8x2_t src2 = {{vld1q_s16(srcPixel2), vld1q_s16(srcPixel2 + 8)}};

            /* add sources */
            int16x8_t dst0 = vqaddq_s16(src1.val[0], src2.val[0]);
            int16x8_t dst1 = vqaddq_s16(src1.val[1], src2.val[1]);

            /* val += 0x40 */
            dst0 = vqaddq_s16(dst0, fractOffset);
            dst1 = vqaddq_s16(dst1, fractOffset);

            /* val >>= 7 */
            dst0 = vshrq_n_s16(dst0, VN_SHIFT_U8());
            dst1 = vshrq_n_s16(dst1, VN_SHIFT_U8());

            /* val += 0x80 */
            dst0 = vaddq_s16(dst0, signOffset);
            dst1 = vaddq_s16(dst1, signOffset);

            /* Saturated cast back to u8 */
            uint8x16_t dstResult = vcombine_u8(vqmovun_s16(dst0), vqmovun_s16(dst1));

            /* Store 16-pixels */
            vst1q_u8(dstPixel, dstResult);
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

/*! \brief Performs an addition of two S16 inputs onto a U16 destination in NEON */
void addUN_NEON(const LdppAddArgs* args, int32_t shift, int16_t roundingOffset, int16_t signOffset,
                int16_t resultMax, LdpFixedPoint outputFP)
{
    FixedPointDemotionFunction sToU = fixedPointGetDemotionFunction(outputFP);

    const int16x8_t shiftDown = vdupq_n_s16(-shift);
    const int16x8_t roundingOffsetV = vdupq_n_s16(roundingOffset);
    const int16x8_t signOffsetV = vdupq_n_s16(signOffset);
    const int16x8_t minV = vdupq_n_s16(0);
    const int16x8_t maxV = vdupq_n_s16(resultMax);

    VN_ADD_SIMD_BOILERPLATE(uint16_t)

    for (uint32_t y = 0; y < count; y++) {
        const int16_t* srcPixel1 = srcRow1;
        const int16_t* srcPixel2 = srcRow2;
        uint16_t* dstPixel = dstRow;
        uint32_t x = 0;

        /* SIMD loop*/
        for (; x < simdWidth; x += kStep, dstPixel += kStep, srcPixel1 += kStep, srcPixel2 += kStep) {
            /* Load 16-pixels from each src */
            int16x8x2_t src1 = {{vld1q_s16(srcPixel1), vld1q_s16(srcPixel1 + 8)}};
            int16x8x2_t src2 = {{vld1q_s16(srcPixel2), vld1q_s16(srcPixel2 + 8)}};

            /* add sources */
            int16x8_t dst0 = vqaddq_s16(src1.val[0], src2.val[0]);
            int16x8_t dst1 = vqaddq_s16(src1.val[1], src2.val[1]);

            /* val += rounding */
            dst0 = vqaddq_s16(dst0, roundingOffsetV);
            dst1 = vqaddq_s16(dst1, roundingOffsetV);

            /* val >>= 5 */
            dst0 = vshlq_s16(dst0, shiftDown);
            dst1 = vshlq_s16(dst1, shiftDown);

            /* val += sign offset */
            dst0 = vaddq_s16(dst0, signOffsetV);
            dst1 = vaddq_s16(dst1, signOffsetV);

            /* clamp to unsigned range */
            dst0 = vmaxq_s16(vminq_s16(dst0, maxV), minV);
            dst1 = vmaxq_s16(vminq_s16(dst1, maxV), minV);

            /* Store 16-pixels */
            vst1q_u16(dstPixel, vreinterpretq_u16_s16(dst0));
            vst1q_u16(dstPixel + 8, vreinterpretq_u16_s16(dst1));
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

static void addU10_NEON(const LdppAddArgs* args) { addUN_NEON(args, 5, 16, 512, 1023, LdpFPU10); }

static void addU12_NEON(const LdppAddArgs* args) { addUN_NEON(args, 3, 4, 2048, 4095, LdpFPU12); }

static void addU14_NEON(const LdppAddArgs* args) { addUN_NEON(args, 1, 1, 8192, 16383, LdpFPU14); }

/*------------------------------------------------------------------------------*/

/* clang-format off */
static const PlaneAddFunction kAddTable[LdpFPCount] = {
	&addU8_NEON,  /* FP_U8 */
	&addU10_NEON, /* FP_U10 */
	&addU12_NEON, /* FP_U12 */
	&addU14_NEON, /* FP_U14 */
	NULL, /* FP_S8_7 */
	NULL, /* FP_S10_5 */
	NULL, /* FP_S12_3 */
	NULL, /* FP_S14_1 */
};
/* clang-format on */

/*------------------------------------------------------------------------------*/

PlaneAddFunction planeAddGetFunctionNEON(LdpFixedPoint dstFP)
{
    assert(!fixedPointIsSigned(dstFP));

    return kAddTable[dstFP];
}

/*------------------------------------------------------------------------------*/

#else

PlaneAddFunction planeAddGetFunctionNEON(LdpFixedPoint dstFP)
{
    VNUnused(dstFP);
    return NULL;
}

#endif
