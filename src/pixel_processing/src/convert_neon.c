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

#if VN_SDK_FEATURE(NEON)

#include <LCEVC/common/neon.h>

/*------------------------------------------------------------------------------*/

static const uint32_t kStep = 16;

/*! \brief Rounds width down to SIMD alignment requirements. */
static inline uint32_t simdAlignment(const uint32_t width) { return alignTruncU32(width, kStep); }

/*------------------------------------------------------------------------------*/

typedef int16x8_t (*U16ToS16Func)(uint16x8_t value);

static inline int16x8_t U10ToS16(uint16x8_t value)
{
    const uint16x8_t kMidpoint = vdupq_n_u16(0x4000);
    return vreinterpretq_s16_u16(vsubq_u16(vshlq_n_u16(value, 5), kMidpoint));
}

static inline int16x8_t U12ToS16(uint16x8_t value)
{
    const uint16x8_t kMidpoint = vdupq_n_u16(0x4000);
    return vreinterpretq_s16_u16(vsubq_u16(vshlq_n_u16(value, 3), kMidpoint));
}

static inline int16x8_t U14ToS16(uint16x8_t value)
{
    const uint16x8_t kMidpoint = vdupq_n_u16(0x4000);
    return vreinterpretq_s16_u16(vsubq_u16(vshlq_n_u16(value, 1), kMidpoint));
}

/*------------------------------------------------------------------------------*/

static void copyU8_S16_NEON(const LdppConvertArgs* args)
{
    const uint16x8_t kMidpoint = vdupq_n_u16(0x4000);

    VN_BLIT_SIMD_BOILERPLATE(uint8_t, int16_t);

    for (uint32_t y = 0; y < count; y++) {
        const uint8_t* srcPixel = srcRow;
        int16_t* dstPixel = dstRow;
        uint32_t x = 0;

        for (; x < simdWidth; x += kStep, srcPixel += kStep, dstPixel += kStep) {
            /* Load 16-pixels & split */
            uint8x8x2_t in = {{vld1_u8(srcPixel), vld1_u8(srcPixel + 8)}};

            int16x8x2_t rounded = {
                {vreinterpretq_s16_u16(vsubq_u16(vshll_n_u8(in.val[0], 7), kMidpoint)),
                 vreinterpretq_s16_u16(vsubq_u16(vshll_n_u8(in.val[1], 7), kMidpoint))}};

            /* Store 16-pixels */
#if defined(__aarch64__)
            vst1q_s16_x2(dstPixel, rounded);
#else
            vst1q_s16(dstPixel, rounded.val[0]);
            vst1q_s16(dstPixel + 8, rounded.val[1]);
#endif
        }

        for (; x < width; x++, srcPixel++, dstPixel++) {
            *dstPixel = fpU8ToS8(*srcPixel);
        }

        srcRow += srcStride;
        dstRow += dstStride;
    }
}

static void copyU16_S16_NEON(const LdppConvertArgs* args, const int16_t shift, U16ToS16Func convert)
{
    VN_BLIT_SIMD_BOILERPLATE(uint16_t, int16_t);

    for (uint32_t y = 0; y < count; y++) {
        const uint16_t* srcPixel = srcRow;
        int16_t* dstPixel = dstRow;
        uint32_t x = 0;

        for (; x < simdWidth; x += kStep, srcPixel += kStep, dstPixel += kStep) {
            /* Load 16-pixels & split */
#if defined(__aarch64__)
            uint16x8x2_t in = vld1q_u16_x2(srcPixel);
#else
            uint16x8x2_t in = {{vld1q_u16(srcPixel), vld1q_u16(srcPixel + 8)}};
#endif

            int16x8x2_t rounded = {{convert(in.val[0]), convert(in.val[1])}};

            /* Store 16-pixels */
#if defined(__aarch64__)
            vst1q_s16_x2(dstPixel, rounded);
#else
            vst1q_s16(dstPixel, rounded.val[0]);
            vst1q_s16(dstPixel + 8, rounded.val[1]);
#endif
        }

        for (; x < width; x++, srcPixel++, dstPixel++) {
            *dstPixel = fpU16ToS16(*srcPixel, shift);
        }

        srcRow += srcStride;
        dstRow += dstStride;
    }
}

/*------------------------------------------------------------------------------*/

static void copyU10_S16_NEON(const LdppConvertArgs* args) { copyU16_S16_NEON(args, 5, U10ToS16); }

static void copyU12_S16_NEON(const LdppConvertArgs* args) { copyU16_S16_NEON(args, 3, U12ToS16); }

static void copyU14_S16_NEON(const LdppConvertArgs* args) { copyU16_S16_NEON(args, 1, U14ToS16); }

/*------------------------------------------------------------------------------*/

/* clang-format off */

static const PlaneConvertFunction kCopyTable[LdpFPCount][LdpFPCount] = {
    /* src/dst   U8                U10               U12               U14               S8.7             S10.5             S12.3             S14.1*/
    /* U8    */ {NULL,             NULL,             NULL,             NULL,             copyU8_S16_NEON, NULL,             NULL,             NULL            },
    /* U10   */ {NULL,             NULL,             NULL,             NULL,             NULL,            copyU10_S16_NEON, NULL,             NULL            },
    /* U12   */ {NULL,             NULL,             NULL,             NULL,             NULL,            NULL,             copyU12_S16_NEON, NULL            },
    /* U14   */ {NULL,             NULL,             NULL,             NULL,             NULL,            NULL,             NULL,             copyU14_S16_NEON},
    /* S8.7  */ {NULL,             NULL,             NULL,             NULL,             NULL,            NULL,             NULL,             NULL            },
    /* S10.5 */ {NULL,             NULL,             NULL,             NULL,             NULL,            NULL,             NULL,             NULL            },
    /* S12.3 */ {NULL,             NULL,             NULL,             NULL,             NULL,            NULL,             NULL,             NULL            },
    /* S14.1 */ {NULL,             NULL,             NULL,             NULL,             NULL,            NULL,             NULL,             NULL            },
};

/* clang-format on */

/*------------------------------------------------------------------------------*/

PlaneConvertFunction planeConvertGetFunctionNEON(LdpFixedPoint srcFP, LdpFixedPoint dstFP, bool isNV12)
{
    if (isNV12) {
        return NULL;
    }
    return kCopyTable[srcFP][dstFP];
}

/*------------------------------------------------------------------------------*/

#else

PlaneConvertFunction planeConvertGetFunctionNEON(LdpFixedPoint srcFP, LdpFixedPoint dstFP, bool isNV12)
{
    VNUnused(dstFP);
    VNUnused(srcFP);
    VNUnused(isNV12);
    return NULL;
}

#endif
