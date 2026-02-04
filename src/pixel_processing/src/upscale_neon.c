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

#include "upscale_neon.h"

#include "upscale_common.h"

#include <LCEVC/build_config.h>
#include <LCEVC/common/platform.h>
//
#include <stddef.h>

#if VN_SDK_FEATURE(NEON)
#include "fp_types.h"
#include "upscale_scalar.h"

#include <LCEVC/common/limit.h>
#include <LCEVC/common/neon.h>
#include <LCEVC/pixel_processing/dither.h>
//
#include <assert.h>

/*------------------------------------------------------------------------------*/

enum UpscaleConstantsNEON
{
    UCHoriStepping = 8,
    UCHoriLoadAlignment = 16,     /* Horizontal requires 16-values loaded. */
    UCHoriLoadAlignmentNV12 = 32, /* Horizontal NV12 requires 32-values loaded. */
    UCHoriLoadAlignmentRGB = 48,  /* Horizontal RGB requires 48-values loaded. */
    UCHoriLoadAlignmentRGBA = 64, /* Horizontal RGBA requires 64-values loaded. */
    UCMaxKernelSize = 6,
    UCInverseShift = 14,
};

/*------------------------------------------------------------------------------*/

static inline int16x8_t U8ToS16(uint8x8_t value)
{
    const uint16x8_t kMidpoint = vdupq_n_u16(0x4000);
    return vreinterpretq_s16_u16(vsubq_u16(vshll_n_u8(value, 7), kMidpoint));
}

static inline int16x8_t U10ToS16(int16x8_t value)
{
    const int16x8_t kMidpoint = vdupq_n_s16(0x4000);
    return vsubq_s16(vshlq_n_s16(value, 5), kMidpoint);
}

static inline int16x8_t U12ToS16(int16x8_t value)
{
    const int16x8_t kMidpoint = vdupq_n_s16(0x4000);
    return vsubq_s16(vshlq_n_s16(value, 3), kMidpoint);
}

static inline int16x8_t U14ToS16(int16x8_t value)
{
    const int16x8_t kMidpoint = vdupq_n_s16(0x4000);
    return vsubq_s16(vshlq_n_s16(value, 1), kMidpoint);
}

static inline int16x8x2_t UNtoS16(int16x8x2_t value, uint16_t shift)
{
    switch (shift) {
        case 5: {
            int16x8x2_t ret = {{U10ToS16(value.val[0]), U10ToS16(value.val[1])}};
            return ret;
        }
        case 3: {
            int16x8x2_t ret = {{U12ToS16(value.val[0]), U12ToS16(value.val[1])}};
            return ret;
        }
        case 1: {
            int16x8x2_t ret = {{U14ToS16(value.val[0]), U14ToS16(value.val[1])}};
            return ret;
        }
        default:
            assert(false);
            int16x8x2_t zero = {{vdupq_n_s16(0), vdupq_n_s16(0)}};
            return zero;
    }
}

static inline int16x8_t horizontalUNtoS16(int16x8_t value, uint16_t shift)
{
    switch (shift) {
        case 5: return U10ToS16(value);
        case 3: return U12ToS16(value);
        case 1: return U14ToS16(value);
        default: assert(false); return vdupq_n_s16(0);
    }
}

static inline uint8x16_t S16ToU8(int16x8x2_t value)
{
    const int16x8_t kOffset = vdupq_n_s16(64);
    const int16x8_t kMidpoint = vdupq_n_s16(128);
    const int16x8_t kS8Min = vdupq_n_s16(INT8_MIN);
    const int16x8_t kU8Max = vdupq_n_s16(UINT8_MAX);

    uint8x8x2_t rounded = {
        {vreinterpret_u8_s8(vmovn_s16(vminq_s16(
             vqaddq_s16(vmaxq_s16(vshrq_n_s16(vqaddq_s16(value.val[0], kOffset), 7), kS8Min), kMidpoint),
             kU8Max))),
         vreinterpret_u8_s8(vmovn_s16(vminq_s16(
             vqaddq_s16(vmaxq_s16(vshrq_n_s16(vqaddq_s16(value.val[1], kOffset), 7), kS8Min), kMidpoint),
             kU8Max)))}};

    return vcombine_u8(rounded.val[0], rounded.val[1]);
}

static inline int16x8x2_t S16ToU10(int16x8x2_t value)
{
    const int16x8_t kOffset = vdupq_n_s16(16);
    const int16x8_t kMidpoint = vdupq_n_s16(512);
    int16x8x2_t ret = {{vaddq_s16(vshrq_n_s16(vaddq_s16(value.val[0], kOffset), 5), kMidpoint),
                        vaddq_s16(vshrq_n_s16(vaddq_s16(value.val[1], kOffset), 5), kMidpoint)}};

    return ret;
}

static inline int16x8x2_t S16ToU12(int16x8x2_t value)
{
    const int16x8_t kOffset = vdupq_n_s16(4);
    const int16x8_t kMidpoint = vdupq_n_s16(2048);
    int16x8x2_t ret = {{vaddq_s16(vshrq_n_s16(vaddq_s16(value.val[0], kOffset), 3), kMidpoint),
                        vaddq_s16(vshrq_n_s16(vaddq_s16(value.val[1], kOffset), 3), kMidpoint)}};

    return ret;
}

static inline int16x8x2_t S16ToU14(int16x8x2_t value)
{
    const int16x8_t kOffset = vdupq_n_s16(1);
    const int16x8_t kMidpoint = vdupq_n_s16(8192);
    int16x8x2_t ret = {{vaddq_s16(vshrq_n_s16(vaddq_s16(value.val[0], kOffset), 1), kMidpoint),
                        vaddq_s16(vshrq_n_s16(vaddq_s16(value.val[1], kOffset), 1), kMidpoint)}};

    return ret;
}

static inline int16x8x2_t S16ToUN(int16x8x2_t value, uint16_t shift)
{
    switch (shift) {
        case 5: return S16ToU10(value);
        case 3: return S16ToU12(value);
        case 1: return S16ToU14(value);
        default:
            assert(false);
            int16x8x2_t zero = {{vdupq_n_s16(0), vdupq_n_s16(0)}};
            return zero;
    }
}

/*------------------------------------------------------------------------------*/

/*!
 * Loads a single channel of pixels in the high-half of a register.
 *
 * \param in       The input row to load from.
 * \param offset   The offset in pixels to load from.
 *
 * \return 8-pixels loaded for 1 channel.
 */
static inline void horizontalGetPelsU8(const uint8_t* in, int32_t offset, int16x8x2_t* pels)
{
    pels->val[1] = U8ToS16(vld1_u8(&in[offset]));
}

/*!
 * Loads a single channel of pixels in the high register.
 *
 * \param in       The input row to load from.
 * \param offset   The offset in pixels to load from.
 * \param pels     The pixels to modify and load into.
 *
 * \return 8-pixels loaded for 1 channel.
 */
static inline void horizontalGetPelsN16(const uint8_t* in, int32_t offset, int16x8x2_t* pels)
{
    const int16_t* in16 = (const int16_t*)in;
    pels->val[1] = vld1q_s16(&in16[offset]);
}

/*!
 * Loads 2 channels of pixels in the high-half of 2 registers.
 *
 * \param in       The input row to load from.
 * \param offset   The offset in "elements" to load from.
 *
 * \return 8-pixels loaded for 2 channels.
 */
static inline int16x8x4_t horizontalGetPelsS16NV12(const uint8_t* in, int32_t offset)
{
    const int16_t* in16 = (const int16_t*)in;
    const int16x8x2_t loaded = vld2q_s16(&in16[offset << 1]);
    const int16x8x4_t res = {{vdupq_n_s16(0), loaded.val[0], vdupq_n_s16(0), loaded.val[1]}};

    return res;
}

static inline int16x8x4_t horizontalGetPelsU8ToS16NV12(const uint8_t* in, int32_t offset)
{
    const uint8x8x2_t loaded = vld2_u8(&in[offset << 1]);
    const int16x8x4_t res = {
        {vdupq_n_s16(0), U8ToS16(loaded.val[0]), vdupq_n_s16(0), U8ToS16(loaded.val[1])}};

    return res;
}

/*!
 * Loads the next pixels for a single channel into the high-half of a register whilst
 * shifting the high half into the low half.
 *
 * \param in       The input row to load from.
 * \param offset   The offset in pixels to load from.
 * \param pels     The pixels to modify and load into.
 *
 * \return Additional 8-pixels loaded for 1 channel.
 */
static inline void horizontalGetNextPelsU8(const uint8_t* in, int32_t offset, int16x8x2_t* pels)
{
    /* Move current high half down to low and load subsequent 8 pels into high half of uint8x16x2_t */
    pels->val[0] = pels->val[1];
    horizontalGetPelsU8(in, offset, pels);
}

/*!
 * Loads the next pixels for a single channel into the high-half of a register whilst
 * shifting the high half into the low half.
 *
 * \param in       The input row to load from.
 * \param offset   The offset in pixels to load from.
 * \param pels     The pixels to modify and load into.
 *
 * \return Additional 8-pixels loaded for 1 channel.
 */
static inline void horizontalGetNextPelsN16(const uint8_t* in, int32_t offset, int16x8x2_t* pels)
{
    const int16_t* in16 = (const int16_t*)in;

    pels->val[0] = pels->val[1];
    pels->val[1] = vld1q_s16(&in16[offset]);
}

/*!
 * Loads the next pixels for 2 channels into the high-half of 2 registers whilst
 * shifting the high half into the low half.
 *
 * \param in       The input row to load from.
 * \param offset   The offset in pixels to load from.
 * \param pels     The pixels to modify and load into.
 */
static inline void horizontalGetNextPelsS16NV12(const uint8_t* in, int32_t offset, int16x8x4_t* pels)
{
    const int16_t* in16 = (const int16_t*)in;
    const int16x8x2_t next = vld2q_s16(&in16[offset << 1]);
    pels->val[0] = pels->val[1];
    pels->val[1] = next.val[0];
    pels->val[2] = pels->val[3];
    pels->val[3] = next.val[1];
}

static inline void horizontalGetNextPelsU8ToS16NV12(const uint8_t* in, int32_t offset, int16x8x4_t* pels)
{
    const uint8x8x2_t next = vld2_u8(&in[offset << 1]);
    pels->val[0] = pels->val[1];
    pels->val[1] = U8ToS16(next.val[0]);
    pels->val[2] = pels->val[3];
    pels->val[3] = U8ToS16(next.val[1]);
}

/*!
 * Performs horizontal convolution of input pels into result applying the forward
 * and reverse kernels accordingly, whereby the first result pixel will have the
 * reverse kernel applied, due to upscaling being off-pixel.
 *
 * This generates 16-pixels worth of output.
 *
 * \param pels            The pixels to upscale from.
 * \param result          Place to store the resultant 16-pixels.
 * \param kernelFwd       The forward kernel.
 * \param kernelRev       The reverse kernel.
 * \param kernelLength    The length of both kernel_fwd and kernel_rev.
 */
static inline void horizontalConvolveN16(int16x8x2_t pels, int16x8x2_t* result, const int16_t* kernelFwd,
                                         const int16_t* kernelRev, int32_t kernelLength)
{
    int32x4_t values[4];
    int16x4x2_t combine[2];

    const int16x8_t minV = vdupq_n_s16(-16384); /* see saturateS15 for explanation */
    const int16x8_t maxV = vdupq_n_s16(16383);

    /* Reverse */
    values[0] = vmull_n_s16(vget_low_s16(pels.val[0]), kernelRev[0]);
    values[2] = vmull_n_s16(vget_high_s16(pels.val[0]), kernelRev[0]);

    /* Shift down */
    pels.val[0] = vextq_s16(pels.val[0], pels.val[1], 1);
    pels.val[1] = vextq_s16(pels.val[1], pels.val[1], 1);

    /* Forward */
    values[1] = vmull_n_s16(vget_low_s16(pels.val[0]), kernelFwd[0]);
    values[3] = vmull_n_s16(vget_high_s16(pels.val[0]), kernelFwd[0]);

    for (int32_t i = 1; i < kernelLength; i++) {
        /* Reverse */
        values[0] = vmlal_n_s16(values[0], vget_low_s16(pels.val[0]), kernelRev[i]);
        values[2] = vmlal_n_s16(values[2], vget_high_s16(pels.val[0]), kernelRev[i]);

        /* Shift */
        pels.val[0] = vextq_s16(pels.val[0], pels.val[1], 1);
        pels.val[1] = vextq_s16(pels.val[1], pels.val[1], 1);

        /* Forward */
        values[1] = vmlal_n_s16(values[1], vget_low_s16(pels.val[0]), kernelFwd[i]);
        values[3] = vmlal_n_s16(values[3], vget_high_s16(pels.val[0]), kernelFwd[i]);
    }

    /* Scale back to 16 bits */
    combine[0].val[0] = vqrshrn_n_s32(values[0], UCInverseShift);
    combine[0].val[1] = vqrshrn_n_s32(values[1], UCInverseShift);
    combine[1].val[0] = vqrshrn_n_s32(values[2], UCInverseShift);
    combine[1].val[1] = vqrshrn_n_s32(values[3], UCInverseShift);

    /* Interleave */
    combine[0] = vzip_s16(combine[0].val[0], combine[0].val[1]);
    combine[1] = vzip_s16(combine[1].val[0], combine[1].val[1]);

    /* Output */
    result->val[0] = vcombine_s16(combine[0].val[0], combine[0].val[1]);
    result->val[1] = vcombine_s16(combine[1].val[0], combine[1].val[1]);

    /* Saturate (clamp) to +/- 2^14 */
    result->val[0] = vmaxq_s16(vminq_s16(result->val[0], maxV), minV);
    result->val[1] = vmaxq_s16(vminq_s16(result->val[1], maxV), minV);
}

/*!
 * Apply 1D predicted-average to values using base for a single row.
 *
 * See `horizontal_apply_pa_2d_precision` for more detail on this specialisation.
 *
 * \param base     The base pixels for the PA calculation.
 * \param values   The upscaled pixels to apply PA to.
 */
static inline void applyPA1D(int16x8_t base, int16x8x2_t* values)
{
    /* avg = base - ((pel_even + pel_odd + 1) >> 1) */
    const int32x4_t tmp0 = vpaddlq_s16(values->val[0]);
    const int32x4_t tmp1 = vpaddlq_s16(values->val[1]);
    const int16x8_t sum = vcombine_s16(vrshrn_n_s32(tmp0, 1), vrshrn_n_s32(tmp1, 1));
    const int16x8_t avg = vsubq_s16(base, sum);

    /* Repeat each avg to apply to values. */
    const int16x8x2_t broadcast = vzipq_s16(avg, avg);
    values->val[0] = vqaddq_s16(values->val[0], broadcast.val[0]);
    values->val[1] = vqaddq_s16(values->val[1], broadcast.val[1]);
}

/*!
 * Apply 2D predicted-average to values using base, this requires 2 upscaled rows.
 *
 * This is a specialised version of the function that promotes the math to 32-bit
 * as the average calculation for S16 & U14 can trivially overflow - the none-S16 variant
 * is intended to consume numbers between U8 and U12 which have enough headroom bits
 * to allow the average to be performed in 16-bit.
 *
 * \param base     The base pixels for the PA calculation.
 * \param values   The upscaled pixels to apply PA to for 2 rows.
 */
static inline void applyPA2D(int16x8_t base, int16x8x2_t values[2])
{
    /* avg = base - ((row0_pel_even + row0_pel_odd + row1_pel_even + row1_pel_odd + 2) >> 2) */
    const int32x4_t tmp0 = vpaddlq_s16(values[0].val[0]);
    const int32x4_t tmp1 = vpaddlq_s16(values[0].val[1]);
    const int32x4_t tmp2 = vpaddlq_s16(values[1].val[0]);
    const int32x4_t tmp3 = vpaddlq_s16(values[1].val[1]);
    const int32x4_t sum0 = vaddq_s32(tmp0, tmp2);
    const int32x4_t sum1 = vaddq_s32(tmp1, tmp3);
    const int16x8_t sum = vcombine_s16(vrshrn_n_s32(sum0, 2), vrshrn_n_s32(sum1, 2));
    const int16x8_t avg = vsubq_s16(base, sum);

    /* Repeat each avg to apply to values. */
    const int16x8x2_t broadcast = vzipq_s16(avg, avg);
    values[0].val[0] = vqaddq_s16(values[0].val[0], broadcast.val[0]);
    values[0].val[1] = vqaddq_s16(values[0].val[1], broadcast.val[1]);
    values[1].val[0] = vqaddq_s16(values[1].val[0], broadcast.val[0]);
    values[1].val[1] = vqaddq_s16(values[1].val[1], broadcast.val[1]);
}

/*! \brief Planar horizontal upscaling of 2 rows. U8 input, U8 output. */
void horizontal1DU8PlanarNEON(LdppDitherSlice* dither, const uint8_t* in[2], uint8_t* out[2],
                              const uint8_t* base[2], uint32_t width, uint32_t xStart,
                              uint32_t xEnd, LdppHorizontalUpscaleParams* params)
{
    const int16_t* kernelFwd = params->kernel->coeffs[0];
    const int16_t* kernelRev = params->kernel->coeffs[1];
    const int32_t kernelLength = (int32_t)params->kernel->length;
    int16x8x2_t pels[2];
    int16x8x2_t values[2];
    const bool paEnabled = (base[0] != NULL);
    const uint16_t* ditherBuffer = NULL;

    UpscaleHorizontalCoords coords = {0};

    assert(kernelLength % 2 == 0);
    assert(kernelLength <= UCMaxKernelSize);

    /* Determine edge-cases that should be run in non-SIMD codepath. */
    upscaleHorizontalGetCoords(width, xStart, xEnd, kernelLength, UCHoriLoadAlignment, &coords);

    /* Run left edge non-SIMD loop */
    if (upscaleHorizontalCoordsIsLeftValid(&coords)) {
        horizontalU8Scalar(dither, in, out, base, width, coords.leftStart, coords.leftEnd, params);
    }

    /* Prime I/O */
    int32_t loadOffset = (int32_t)(coords.start - (kernelLength >> 1));
    horizontalGetPelsU8(in[0], loadOffset, &pels[0]);
    horizontalGetPelsU8(in[1], loadOffset, &pels[1]);
    loadOffset += UCHoriStepping;
    int32_t storeOffset = (int32_t)(coords.start << 1);

    /* Prepare dither buffer containing enough values for 2 fully upscaled rows. */
    if (dither != NULL) {
        ditherBuffer = ldppDitherGetBuffer(dither, alignU32(4 * (xEnd - xStart), 16));
    }

    /* Run middle SIMD loop */
    for (uint32_t x = coords.start; x < coords.end; x += UCHoriStepping) {
        horizontalGetNextPelsU8(in[0], loadOffset, &pels[0]);
        horizontalGetNextPelsU8(in[1], loadOffset, &pels[1]);

        horizontalConvolveN16(pels[0], &values[0], kernelFwd, kernelRev, kernelLength);
        horizontalConvolveN16(pels[1], &values[1], kernelFwd, kernelRev, kernelLength);

        if (paEnabled) {
            const int16x8_t basePels0 = U8ToS16(vld1_u8(&base[0][x]));
            const int16x8_t basePels1 = U8ToS16(vld1_u8(&base[1][x]));

            applyPA1D(basePels0, &values[0]);
            applyPA1D(basePels1, &values[1]);
        }

        if (ditherBuffer) {
            ldppDitherApplyNEON(&values[0], &ditherBuffer, 0, dither->strength);
            ldppDitherApplyNEON(&values[1], &ditherBuffer, 0, dither->strength);
        }

        vst1q_u8(&out[0][storeOffset], S16ToU8(values[0]));
        vst1q_u8(&out[1][storeOffset], S16ToU8(values[1]));

        loadOffset += UCHoriStepping;
        storeOffset += (UCHoriStepping << 1);
    }

    /* Run right edge non-SIMD loop */
    if (upscaleHorizontalCoordsIsRightValid(&coords)) {
        horizontalU8Scalar(dither, in, out, base, width, coords.rightStart, coords.rightEnd, params);
    }
}

/*! \brief Planar horizontal upscaling of 2 rows. S16 input, U8 output. */
void horizontal2DU8PlanarNEON(LdppDitherSlice* dither, const uint8_t* in[2], uint8_t* out[2],
                              const uint8_t* base[2], uint32_t width, uint32_t xStart,
                              uint32_t xEnd, LdppHorizontalUpscaleParams* params)
{
    const int16_t* kernelFwd = params->kernel->coeffs[0];
    const int16_t* kernelRev = params->kernel->coeffs[1];
    const int32_t kernelLength = (int32_t)params->kernel->length;
    int16x8x2_t pels[2];
    int16x8x2_t values[2];
    const bool paEnabled = (base[0] != NULL);
    const uint16_t* ditherBuffer = NULL;

    UpscaleHorizontalCoords coords = {0};

    assert(kernelLength % 2 == 0);
    assert(kernelLength <= UCMaxKernelSize);

    /* Determine edge-cases that should be run in non-SIMD codepath. */
    upscaleHorizontalGetCoords(width, xStart, xEnd, kernelLength, UCHoriLoadAlignment, &coords);

    /* Run left edge non-SIMD loop */
    if (upscaleHorizontalCoordsIsLeftValid(&coords)) {
        horizontalU8Scalar(dither, in, out, base, width, coords.leftStart, coords.leftEnd, params);
    }

    /* Prime I/O */
    int32_t loadOffset = (int32_t)(coords.start - (kernelLength >> 1));
    horizontalGetPelsN16(in[0], loadOffset, &pels[0]);
    horizontalGetPelsN16(in[1], loadOffset, &pels[1]);
    loadOffset += UCHoriStepping;
    int32_t storeOffset = (int32_t)(coords.start << 1);

    /* Prepare dither buffer containing enough values for 2 fully upscaled rows. */
    if (dither != NULL) {
        ditherBuffer = ldppDitherGetBuffer(dither, alignU32(4 * (xEnd - xStart), 16));
    }

    /* Run middle SIMD loop */
    for (uint32_t x = coords.start; x < coords.end; x += UCHoriStepping) {
        horizontalGetNextPelsN16(in[0], loadOffset, &pels[0]);
        horizontalGetNextPelsN16(in[1], loadOffset, &pels[1]);

        horizontalConvolveN16(pels[0], &values[0], kernelFwd, kernelRev, kernelLength);
        horizontalConvolveN16(pels[1], &values[1], kernelFwd, kernelRev, kernelLength);

        if (paEnabled) {
            const int16x8_t basePels = U8ToS16(vld1_u8(&base[0][x]));
            applyPA2D(basePels, values);
        }

        if (ditherBuffer) {
            ldppDitherApplyNEON(&values[0], &ditherBuffer, 0, dither->strength);
            ldppDitherApplyNEON(&values[1], &ditherBuffer, 0, dither->strength);
        }

        vst1q_u8(&out[0][storeOffset], S16ToU8(values[0]));
        vst1q_u8(&out[1][storeOffset], S16ToU8(values[1]));

        loadOffset += UCHoriStepping;
        storeOffset += (UCHoriStepping << 1);
    }

    /* Run right edge non-SIMD loop */
    if (upscaleHorizontalCoordsIsRightValid(&coords)) {
        horizontalU8Scalar(dither, in, out, base, width, coords.rightStart, coords.rightEnd, params);
    }
}

/*! \brief Planar horizontal upscaling of 2 rows. S16 input, S16 output. */
void horizontalS16PlanarNEON(LdppDitherSlice* dither, const uint8_t* in[2], uint8_t* out[2],
                             const uint8_t* base[2], uint32_t width, uint32_t xStart, uint32_t xEnd,
                             LdppHorizontalUpscaleParams* params)
{
    const int16_t* kernelFwd = params->kernel->coeffs[0];
    const int16_t* kernelRev = params->kernel->coeffs[1];
    const int32_t kernelLength = (int32_t)params->kernel->length;
    int16x8x2_t pels[2];
    int16x8x2_t values[2];
    const bool paEnabled = (base[0] != NULL);
    const bool paEnabled1D = paEnabled && (base[1] != NULL);
    const uint16_t* ditherBuffer = NULL;
    int16_t* out16[2] = {(int16_t*)out[0], (int16_t*)out[1]};
    const int16_t* base16[2] = {(const int16_t*)base[0], (const int16_t*)base[1]};

    UpscaleHorizontalCoords coords = {0};

    assert(kernelLength % 2 == 0);
    assert(kernelLength <= UCMaxKernelSize);

    /* Determine edge-cases that should be run in non-SIMD codepath. */
    upscaleHorizontalGetCoords(width, xStart, xEnd, kernelLength, UCHoriLoadAlignment, &coords);

    /* Run left edge non-SIMD loop */
    if (upscaleHorizontalCoordsIsLeftValid(&coords)) {
        horizontalS16Scalar(dither, in, out, base, width, coords.leftStart, coords.leftEnd, params);
    }

    /* Prime I/O */
    int32_t loadOffset = (int32_t)(coords.start - (kernelLength >> 1));
    horizontalGetPelsN16(in[0], loadOffset, &pels[0]);
    horizontalGetPelsN16(in[1], loadOffset, &pels[1]);
    loadOffset += UCHoriStepping;
    int32_t storeOffset = (int32_t)(coords.start << 1);

    /* Prepare dither buffer containing enough values for 2 fully upscaled rows. */
    if (dither != NULL) {
        ditherBuffer = ldppDitherGetBuffer(dither, alignU32(4 * (xEnd - xStart), 16));
    }

    /* Run middle SIMD loop */
    for (uint32_t x = coords.start; x < coords.end; x += UCHoriStepping) {
        horizontalGetNextPelsN16(in[0], loadOffset, &pels[0]);
        horizontalGetNextPelsN16(in[1], loadOffset, &pels[1]);

        horizontalConvolveN16(pels[0], &values[0], kernelFwd, kernelRev, kernelLength);
        horizontalConvolveN16(pels[1], &values[1], kernelFwd, kernelRev, kernelLength);

        if (paEnabled1D) {
            const int16x8_t basePels0 = vld1q_s16(&base16[0][x]);
            const int16x8_t basePels1 = vld1q_s16(&base16[1][x]);

            applyPA1D(basePels0, &values[0]);
            applyPA1D(basePels1, &values[1]);
        } else if (paEnabled) {
            const int16x8_t basePels = vld1q_s16(&base16[0][x]);
            applyPA2D(basePels, values);
        }

        if (ditherBuffer) {
            ldppDitherApplyNEON(&values[0], &ditherBuffer, params->shift, dither->strength);
            ldppDitherApplyNEON(&values[1], &ditherBuffer, params->shift, dither->strength);
        }

        /* Write out. */
        vst1q_s16(&out16[0][storeOffset], values[0].val[0]);
        vst1q_s16(&out16[0][storeOffset + 8], values[0].val[1]);

        vst1q_s16(&out16[1][storeOffset], values[1].val[0]);
        vst1q_s16(&out16[1][storeOffset + 8], values[1].val[1]);

        loadOffset += UCHoriStepping;
        storeOffset += (UCHoriStepping << 1);
    }

    /* Run right edge non-SIMD loop */
    if (upscaleHorizontalCoordsIsRightValid(&coords)) {
        horizontalS16Scalar(dither, in, out, base, width, coords.rightStart, coords.rightEnd, params);
    }
}

/*! \brief Planar horizontal upscaling of 2 rows. S16 or UN input, UN output. */
static inline void horizontalUNPlanarNEON(LdppDitherSlice* dither, const uint8_t* in[2], uint8_t* out[2],
                                          const uint8_t* base[2], uint32_t width, uint32_t xStart,
                                          uint32_t xEnd, LdppHorizontalUpscaleParams* params)
{
    const int16_t* kernelFwd = params->kernel->coeffs[0];
    const int16_t* kernelRev = params->kernel->coeffs[1];
    const int32_t kernelLength = (int32_t)params->kernel->length;
    int16x8x2_t pels[2];
    int16x8x2_t basePels = {{vdupq_n_s16(0), vdupq_n_s16(0)}};
    int16x8x2_t values[2];
    const bool paEnabled = (base[0] != NULL);
    const bool paEnabled1D = paEnabled && (base[1] != NULL);
    const uint16_t* ditherBuffer = NULL;
    uint16_t* out16[2] = {(uint16_t*)out[0], (uint16_t*)out[1]};
    const int16_t* base16[2] = {(const int16_t*)base[0], (const int16_t*)base[1]};
    const int16x8_t minV = vdupq_n_s16(0);
    const int16x8_t maxV = vdupq_n_s16((int16_t)params->maxValue);

    UpscaleHorizontalCoords coords = {0};

    assert(kernelLength % 2 == 0);
    assert(kernelLength <= UCMaxKernelSize);

    /* Determine edge-cases that should be run in non-SIMD codepath. */
    upscaleHorizontalGetCoords(width, xStart, xEnd, kernelLength, UCHoriLoadAlignment, &coords);

    /* Run left edge non-SIMD loop */
    if (upscaleHorizontalCoordsIsLeftValid(&coords)) {
        horizontalUNScalar(dither, in, out, base, width, coords.leftStart, coords.leftEnd, params);
    }

    /* Prime I/O */
    int32_t loadOffset = (int32_t)(coords.start - (kernelLength >> 1));
    horizontalGetPelsN16(in[0], loadOffset, &pels[0]);
    horizontalGetPelsN16(in[1], loadOffset, &pels[1]);
    if (!params->is2D) {
        pels[0].val[1] = horizontalUNtoS16(pels[0].val[1], params->shift);
        pels[1].val[1] = horizontalUNtoS16(pels[1].val[1], params->shift);
    }
    loadOffset += UCHoriStepping;
    int32_t storeOffset = (int32_t)(coords.start << 1);

    /* Prepare dither buffer containing enough values for 2 fully upscaled rows. */
    if (dither != NULL) {
        ditherBuffer = ldppDitherGetBuffer(dither, alignU32(4 * (xEnd - xStart), 16));
    }

    /* Run middle SIMD loop */
    for (uint32_t x = coords.start; x < coords.end; x += UCHoriStepping) {
        horizontalGetNextPelsN16(in[0], loadOffset, &pels[0]);
        horizontalGetNextPelsN16(in[1], loadOffset, &pels[1]);
        if (!params->is2D) {
            pels[0].val[1] = horizontalUNtoS16(pels[0].val[1], params->shift);
            pels[1].val[1] = horizontalUNtoS16(pels[1].val[1], params->shift);
        }

        horizontalConvolveN16(pels[0], &values[0], kernelFwd, kernelRev, kernelLength);
        horizontalConvolveN16(pels[1], &values[1], kernelFwd, kernelRev, kernelLength);

        if (paEnabled1D) {
            basePels.val[0] = vld1q_s16(&base16[0][x]);
            basePels.val[1] = vld1q_s16(&base16[1][x]);
            basePels = UNtoS16(basePels, params->shift);

            applyPA1D(basePels.val[0], &values[0]);
            applyPA1D(basePels.val[1], &values[1]);
        } else if (paEnabled) {
            basePels.val[0] = vld1q_s16(&base16[0][x]);
            basePels = UNtoS16(basePels, params->shift);

            applyPA2D(basePels.val[0], values);
        }

        if (ditherBuffer) {
            ldppDitherApplyNEON(&values[0], &ditherBuffer, 0, dither->strength);
            ldppDitherApplyNEON(&values[1], &ditherBuffer, 0, dither->strength);
        }

        values[0] = S16ToUN(values[0], params->shift);
        values[1] = S16ToUN(values[1], params->shift);

        vst1q_u16(&out16[0][storeOffset], clampS16ToU16(values[0].val[0], minV, maxV));
        vst1q_u16(&out16[0][storeOffset + 8], clampS16ToU16(values[0].val[1], minV, maxV));

        vst1q_u16(&out16[1][storeOffset], clampS16ToU16(values[1].val[0], minV, maxV));
        vst1q_u16(&out16[1][storeOffset + 8], clampS16ToU16(values[1].val[1], minV, maxV));

        loadOffset += UCHoriStepping;
        storeOffset += (UCHoriStepping << 1);
    }

    /* Run right edge non-SIMD loop */
    if (upscaleHorizontalCoordsIsRightValid(&coords)) {
        horizontalUNScalar(dither, in, out, base, width, coords.rightStart, coords.rightEnd, params);
    }
}

/*! \brief NV12 horizontal upscaling of 2 rows. NV12 S16 input, NV12 U8 output. */
void horizontal2DU8NV12NEON(LdppDitherSlice* dither, const uint8_t* in[2], uint8_t* out[2],
                            const uint8_t* base[2], uint32_t width, uint32_t xStart, uint32_t xEnd,
                            LdppHorizontalUpscaleParams* params)
{
    const int16_t* kernelFwd = params->kernel->coeffs[0];
    const int16_t* kernelRev = params->kernel->coeffs[1];
    const int32_t kernelLength = (int32_t)params->kernel->length;
    int16x8x4_t pels[2];
    int16x8x2_t values[2];
    const bool paEnabled = (base[0] != NULL);
    const uint16_t* ditherBuffer = NULL;
    uint32_t channelIdx = 0;
    uint8x8x2_t basePels;
    int16x8x2_t result[2][2];
    uint8x16x2_t packed;

    UpscaleHorizontalCoords coords = {0};

    assert(kernelLength % 2 == 0);
    assert(kernelLength <= UCMaxKernelSize);

    /* Determine edge-cases that should be run in non-SIMD codepath. */
    upscaleHorizontalGetCoords(width, xStart, xEnd, kernelLength, UCHoriLoadAlignmentNV12, &coords);

    /* Run left edge non-SIMD loop */
    if (upscaleHorizontalCoordsIsLeftValid(&coords)) {
        horizontalU8Scalar(dither, in, out, base, width, coords.leftStart, coords.leftEnd, params);
    }

    /* Prime I/O */
    int32_t loadOffset = (int32_t)(coords.start - (kernelLength >> 1));
    pels[0] = horizontalGetPelsS16NV12(in[0], loadOffset);
    pels[1] = horizontalGetPelsS16NV12(in[1], loadOffset);
    loadOffset += UCHoriStepping;
    int32_t storeOffset = (int32_t)(coords.start << 2);

    /* Prepare dither buffer containing enough values for 2 fully upscaled rows. */
    if (dither != NULL) {
        ditherBuffer = ldppDitherGetBuffer(dither, alignU32(4 * (xEnd - xStart), 16));
    }

    /* Run middle SIMD loop */
    for (uint32_t x = coords.start; x < coords.end; x += UCHoriStepping) {
        horizontalGetNextPelsS16NV12(in[0], loadOffset, &pels[0]);
        horizontalGetNextPelsS16NV12(in[1], loadOffset, &pels[1]);

        if (paEnabled) {
            basePels = vld2_u8(&base[0][x << 1]);
        }

        for (channelIdx = 0; channelIdx < params->channelCount; ++channelIdx) {
            int16x8x2_t channelPels[2];
            if (channelIdx == 0) {
                channelPels[0].val[0] = pels[0].val[0];
                channelPels[0].val[1] = pels[0].val[1];
                channelPels[1].val[0] = pels[1].val[0];
                channelPels[1].val[1] = pels[1].val[1];
            } else {
                channelPels[0].val[0] = pels[0].val[2];
                channelPels[0].val[1] = pels[0].val[3];
                channelPels[1].val[0] = pels[1].val[2];
                channelPels[1].val[1] = pels[1].val[3];
            }
            horizontalConvolveN16(channelPels[0], &values[0], kernelFwd, kernelRev, kernelLength);
            horizontalConvolveN16(channelPels[1], &values[1], kernelFwd, kernelRev, kernelLength);

            if (paEnabled) {
                applyPA2D(U8ToS16(basePels.val[channelIdx]), values);
            }

            if (ditherBuffer) {
                ldppDitherApplyNEON(&values[0], &ditherBuffer, 0, dither->strength);
                ldppDitherApplyNEON(&values[1], &ditherBuffer, 0, dither->strength);
            }

            /* Stash result */
            result[0][channelIdx] = values[0];
            result[1][channelIdx] = values[1];
        }

        packed.val[0] = S16ToU8(result[0][0]);
        packed.val[1] = S16ToU8(result[0][1]);
        vst2q_u8(&out[0][storeOffset], packed);

        packed.val[0] = S16ToU8(result[1][0]);
        packed.val[1] = S16ToU8(result[1][1]);
        vst2q_u8(&out[1][storeOffset], packed);

        loadOffset += UCHoriStepping;
        storeOffset += (UCHoriStepping << 2);
    }

    /* Run right edge non-SIMD loop */
    if (upscaleHorizontalCoordsIsRightValid(&coords)) {
        horizontalU8Scalar(dither, in, out, base, width, coords.rightStart >> 1, coords.rightEnd, params);
    }
}

/*! \brief NV12 horizontal upscaling of 2 rows. NV12 U8 input, NV12 U8 output. */
void horizontal1DU8NV12NEON(LdppDitherSlice* dither, const uint8_t* in[2], uint8_t* out[2],
                            const uint8_t* base[2], uint32_t width, uint32_t xStart, uint32_t xEnd,
                            LdppHorizontalUpscaleParams* params)
{
    const int16_t* kernelFwd = params->kernel->coeffs[0];
    const int16_t* kernelRev = params->kernel->coeffs[1];
    const int32_t kernelLength = (int32_t)params->kernel->length;
    int16x8x4_t pels[2];
    int16x8x2_t values[2];
    const bool paEnabled = (base[0] != NULL) && (base[1] != NULL);
    const uint16_t* ditherBuffer = NULL;
    uint8_t channelIdx = 0;
    uint8x8x2_t basePels[2];
    int16x8x2_t result[2][2];
    uint8x16x2_t packed;

    UpscaleHorizontalCoords coords = {0};

    assert(kernelLength % 2 == 0);
    assert(kernelLength <= UCMaxKernelSize);

    /* Determine edge-cases that should be run in non-SIMD codepath. */
    upscaleHorizontalGetCoords(width, xStart, xEnd, kernelLength, UCHoriLoadAlignmentNV12, &coords);

    /* Run left edge non-SIMD loop */
    if (upscaleHorizontalCoordsIsLeftValid(&coords)) {
        horizontalU8Scalar(dither, in, out, base, width, coords.leftStart, coords.leftEnd, params);
    }

    /* Prime I/O */
    int32_t loadOffset = (int32_t)(coords.start - (kernelLength >> 1));
    pels[0] = horizontalGetPelsU8ToS16NV12(in[0], loadOffset);
    pels[1] = horizontalGetPelsU8ToS16NV12(in[1], loadOffset);
    loadOffset += UCHoriStepping;
    int32_t storeOffset = (int32_t)(coords.start << 2);

    /* Prepare dither buffer containing enough values for 2 fully upscaled rows. */
    if (dither != NULL) {
        ditherBuffer = ldppDitherGetBuffer(dither, alignU32(4 * (xEnd - xStart), 16));
    }

    /* Run middle SIMD loop */
    for (uint32_t x = coords.start; x < coords.end; x += UCHoriStepping) {
        horizontalGetNextPelsU8ToS16NV12(in[0], loadOffset, &pels[0]);
        horizontalGetNextPelsU8ToS16NV12(in[1], loadOffset, &pels[1]);

        if (paEnabled) {
            basePels[0] = vld2_u8(&base[0][x << 1]);
            basePels[1] = vld2_u8(&base[1][x << 1]);
        }

        for (channelIdx = 0; channelIdx < params->channelCount; ++channelIdx) {
            int16x8x2_t channelPels[2];
            if (channelIdx == 0) {
                channelPels[0].val[0] = pels[0].val[0];
                channelPels[0].val[1] = pels[0].val[1];
                channelPels[1].val[0] = pels[1].val[0];
                channelPels[1].val[1] = pels[1].val[1];
            } else {
                channelPels[0].val[0] = pels[0].val[2];
                channelPels[0].val[1] = pels[0].val[3];
                channelPels[1].val[0] = pels[1].val[2];
                channelPels[1].val[1] = pels[1].val[3];
            }
            horizontalConvolveN16(channelPels[0], &values[0], kernelFwd, kernelRev, kernelLength);
            horizontalConvolveN16(channelPels[1], &values[1], kernelFwd, kernelRev, kernelLength);

            if (paEnabled) {
                applyPA1D(U8ToS16(basePels[0].val[channelIdx]), &values[0]);
                applyPA1D(U8ToS16(basePels[1].val[channelIdx]), &values[1]);
            }

            if (ditherBuffer) {
                ldppDitherApplyNEON(&values[0], &ditherBuffer, 0, dither->strength);
                ldppDitherApplyNEON(&values[1], &ditherBuffer, 0, dither->strength);
            }

            /* Stash result */
            result[0][channelIdx] = values[0];
            result[1][channelIdx] = values[1];
        }

        packed.val[0] = S16ToU8(result[0][0]);
        packed.val[1] = S16ToU8(result[0][1]);
        vst2q_u8(&out[0][storeOffset], packed);

        packed.val[0] = S16ToU8(result[1][0]);
        packed.val[1] = S16ToU8(result[1][1]);
        vst2q_u8(&out[1][storeOffset], packed);

        loadOffset += UCHoriStepping;
        storeOffset += (UCHoriStepping << 2);
    }

    /* Run right edge non-SIMD loop */
    if (upscaleHorizontalCoordsIsRightValid(&coords)) {
        horizontalU8Scalar(dither, in, out, base, width, coords.rightStart >> 1, coords.rightEnd, params);
    }
}

/*------------------------------------------------------------------------------*/

/*!
 * Loads kernel-length rows of initial upscale input data ensuring that edge extension
 * is performed.
 *
 * \param  in       The input source surface to load from.
 * \param  height   The height of the input surface being loaded.
 * \param  stride   The stride of the input surface being loaded.
 * \param  offset   The row offset to start loading from.
 * \param  count    The number of rows to load in.
 * \param  pels     The destination to load the pixels into.
 */
static inline void verticalGetPelsU8(const uint8_t* in, uint32_t height, uint32_t stride,
                                     int32_t offset, int32_t count, int16x8x2_t pels[UCMaxKernelSize])
{
    for (int32_t i = 0; i < count; ++i) {
        const int32_t rowIdx = clampS32(offset + i, 0, (int32_t)height - 1);
        pels[i].val[0] = U8ToS16(vget_low_u8(vld1q_u8(&in[rowIdx * stride])));
        pels[i].val[1] = U8ToS16(vget_high_u8(vld1q_u8(&in[rowIdx * stride])));
    }
}

/*!
 * Loads kernel-length rows of initial upscale input data ensuring that edge extension
 * is performed.
 *
 * \param  in       The input source surface to load from.
 * \param  height   The height of the input surface being loaded.
 * \param  stride   The stride of the input surface being loaded.
 * \param  offset   The row offset to start loading from.
 * \param  count    The number of rows to load in.
 * \param  pels     The destination to load the pixels into.
 */
static inline void verticalGetPelsN16(const uint8_t* in, uint32_t height, uint32_t stride,
                                      int32_t offset, int32_t count, int16x8x2_t pels[UCMaxKernelSize])
{
    const int16_t* in16 = (const int16_t*)in;

    for (int32_t i = 0; i < count; ++i) {
        const int32_t rowIdx = clampS32(offset + i, 0, (int32_t)height - 1);
        const size_t rowOffset = (size_t)rowIdx * stride;
        pels[i].val[0] = vld1q_s16(&in16[rowOffset]);
        pels[i].val[1] = vld1q_s16(&in16[rowOffset + 8]);
    }
}

/*!
 * Loads the next row of upscale input data by shuffling the pels down 1, and loading
 * next row into the last entry. This function ensures that edge extension is performed.
 *
 * \param  in       The input source surface to load from.
 * \param  height   The height of the input surface being loaded.
 * \param  stride   The stride of the input surface being loaded.
 * \param  offset   The row to load from.
 * \param  count    The number of rows loaded in.
 * \param  pels     The destination to load the pixels into.
 */
static inline void verticalGetNextPelsU8(const uint8_t* in, uint32_t height, uint32_t stride,
                                         int32_t offset, int32_t count, int16x8x2_t pels[UCMaxKernelSize])
{
    /* Shift pels */
    for (int32_t i = 1; i < count; ++i) {
        pels[i - 1] = pels[i];
    }

    const size_t rowIdx = (size_t)clampS32(offset + count - 1, 0, (int32_t)height - 1);
    pels[count - 1].val[0] = U8ToS16(vget_low_u8(vld1q_u8(&in[rowIdx * stride])));
    pels[count - 1].val[1] = U8ToS16(vget_high_u8(vld1q_u8(&in[rowIdx * stride])));
}

static inline void verticalGetNextPelsN16(const uint8_t* in, uint32_t height, uint32_t stride,
                                          int32_t offset, int32_t count, int16x8x2_t pels[UCMaxKernelSize])
{
    /* Shift pels */
    for (int32_t i = 1; i < count; ++i) {
        pels[i - 1] = pels[i];
    }

    const int32_t rowIndex = clampS32(offset + count - 1, 0, (int32_t)height - 1);
    const size_t rowOffset = (size_t)rowIndex * stride;
    const int16_t* in16 = (const int16_t*)in;
    pels[count - 1].val[0] = vld1q_s16(&in16[rowOffset]);
    pels[count - 1].val[1] = vld1q_s16(&in16[rowOffset + 8]);
}

/*!
 * Performs vertical convolution of input pels applying the kernel and returns the
 * result.
 *
 * This generates 16-pixels worth of output.
 *
 * \param pels            The pixels to upscale from.
 * \param result          Place to store the resultant 16-pixels.
 * \param kernel          The kernel to upscale with
 * \param kernelLength    The length of kernel.
 * \param result          Place to store the resultant 16-pixels.
 */
static inline void verticalConvolveS16(int16x8x2_t pels[UCMaxKernelSize], const int16_t* kernel,
                                       int32_t kernelLength, int16x8x2_t* result)
{
    int32x4_t values[4];
    const int16x8_t minV = vdupq_n_s16(-16384); /* see saturateS15 for explanation */
    const int16x8_t maxV = vdupq_n_s16(16383);

    /* Prime with initial multiply */
    values[0] = vmull_n_s16(vget_low_s16(pels[0].val[0]), kernel[0]);
    values[1] = vmull_n_s16(vget_high_s16(pels[0].val[0]), kernel[0]);
    values[2] = vmull_n_s16(vget_low_s16(pels[0].val[1]), kernel[0]);
    values[3] = vmull_n_s16(vget_high_s16(pels[0].val[1]), kernel[0]);

    /* Multiply and accumulate the rest of the kernel */
    for (int32_t i = 1; i < kernelLength; ++i) {
        values[0] = vmlal_n_s16(values[0], vget_low_s16(pels[i].val[0]), kernel[i]);
        values[1] = vmlal_n_s16(values[1], vget_high_s16(pels[i].val[0]), kernel[i]);
        values[2] = vmlal_n_s16(values[2], vget_low_s16(pels[i].val[1]), kernel[i]);
        values[3] = vmlal_n_s16(values[3], vget_high_s16(pels[i].val[1]), kernel[i]);
    }

    /* Scale back & combine with a shift from int32_t to unsigned N-bits. */
    result->val[0] = vcombine_s16(vqrshrn_n_s32(values[0], UCInverseShift),
                                  vqrshrn_n_s32(values[1], UCInverseShift));

    result->val[1] = vcombine_s16(vqrshrn_n_s32(values[2], UCInverseShift),
                                  vqrshrn_n_s32(values[3], UCInverseShift));

    /* Saturate (clamp) to +/- 2^14 */
    result->val[0] = vmaxq_s16(vminq_s16(result->val[0], maxV), minV);
    result->val[1] = vmaxq_s16(vminq_s16(result->val[1], maxV), minV);
}

/*! \brief Vertical upscaling of 16 columns. */
static void verticalU8NEON(const uint8_t* in, uint32_t inStride, uint8_t* out, uint32_t outStride,
                           uint32_t y, uint32_t rows, uint32_t height, const LdeKernel* kernel)
{
    const int16_t* kernelFwd = kernel->coeffs[0];
    const int16_t* kernelRev = kernel->coeffs[1];
    const int32_t kernelLength = (int32_t)kernel->length;

    uint32_t rowIndex = 0;
    const uint32_t outSkip = 2 * outStride;
    int16_t* out0 = (int16_t*)out + ((size_t)y * outSkip);
    int16_t* out1 = (int16_t*)out + ((size_t)y * outSkip) + outStride;
    int32_t loadOffset = (int32_t)y - (kernelLength / 2);
    int16x8x2_t pels[UCMaxKernelSize];
    int16x8x2_t result;

    /* Prime rows. */
    verticalGetPelsU8(in, height, inStride, loadOffset, kernelLength, pels);
    loadOffset += 1;

    for (rowIndex = 0; rowIndex < rows; ++rowIndex) {
        /* Reverse filter */
        verticalConvolveS16(pels, kernelRev, kernelLength, &result);
        vst1q_s16(out0, result.val[0]);
        vst1q_s16(out0 + 8, result.val[1]);

        /* Next input due to being off-pixel */
        verticalGetNextPelsU8(in, height, inStride, loadOffset, kernelLength, pels);
        loadOffset += 1;

        /* Forward filter */
        verticalConvolveS16(pels, kernelFwd, kernelLength, &result);
        vst1q_s16(out1, result.val[0]);
        vst1q_s16(out1 + 8, result.val[1]);

        out0 += outSkip;
        out1 += outSkip;
    }
}

/*! \brief S16 vertical upscaling of 16 columns. */
static void verticalS16NEON(const uint8_t* in, uint32_t inStride, uint8_t* out, uint32_t outStride,
                            uint32_t y, uint32_t rows, uint32_t height, const LdeKernel* kernel)
{
    const int16_t* kernelFwd = kernel->coeffs[0];
    const int16_t* kernelRev = kernel->coeffs[1];
    const int32_t kernelLength = (int32_t)kernel->length;

    uint32_t rowIndex = 0;
    const uint32_t outSkip = 2 * outStride;
    int16_t* out16 = (int16_t*)out;
    int16_t* out0 = out16 + ((size_t)y * outSkip);
    int16_t* out1 = out0 + outStride;
    int32_t loadOffset = (int32_t)y - (kernelLength / 2);
    int16x8x2_t pels[UCMaxKernelSize];
    int16x8x2_t result;

    /* Prime rows. */
    verticalGetPelsN16(in, height, inStride, loadOffset, kernelLength, pels);
    loadOffset += 1;

    for (rowIndex = 0; rowIndex < rows; ++rowIndex) {
        /* Reverse filter */
        verticalConvolveS16(pels, kernelRev, kernelLength, &result);
        vst1q_s16(out0, result.val[0]);
        vst1q_s16(out0 + 8, result.val[1]);

        /* Next input due to being off-pixel */
        verticalGetNextPelsN16(in, height, inStride, loadOffset, kernelLength, pels);
        loadOffset += 1;

        /* Forward filter */
        verticalConvolveS16(pels, kernelFwd, kernelLength, &result);
        vst1q_s16(out1, result.val[0]);
        vst1q_s16(out1 + 8, result.val[1]);

        out0 += outSkip;
        out1 += outSkip;
    }
}

/*! \brief U16 vertical upscaling of 16 columns. */
static inline void verticalU16NEON(const uint8_t* in, uint32_t inStride, uint8_t* out,
                                   uint32_t outStride, uint32_t y, uint32_t rows, uint32_t height,
                                   const LdeKernel* kernel, uint16_t shift)
{
    const int16_t* kernelFwd = kernel->coeffs[0];
    const int16_t* kernelRev = kernel->coeffs[1];
    const int32_t kernelLength = (int32_t)kernel->length;

    uint32_t rowIndex = 0;
    const uint32_t outSkip = 2 * outStride;
    int16_t* out16 = (int16_t*)out;
    int16_t* out0 = out16 + ((size_t)y * outSkip);
    int16_t* out1 = out0 + outStride;
    int32_t loadOffset = (int32_t)y - (kernelLength / 2);
    int16x8x2_t pels[UCMaxKernelSize];
    int16x8x2_t result;

    /* Prime rows. */
    verticalGetPelsN16(in, height, inStride, loadOffset, kernelLength, pels);
    for (uint8_t i = 0; i < kernelLength; i++) {
        pels[i] = UNtoS16(pels[i], shift);
    }

    loadOffset += 1;

    for (rowIndex = 0; rowIndex < rows; ++rowIndex) {
        /* Reverse filter */
        verticalConvolveS16(pels, kernelRev, kernelLength, &result);
        vst1q_s16(out0, result.val[0]);
        vst1q_s16(out0 + 8, result.val[1]);

        /* Next input due to being off-pixel */
        verticalGetNextPelsN16(in, height, inStride, loadOffset, kernelLength, pels);
        pels[kernelLength - 1] = UNtoS16(pels[kernelLength - 1], shift);
        loadOffset += 1;

        /* Forward filter */
        verticalConvolveS16(pels, kernelFwd, kernelLength, &result);
        vst1q_s16(out1, result.val[0]);
        vst1q_s16(out1 + 8, result.val[1]);

        out0 += outSkip;
        out1 += outSkip;
    }
}

/*! \brief U10 vertical upscaling of 16 columns. */
void verticalU10NEON(const uint8_t* in, uint32_t inStride, uint8_t* out, uint32_t outStride,
                     uint32_t y, uint32_t rows, uint32_t height, const LdeKernel* kernel)
{
    verticalU16NEON(in, inStride, out, outStride, y, rows, height, kernel, 5);
}

/*! \brief U12 vertical upscaling of 16 columns. */
void verticalU12NEON(const uint8_t* in, uint32_t inStride, uint8_t* out, uint32_t outStride,
                     uint32_t y, uint32_t rows, uint32_t height, const LdeKernel* kernel)
{
    verticalU16NEON(in, inStride, out, outStride, y, rows, height, kernel, 3);
}

/*! \brief U14 vertical upscaling of 16 columns. */
void verticalU14NEON(const uint8_t* in, uint32_t inStride, uint8_t* out, uint32_t outStride,
                     uint32_t y, uint32_t rows, uint32_t height, const LdeKernel* kernel)
{
    verticalU16NEON(in, inStride, out, outStride, y, rows, height, kernel, 1);
}

/*------------------------------------------------------------------------------*/

/* clang-format off */

/* Conversion is not currently supported in NEON, will fallthrough to scalar for 'relative bit depths' */
static const UpscaleHorizontalFunction kHorizontalPlanarFunctionTable[LdpFPCount][LdpFPCount] = {
	/* src  /  U8,                       U10,                    U12,                    U14,                    S8.7,                    S10.5,                   S12.3,                   S14.1 */
	/* U8  */ {horizontal1DU8PlanarNEON, NULL,                   NULL,                   NULL,                   NULL,                    NULL,                    NULL,                    NULL},
	/* U10 */ {NULL,                     horizontalUNPlanarNEON, NULL,                   NULL,                   NULL,                    NULL,                    NULL,                    NULL},
	/* U12 */ {NULL,                     NULL,                   horizontalUNPlanarNEON, NULL,                   NULL,                    NULL,                    NULL,                    NULL},
	/* U14 */ {NULL,                     NULL,                   NULL,                   horizontalUNPlanarNEON, NULL,                    NULL,                    NULL,                    NULL},
	/* S8  */ {horizontal2DU8PlanarNEON, NULL,                   NULL,                   NULL,                   horizontalS16PlanarNEON, horizontalS16PlanarNEON, horizontalS16PlanarNEON, horizontalS16PlanarNEON},
	/* S10 */ {NULL,                     horizontalUNPlanarNEON, NULL,                   NULL,                   horizontalS16PlanarNEON, horizontalS16PlanarNEON, horizontalS16PlanarNEON, horizontalS16PlanarNEON},
    /* S12 */ {NULL,                     NULL,                   horizontalUNPlanarNEON, NULL,                   horizontalS16PlanarNEON, horizontalS16PlanarNEON, horizontalS16PlanarNEON, horizontalS16PlanarNEON},
    /* S14 */ {NULL,                     NULL,                   NULL,                   horizontalUNPlanarNEON, horizontalS16PlanarNEON, horizontalS16PlanarNEON, horizontalS16PlanarNEON, horizontalS16PlanarNEON},
};

/* kVerticalFunctionTable[fp] */
static const UpscaleVerticalFunction kVerticalFunctionTable[LdpFPCount] = {
	verticalU8NEON,  /* U8 */
	verticalU10NEON, /* U10 */
	verticalU12NEON, /* U12 */
	verticalU14NEON, /* U14 */
	verticalS16NEON, /* S8.7 */
	verticalS16NEON, /* S10.5 */
	verticalS16NEON, /* S12.3 */
	verticalS16NEON, /* S14.1 */
};

/* clang-format on */

/*------------------------------------------------------------------------------*/

UpscaleHorizontalFunction upscaleGetHorizontalFunctionNEON(Interleaving interleaving,
                                                           LdpFixedPoint srcFP, LdpFixedPoint dstFP)
{
    if (interleaving == ILNone) {
        return kHorizontalPlanarFunctionTable[srcFP][dstFP];
    }
    if (interleaving == ILNV12 && dstFP == LdpFPU8) {
        if (srcFP == dstFP) {
            return horizontal1DU8NV12NEON;
        }
        return horizontal2DU8NV12NEON;
    }

    return NULL;
}

UpscaleVerticalFunction upscaleGetVerticalFunctionNEON(LdpFixedPoint srcFP, LdpFixedPoint dstFP)
{
    /* Conversion is not currently supported in SIMD. */
    if (bitdepthFromFixedPoint(srcFP) != bitdepthFromFixedPoint(dstFP)) {
        return NULL;
    }

    return kVerticalFunctionTable[srcFP];
}

/*------------------------------------------------------------------------------*/

#else

UpscaleHorizontalFunction upscaleGetHorizontalFunctionNEON(Interleaving ilv, LdpFixedPoint srcFP,
                                                           LdpFixedPoint dstFP)
{
    VNUnused(ilv);
    VNUnused(srcFP);
    VNUnused(dstFP);
    return NULL;
}

UpscaleVerticalFunction upscaleGetVerticalFunctionNEON(LdpFixedPoint srcFP, LdpFixedPoint dstFP)
{
    VNUnused(srcFP);
    VNUnused(dstFP);
    return NULL;
}

#endif
