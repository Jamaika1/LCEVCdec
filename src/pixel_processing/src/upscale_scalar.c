/* Copyright (c) V-Nova International Limited 2022-2025. All rights reserved.
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

#include "upscale_scalar.h"
//
#include <LCEVC/common/limit.h>
#include <LCEVC/pixel_processing/dither.h>
//
#include <assert.h>
#include <stdint.h>

/*------------------------------------------------------------------------------*/

/*! Constants for upscaling. */
enum UpscaleConstants
{
    UCShift = 14,
    UCCeilRounding = 1 << (UCShift - 1),
    UCS16Midpoint = 0x4000,
};

/*! Performs the inversion shift for the calculated convolution result.
 *
 * The result is unsaturated, as it is expected to be saturated by the caller.
 */
static inline int32_t shiftResultUnsaturated(int32_t value)
{
    const int32_t result = (value + UCCeilRounding) >> UCShift;
    return result;
}

/*! Perform the inversion shift for the calculated convolution result and saturate to 15-bit.
 *
 * 15 bits is the saturation performed in the middle of upscale operations, to ensure that residuals
 * (which have, at most, 16 bits) can correct all upscaling differences.
 */
static inline int16_t shiftResultSaturated(int32_t value)
{
    value = shiftResultUnsaturated(value);
    return saturateS15(value);
}

/*! Convert an unsigned 8-14 bit value to signed 16-bit.
 *
 * \param value         Value to convert.
 * \param shift         Appropriate shift for input bit-depth.
 *
 * \return              Converted signed 16-bit value
 */
static inline int16_t UNToS16(uint16_t value, uint16_t shift)
{
    return (int16_t)(((int32_t)value << shift) - UCS16Midpoint);
}

/*! Convert a signed 16-bit to unsigned 8-14 bit value.
 *
 * \param value         Value to convert.
 * \param shift         Appropriate shift for output bit-depth.
 * \param offset        Appropriate pre-shift offset for output bit-depth.
 * \param midpoint      Midpoint value of output bit-depth.
 *
 * \return              Converted signed 16-bit value
 */
static inline int32_t S16ToUN(int32_t value, uint16_t shift, uint16_t offset, uint16_t midpoint)
{
    return ((value + offset) >> shift) + midpoint;
}

/*------------------------------------------------------------------------------*/

static inline int32_t getPelsOffset(int32_t offset, uint32_t length, int32_t pelsLength)
{
    return clampS32(offset + pelsLength - 1, 0, (int32_t)length - 1);
}

static inline void getPelsU8ToS16(const uint8_t* in, uint32_t inSize, uint32_t stride,
                                  int32_t offset, int16_t* pels, int32_t count)
{
    for (int32_t i = 0; i < count; i++) {
        const int32_t rowIndex = clampS32(offset + i, 0, (int32_t)inSize - 1);

        pels[i] = UNToS16(in[(size_t)rowIndex * stride], 7);
    }
}

static inline void getNextPelsU8ToS16(const uint8_t* in, uint32_t inSize, uint32_t stride,
                                      int32_t offset, int16_t* pels, int32_t pelsLength)
{
    /* Shuffle pels */
    for (int32_t i = 1; i < pelsLength; i++) {
        pels[i - 1] = pels[i];
    }

    /* Load the new one */
    const size_t rowIndex = (size_t)getPelsOffset(offset, inSize, pelsLength);
    pels[pelsLength - 1] = UNToS16(in[rowIndex * stride], 7);
}

static inline void getPelsUNToS16(const uint16_t* in, uint32_t inSize, uint32_t stride,
                                  int32_t offset, int16_t* pels, int32_t pelsLength, uint16_t shift)
{
    for (int32_t i = 0; i < pelsLength; i++) {
        const int32_t rowIndex = clampS32(offset + i, 0, (int32_t)inSize - 1);
        pels[i] = UNToS16(in[(size_t)rowIndex * stride], shift);
    }
}

static inline void getNextPelsUNToS16(const uint16_t* in, uint32_t inSize, uint32_t stride,
                                      int32_t offset, int16_t* pels, int32_t pelsLength, uint16_t shift)
{
    /* Shuffle pels */
    for (int32_t i = 1; i < pelsLength; i++) {
        pels[i - 1] = pels[i];
    }

    /* Load the new one */
    const int32_t rowIndex = getPelsOffset(offset, inSize, pelsLength);
    pels[pelsLength - 1] = UNToS16(in[(size_t)rowIndex * stride], shift);
}

static inline void getPelsS16(const int16_t* in, uint32_t inSize, uint32_t stride, int32_t offset,
                              int16_t* pels, int32_t pelsLength)
{
    for (int32_t i = 0; i < pelsLength; i++) {
        const int32_t rowIndex = clampS32(offset + i, 0, (int32_t)inSize - 1);
        pels[i] = in[(size_t)rowIndex * stride];
    }
}

static inline void getNextPelsS16(const int16_t* in, uint32_t inSize, uint32_t stride,
                                  int32_t offset, int16_t* pels, int32_t pelsLength)
{
    /* Shuffle pels */
    for (int32_t i = 1; i < pelsLength; i++) {
        pels[i - 1] = pels[i];
    }

    /* Load the new one */
    const int32_t rowIndex = getPelsOffset(offset, inSize, pelsLength);
    pels[pelsLength - 1] = in[(size_t)rowIndex * stride];
}

/*------------------------------------------------------------------------------*/

/*!
 * Perform horizontal upscaling of 2 lines at a time for an interleaved unsigned 8-bit surface of up
 * to 2 channels.
 *
 * This function supports multiple features:
 *
 *    - Off-pixel convolution upscaling
 *    - 1D predicted average
 *    - 2D predicted average for when horizontal is the second upscale pass (hence
 *      requiring 2 lines at a time).
 *    - dithering application
 *    - simultaneous upscaling of interleaved planes for NV12.
 *
 * \param dither        Dither data (NULL for no dithering)
 * \param in            Byte pointers to the 2 input rows to upscale from.
 * \param out           Byte pointers to the 2 output rows to upscale to.
 * \param base          Byte pointers to 2 rows to use for PA, if the second pointer is NULL then it
 *                      is assumed that 2D PA is expected.
 * \param width         The pixel width of the input rows.
 * \param xStart        The x-coordinate to start upscaling from.
 * \param xEnd          The x-coordinate to end upscaling from.
 * \param params        Upscale param struct including kernel, channels constants and bit-depth
 *                      conversion constants
 */
void horizontalU8Scalar(LdppDitherSlice* dither, const uint8_t* in[2], uint8_t* out[2],
                        const uint8_t* base[2], uint32_t width, uint32_t xStart, uint32_t xEnd,
                        LdppHorizontalUpscaleParams* params)
{
    const int16_t* kernelFwd = params->kernel->coeffs[0];
    const int16_t* kernelRev = params->kernel->coeffs[1];
    const int32_t kernelLength = (int32_t)params->kernel->length;
    const uint8_t channelCount = params->channelCount;
    const uint8_t channelMap[2] = {params->channelMap[0], params->channelMap[1]};
    const bool is2D = params->is2D;
    const bool paEnabled = (*base != NULL);
    const uint16_t* ditherBuffer = NULL;
    const int16_t* channelInS16[2][2] = {{0}};
    const uint8_t* channelInU8[2][2] = {{0}};
    const size_t initialBaseOffset = (size_t)xStart * channelCount;
    const uint8_t* basePtr[2] = {NULL, NULL};
    if (*base != NULL) {
        basePtr[0] = &base[0][initialBaseOffset];
        basePtr[1] = &base[1][initialBaseOffset];
    }
    int16_t pels[2][2][8];
    int32_t values[4];
    int32_t channelLoadOffset[2] = {(int32_t)xStart - (kernelLength >> 1),
                                    (int32_t)xStart - (kernelLength >> 1)};
    const int32_t initialStoreOffset = (int32_t)((xStart << 1) * channelCount);
    int32_t channelStoreOffset[2] = {initialStoreOffset, initialStoreOffset + 1};

    assert((channelCount > 0) && (channelCount <= 2));

    /* Prime pels with initial values. */
    for (uint8_t channelIdx = 0; channelIdx < channelCount; ++channelIdx) {
        const uint8_t channel = channelMap[channelIdx];

        /* For channels where they map load/store to another channels PELs then skip the initial load. */
        if (channel == channelIdx) {
            const int32_t loadOffset = channelLoadOffset[channel];
            const uint32_t skip = params->channelSkip[channel];
            assert(skip > 0);

            if (is2D) {
                channelInS16[channel][0] = (const int16_t*)in[0] + channelIdx;
                channelInS16[channel][1] = (const int16_t*)in[1] + channelIdx;
                getPelsS16(channelInS16[channel][0], width, skip, loadOffset, pels[channel][0], kernelLength);
                getPelsS16(channelInS16[channel][1], width, skip, loadOffset, pels[channel][1], kernelLength);
            } else {
                channelInU8[channel][0] = in[0] + channelIdx;
                channelInU8[channel][1] = in[1] + channelIdx;
                getPelsU8ToS16(channelInU8[channel][0], width, skip, loadOffset, pels[channel][0],
                               kernelLength);
                getPelsU8ToS16(channelInU8[channel][1], width, skip, loadOffset, pels[channel][1],
                               kernelLength);
            }

            channelLoadOffset[channel]++;
        }
    }

    /* Prepare dither buffer containing enough values for 2 fully upscaled rows
     * for each channel */
    if (dither != NULL) {
        ditherBuffer = ldppDitherGetBuffer(dither, (xEnd - xStart) * (size_t)channelCount * 4);
    }

    for (uint32_t x = xStart; x < xEnd; ++x) {
        for (uint32_t channelIdx = 0; channelIdx < channelCount; ++channelIdx) {
            const uint32_t channel = channelMap[channelIdx];
            const int32_t loadOffset = channelLoadOffset[channel];
            const int32_t storeOffset = channelStoreOffset[channel];
            const uint32_t skip = params->channelSkip[channel];
            memset(values, 0, sizeof(values));

            /* Reverse filter */
            for (int32_t i = 0; i < kernelLength; ++i) {
                values[0] += kernelRev[i] * pels[channel][0][i];
                values[2] += kernelRev[i] * pels[channel][1][i];
            }

            /* Next input after reverse-phase as we are off pixel */
            if (is2D) {
                getNextPelsS16(channelInS16[channel][0], width, skip, loadOffset, pels[channel][0],
                               kernelLength);
                getNextPelsS16(channelInS16[channel][1], width, skip, loadOffset, pels[channel][1],
                               kernelLength);
            } else {
                getNextPelsU8ToS16(channelInU8[channel][0], width, skip, loadOffset,
                                   pels[channel][0], kernelLength);
                getNextPelsU8ToS16(channelInU8[channel][1], width, skip, loadOffset,
                                   pels[channel][1], kernelLength);
            }

            /* Forward filter */
            for (int32_t i = 0; i < kernelLength; ++i) {
                values[1] += kernelFwd[i] * pels[channel][0][i];
                values[3] += kernelFwd[i] * pels[channel][1][i];
            }

            values[0] = shiftResultSaturated(values[0]);
            values[1] = shiftResultSaturated(values[1]);
            values[2] = shiftResultSaturated(values[2]);
            values[3] = shiftResultSaturated(values[3]);

            /* Apply predicted average */
            if (paEnabled && is2D) {
                const int16_t basePel = UNToS16(*basePtr[0]++, params->shift);
                const int32_t avg = basePel - ((values[0] + values[1] + values[2] + values[3] + 2) >> 2);

                values[0] += avg;
                values[1] += avg;
                values[2] += avg;
                values[3] += avg;
            } else if (paEnabled && !is2D) {
                const int16_t basePel0 = UNToS16(*basePtr[0]++, params->shift);
                const int16_t basePel1 = UNToS16(*basePtr[1]++, params->shift);
                const int32_t avg0 = basePel0 - ((values[0] + values[1] + 1) >> 1);
                const int32_t avg1 = basePel1 - ((values[2] + values[3] + 1) >> 1);

                values[0] += avg0;
                values[1] += avg0;
                values[2] += avg1;
                values[3] += avg1;
            }

            values[0] = S16ToUN(values[0], params->shift, params->offset, params->midpoint);
            values[1] = S16ToUN(values[1], params->shift, params->offset, params->midpoint);
            values[2] = S16ToUN(values[2], params->shift, params->offset, params->midpoint);
            values[3] = S16ToUN(values[3], params->shift, params->offset, params->midpoint);

            /* Apply dithering */
            if (ditherBuffer) {
                ldppDitherApply(&values[0], &ditherBuffer, 0, dither->strength);
                ldppDitherApply(&values[1], &ditherBuffer, 0, dither->strength);
                ldppDitherApply(&values[2], &ditherBuffer, 0, dither->strength);
                ldppDitherApply(&values[3], &ditherBuffer, 0, dither->strength);
            }

            out[0][storeOffset] = saturateU8(values[0]);
            out[0][storeOffset + skip] = saturateU8(values[1]);
            out[1][storeOffset] = saturateU8(values[2]);
            out[1][storeOffset + skip] = saturateU8(values[3]);

            channelStoreOffset[channel] += (int32_t)(skip * 2);
            channelLoadOffset[channel]++;
        }
    }
}

/*!
 * Perform horizontal upscaling of 2 lines at a time for an interleaved signed 16-bit surface of up
 * to 2 channels.
 *
 * This function supports multiple features:
 *
 *    - Off-pixel convolution upscaling
 *    - 1D predicted average
 *    - 2D predicted average for when horizontal is the second upscale pass (hence
 *      requiring 2 lines at a time).
 *    - dithering application
 *
 * \param dither        Dither data (NULL for no dithering)
 * \param in            Byte pointers to the 2 input rows to upscale from.
 * \param out           Byte pointers to the 2 output rows to upscale to.
 * \param base          Byte pointers to 2 rows to use for PA, if the second pointer is NULL then it
 *                      is assumed that 2D PA is expected.
 * \param width         The pixel width of the input rows.
 * \param xStart        The x-coordinate to start upscaling from.
 * \param xEnd          The x-coordinate to end upscaling from.
 * \param params        Upscale param struct including kernel, channels constants and bit-depth
 *                      conversion constants
 */
void horizontalS16Scalar(LdppDitherSlice* dither, const uint8_t* in[2], uint8_t* out[2],
                         const uint8_t* base[2], uint32_t width, uint32_t xStart, uint32_t xEnd,
                         LdppHorizontalUpscaleParams* params)
{
    const int16_t* kernelFwd = params->kernel->coeffs[0];
    const int16_t* kernelRev = params->kernel->coeffs[1];
    const int32_t kernelLength = (int32_t)params->kernel->length;
    const bool paEnabled = (base[0] != NULL);
    const bool paEnabled1D = paEnabled && (base[1] != NULL);
    const uint16_t* ditherBuffer = NULL;
    const int16_t* inS16[2] = {(const int16_t*)in[0], (const int16_t*)in[1]};
    const int16_t* baseS16[2] = {NULL, NULL};
    if (*base != NULL) {
        baseS16[0] = &((const int16_t*)base[0])[xStart];
        baseS16[1] = &((const int16_t*)base[1])[xStart];
    }
    int32_t values[4] = {0};
    int16_t pels[2][8];
    int16_t* outS16[2] = {(int16_t*)out[0], (int16_t*)out[1]};
    int32_t loadOffset = (int32_t)xStart - (kernelLength >> 1);
    int32_t storeOffset = (int32_t)(xStart << 1);

    /* Prime pels with initial values. */
    getPelsS16(inS16[0], width, 1, loadOffset, pels[0], kernelLength);
    getPelsS16(inS16[1], width, 1, loadOffset, pels[1], kernelLength);
    loadOffset++;

    /* Prepare dither buffer containing enough values for 2 fully upscaled rows for each channel */
    if (dither != NULL) {
        ditherBuffer = ldppDitherGetBuffer(dither, (xEnd - xStart) * 4);
    }

    for (uint32_t x = xStart; x < xEnd; ++x) {
        memset(values, 0, sizeof(values));

        /* Reverse filter */
        for (int32_t i = 0; i < kernelLength; ++i) {
            values[0] += (kernelRev[i] * pels[0][i]);
            values[2] += (kernelRev[i] * pels[1][i]);
        }

        /* Next input after reverse-phase as we are off pixel */
        getNextPelsS16(inS16[0], width, 1, loadOffset, pels[0], kernelLength);
        getNextPelsS16(inS16[1], width, 1, loadOffset, pels[1], kernelLength);

        /* Forward filter */
        for (int32_t i = 0; i < kernelLength; ++i) {
            values[1] += (kernelFwd[i] * pels[0][i]);
            values[3] += (kernelFwd[i] * pels[1][i]);
        }

        values[0] = shiftResultSaturated(values[0]);
        values[1] = shiftResultSaturated(values[1]);
        values[2] = shiftResultSaturated(values[2]);
        values[3] = shiftResultSaturated(values[3]);

        /* Apply predicted average */
        if (paEnabled1D) {
            const int32_t avg0 = *baseS16[0]++ - ((values[0] + values[1] + 1) >> 1);
            const int32_t avg1 = *baseS16[1]++ - ((values[2] + values[3] + 1) >> 1);

            values[0] += avg0;
            values[1] += avg0;
            values[2] += avg1;
            values[3] += avg1;
        } else if (paEnabled) {
            const int32_t avg =
                *baseS16[0]++ - ((values[0] + values[1] + values[2] + values[3] + 2) >> 2);

            values[0] += avg;
            values[1] += avg;
            values[2] += avg;
            values[3] += avg;
        }

        /* Apply dithering */
        if (ditherBuffer) {
            ldppDitherApply(&values[0], &ditherBuffer, (uint8_t)params->shift, dither->strength);
            ldppDitherApply(&values[1], &ditherBuffer, (uint8_t)params->shift, dither->strength);
            ldppDitherApply(&values[2], &ditherBuffer, (uint8_t)params->shift, dither->strength);
            ldppDitherApply(&values[3], &ditherBuffer, (uint8_t)params->shift, dither->strength);
        }

        outS16[0][storeOffset] = saturateS16(values[0]);
        outS16[0][storeOffset + 1] = saturateS16(values[1]);
        outS16[1][storeOffset] = saturateS16(values[2]);
        outS16[1][storeOffset + 1] = saturateS16(values[3]);

        storeOffset += 2;
        loadOffset++;
    }
}

/*!
 * Perform horizontal upscaling of 2 lines at a time for an interleaved unsigned 16-bit surface of
 * up to 2 channels.
 *
 * This function supports multiple features:
 *
 *    - Off-pixel convolution upscaling
 *    - 1D predicted average
 *    - 2D predicted average for when horizontal is the second upscale pass (hence
 *      requiring 2 lines at a time).
 *    - dithering application
 *
 * \param dither        Dither data (NULL for no dithering)
 * \param in            Byte pointers to the 2 input rows to upscale from.
 * \param out           Byte pointers to the 2 output rows to upscale to.
 * \param base          Byte pointers to 2 rows to use for PA, if the second pointer is NULL then it
 *                      is assumed that 2D PA is expected.
 * \param width         The pixel width of the input rows.
 * \param xStart        The x-coordinate to start upscaling from.
 * \param xEnd          The x-coordinate to end upscaling from.
 * \param params        Upscale param struct including kernel, channels constants and bit-depth
 *                      conversion constants
 */
void horizontalUNScalar(LdppDitherSlice* dither, const uint8_t* in[2], uint8_t* out[2],
                        const uint8_t* base[2], uint32_t width, uint32_t xStart, uint32_t xEnd,
                        LdppHorizontalUpscaleParams* params)
{
    const int16_t* kernelFwd = params->kernel->coeffs[0];
    const int16_t* kernelRev = params->kernel->coeffs[1];
    const int32_t kernelLength = (int32_t)params->kernel->length;
    const bool is2D = params->is2D;
    const bool paEnabled = (*base != NULL);
    const uint16_t* ditherBuffer = NULL;
    const uint16_t* baseU16[2] = {NULL, NULL};
    if (*base != NULL) {
        baseU16[0] = &((const uint16_t*)base[0])[xStart];
        baseU16[1] = &((const uint16_t*)base[1])[xStart];
    }
    const int16_t* inS16[2] = {(const int16_t*)in[0], (const int16_t*)in[1]};
    const uint16_t* inU16[2] = {(const uint16_t*)in[0], (const uint16_t*)in[1]};
    uint16_t* outU16[2] = {(uint16_t*)out[0], (uint16_t*)out[1]};
    int16_t pels[2][8];
    int32_t values[4];
    int32_t loadOffset = (int32_t)xStart - (kernelLength >> 1);
    int32_t storeOffset = (int32_t)(xStart << 1);

    /* Prime pels with initial values. */
    if (is2D) {
        getPelsS16(inS16[0], width, 1, loadOffset, pels[0], kernelLength);
        getPelsS16(inS16[1], width, 1, loadOffset, pels[1], kernelLength);
    } else {
        getPelsUNToS16(inU16[0], width, 1, loadOffset, pels[0], kernelLength, params->shift);
        getPelsUNToS16(inU16[1], width, 1, loadOffset, pels[1], kernelLength, params->shift);
    }
    loadOffset += 1;

    /* Prepare dither buffer containing enough values for 2 fully upscaled rows for each channel */
    if (dither != NULL) {
        ditherBuffer = ldppDitherGetBuffer(dither, (size_t)(xEnd - xStart) * 4);
    }

    for (uint32_t x = xStart; x < xEnd; ++x) {
        memset(values, 0, sizeof(int32_t) * 4);

        /* Reverse filter */
        for (int32_t i = 0; i < kernelLength; ++i) {
            values[0] += kernelRev[i] * pels[0][i];
            values[2] += kernelRev[i] * pels[1][i];
        }

        /* Next input after reverse-phase as we are off pixel */
        if (is2D) {
            getNextPelsS16(inS16[0], width, 1, loadOffset, pels[0], kernelLength);
            getNextPelsS16(inS16[1], width, 1, loadOffset, pels[1], kernelLength);
        } else {
            getNextPelsUNToS16(inU16[0], width, 1, loadOffset, pels[0], kernelLength, params->shift);
            getNextPelsUNToS16(inU16[1], width, 1, loadOffset, pels[1], kernelLength, params->shift);
        }

        /* Forward filter */
        for (int32_t i = 0; i < kernelLength; ++i) {
            values[1] += kernelFwd[i] * pels[0][i];
            values[3] += kernelFwd[i] * pels[1][i];
        }

        values[0] = shiftResultSaturated(values[0]);
        values[1] = shiftResultSaturated(values[1]);
        values[2] = shiftResultSaturated(values[2]);
        values[3] = shiftResultSaturated(values[3]);

        /* Apply predicted average */
        if (paEnabled && is2D) {
            const int16_t basePel = UNToS16(*baseU16[0]++, params->shift);
            const int32_t avg = basePel - ((values[0] + values[1] + values[2] + values[3] + 2) >> 2);

            values[0] += avg;
            values[1] += avg;
            values[2] += avg;
            values[3] += avg;
        } else if (paEnabled && !is2D) {
            const int16_t basePel0 = UNToS16(*baseU16[0]++, params->shift);
            const int16_t basePel1 = UNToS16(*baseU16[1]++, params->shift);
            const int32_t avg0 = basePel0 - ((values[0] + values[1] + 1) >> 1);
            const int32_t avg1 = basePel1 - ((values[2] + values[3] + 1) >> 1);

            values[0] += avg0;
            values[1] += avg0;
            values[2] += avg1;
            values[3] += avg1;
        }

        values[0] = S16ToUN(values[0], params->shift, params->offset, params->midpoint);
        values[1] = S16ToUN(values[1], params->shift, params->offset, params->midpoint);
        values[2] = S16ToUN(values[2], params->shift, params->offset, params->midpoint);
        values[3] = S16ToUN(values[3], params->shift, params->offset, params->midpoint);

        /* Apply dithering */
        if (ditherBuffer) {
            ldppDitherApply(&values[0], &ditherBuffer, 0, dither->strength);
            ldppDitherApply(&values[1], &ditherBuffer, 0, dither->strength);
            ldppDitherApply(&values[2], &ditherBuffer, 0, dither->strength);
            ldppDitherApply(&values[3], &ditherBuffer, 0, dither->strength);
        }

        outU16[0][storeOffset] = saturateUN(values[0], params->maxValue);
        outU16[0][storeOffset + 1] = saturateUN(values[1], params->maxValue);
        outU16[1][storeOffset] = saturateUN(values[2], params->maxValue);
        outU16[1][storeOffset + 1] = saturateUN(values[3], params->maxValue);

        storeOffset += 2;
        loadOffset += 1;
    }
}

/*!
 * Perform vertical upscaling of 2 columns at a time for unsigned 8-bit surfaces.
 *
 * \param in           Byte pointer to the input surface data pointer offset by the first column to
 *                     upscale from.
 * \param inStride     The pixel stride of the input surface.
 * \param out          Byte pointer to the output surface data pointer offset by the first column
 *                     to upscale to.
 * \param outStride    The pixel stride of the output surface.
 * \param y            The y coordinate to start upscaling from on the input surface.
 * \param rows         The number of rows in the column to upscale starting from y.
 * \param height       The pixel height of the input surface.
 * \param kernel       The kernel to perform convolution upscaling with.
 *
 * \note To reiterate, the `in` and `out` pointers are pointers to the allocated surface, offset by
 *       the pixel column that you're upscaling from/to.
 *
 *       These pointers are then offset by (y * <>_stride). This is because the `in` surface is
 *       indexed like so:
 *           index_range = clamp([y - (kernel_length / 2) <-> y + rows + (kernel_length / 2)],
 *                               0,
 *                               height - 1);
 */
static void verticalU8(const uint8_t* in, uint32_t inStride, uint8_t* out, uint32_t outStride,
                       uint32_t y, uint32_t rows, uint32_t height, const LdeKernel* kernel)
{
    int16_t pels[2][8];
    const int16_t* kernelFwd = kernel->coeffs[0];
    const int16_t* kernelRev = kernel->coeffs[1];
    const int32_t kernelLength = (int32_t)kernel->length;
    int32_t values[4];
    const uint32_t outSkip = 2 * outStride;
    int16_t* outS16[2] = {(int16_t*)out + ((size_t)y * outSkip),
                          (int16_t*)out + ((size_t)y * outSkip) + outStride};
    int32_t loadOffset = (int32_t)y - (kernelLength >> 1);

    getPelsU8ToS16(in, height, inStride, loadOffset, pels[0], kernelLength);
    getPelsU8ToS16(in + 1, height, inStride, loadOffset, pels[1], kernelLength);
    loadOffset += 1;

    for (uint32_t rowIndex = 0; rowIndex < rows; ++rowIndex) {
        memset(values, 0, sizeof(int32_t) * 4);

        /* Reverse filter on the S16 pels */
        for (int32_t i = 0; i < kernelLength; ++i) {
            values[0] += (kernelRev[i] * pels[0][i]);
            values[1] += (kernelRev[i] * pels[1][i]);
        }

        /* Next input after reverse-phase as we are off pixel */
        getNextPelsU8ToS16(in, height, inStride, loadOffset, pels[0], kernelLength);
        getNextPelsU8ToS16(in + 1, height, inStride, loadOffset, pels[1], kernelLength);
        loadOffset += 1;

        /* Forward filter */
        for (int32_t i = 0; i < kernelLength; ++i) {
            values[2] += (kernelFwd[i] * pels[0][i]);
            values[3] += (kernelFwd[i] * pels[1][i]);
        }

        // Just like the S16 function, shift the S32 result to S16 output
        outS16[0][0] = shiftResultSaturated(values[0]);
        outS16[0][1] = shiftResultSaturated(values[1]);
        outS16[1][0] = shiftResultSaturated(values[2]);
        outS16[1][1] = shiftResultSaturated(values[3]);

        outS16[0] += outSkip;
        outS16[1] += outSkip;
    }
}

/*!
 * Perform vertical upscaling of 2 columns at a time for signed 16-bit surfaces.
 *
 * \param in           Byte pointer to the input surface data pointer offset by
 *                     the first column to upscale from.
 * \param inStride     The pixel stride of the input surface.
 * \param out          Byte pointer to the output surface data pointer offset by
 *                     the first column to upscale to.
 * \param outStride    The pixel stride of the output surface.
 * \param y            The y coordinate to start upscaling from on the input surface.
 * \param rows         The number of rows in the column to upscale starting from y.
 * \param height       The pixel height of the input surface.
 * \param kernel       The kernel to perform convolution upscaling with.
 *
 * \note See `upscale_vertical_u8` for more details.
 */
static void verticalS16(const uint8_t* in, uint32_t inStride, uint8_t* out, uint32_t outStride,
                        uint32_t y, uint32_t rows, uint32_t height, const LdeKernel* kernel)
{
    int16_t pels[2][8];
    const int16_t* inI16 = (const int16_t*)in;
    const int16_t* kernelFwd = kernel->coeffs[0];
    const int16_t* kernelRev = kernel->coeffs[1];
    const int32_t kernelLength = (int32_t)kernel->length;
    int32_t values[4];
    const uint32_t outSkip = 2 * outStride;
    int16_t* out0 = (int16_t*)out + ((size_t)y * outSkip);
    int16_t* out1 = out0 + outStride;
    int32_t loadOffset = (int32_t)y - (kernelLength / 2);

    getPelsS16(inI16, height, inStride, loadOffset, pels[0], kernelLength);
    getPelsS16(inI16 + 1, height, inStride, loadOffset, pels[1], kernelLength);
    loadOffset += 1;

    for (uint32_t rowIndex = 0; rowIndex < rows; ++rowIndex) {
        memset(values, 0, sizeof(int32_t) * 4);

        /* Reverse filter */
        for (int32_t i = 0; i < kernelLength; ++i) {
            values[0] += (kernelRev[i] * pels[0][i]);
            values[1] += (kernelRev[i] * pels[1][i]);
        }

        /* Next input after reverse-phase as we are off pixel */
        getNextPelsS16(inI16, height, inStride, loadOffset, pels[0], kernelLength);
        getNextPelsS16(inI16 + 1, height, inStride, loadOffset, pels[1], kernelLength);
        loadOffset += 1;

        /* Forward filter */
        for (int32_t i = 0; i < kernelLength; ++i) {
            values[2] += (kernelFwd[i] * pels[0][i]);
            values[3] += (kernelFwd[i] * pels[1][i]);
        }

        out0[0] = shiftResultSaturated(values[0]);
        out0[1] = shiftResultSaturated(values[1]);
        out1[0] = shiftResultSaturated(values[2]);
        out1[1] = shiftResultSaturated(values[3]);

        out0 += outSkip;
        out1 += outSkip;
    }
}

/*!
 * Perform vertical upscaling of 2 columns at a time for unsigned 16-bit surfaces.
 *
 * \param in           Byte pointer to the input surface data pointer offset by
 *                     the first column to upscale from.
 * \param inStride     The pixel stride of the input surface.
 * \param out          Byte pointer to the output surface data pointer offset by
 *                     the first column to upscale to.
 * \param outStride    The pixel stride of the output surface.
 * \param y            The y coordinate to start upscaling from on the input surface.
 * \param rows         The number of rows in the column to upscale starting from y.
 * \param height       The pixel height of the input surface.
 * \param kernel       The kernel to perform convolution upscaling with.
 * \param shift        Shift value from unsigned input bit-depth to signed 16-bit
 *
 * \note See `upscale_vertical_u8` for more details.
 */
static void verticalU16(const uint8_t* in, uint32_t inStride, uint8_t* out, uint32_t outStride, uint32_t y,
                        uint32_t rows, uint32_t height, const LdeKernel* kernel, const uint16_t shift)
{
    int16_t pels[2][8];
    const uint16_t* inU16 = (const uint16_t*)in;
    const int16_t* kernelFwd = kernel->coeffs[0];
    const int16_t* kernelRev = kernel->coeffs[1];
    const int32_t kernelLength = (int32_t)kernel->length;
    int32_t values[4];
    const uint32_t outSkip = 2 * outStride;
    int16_t* outS16[2] = {(int16_t*)out + ((size_t)y * outSkip),
                          (int16_t*)out + ((size_t)y * outSkip) + outStride};
    int32_t loadOffset = (int32_t)y - (kernelLength / 2);

    getPelsUNToS16(inU16, height, inStride, loadOffset, pels[0], kernelLength, shift);
    getPelsUNToS16(inU16 + 1, height, inStride, loadOffset, pels[1], kernelLength, shift);
    loadOffset += 1;

    for (uint32_t rowIndex = 0; rowIndex < rows; ++rowIndex) {
        memset(values, 0, sizeof(int32_t) * 4);

        /* Reverse filter */
        for (int32_t i = 0; i < kernelLength; ++i) {
            values[0] += (kernelRev[i] * pels[0][i]);
            values[1] += (kernelRev[i] * pels[1][i]);
        }

        /* Next input after reverse-phase as we are off pixel */
        getNextPelsUNToS16(inU16, height, inStride, loadOffset, pels[0], kernelLength, shift);
        getNextPelsUNToS16(inU16 + 1, height, inStride, loadOffset, pels[1], kernelLength, shift);
        loadOffset += 1;

        /* Forward filter */
        for (int32_t i = 0; i < kernelLength; ++i) {
            values[2] += (kernelFwd[i] * pels[0][i]);
            values[3] += (kernelFwd[i] * pels[1][i]);
        }

        outS16[0][0] = (int16_t)shiftResultUnsaturated(values[0]);
        outS16[0][1] = (int16_t)shiftResultUnsaturated(values[1]);
        outS16[1][0] = (int16_t)shiftResultUnsaturated(values[2]);
        outS16[1][1] = (int16_t)shiftResultUnsaturated(values[3]);

        outS16[0] += outSkip;
        outS16[1] += outSkip;
    }
}

void verticalU10(const uint8_t* in, uint32_t inStride, uint8_t* out, uint32_t outStride, uint32_t y,
                 uint32_t rows, uint32_t height, const LdeKernel* kernel)
{
    verticalU16(in, inStride, out, outStride, y, rows, height, kernel, 5);
}

void verticalU12(const uint8_t* in, uint32_t inStride, uint8_t* out, uint32_t outStride, uint32_t y,
                 uint32_t rows, uint32_t height, const LdeKernel* kernel)
{
    verticalU16(in, inStride, out, outStride, y, rows, height, kernel, 3);
}

void verticalU14(const uint8_t* in, uint32_t inStride, uint8_t* out, uint32_t outStride, uint32_t y,
                 uint32_t rows, uint32_t height, const LdeKernel* kernel)
{
    verticalU16(in, inStride, out, outStride, y, rows, height, kernel, 1);
}

/*------------------------------------------------------------------------------*/
/* clang-format off */

/*! Tables for selecting scalar upscale functions based on src (input) and dst (output) FP types
 *  (plane bit depths and sign). The default mode for LCEVC is to perform all pixel processing
 *  operations on a signed 16bit plane - this ensures the output will be conformant and free from
 *  rounding errors. However, some shortcuts can be taken when not applying residuals to a given
 *  plane to upscale it directly from the same src to dst format so additional functions are
 *  provided for this.
 *
 *  When performing a 2D upscale, the vertical upscale must run first, followed by the horizontal
 *  upscale. A 1D upscale only runs the horizontal upscale and there is no case in the MPEG-5 Part 2
 *  LCEVC specification where a vertical-only upscale can happen. A 16-bit intermediate plane is
 *  used to retain precision between the two upscales in 2D mode which is why a vertical upscale
 *  always outputs to S16 format and a horizontal upscale supports reading from S16 or the input
 *  unsigned format.
 *
 *  The MPEG-5 Part 2 LCEVC specification also supports bit-depth promotion (and demotion) e.g.
 *  8bit base to 10bit output. This is currently handled by a full image conversion to S16, S16->S16
 *  scaling operations and a final conversion back to the final enhanced bit-depth. These functions
 *  could be handled here (and in SIMD upscales) in future by populating the missing combinations.
 */

/*! kHorizontalFunctionTable[srcFP][dstFP] */
static const UpscaleHorizontalFunction kHorizontalFunctionTable[LdpFPCount][LdpFPCount] = {
    /* src  /  U8                  U10                 U12                 U14                 S8                   S10                  S12                  S14 */
    /* U8  */ {horizontalU8Scalar, NULL              , NULL              , NULL              , NULL               , NULL               , NULL               , NULL               },
    /* U10 */ {NULL,               horizontalUNScalar, NULL              , NULL              , NULL               , NULL               , NULL               , NULL               },
    /* U12 */ {NULL,               NULL              , horizontalUNScalar, NULL              , NULL               , NULL               , NULL               , NULL               },
    /* U14 */ {NULL,               NULL              , NULL              , horizontalUNScalar, NULL               , NULL               , NULL               , NULL               },
    /* S8  */ {horizontalU8Scalar, NULL              , NULL              , NULL              , horizontalS16Scalar, horizontalS16Scalar, horizontalS16Scalar, horizontalS16Scalar},
    /* S10 */ {NULL,               horizontalUNScalar, NULL              , NULL              , horizontalS16Scalar, horizontalS16Scalar, horizontalS16Scalar, horizontalS16Scalar},
    /* S12 */ {NULL,               NULL              , horizontalUNScalar, NULL              , horizontalS16Scalar, horizontalS16Scalar, horizontalS16Scalar, horizontalS16Scalar},
    /* S14 */ {NULL,               NULL              , NULL              , horizontalUNScalar, horizontalS16Scalar, horizontalS16Scalar, horizontalS16Scalar, horizontalS16Scalar}
};

/*! kVerticalFunctionTable[srcFP][dstFP] */
static const UpscaleVerticalFunction kVerticalFunctionTable[LdpFPCount][LdpFPCount] = {
    /* src  /  U8    U10   U12   U14   S8           S10              S12               S14 */
    /* U8  */ {NULL, NULL, NULL, NULL, verticalU8 , NULL           , NULL            , NULL             },
    /* U10 */ {NULL, NULL, NULL, NULL, NULL       , verticalU10    , NULL            , NULL             },
    /* U12 */ {NULL, NULL, NULL, NULL, NULL       , NULL           , verticalU12     , NULL             },
    /* U14 */ {NULL, NULL, NULL, NULL, NULL       , NULL           , NULL            , verticalU14      },
    /* S8  */ {NULL, NULL, NULL, NULL, verticalS16, verticalS16    , verticalS16     , verticalS16      },
    /* S10 */ {NULL, NULL, NULL, NULL, verticalS16, verticalS16    , verticalS16     , verticalS16      },
    /* S12 */ {NULL, NULL, NULL, NULL, verticalS16, verticalS16    , verticalS16     , verticalS16      },
    /* S14 */ {NULL, NULL, NULL, NULL, verticalS16, verticalS16    , verticalS16     , verticalS16      }
};

/* clang-format on */
/*------------------------------------------------------------------------------*/

UpscaleHorizontalFunction upscaleGetHorizontalFunctionScalar(LdpFixedPoint srcFP, LdpFixedPoint dstFP)
{
    return kHorizontalFunctionTable[srcFP][dstFP];
}

UpscaleVerticalFunction upscaleGetVerticalFunctionScalar(LdpFixedPoint srcFP, LdpFixedPoint dstFP)
{
    return kVerticalFunctionTable[srcFP][dstFP];
}

/*------------------------------------------------------------------------------*/
