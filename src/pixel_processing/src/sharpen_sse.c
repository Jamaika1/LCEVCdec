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

#include "sharpen_common.h"

#include <LCEVC/build_config.h>
#include <stddef.h>

#if VN_SDK_FEATURE(SSE)

#include <LCEVC/common/sse.h>
#include <LCEVC/pixel_processing/dither.h>
//
#include "fp_types.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/*------------------------------------------------------------------------------*/

static const uint32_t kStepU8 = 16;
static const uint32_t kStepU16 = 8;

static inline __m128i fromU16S32(__m128i val)
{
    return _mm_srai_epi32(_mm_add_epi32(val, _mm_set1_epi32(1 << 15)), 16);
}

static inline __m128i sseGenerateDither(const uint16_t** ditherBuffer, const __m128i ditherOffset,
                                        const __m128i ditherMul)
{
    __m128i noise = _mm_loadu_si128((const __m128i*)*ditherBuffer);
    *ditherBuffer += 8;

    noise = _mm_mulhi_epu16(noise, ditherMul);
    return _mm_sub_epi16(ditherOffset, noise);
}

static inline __m128i sseApplyDitherU8(const __m128i values, const uint16_t** ditherBuffer,
                                       const __m128i ditherOffset, const __m128i ditherMul)
{
    const __m128i ditherLo = sseGenerateDither(ditherBuffer, ditherOffset, ditherMul);
    const __m128i ditherHi = sseGenerateDither(ditherBuffer, ditherOffset, ditherMul);

    m128ix2 valuesS16 = expandU8ToS16SSE(values);
    valuesS16.val[0] = _mm_adds_epi16(valuesS16.val[0], ditherLo);
    valuesS16.val[1] = _mm_adds_epi16(valuesS16.val[1], ditherHi);

    return _mm_packus_epi16(valuesS16.val[0], valuesS16.val[1]);
}

static inline __m128i sseApplyDitherU16(const __m128i values, const uint16_t** ditherBuffer,
                                        const __m128i ditherOffset, const __m128i ditherMul,
                                        const __m128i clampS32)
{
    const __m128i zero = _mm_setzero_si128();
    const __m128i dither = sseGenerateDither(ditherBuffer, ditherOffset, ditherMul);
    const m128ix2 ditherS32 = expandS16ToS32SSE(dither);

    __m128i valueLo = _mm_cvtepu16_epi32(values);
    __m128i valueHi = _mm_cvtepu16_epi32(_mm_srli_si128(values, 8));

    valueLo = _mm_add_epi32(valueLo, ditherS32.val[0]);
    valueHi = _mm_add_epi32(valueHi, ditherS32.val[1]);

    valueLo = _mm_max_epi32(_mm_min_epi32(valueLo, clampS32), zero);
    valueHi = _mm_max_epi32(_mm_min_epi32(valueHi, clampS32), zero);

    return _mm_packus_epi32(valueLo, valueHi);
}

static inline __m128i sharpenCoeffU8x8(const __m128i center, const __m128i left,
                                       const __m128i right, const __m128i top, const __m128i bottom,
                                       const __m128i strengthS32, __m128i* weightOut)
{
    const __m128i neighbours = _mm_add_epi16(_mm_add_epi16(left, right), _mm_add_epi16(top, bottom));
    const __m128i weight = _mm_sub_epi16(_mm_slli_epi16(center, 2), neighbours);

    if (weightOut) {
        *weightOut = weight;
    }

    const m128ix2 weightS32 = expandS16ToS32SSE(weight);
    const __m128i prod0 = _mm_mullo_epi32(weightS32.val[0], strengthS32);
    const __m128i prod1 = _mm_mullo_epi32(weightS32.val[1], strengthS32);
    const __m128i coeff0 = fromU16S32(prod0);
    const __m128i coeff1 = fromU16S32(prod1);
    return _mm_packs_epi32(coeff0, coeff1);
}

static inline __m128i sharpenKernelU8x8(const __m128i center, const __m128i left, const __m128i right,
                                        const __m128i top, const __m128i bottom, const __m128i strengthS32)
{
    const __m128i coeff = sharpenCoeffU8x8(center, left, right, top, bottom, strengthS32, NULL);
    return _mm_add_epi16(center, coeff);
}

static inline __m128i sharpenKernelU8x8AddWeight(const __m128i center, const __m128i left,
                                                 const __m128i right, const __m128i top,
                                                 const __m128i bottom, const __m128i strengthS32)
{
    __m128i weight;
    const __m128i coeff = sharpenCoeffU8x8(center, left, right, top, bottom, strengthS32, &weight);
    return _mm_add_epi16(_mm_add_epi16(center, coeff), weight);
}

static inline __m128i sharpenKernelU8x16(const __m128i center, const __m128i left, const __m128i right,
                                         const __m128i top, const __m128i bottom, const __m128i strengthS32)
{
    const m128ix2 center16 = expandU8ToS16SSE(center);
    const m128ix2 left16 = expandU8ToS16SSE(left);
    const m128ix2 right16 = expandU8ToS16SSE(right);
    const m128ix2 top16 = expandU8ToS16SSE(top);
    const m128ix2 bottom16 = expandU8ToS16SSE(bottom);

    const __m128i dstLo = sharpenKernelU8x8(center16.val[0], left16.val[0], right16.val[0],
                                            top16.val[0], bottom16.val[0], strengthS32);
    const __m128i dstHi = sharpenKernelU8x8(center16.val[1], left16.val[1], right16.val[1],
                                            top16.val[1], bottom16.val[1], strengthS32);
    return _mm_packus_epi16(dstLo, dstHi);
}

static inline __m128i sharpenKernelU8x16AddWeight(const __m128i center, const __m128i left,
                                                  const __m128i right, const __m128i top,
                                                  const __m128i bottom, const __m128i strengthS32)
{
    const m128ix2 center16 = expandU8ToS16SSE(center);
    const m128ix2 left16 = expandU8ToS16SSE(left);
    const m128ix2 right16 = expandU8ToS16SSE(right);
    const m128ix2 top16 = expandU8ToS16SSE(top);
    const m128ix2 bottom16 = expandU8ToS16SSE(bottom);

    const __m128i dstLo = sharpenKernelU8x8AddWeight(center16.val[0], left16.val[0], right16.val[0],
                                                     top16.val[0], bottom16.val[0], strengthS32);
    const __m128i dstHi = sharpenKernelU8x8AddWeight(center16.val[1], left16.val[1], right16.val[1],
                                                     top16.val[1], bottom16.val[1], strengthS32);
    return _mm_packus_epi16(dstLo, dstHi);
}

static inline m128ix2 sharpenWeightU16x8(const __m128i center, const __m128i left,
                                         const __m128i right, const __m128i top, const __m128i bottom)
{
    m128ix2 weight;

    const __m128i centerLo = _mm_cvtepu16_epi32(center);
    const __m128i centerHi = _mm_cvtepu16_epi32(_mm_srli_si128(center, 8));
    const __m128i leftLo = _mm_cvtepu16_epi32(left);
    const __m128i leftHi = _mm_cvtepu16_epi32(_mm_srli_si128(left, 8));
    const __m128i rightLo = _mm_cvtepu16_epi32(right);
    const __m128i rightHi = _mm_cvtepu16_epi32(_mm_srli_si128(right, 8));
    const __m128i topLo = _mm_cvtepu16_epi32(top);
    const __m128i topHi = _mm_cvtepu16_epi32(_mm_srli_si128(top, 8));
    const __m128i bottomLo = _mm_cvtepu16_epi32(bottom);
    const __m128i bottomHi = _mm_cvtepu16_epi32(_mm_srli_si128(bottom, 8));

    const __m128i neighboursLo =
        _mm_add_epi32(_mm_add_epi32(leftLo, rightLo), _mm_add_epi32(topLo, bottomLo));
    const __m128i neighboursHi =
        _mm_add_epi32(_mm_add_epi32(leftHi, rightHi), _mm_add_epi32(topHi, bottomHi));

    weight.val[0] = _mm_sub_epi32(_mm_slli_epi32(centerLo, 2), neighboursLo);
    weight.val[1] = _mm_sub_epi32(_mm_slli_epi32(centerHi, 2), neighboursHi);
    return weight;
}

static inline __m128i sharpenKernelU16x8(const __m128i center, const __m128i left,
                                         const __m128i right, const __m128i top, const __m128i bottom,
                                         const __m128i strengthS32, const __m128i clampS32)
{
    const __m128i zero = _mm_setzero_si128();
    const __m128i centerLo = _mm_cvtepu16_epi32(center);
    const __m128i centerHi = _mm_cvtepu16_epi32(_mm_srli_si128(center, 8));
    const m128ix2 weight = sharpenWeightU16x8(center, left, right, top, bottom);

    __m128i resultLo = _mm_add_epi32(centerLo, fromU16S32(_mm_mullo_epi32(weight.val[0], strengthS32)));
    __m128i resultHi = _mm_add_epi32(centerHi, fromU16S32(_mm_mullo_epi32(weight.val[1], strengthS32)));

    resultLo = _mm_max_epi32(_mm_min_epi32(resultLo, clampS32), zero);
    resultHi = _mm_max_epi32(_mm_min_epi32(resultHi, clampS32), zero);

    return _mm_packus_epi32(resultLo, resultHi);
}

static inline __m128i sharpenKernelU16x8AddWeight(const __m128i center, const __m128i left,
                                                  const __m128i right, const __m128i top,
                                                  const __m128i bottom, const __m128i strengthS32,
                                                  const __m128i clampS32)
{
    const __m128i zero = _mm_setzero_si128();
    const __m128i centerLo = _mm_cvtepu16_epi32(center);
    const __m128i centerHi = _mm_cvtepu16_epi32(_mm_srli_si128(center, 8));
    const m128ix2 weight = sharpenWeightU16x8(center, left, right, top, bottom);

    __m128i resultLo = _mm_add_epi32(
        _mm_add_epi32(centerLo, fromU16S32(_mm_mullo_epi32(weight.val[0], strengthS32))), weight.val[0]);
    __m128i resultHi = _mm_add_epi32(
        _mm_add_epi32(centerHi, fromU16S32(_mm_mullo_epi32(weight.val[1], strengthS32))), weight.val[1]);

    resultLo = _mm_max_epi32(_mm_min_epi32(resultLo, clampS32), zero);
    resultHi = _mm_max_epi32(_mm_min_epi32(resultHi, clampS32), zero);

    return _mm_packus_epi32(resultLo, resultHi);
}

static inline void sharpenU8_SSE(uint8_t* src, uint32_t stride, uint32_t width, uint32_t height,
                                 float strengthFloat, LdppDitherSlice* dither)
{
    const uint16_t strengthInt = f32ToU16(strengthFloat);
    const int16_t strength = (int16_t)strengthInt;
    const bool addWeight = (strengthInt & 0x8000) != 0;
    const bool ditherEnabled = (dither != NULL);
    const __m128i strengthS32 = _mm_set1_epi32((int32_t)strength);

    const uint32_t tmpSize = width - 2;
    uint8_t* tmpRow[2] = {alloca(tmpSize), alloca(tmpSize)};
    memcpy(tmpRow[0], getRowU8(src, stride, 0) + 1, tmpSize);
    const uint32_t simdWidth = tmpSize & ~(kStepU8 - 1);
    const uint32_t simdEnd = 1 + simdWidth;
    const uint32_t ditherSpan = simdWidth + kStepU8;
    const __m128i ditherOffset = _mm_set1_epi16((int16_t)(ditherEnabled ? dither->strength : 0));
    const __m128i ditherMul =
        _mm_set1_epi16((int16_t)(ditherEnabled ? (dither->strength * 2u + 1u) : 0));

    for (uint32_t y = 1; y < height - 1; ++y) {
        uint8_t* srcRows[3] = {
            getRowU8(src, stride, y - 1),
            getRowU8(src, stride, y),
            getRowU8(src, stride, y + 1),
        };
        const uint16_t* ditherBuffer = NULL;
        if (ditherEnabled) {
            ditherBuffer = ldppDitherGetBuffer(dither, ditherSpan);
        }

        uint32_t x = 1;
        if (addWeight) {
            for (; x < simdEnd; x += kStepU8) {
                const __m128i top = _mm_loadu_si128((const __m128i*)(srcRows[0] + x));
                const __m128i left = _mm_loadu_si128((const __m128i*)(srcRows[1] + x - 1));
                const __m128i center = _mm_loadu_si128((const __m128i*)(srcRows[1] + x));
                const __m128i right = _mm_loadu_si128((const __m128i*)(srcRows[1] + x + 1));
                const __m128i bottom = _mm_loadu_si128((const __m128i*)(srcRows[2] + x));

                __m128i dst = sharpenKernelU8x16AddWeight(center, left, right, top, bottom, strengthS32);
                if (ditherBuffer) {
                    dst = sseApplyDitherU8(dst, &ditherBuffer, ditherOffset, ditherMul);
                }
                _mm_storeu_si128((__m128i*)(tmpRow[1] + x - 1), dst);
            }
        } else {
            for (; x < simdEnd; x += kStepU8) {
                const __m128i top = _mm_loadu_si128((const __m128i*)(srcRows[0] + x));
                const __m128i left = _mm_loadu_si128((const __m128i*)(srcRows[1] + x - 1));
                const __m128i center = _mm_loadu_si128((const __m128i*)(srcRows[1] + x));
                const __m128i right = _mm_loadu_si128((const __m128i*)(srcRows[1] + x + 1));
                const __m128i bottom = _mm_loadu_si128((const __m128i*)(srcRows[2] + x));

                __m128i dst = sharpenKernelU8x16(center, left, right, top, bottom, strengthS32);
                if (ditherBuffer) {
                    dst = sseApplyDitherU8(dst, &ditherBuffer, ditherOffset, ditherMul);
                }
                _mm_storeu_si128((__m128i*)(tmpRow[1] + x - 1), dst);
            }
        }

        // Final right edge
        x = width - kStepU8 - 1;
        const __m128i top = _mm_loadu_si128((const __m128i*)(srcRows[0] + x));
        const __m128i left = _mm_loadu_si128((const __m128i*)(srcRows[1] + x - 1));
        const __m128i center = _mm_loadu_si128((const __m128i*)(srcRows[1] + x));
        const __m128i right = _mm_loadu_si128((const __m128i*)(srcRows[1] + x + 1));
        const __m128i bottom = _mm_loadu_si128((const __m128i*)(srcRows[2] + x));

        if (addWeight) {
            __m128i dst = sharpenKernelU8x16AddWeight(center, left, right, top, bottom, strengthS32);
            if (ditherBuffer) {
                dst = sseApplyDitherU8(dst, &ditherBuffer, ditherOffset, ditherMul);
            }
            _mm_storeu_si128((__m128i*)(tmpRow[1] + x - 1), dst);
        } else {
            __m128i dst = sharpenKernelU8x16(center, left, right, top, bottom, strengthS32);
            if (ditherBuffer) {
                dst = sseApplyDitherU8(dst, &ditherBuffer, ditherOffset, ditherMul);
            }
            _mm_storeu_si128((__m128i*)(tmpRow[1] + x - 1), dst);
        }

        memcpy(getRowU8(src, stride, y - 1) + 1, tmpRow[0], tmpSize);
        uint8_t* const swap = tmpRow[0];
        tmpRow[0] = tmpRow[1];
        tmpRow[1] = swap;
    }

    memcpy(getRowU8(src, stride, height - 2) + 1, tmpRow[0], tmpSize);
}

static inline void sharpenU16_SSE(uint8_t* src, uint32_t stride, uint32_t width, uint32_t height,
                                  float strengthFloat, int32_t resultClamp, LdppDitherSlice* dither)
{
    const uint16_t strengthInt = f32ToU16(strengthFloat);
    const int16_t strength = (int16_t)strengthInt;
    const bool addWeight = (strengthInt & 0x8000) != 0;
    const bool ditherEnabled = (dither != NULL);
    const __m128i strengthS32 = _mm_set1_epi32((int32_t)strength);
    const __m128i clampS32 = _mm_set1_epi32(resultClamp);

    const uint32_t tmpSize = (width - 2) * sizeof(uint16_t);
    uint16_t* tmpRow[2] = {(uint16_t*)alloca(tmpSize), (uint16_t*)alloca(tmpSize)};

    memcpy(tmpRow[0], ((uint16_t*)getRowU8(src, stride, 0)) + 1, tmpSize);
    const uint32_t simdWidth = (width - 2) & ~(kStepU16 - 1);
    const uint32_t simdEnd = 1 + simdWidth;
    const uint32_t ditherSpan = simdWidth + kStepU16;
    const __m128i ditherOffset = _mm_set1_epi16((int16_t)(ditherEnabled ? dither->strength : 0));
    const __m128i ditherMul =
        _mm_set1_epi16((int16_t)(ditherEnabled ? (dither->strength * 2u + 1u) : 0));

    for (uint32_t y = 1; y < height - 1; ++y) {
        uint16_t* srcRows[3] = {
            (uint16_t*)getRowU8(src, stride, y - 1),
            (uint16_t*)getRowU8(src, stride, y),
            (uint16_t*)getRowU8(src, stride, y + 1),
        };
        const uint16_t* ditherBuffer = NULL;
        if (ditherEnabled) {
            ditherBuffer = ldppDitherGetBuffer(dither, ditherSpan);
        }

        uint32_t x = 1;
        if (addWeight) {
            for (; x < simdEnd; x += kStepU16) {
                const __m128i top = _mm_loadu_si128((const __m128i*)(srcRows[0] + x));
                const __m128i left = _mm_loadu_si128((const __m128i*)(srcRows[1] + x - 1));
                const __m128i center = _mm_loadu_si128((const __m128i*)(srcRows[1] + x));
                const __m128i right = _mm_loadu_si128((const __m128i*)(srcRows[1] + x + 1));
                const __m128i bottom = _mm_loadu_si128((const __m128i*)(srcRows[2] + x));

                __m128i dst = sharpenKernelU16x8AddWeight(center, left, right, top, bottom,
                                                          strengthS32, clampS32);
                if (ditherBuffer) {
                    dst = sseApplyDitherU16(dst, &ditherBuffer, ditherOffset, ditherMul, clampS32);
                }
                _mm_storeu_si128((__m128i*)(tmpRow[1] + x - 1), dst);
            }
        } else {
            for (; x < simdEnd; x += kStepU16) {
                const __m128i top = _mm_loadu_si128((const __m128i*)(srcRows[0] + x));
                const __m128i left = _mm_loadu_si128((const __m128i*)(srcRows[1] + x - 1));
                const __m128i center = _mm_loadu_si128((const __m128i*)(srcRows[1] + x));
                const __m128i right = _mm_loadu_si128((const __m128i*)(srcRows[1] + x + 1));
                const __m128i bottom = _mm_loadu_si128((const __m128i*)(srcRows[2] + x));

                __m128i dst = sharpenKernelU16x8(center, left, right, top, bottom, strengthS32, clampS32);
                if (ditherBuffer) {
                    dst = sseApplyDitherU16(dst, &ditherBuffer, ditherOffset, ditherMul, clampS32);
                }
                _mm_storeu_si128((__m128i*)(tmpRow[1] + x - 1), dst);
            }
        }

        // Final right edge
        x = width - kStepU16 - 1;
        const __m128i top = _mm_loadu_si128((const __m128i*)(srcRows[0] + x));
        const __m128i left = _mm_loadu_si128((const __m128i*)(srcRows[1] + x - 1));
        const __m128i center = _mm_loadu_si128((const __m128i*)(srcRows[1] + x));
        const __m128i right = _mm_loadu_si128((const __m128i*)(srcRows[1] + x + 1));
        const __m128i bottom = _mm_loadu_si128((const __m128i*)(srcRows[2] + x));

        if (addWeight) {
            __m128i dst =
                sharpenKernelU16x8AddWeight(center, left, right, top, bottom, strengthS32, clampS32);
            if (ditherBuffer) {
                dst = sseApplyDitherU16(dst, &ditherBuffer, ditherOffset, ditherMul, clampS32);
            }
            _mm_storeu_si128((__m128i*)(tmpRow[1] + x - 1), dst);
        } else {
            __m128i dst = sharpenKernelU16x8(center, left, right, top, bottom, strengthS32, clampS32);
            if (ditherBuffer) {
                dst = sseApplyDitherU16(dst, &ditherBuffer, ditherOffset, ditherMul, clampS32);
            }
            _mm_storeu_si128((__m128i*)(tmpRow[1] + x - 1), dst);
        }

        memcpy(((uint16_t*)getRowU8(src, stride, y - 1)) + 1, tmpRow[0], tmpSize);
        uint16_t* const swap = tmpRow[0];
        tmpRow[0] = tmpRow[1];
        tmpRow[1] = swap;
    }

    memcpy(((uint16_t*)getRowU8(src, stride, height - 2)) + 1, tmpRow[0], tmpSize);
}

static inline void sharpenU10_SSE(uint8_t* src, uint32_t stride, uint32_t width, uint32_t height,
                                  float strength, LdppDitherSlice* dither)
{
    sharpenU16_SSE(src, stride, width, height, strength, fixedPointHighlightValue(LdpFPU10), dither);
}

static inline void sharpenU12_SSE(uint8_t* src, uint32_t stride, uint32_t width, uint32_t height,
                                  float strength, LdppDitherSlice* dither)
{
    sharpenU16_SSE(src, stride, width, height, strength, fixedPointHighlightValue(LdpFPU12), dither);
}

static inline void sharpenU14_SSE(uint8_t* src, uint32_t stride, uint32_t width, uint32_t height,
                                  float strength, LdppDitherSlice* dither)
{
    sharpenU16_SSE(src, stride, width, height, strength, fixedPointHighlightValue(LdpFPU14), dither);
}

/*------------------------------------------------------------------------------*/

SharpenFunction sharpenGetFunctionSSE(LdpFixedPoint fixedPoint)
{
    static const SharpenFunction kTable[LdpFPCount] = {
        sharpenU8_SSE,  /* U8 */
        sharpenU10_SSE, /* U10 */
        sharpenU12_SSE, /* U12 */
        sharpenU14_SSE, /* U14 */
        NULL,           /* S8.7 */
        NULL,           /* S10.5 */
        NULL,           /* S12.3 */
        NULL,           /* S14.1 */
    };
    return kTable[fixedPoint];
}

/*------------------------------------------------------------------------------*/

#else

SharpenFunction sharpenGetFunctionSSE(LdpFixedPoint fixedPoint)
{
    VNUnused(fixedPoint);
    return NULL;
}

#endif
