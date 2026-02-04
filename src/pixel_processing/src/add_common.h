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

#ifndef VN_LCEVC_PIXEL_PROCESSING_ADD_COMMON_H
#define VN_LCEVC_PIXEL_PROCESSING_ADD_COMMON_H

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct LdpPicturePlaneDesc LdpPicturePlaneDesc;

/*------------------------------------------------------------------------------*/

/*! \brief Helper macro for adding two signed 16bit source planes to a destination plane in scalar
 *         functions.
 *
 * Dst_t     Integer type of the destination plane. eg. uint8_t for LdpFPU8
 * storeOp   The conversion operation to perform on the destination value, eg. fpS8ToU8 for LdpFPU8;
 */
#define VN_PLANE_GETLINE(pl, offset) (pl->firstSample + (offset * pl->rowByteStride))

#define VN_ADD_PER_PIXEL_BODY(Dst_t, storeOp)                                               \
    const LdpPicturePlaneDesc* src1 = args->src1;                                           \
    const LdpPicturePlaneDesc* src2 = args->src2;                                           \
    const LdpPicturePlaneDesc* dst = args->dst;                                             \
    const uint32_t src1Stride = src1->rowByteStride / sizeof(int16_t);                      \
    const uint32_t src2Stride = src2->rowByteStride / sizeof(int16_t);                      \
    const uint32_t dstStride = dst->rowByteStride / sizeof(Dst_t);                          \
    const int16_t* restrict src1Row = (const int16_t*)VN_PLANE_GETLINE(src1, args->offset); \
    const int16_t* restrict src2Row = (const int16_t*)VN_PLANE_GETLINE(src2, args->offset); \
    Dst_t* restrict dstRow = (Dst_t*)VN_PLANE_GETLINE(dst, args->offset);                   \
    int32_t dstValue = 0;                                                                   \
    const uint32_t minWidth = args->minWidth;                                               \
    const uint32_t count = args->count;                                                     \
    for (uint32_t y = 0; y < count; ++y) {                                                  \
        const int16_t* restrict src1Pixel = src1Row;                                        \
        const int16_t* restrict src2Pixel = src2Row;                                        \
        Dst_t* restrict dstPixel = dstRow;                                                  \
        for (uint32_t x = 0; x < minWidth; ++x) {                                           \
            int32_t src1Value = (int32_t)*src1Pixel++;                                      \
            int32_t src2Value = (int32_t)*src2Pixel++;                                      \
            dstValue = storeOp(src1Value + src2Value);                                      \
            *dstPixel++ = (Dst_t)dstValue;                                                  \
        }                                                                                   \
        src1Row += src1Stride;                                                              \
        src2Row += src2Stride;                                                              \
        dstRow += dstStride;                                                                \
    }

/*! \brief Helper macro for adding two signed 16bit source planes to a destination plane in SIMD
 *         functions. It initializes several variables that each implementation will need:
 *
 *  src        The source surface to read from.
 *  dst        The destination surface to write to.
 *  width      The overall width to copy.
 *  simdWidth  The number of pixels to operate on in the SIMD loop.
 *  srcRow     Pointer of const Src_t type pointing to first pixel of the
 *             row to copy from.
 *  dstRow     Pointer of Dst_t type pointing to the first pixel of the row
 *             to copy to.
 *
 * Note: This functions requires the declaration of a function with the following
 *       signature:
 *
 *           uint32_t simdAlignment(const uint32_t);
 *
 *       It is intended to take the width to process, and it will return the lower
 *       aligned width for the number of pixels to process in the SIMD loop.
 */
#define VN_ADD_SIMD_BOILERPLATE(Dst_t)                                                  \
    const LdpPicturePlaneDesc* srcPlane1 = args->src1;                                  \
    const LdpPicturePlaneDesc* srcPlane2 = args->src2;                                  \
    const LdpPicturePlaneDesc* dst = args->dst;                                         \
    const uint32_t width = args->minWidth;                                              \
    const uint32_t simdWidth = simdAlignment(width);                                    \
    const uint32_t src1Stride = srcPlane1->rowByteStride / sizeof(int16_t);             \
    const uint32_t src2Stride = srcPlane2->rowByteStride / sizeof(int16_t);             \
    const uint32_t dstStride = dst->rowByteStride / sizeof(Dst_t);                      \
    const int16_t* srcRow1 = (const int16_t*)VN_PLANE_GETLINE(srcPlane1, args->offset); \
    const int16_t* srcRow2 = (const int16_t*)VN_PLANE_GETLINE(srcPlane2, args->offset); \
    Dst_t* dstRow = (Dst_t*)VN_PLANE_GETLINE(dst, args->offset);                        \
    const uint32_t count = args->count;

/*------------------------------------------------------------------------------*/

/*! \brief Arguments passed to the specialised blit function implementations. */
typedef struct LdppAddArgs
{
    const LdpPicturePlaneDesc* src1; /**< First source plane to add. */
    const LdpPicturePlaneDesc* src2; /**< Second source plane to add. */
    const LdpPicturePlaneDesc* dst;  /**< Destination plane to blit to. */
    uint32_t minWidth;               /**< Minimum plane width. */
    uint32_t offset;                 /**< Row offset to start processing from. */
    uint32_t count;                  /**< Number of rows to process. */
} LdppAddArgs;

typedef void (*PlaneAddFunction)(const LdppAddArgs* args);

/*------------------------------------------------------------------------------*/

#ifdef __cplusplus
}
#endif

#endif // VN_LCEVC_PIXEL_PROCESSING_ADD_COMMON_H
