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

#ifndef VN_LCEVC_PIXEL_PROCESSING_SHARPEN_H
#define VN_LCEVC_PIXEL_PROCESSING_SHARPEN_H

#include <LCEVC/pipeline/picture_layout.h>
#include <LCEVC/pipeline/types.h>
#include <LCEVC/pixel_processing/dither.h>

#ifdef __cplusplus
extern "C"
{
#endif

/*! \brief Applies sharpening (S-filter) to a plane in-place. Note that this LDPP function does not
 *         currently support multi-threading.
 *
 * \param layout         The picture layout describing the plane dimensions and fixed-point type.
 * \param plane          The plane to sharpen.
 * \param planeIndex     The plane index in the picture layout.
 * \param strength       Sharpen strength in the range [0.0, 1.0].
 * \param frameDither    Per-frame dither context. When NULL, no dither is applied after sharpening.
 * \param diagInfo       Any diagnostic information for tracing.
 *
 * \return True if the sharpen operation was successful. */
bool ldppSharpen(const LdpPictureLayout* layout, LdpPicturePlaneDesc* plane, uint32_t planeIndex,
                 float strength, const LdppDitherFrame* frameDither, const LdpPipelineDiagInfo* diagInfo);

/*------------------------------------------------------------------------------*/

#ifdef __cplusplus
}
#endif

#endif // VN_LCEVC_PIXEL_PROCESSING_SHARPEN_H
