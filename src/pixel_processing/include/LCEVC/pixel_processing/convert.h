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

#ifndef VN_LCEVC_PIXEL_PROCESSING_CONVERT_H
#define VN_LCEVC_PIXEL_PROCESSING_CONVERT_H

#include <LCEVC/common/task_pool.h>
#include <LCEVC/pipeline/picture_layout.h>
#include <LCEVC/pipeline/types.h>

#ifdef __cplusplus
extern "C"
{
#endif

/*! \brief Copies and converts a source plane to a destination plane. Primarily used to copy planes
 *         into the pipeline from their source layout to signed 16bit for LCEVC operations, then
 *         back out to the requested output format. Various common formats are written with SSE and
 *         NEON accelerated functions, some may fallback to scalar.
 *
 * \param taskPool       The task pool to create a sliced blit task from
 * \param parent         If not NULL, task that deferred tasks inherit dependencies from
 * \param planeIndex     The plane index in src/dst layout
 * \param srcLayout      The source plane picture layout
 * \param dstLayout      The destination picture layout
 * \param srcPlane       The source plane to blit from.
 * \param dstPlane       The destination plane to blit to.
 * \param diagInfo       Any diagnostic information for tracing.
 *
 * \return True if the copy/conversion was successful. */
bool ldppPlaneConvert(LdcTaskPool* taskPool, LdcTask* parent, uint32_t planeIndex,
                      const LdpPictureLayout* srcLayout, const LdpPictureLayout* dstLayout,
                      LdpPicturePlaneDesc* srcPlane, LdpPicturePlaneDesc* dstPlane,
                      const LdpPipelineDiagInfo* diagInfo);

/*------------------------------------------------------------------------------*/

#ifdef __cplusplus
}
#endif

#endif // VN_LCEVC_PIXEL_PROCESSING_CONVERT_H
