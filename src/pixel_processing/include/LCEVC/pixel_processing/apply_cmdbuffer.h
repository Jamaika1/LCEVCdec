/* Copyright (c) V-Nova International Limited 2025-2026. All rights reserved.
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

#ifndef VN_LCEVC_PIXEL_PROCESSING_APPLY_CMDBUFFER_H
#define VN_LCEVC_PIXEL_PROCESSING_APPLY_CMDBUFFER_H

#include <LCEVC/common/task_pool.h>
#include <LCEVC/pipeline/enhancement_tile.h>
#include <LCEVC/pipeline/types.h>

#ifdef __cplusplus
extern "C"
{
#endif

/*! \brief Applies a CPU cmdbuffer to a plane
 *
 * \param taskPool        A task pool for multi-threaded apply of a cmdbuffer with entry points.
 *                        Can be NULL if the cmdbuffer does not have entry points.
 * \param parent          If not NULL, task that deferred tasks inherit dependencies from.
 * \param enhancementTile Structure containing CPU cmdbuffer and tile metadata for tiling mode.
 * \param fixedPoint      Datatype of the plane.
 * \param plane           Plane of pixels to apply residuals to.
 * \param rasterOrder     Toggle between block-order or raster-order apply, given by
 *                        `temporalEnabled` in the global config.
 * \param highlight       Set to true to ignore residual values and apply maximum values at
 *                        residual locations for debugging residual distribution.
 * \param diagInfo        Any diagnostic information for tracing.
 */
bool ldppApplyCmdBuffer(LdcTaskPool* taskPool, LdcTask* parent, LdpEnhancementTile* enhancementTile,
                        LdpFixedPoint fixedPoint, const LdpPicturePlaneDesc* plane,
                        bool rasterOrder, bool highlight, const LdpPipelineDiagInfo* diagInfo);

#ifdef __cplusplus
}
#endif

#endif // VN_LCEVC_PIXEL_PROCESSING_APPLY_CMDBUFFER_H
