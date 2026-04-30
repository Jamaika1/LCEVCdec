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

#ifndef VN_LCEVC_PIXEL_PROCESSING_UPSCALE_H
#define VN_LCEVC_PIXEL_PROCESSING_UPSCALE_H

#include <LCEVC/common/task_pool.h>
#include <LCEVC/enhancement/bitstream_types.h>
#include <LCEVC/pipeline/picture_layout.h>
#include <LCEVC/pipeline/types.h>
#include <LCEVC/pixel_processing/dither.h>
//
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

/*------------------------------------------------------------------------------*/

/*! \brief Upscale parameters for input to ldppUpscale. */
typedef struct ldppUpscaleArgs
{
    uint32_t planeIndex;               /**< Plane index being upscaled */
    const LdpPictureLayout* srcLayout; /**< Source picture layout for the input plane */
    const LdpPictureLayout* dstLayout; /**< Destination picture layout for the output plane */
    LdpPicturePlaneDesc srcPlane;      /**< Source plane to read from */
    LdpPicturePlaneDesc dstPlane;      /**< Destination plane to write to */
    const LdeKernel* kernel;           /**< Upscaling kernel coefficients */
    bool applyPA;                      /**< Predicted-average mode toggle */
    const LdppDitherFrame* frameDither; /**< Populated dither struct of random noise, NULL for dithering off */
    LdeScalingMode mode;                /**< The type of scaling to perform (1D or 2D) */
} LdppUpscaleArgs;

/*------------------------------------------------------------------------------*/

/*! \brief Upscales a source surface to a destination surface using the supplied args. Operates in
 *         all combinations of predicted-average, dithering, and scaling modes for 8, 10, 12 &
 *         14bit inputs as well as NV12. Highly optimized SIMD implementations for SSE and NEON.
 *         Implicit bitdepth promotion to a higher dstLayout supported in scalar operation only.
 *         All upscaling is fully LCEVC standard compliant and performed in 16-bit fixed-point
 *         arithmetic. Approximated PA 3-tap kernels are not supported.
 *
 *  \param taskPool       The task pool to create a sliced blit task from
 *  \param parent         If not NULL, task that deferred tasks inherit dependencies from
 *  \param params         The arguments to use for upscaling.
 *  \param diagInfo       Any diagnostic information for tracing.
 *
 *  \return True if the upscale operation was successful. */
bool ldppUpscale(LdcTaskPool* taskPool, LdcTask* parent, const LdppUpscaleArgs* params,
                 const LdpPipelineDiagInfo* diagInfo);

/*------------------------------------------------------------------------------*/

#ifdef __cplusplus
}
#endif

#endif // VN_LCEVC_PIXEL_PROCESSING_UPSCALE_H
