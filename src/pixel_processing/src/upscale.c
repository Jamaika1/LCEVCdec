/* Copyright (c) V-Nova International Limited 2025. All rights reserved.
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

#include <LCEVC/common/acceleration.h>
#include <LCEVC/common/diagnostics.h>
#include <LCEVC/common/limit.h>
#include <LCEVC/common/log.h>
#include <LCEVC/pixel_processing/upscale.h>
//
#include "fp_types.h"
#include "upscale_neon.h"
#include "upscale_scalar.h"
#include "upscale_sse.h"

#include <assert.h>

/*------------------------------------------------------------------------------*/

/*! \brief  Helper function used to query the horizontal function look-up tables.
 *
 * It has a fallback mechanism when SIMD is desired to provide the non-SIMD
 * function if a SIMD function does not yet exist.
 *
 * \return A valid function pointer on success, otherwise NULL. */
UpscaleHorizontalFunction getHorizontalFunction(LdpFixedPoint srcFP, LdpFixedPoint dstFP,
                                                Interleaving interleaving, bool forceScalar)
{
    if (!fixedPointIsValid(srcFP) || !fixedPointIsValid(dstFP)) {
        VNLogError("Invalid horizontal function request - src_fp, dst_fp is invalid");
        return NULL;
    }

    UpscaleHorizontalFunction res = NULL;
    const LdcAcceleration* acceleration = ldcAccelerationGet();

    /* Find a SIMD functions */

    if (!forceScalar && acceleration->SSE) {
        res = upscaleGetHorizontalFunctionSSE(interleaving, srcFP, dstFP);
    }

    if (!forceScalar && acceleration->NEON) {
        assert(res == NULL);
        res = upscaleGetHorizontalFunctionNEON(interleaving, srcFP, dstFP);
    }

    /* Find a non-SIMD function */
    if (!res) {
        res = upscaleGetHorizontalFunctionScalar(srcFP, dstFP);
    }

    return res;
}

/*!
 * Helper function used to query the vertical function look-up tables. It has a
 * fallback mechanism when SIMD is desired to provide the non-SIMD function if a
 * SIMD function does not yet exist.
 *
 * \return A valid function pointer on success, otherwise NULL. */
UpscaleVerticalFunction getVerticalFunction(LdpFixedPoint srcFP, LdpFixedPoint dstFP,
                                            bool forceScalar, uint32_t* xStep)
{
    if (!fixedPointIsValid(srcFP) || !fixedPointIsValid(dstFP)) {
        VNLogError("Invalid vertical function request - src_fp or dst_fp is invalid");
        return NULL;
    }

    UpscaleVerticalFunction res = NULL;
    const LdcAcceleration* acceleration = ldcAccelerationGet();

    /* Find a SIMD function */
    if (!forceScalar && acceleration->SSE) {
        res = upscaleGetVerticalFunctionSSE(srcFP, dstFP);
        *xStep = 16;
    }

    if (!forceScalar && acceleration->NEON) {
        assert(res == NULL);
        res = upscaleGetVerticalFunctionNEON(srcFP, dstFP);
        *xStep = 16;
    }

    /* Find a non-SIMD function */
    if (!res) {
        res = upscaleGetVerticalFunctionScalar(srcFP, dstFP);
        *xStep = 2;
    }

    return res;
}

static bool interleavingEqual(const LdpPictureLayoutInfo* layoutLeft, const LdpPictureLayoutInfo* layoutRight)
{
    return 0 == memcmp(layoutLeft->interleave, layoutRight->interleave, kLdpPictureMaxNumPlanes);
}

static Interleaving getInterleaving(const LdpPictureLayoutInfo* layout, const uint32_t planeIndex)
{
    switch (layout->interleave[planeIndex]) {
        case 1: return ILNone;
        case 2: return ILNV12;
    }

    return ILCount;
}

/*------------------------------------------------------------------------------*/

/*! \brief defined predicted average modes of operation. */
typedef enum PAMode
{
    PAMDisabled,
    PAM1D,
    PAM2D
} PAMode;

/*!
 * Helper function to determine the predicted_average_mode to apply.
 *
 * \param paEnabled   Whether predicted-average is enabled or not.
 * \param is2D        Whether predicted-average is 2D or 1D.
 *
 * \return The calculated predicted average mode. */
static inline PAMode getPAMode(bool paEnabled, bool is2D)
{
    if (!paEnabled) {
        return PAMDisabled;
    }

    return is2D ? PAM2D : PAM1D;
}

static inline uint8_t* surfaceGetLine(const LdpPicturePlaneDesc* desc, const uint32_t lineOffset)
{
    return desc->firstSample + (lineOffset * desc->rowByteStride);
}

/* Upscale threading shared state. */
typedef struct UpscaleSlicedJobContext
{
    uint32_t planeIndex;
    const LdpPictureLayout* srcLayout;
    const LdpPictureLayout* dstLayout;
    const LdpPictureLayout* intermediateLayout;
    LdpPicturePlaneDesc srcPlane;
    LdpPicturePlaneDesc dstPlane;
    LdpPicturePlaneDesc intermediatePlane;
    UpscaleHorizontalFunction lineFunction;
    UpscaleVerticalFunction colFunction;
    LdeKernel kernel;
    bool applyPA;
    const LdppDitherFrame* frameDither;
    uint32_t colStepping;
} UpscaleSlicedJobContext;

/*------------------------------------------------------------------------------*/

/*! \brief Populates LdppHorizontalUpscaleParams with constants required for horizontal upscaling
 *         and inline conversion to and from dst picture format and internal signed 16-bit format.
 *         See `LdppHorizontalUpscaleParams` docs for details on each param.
 */
static LdppHorizontalUpscaleParams generateParams(const LdeKernel* kernel, LdpFixedPoint dstFp,
                                                  bool is2D, Interleaving interleaving)
{
    LdppHorizontalUpscaleParams params = {0};
    params.kernel = kernel;
    params.is2D = is2D;
    params.channelCount = 1;

    switch (dstFp) {
        // Unsigned planes (direct upscale)
        case LdpFPU8: {
            params.shift = 7;
            params.offset = 64;
            params.midpoint = 128;
            params.maxValue = 255;
            switch (interleaving) {
                case ILNone: params.channelSkip[0] = 1; break;
                case ILNV12:
                    params.channelCount = 2;
                    params.channelSkip[0] = 2;
                    params.channelSkip[1] = 2;
                    params.channelMap[1] = 1;
                    break;
                default: break;
            }
            break;
        }
        case LdpFPU10:
            params.shift = 5;
            params.offset = 16;
            params.midpoint = 512;
            params.maxValue = 1023;
            break;
        case LdpFPU12:
            params.shift = 3;
            params.offset = 4;
            params.midpoint = 2048;
            params.maxValue = 4096;
            break;
        case LdpFPU14:
            params.shift = 1;
            params.offset = 1;
            params.midpoint = 8192;
            params.maxValue = 16383;
            break;
        // Signed planes (upscale converted planes)
        case LdpFPS8:
            params.maxValue = 255;
            params.shift = 7;
            break;
        case LdpFPS10:
            params.maxValue = 1023;
            params.shift = 5;
            break;
        case LdpFPS12:
            params.maxValue = 4096;
            params.shift = 3;
            break;
        case LdpFPS14:
            params.maxValue = 16383;
            params.shift = 1;
            break;
        case LdpFPCount: break;
    }

    return params;
}

/*!
 * Helper function that performs horizontal upscaling for a given job.
 *
 * This performs upscaling down a slice of src surface, where each invocation of
 * hori_func will upscale 2 full width lines at a time, with optional predicted-average
 * and dithering applied.
 *
 * \param context        Upscale context
 * \param yStart         The row to start upscaling from.
 * \param yEnd           The row to end upscaling from (exclusive).
 * \param paMode         The predicted-average mode to use. */
static void horizontalTask(const UpscaleSlicedJobContext* context, uint32_t yStart, uint32_t yEnd,
                           PAMode paMode)
{
    bool is2D = context->colFunction != NULL;
    const uint32_t channelWidth = context->srcLayout->width >>
                                  context->srcLayout->layoutInfo->planeWidthShift[context->planeIndex];
    const LdpPicturePlaneDesc* horizontalInputPlane =
        is2D ? &context->intermediatePlane : &context->srcPlane;

    LdppDitherSlice sliceDither;

    uint8_t* dstPtrs[2];
    const uint8_t* srcPtrs[2];
    const uint8_t* basePtrs[2] = {NULL, NULL};
    LdppHorizontalUpscaleParams params =
        generateParams(&context->kernel, context->dstLayout->layoutInfo->fixedPoint, is2D,
                       getInterleaving(context->dstLayout->layoutInfo, context->planeIndex));

    if (context->frameDither) {
        ldppDitherSliceInitialise(&sliceDither, context->frameDither, yStart, context->planeIndex);
    }

    for (uint32_t y = yStart; y < yEnd; y += 2) {
        srcPtrs[0] = surfaceGetLine(horizontalInputPlane, y);
        dstPtrs[0] = surfaceGetLine(&context->dstPlane, y);

        /* y_end is aligned to even so can always expect there to be 2 lines available
         * except for last job which deals with the remainder */
        if (y + 1 < yEnd) {
            srcPtrs[1] = surfaceGetLine(horizontalInputPlane, y + 1);
            dstPtrs[1] = surfaceGetLine(&context->dstPlane, y + 1);
        } else {
            /* Maintain valid pointers, this will simply duplicate work on last line and
             * prevents the need for each specific implementation to have to check for
             * pointer validity. */
            srcPtrs[1] = srcPtrs[0];
            dstPtrs[1] = dstPtrs[0];
        }

        /* The presence of valid base_ptrs informs the horizontalFunction implementation
         * of what mode of PA to apply. */
        switch (paMode) {
            case PAM1D: {
                basePtrs[0] = srcPtrs[0];
                basePtrs[1] = srcPtrs[1];
                break;
            }
            case PAM2D: {
                basePtrs[0] = surfaceGetLine(&context->srcPlane, y >> 1);
                break;
            }
            case PAMDisabled:;
        }

        context->lineFunction(context->frameDither ? &sliceDither : NULL, srcPtrs, dstPtrs,
                              basePtrs, channelWidth, 0, channelWidth, &params);
    }
}

/*!
 * Helper function that performs vertical upscaling for a given job.
 *
 * This performs upscaling across a slice of src surface, where each invocation
 * of the vert_func will upscale some number of columns, determined by x_step.
 *
 * \param context        Upscale context
 * \param yStart       The row to start upscaling from on the input surface.
 * \param yEnd         The row to end upscaling from on the input surface (exclusive).
 * \param xStep        The number of columns upscaled per invocation of vert_func. */
static void verticalTask(const UpscaleSlicedJobContext* context, uint32_t yStart, uint32_t yEnd, uint32_t xStep)
{
    UpscaleVerticalFunction vertFunction = context->colFunction;
    const LdpFixedPoint srcFP = context->srcLayout->layoutInfo->fixedPoint;
    const LdpFixedPoint dstFP = context->intermediateLayout->layoutInfo->fixedPoint;
    const uint32_t srcPelSize = fixedPointByteSize(srcFP);
    const uint32_t dstPelSize = sizeof(int16_t);
    uint32_t srcStep = xStep * srcPelSize;
    uint32_t dstStep = xStep * dstPelSize;
    const uint32_t rowCount = yEnd - yStart;

    /* Assume that src and dst interleaving is the same. */
    const uint8_t* srcPtr = context->srcPlane.firstSample;
    uint8_t* dstPtr = context->intermediatePlane.firstSample;
    const uint32_t width = (context->srcLayout->width >>
                            context->srcLayout->layoutInfo->planeWidthShift[context->planeIndex]) *
                           context->srcLayout->layoutInfo->interleave[context->planeIndex];
    const uint32_t height = context->srcLayout->height >>
                            context->srcLayout->layoutInfo->planeHeightShift[context->planeIndex];
    const uint32_t srcStride = context->srcPlane.rowByteStride / srcPelSize;
    const uint32_t dstStride = context->intermediatePlane.rowByteStride / dstPelSize;

    for (uint32_t x = 0; x < width; x += xStep) {
        /* Check if there's a potential overflow in the current step */
        if ((x + xStep) > width) {
            /* If overflow is detected, set up for scalar mode by default */
            vertFunction = upscaleGetVerticalFunctionScalar(srcFP, dstFP);
            assert(vertFunction != NULL);

            xStep = 2;
            srcStep = xStep * srcPelSize;
            dstStep = xStep * dstPelSize;

            /* If there's only one last pixel to be upscaled due to odd width, move
            back one pixel to upscale the last pixel */
            if ((x + xStep) - width == 1) {
                srcPtr -= srcPelSize;
                dstPtr -= dstPelSize;
            }
        }
        vertFunction(srcPtr, srcStride, dstPtr, dstStride, yStart, rowCount, height, &context->kernel);
        srcPtr += srcStep;
        dstPtr += dstStep;
    }
}

/*------------------------------------------------------------------------------*/

/* Callback that is invoked on each thread during upscaling. */
static bool upscaleSlicedJob(void* argument, uint32_t offset, uint32_t count)
{
    VNTraceScopedBegin();

    const UpscaleSlicedJobContext* context = (const UpscaleSlicedJobContext*)argument;

    const bool is2D = (context->colFunction != NULL);
    const uint32_t horiStart = offset << (is2D ? 1 : 0);
    const uint32_t horiEnd = (offset + count) << (is2D ? 1 : 0);
    const PAMode paMode = getPAMode(context->applyPA, is2D);

    if (is2D) { // 1D scale mode only runs the horizontal upscale
        const uint32_t vertStart = offset;
        const uint32_t vertEnd = offset + count;
        const uint32_t vertStep = context->colStepping;

        verticalTask(context, vertStart, vertEnd, vertStep);
    }

    horizontalTask(context, horiStart, horiEnd, paMode);

    VNTraceScopedEnd();
    return true;
}

/*! Execute a multi-threaded upscale operation. */
static bool upscaleExecute(LdcTaskPool* taskPool, LdcTask* parent, const LdppUpscaleArgs* params,
                           const LdeKernel* kernel)
{
    assert(params->mode != Scale0D);

    UpscaleSlicedJobContext slicedJobContext = {0};

    const bool is2D = (params->mode == Scale2D);

    const LdpPictureLayoutInfo* srcLayoutInfo = params->srcLayout->layoutInfo;
    const LdpPictureLayoutInfo* dstLayoutInfo = params->dstLayout->layoutInfo;
    const LdpFixedPoint intermediateFP = is2D && params->intermediateLayout
                                             ? params->intermediateLayout->layoutInfo->fixedPoint
                                             : srcLayoutInfo->fixedPoint;

    slicedJobContext.planeIndex = params->planeIndex;
    slicedJobContext.srcLayout = params->srcLayout;
    slicedJobContext.dstLayout = params->dstLayout;
    slicedJobContext.intermediateLayout = params->intermediateLayout;
    slicedJobContext.srcPlane = params->srcPlane;
    slicedJobContext.dstPlane = params->dstPlane;
    slicedJobContext.intermediatePlane = params->intermediatePlane;

    slicedJobContext.colFunction = NULL;
    if (is2D) {
        slicedJobContext.colFunction =
            getVerticalFunction(srcLayoutInfo->fixedPoint, intermediateFP, params->forceScalar,
                                &slicedJobContext.colStepping);
    }
    slicedJobContext.lineFunction =
        getHorizontalFunction(intermediateFP, dstLayoutInfo->fixedPoint,
                              getInterleaving(srcLayoutInfo, params->planeIndex), params->forceScalar);
    slicedJobContext.kernel = *kernel;
    slicedJobContext.applyPA = params->applyPA;
    slicedJobContext.frameDither = params->frameDither;

    if (!slicedJobContext.lineFunction) {
        VNLogError("Failed to find upscale horizontal function");
        return false;
    }

    if (is2D && !slicedJobContext.colFunction) {
        VNLogError("Failed to find upscale vertical function");
        return false;
    }

    const uint32_t srcHeight = params->srcLayout->height >>
                               params->srcLayout->layoutInfo->planeHeightShift[params->planeIndex];

    return ldcTaskPoolAddSlicedDeferred(taskPool, parent, &upscaleSlicedJob, NULL,
                                        &slicedJobContext, sizeof(slicedJobContext), srcHeight);
}

/*------------------------------------------------------------------------------*/

bool ldppUpscale(LdcTaskPool* taskPool, LdcTask* parent, const LdeKernel* kernel,
                 const LdppUpscaleArgs* params)
{
    const LdpPictureLayout* srcLayout = params->srcLayout;
    const LdpPictureLayout* dstLayout = params->dstLayout;

    if (!interleavingEqual(srcLayout->layoutInfo, dstLayout->layoutInfo)) {
        VNLogError("Upscale: src and dst must be the same interleaving type");
        return false;
    }

    if (kernel->length & 1 || kernel->length > 8 || !kernel->length) {
        VNLogError("Upscale: kernel length must be multiple of 2 and max 8 and non-zero");
        return false;
    }

    const LdpFixedPoint srcFP = srcLayout->layoutInfo->fixedPoint;
    const LdpFixedPoint dstFP = dstLayout->layoutInfo->fixedPoint;

    if (fixedPointIsSigned(srcFP) != fixedPointIsSigned(dstFP)) {
        VNLogError("Upscale: cannot convert between signed and unsigned formats");
        return false;
    }

    if (!fixedPointIsSigned(srcFP) && (bitdepthFromFixedPoint(srcFP) > bitdepthFromFixedPoint(dstFP))) {
        VNLogError("Upscale: src bitdepth must be less than or equal to dst bitdepth - LCEVC"
                   "doesn't allow bit-depth reduction");
        return false;
    }

    return upscaleExecute(taskPool, parent, params, kernel);
}

/*------------------------------------------------------------------------------*/
