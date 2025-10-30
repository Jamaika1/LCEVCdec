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

#include "frame_cpu.h"
//
#include "picture_cpu.h"
#include "pipeline_config_cpu.h"
#include "temporal_buffer_cpu.h"

#include <LCEVC/common/log.h>
#include <LCEVC/common/memory.h>
#include <LCEVC/common/platform.h>
#include <LCEVC/common/return_code.h>
#include <LCEVC/common/task_pool.h>
#include <LCEVC/enhancement/bitstream_types.h>
#include <LCEVC/enhancement/cmdbuffer_cpu.h>
#include <LCEVC/enhancement/config_parser.h>
#include <LCEVC/enhancement/config_types.h>
#include <LCEVC/enhancement/dimensions.h>
#include <LCEVC/pipeline/buffer.h>
#include <LCEVC/pipeline/types.h>
#include <LCEVC/pixel_processing/dither.h>

#include <algorithm>
#include <cstdint>

namespace lcevc_dec::pipeline_cpu {

FrameCPU::FrameCPU(LdcMemoryAllocator* enhancementAllocator, LdcMemoryAllocator* bufferAllocator,
                   uint64_t timestamp)
    : LdpFrame{}
    , m_enhancementAllocator(enhancementAllocator)
    , m_bufferAllocator(bufferAllocator)
{
    // Only fill in timestamp at this point - full initialization happens after a good config is seen
    this->timestamp = timestamp;
    this->deadline = UINT64_MAX;
}

// Initialize frame, given the resolved configuration.
//
bool FrameCPU::initialize(const PipelineConfigCPU& configuration, LdcTaskPool* taskPool,
                          LdppDitherGlobal* ditherBuffer)
{
    // Set up the task group
    unsigned maxDependencies = kTaskPoolMaxDependencies;
    ldcTaskGroupInitialize(&m_taskGroup, taskPool, maxDependencies, timestamp);

    // Generate task dependencies for inputs
    m_depBasePicture = ldcTaskDependencyAdd(&m_taskGroup); // NOLINT(cppcoreguidelines-prefer-member-initializer)
    if (basePicture) {
        if (!m_intermediateInitialized) {
            initializeIntermediateBuffers();
        }
        ldcTaskDependencyMet(&m_taskGroup, m_depBasePicture, basePicture);
    }

    m_depOutputPicture = ldcTaskDependencyAdd(&m_taskGroup); // NOLINT(cppcoreguidelines-prefer-member-initializer)
    if (outputPicture) {
        ldcTaskDependencyMet(&m_taskGroup, m_depOutputPicture, outputPicture);
    }

    if (!initializeCommandBuffers()) {
        return false;
    }

    // Figure out dithering strength either from the frame or local config override
    uint8_t strength = 0;
    if (configuration.ditherEnabled) {
        if (configuration.ditherOverrideStrength == -1) {
            strength = static_cast<uint8_t>(config.ditherStrength * config.ditherEnabled *
                                            (config.ditherType != DTNone));
        } else {
            strength = static_cast<uint8_t>(configuration.ditherOverrideStrength);
        }
    }

    return ldppDitherFrameInitialise(&m_frameDither, ditherBuffer, timestamp, strength);
}

// Release resources associated with a frame
//
void FrameCPU::release(bool wait)
{
    if (m_taskGroup.pool) {
        if (wait) {
            // Make sure task group is finished
            ldcTaskGroupWait(&m_taskGroup);
        }

        ldcTaskGroupDestroy(&m_taskGroup);
    }

    if (VNIsAllocated(m_enhancementData)) {
        VNFree(m_enhancementAllocator, &m_enhancementData);
    }

    ldeConfigsReleaseFrame(&config);
    releaseCommandBuffers();
    releaseIntermediateBuffers();
}

void FrameCPU::unblockDependency(LdcTaskDependency dep)
{
    if (dep != kTaskDependencyInvalid) {
        ldcTaskDependencyMet(&m_taskGroup, dep, nullptr);
    }
}

void FrameCPU::unblockForSkip()
{
    // Mark the base and output dependencies for this frame as met
    unblockDependency(m_depBasePicture);
    unblockDependency(m_depOutputPicture);
}

void FrameCPU::unblockForFlush()
{
    // Mark all dependencies as met
    unblockDependency(m_depBasePicture);
    unblockDependency(m_depOutputPicture);
    for (uint8_t i = 0; i < RCMaxPlanes; ++i) {
        unblockDependency(m_depTemporalBuffer[i]);
    }
}

uint8_t* FrameCPU::setEnhancementData(const uint8_t* data, uint32_t byteSize)
{
    VNAllocateIdArray(m_enhancementAllocator, &m_enhancementData, uint8_t, byteSize,
                      "FrameCPU_EnhancementData", timestamp);
    uint8_t* const enhancement{VNAllocationPtr(m_enhancementData, uint8_t)};

    if (!enhancement) {
        return nullptr;
    }

    memcpy(enhancement, data, byteSize);
    return enhancement;
}

// Parse the LCEVC configuration into distinct per-frame data
// Switch to pass-through if configuration parse failed.
bool FrameCPU::parseEnhancementData(LdeConfigPool* configPool)
{
    return ldeConfigPoolFrameInsert(configPool, timestamp, VNAllocationPtr(m_enhancementData, uint8_t),
                                    VNAllocationSize(m_enhancementData, uint8_t), &globalConfig, &config);
}

LdcTaskDependency FrameCPU::taskAdd(const LdcTaskDependency* inputs, uint32_t inputsCount,
                                    LdcTaskFunction task, const void* data, size_t dataSize,
                                    const char* name)
{
    const LdcTaskDependency output{ldcTaskDependencyAdd(&m_taskGroup)};

    ldcTaskGroupAdd(&m_taskGroup, inputs, inputsCount, output, task, nullptr, 1, 1, dataSize, data, name);

    return output;
}

void FrameCPU::taskAddSink(const LdcTaskDependency* inputs, uint32_t inputsCount,
                           LdcTaskFunction task, const void* data, size_t dataSize, const char* name)
{
    ldcTaskGroupAdd(&m_taskGroup, inputs, inputsCount, kTaskDependencyInvalid, task, nullptr, 1, 1,
                    dataSize, data, name);
}

bool FrameCPU::findOutputSetFromBase(LdcTaskDependency* outputs, uint32_t outputsMax,
                                     uint32_t* outputsCount) const
{
    return ldcTaskGroupFindOutputSetFromInput(&m_taskGroup, m_depBasePicture, outputs, outputsMax,
                                              outputsCount);
}

void FrameCPU::setOutputPicture(LdpPicture* picture)
{
    assert(m_depOutputPicture != kTaskDependencyInvalid);

    outputPicture = picture;
    ldcTaskDependencyMet(&m_taskGroup, m_depOutputPicture, picture);
}

void FrameCPU::waitForCompletableTasks()
{
    if (!canComplete()) {
        return;
    }

    ldcTaskGroupWait(&m_taskGroup);
}

void FrameCPU::waitForTasks() { ldcTaskGroupWait(&m_taskGroup); }

// Set up command buffers
//
bool FrameCPU::initializeCommandBuffers()
{
    assert(globalConfig->numPlanes <= 3);

    // Quick scan to find how many enhancement tiles are needed
    //
    // Don't include LoQs or planes that don't have enhancement
    uint32_t count = 0;
    for (int8_t loq = LOQ1; loq >= LOQ0; --loq) {
        if (!config.loqEnabled[loq]) {
            continue;
        }
        for (uint8_t plane = 0; plane < numEnhancedPlanes(); ++plane) {
            count += globalConfig->numTiles[plane][loq];
        }
    }

    // Allocate the enhancement tiles
    enhancementTileCount = count;

    if (enhancementTileCount == 0) {
        enhancementTiles = nullptr;
        return true;
    }

    VNAllocateIdArray(m_enhancementAllocator, &m_enhancementTilesAllocation, LdpEnhancementTile,
                      enhancementTileCount, "FrameCPU_EnhancementTiles", timestamp);
    enhancementTiles = VNAllocationPtr(m_enhancementTilesAllocation, LdpEnhancementTile);
    if (!enhancementTiles) {
        return false;
    }

    LdpEnhancementTile* et = enhancementTiles;

    // Fill in locations for command buffers
    for (int8_t loq = LOQ1; loq >= LOQ0; --loq) {
        if (!config.loqEnabled[loq]) {
            continue;
        }
        const uint8_t numPlanes = std::min(numEnhancedPlanes(), static_cast<uint8_t>(RCMaxPlanes));
        for (uint8_t plane = 0; plane < numPlanes; ++plane) {
            uint16_t planeWidth = 0;
            uint16_t planeHeight = 0;
            ldePlaneDimensionsFromConfig(globalConfig, static_cast<LdeLOQIndex>(loq), plane,
                                         &planeWidth, &planeHeight);
            const uint32_t numPlaneTiles = globalConfig->numTiles[plane][loq];
            for (uint32_t tile = 0; tile < numPlaneTiles; ++tile) {
                VNClear(et);
                et->loq = static_cast<LdeLOQIndex>(loq);
                et->plane = plane;
                et->tile = tile;
                ldeTileDimensionsFromConfig(globalConfig, static_cast<LdeLOQIndex>(loq), plane,
                                            tile, &et->tileWidth, &et->tileHeight);
                ldeTileStartFromConfig(globalConfig, static_cast<LdeLOQIndex>(loq), plane, tile,
                                       &et->tileX, &et->tileY);
                et->planeWidth = planeWidth;
                et->planeHeight = planeHeight;

                if (!ldeCmdBufferCpuInitializeId(m_enhancementAllocator, &et->buffer, 0, timestamp)) {
                    return false;
                }
                if (!ldeCmdBufferCpuReset(&et->buffer, globalConfig->numLayers)) {
                    return false;
                }
                et++;
            }
        }
    }

    assert((et - enhancementTiles) == enhancementTileCount);
    return true;
}

void FrameCPU::releaseCommandBuffers()
{
    // Release command buffers
    for (uint32_t i = 0; i < enhancementTileCount; ++i) {
        ldeCmdBufferCpuFree(&enhancementTiles[i].buffer);
    }
    enhancementTileCount = 0;

    if (VNIsAllocated(m_enhancementTilesAllocation)) {
        VNFree(m_enhancementAllocator, &m_enhancementTilesAllocation);
    }
}

// Set up intermediate buffers
//
// Allocate intermediate buffers for each LOQ/plane that needs it.
//
bool FrameCPU::initializeIntermediateBuffers()
{
    if (!hasGoodConfig()) {
        return false;
    }

    const LdpColorFormat format = getBaseColorFormat();

    // Allocate buffers starting at LOQ0, down to LOQ2 - As we go down, if there is no scaling
    // between layers, then the buffer will be shared with lower LOQ.
    for (int8_t loq = LOQ0; loq <= LOQ2; loq++) {
        uint16_t width = 0;
        uint16_t height = 0;
        if (globalConfig->initialized) {
            ldePlaneDimensionsFromConfig(globalConfig, static_cast<LdeLOQIndex>(loq), 0, &width, &height);
        } else {
            // SOme sort of passthrough - use base size
            width = static_cast<uint16_t>(baseWidth);
            height = static_cast<uint16_t>(baseHeight);
        }

        ldpInternalPictureLayoutInitialize(&m_intermediateLayout[loq], format, width, height,
                                           kBufferRowAlignment);
        if (loq < LOQ2) {
            // Initialize a signed 16bit plane with half the width of the output image for storing
            // the vertical upscale result before the horizontal upscale as per needsUpscaleBuffer.
            ldpInternalPictureLayoutInitialize(&m_upscaleLayout[loq], getUpscaleColorFormat(),
                                               width >> 1, height, kBufferRowAlignment);
        }

        const uint8_t numPlanes = std::min(ldpPictureLayoutPlanes(&m_intermediateLayout[loq]),
                                           static_cast<uint8_t>(RCMaxPlanes));

        for (uint8_t plane = 0; plane < numPlanes; plane++) {
            if (needsIntermediateBuffer(static_cast<LdeLOQIndex>(loq), plane)) {
                // Create an internal buffer for this LoQ/plane
                const uint32_t loqSize = ldpPictureLayoutPlaneSize(&m_intermediateLayout[loq], plane);
                VNAllocateIdAlignedArray(m_bufferAllocator, &m_intermediateBufferAllocation[plane][loq],
                                         uint8_t, kBufferRowAlignment, loqSize,
                                         "FrameCPU_IntermediateBuffer", timestamp);
                if (!VNIsAllocated(m_intermediateBufferAllocation[plane][loq])) {
                    return false;
                }
                m_intermediateBufferPtr[plane][loq] =
                    VNAllocationPtr(m_intermediateBufferAllocation[plane][loq], uint8_t);

                VNLogVerbose("Intermediate buffer %" PRIx64 ": LoQ:%d Plane:%d %ux%u:%d %p", timestamp,
                             loq, plane, ldpPictureLayoutPlaneWidth(&m_intermediateLayout[loq], plane),
                             ldpPictureLayoutPlaneHeight(&m_intermediateLayout[loq], plane),
                             (int)ldpPictureLayoutFormat(&m_intermediateLayout[loq]),
                             (void*)m_intermediateBufferPtr[plane][loq]);
            } else {
                // Share internal buffer from higher LoQ
                if (loq != LOQ0) {
                    m_intermediateBufferPtr[plane][loq] = m_intermediateBufferPtr[plane][loq - 1];
                } else {
                    m_intermediateBufferPtr[plane][loq] = nullptr;
                }
            }
        }

        if (needsUpscaleBuffer(static_cast<LdeLOQIndex>(loq))) {
            for (uint8_t plane = 0; plane < ldpPictureLayoutPlanes(&m_upscaleLayout[loq]); plane++) {
                // Add 16 bytes padding to stop upsampler SIMD loads falling off end of page
                const uint32_t loqSize = ldpPictureLayoutPlaneSize(&m_upscaleLayout[loq], plane) + 16;
                VNAllocateIdAlignedArray(m_bufferAllocator, &m_upscaleBufferAllocation[plane][loq],
                                         uint8_t, kBufferRowAlignment, loqSize,
                                         "FrameCPU_UpscaleBuffer", timestamp);
                if (!VNIsAllocated(m_upscaleBufferAllocation[plane][loq])) {
                    return false;
                }
            }
        }
    }

    m_intermediateInitialized = true;
    return true;
}

void FrameCPU::releaseIntermediateBuffers()
{
    // Release intermediate buffers
    for (uint8_t plane = 0; plane < RCMaxPlanes; plane++) {
        for (int8_t loq = LOQ0; loq <= LOQ2; loq++) {
            if (VNIsAllocated(m_intermediateBufferAllocation[plane][loq])) {
                VNFree(m_bufferAllocator, &m_intermediateBufferAllocation[plane][loq]);
            }
            if (loq < LOQ2 && VNIsAllocated(m_upscaleBufferAllocation[plane][loq])) {
                VNFree(m_bufferAllocator, &m_upscaleBufferAllocation[plane][loq]);
            }
        }
    }

    m_intermediateInitialized = false;
}

// Return true if frame needs an intermediate buffer for given loq/plane
//
bool FrameCPU::needsIntermediateBuffer(LdeLOQIndex loq, uint8_t plane) const
{
    if (plane > globalConfig->numPlanes &&
        ((globalConfig->scalingModes[LOQ0] != Scale0D) != (globalConfig->scalingModes[LOQ1] != Scale0D)) &&
        globalConfig->baseDepth == globalConfig->enhancedDepth) {
        return false;
    }

    if (loq == LOQ0) {
        return true;
    }

    if (globalConfig->scalingModes[loq - 1] != Scale0D) {
        return true;
    }

    return false;
}

// Return true if frame needs an upscale buffer for given loq/plane
//
bool FrameCPU::needsUpscaleBuffer(LdeLOQIndex loq) const
{
    if (loq == LOQ2) {
        return false;
    }

    if (globalConfig->scalingModes[loq] == Scale2D) {
        return true;
    }

    return false;
}

LdcReturnCode FrameCPU::setBasePicture(LdpPicture* picture, uint64_t frameDeadline, void* baseUserData)
{
    // Can only set base once
    if (basePicture != nullptr) {
        return LdcReturnCodeInvalidParam;
    }

    basePicture = picture;

    // Record metadata for output decoder info
    userData = baseUserData;

    if (basePicture) {
        baseWidth = ldpPictureLayoutWidth(&basePicture->layout);
        baseHeight = ldpPictureLayoutHeight(&basePicture->layout);
        baseBitdepth = ldpPictureLayoutSampleBits(&basePicture->layout);
        baseFormat = ldpPictureLayoutFormat(&basePicture->layout);
    }
    deadline = frameDeadline;

    if (m_state != FrameStateReorder && !m_intermediateInitialized) {
        initializeIntermediateBuffers();
    }

    // Mark dependency as met if task group is initialised
    if (m_depBasePicture != kTaskDependencyInvalid) {
        assert(m_taskGroup.pool);
        ldcTaskDependencyMet(&m_taskGroup, m_depBasePicture, basePicture);
    }
    return LdcReturnCodeSuccess;
}

void FrameCPU::getBasePlaneDesc(uint32_t plane, LdpPicturePlaneDesc& planeDesc) const
{
    const PictureCPU* picture = static_cast<const PictureCPU*>(basePicture);
    picture->getPlaneDescInternal(plane, planeDesc);
}

void FrameCPU::getOutputPlaneDesc(uint32_t plane, LdpPicturePlaneDesc& planeDesc) const
{
    const PictureCPU* picture = static_cast<const PictureCPU*>(outputPicture);
    picture->getPlaneDescInternal(plane, planeDesc);
}

void FrameCPU::getIntermediatePlaneDesc(uint32_t plane, LdeLOQIndex loq, LdpPicturePlaneDesc& planeDesc) const
{
    assert(loq < LOQMaxCount);
    assert(plane < RCMaxPlanes);
    assert(m_intermediateBufferPtr[plane][loq]);

    planeDesc.firstSample = m_intermediateBufferPtr[plane][loq];
    planeDesc.rowByteStride = ldpPictureLayoutRowStride(&m_intermediateLayout[loq], plane);
}

void FrameCPU::getTemporalBufferPlaneDesc(uint32_t plane, LdpPicturePlaneDesc& planeDesc) const
{
    assert(plane < RCMaxPlanes);
    assert(m_temporalBuffer[plane]);

    planeDesc = m_temporalBuffer[plane]->planeDesc;
}

void FrameCPU::getUpscalePlaneDesc(uint32_t plane, LdeLOQIndex loq, LdpPicturePlaneDesc& planeDesc) const
{
    assert(loq < LOQMaxCount);
    assert(plane < RCMaxPlanes);

    planeDesc.firstSample = VNAllocationPtr(m_upscaleBufferAllocation[plane][loq], uint8_t);
    planeDesc.rowByteStride = ldpPictureLayoutRowStride(&m_upscaleLayout[loq], plane);
}

bool FrameCPU::canComplete() const
{
    if (m_state != FrameStateProcessing) {
        return false;
    }

    // Collect all the input dependencies
    LdcTaskDependency deps[2 + RCMaxPlanes] = {m_depOutputPicture, m_depBasePicture};
    uint32_t depsCount = 2;

    for (uint8_t i = 0; i < RCMaxPlanes; i++) {
        if (m_depTemporalBuffer[i] != kTaskDependencyInvalid) {
            deps[depsCount++] = m_depTemporalBuffer[i];
        }
    }

    return ldcTaskDependencySetIsMet(&m_taskGroup, deps, depsCount);
}

LdcTaskDependency FrameCPU::needsTemporalBuffer(uint64_t previousTimestamp, uint32_t plane)
{
    const auto [width, height] = temporalDimensions(plane);

    // Fill in requirements
    m_temporalBufferDesc[plane].timestamp = previousTimestamp;
    m_temporalBufferDesc[plane].clear = config.nalType == NTIDR || config.temporalRefresh;
    m_temporalBufferDesc[plane].width = width;
    m_temporalBufferDesc[plane].height = height;
    m_temporalBufferDesc[plane].plane = plane;

    VNLogDebug("needsTemporalBuffer: %" PRIx64 " wants %" PRIx64 " plane %" PRIu32 " (%d %dx%d)",
               timestamp, previousTimestamp, plane, m_temporalBufferDesc[plane].clear, width, height);

    const LdcTaskDependency dep{ldcTaskDependencyAdd(&m_taskGroup)};
    m_depTemporalBuffer[plane] = dep;
    return dep;
}

bool FrameCPU::tryAttachTemporalBuffer(uint32_t plane, TemporalBuffer* temporalBuffer)
{
    assert(temporalBuffer);
    assert(plane < VNArraySize(m_temporalBuffer));
    assert(m_temporalBuffer[plane] == nullptr);
    assert(temporalBuffer->frame == nullptr);

    // Plane must match
    if (temporalBuffer->desc.plane != plane) {
        return false;
    }

    // Match and connect:
    //  Either:  A 'temporal clear' with invalid temporal buffer
    //  Or:      Active temporal buffer with required timestamp
    //
    if ((m_temporalBufferDesc[plane].clear && temporalBuffer->desc.timestamp == kInvalidTimestamp) ||
        (temporalBuffer->desc.timestamp == m_temporalBufferDesc[plane].timestamp)) {
        m_temporalBuffer[plane] = temporalBuffer;
        temporalBuffer->frame = this;
        return true;
    }

    return false;
}

// Make sure temporal buffer match required dimensions, and mark it's dependency as met
void FrameCPU::updateTemporalBuffer(uint32_t plane)
{
    assert(plane < VNArraySize(m_temporalBuffer));

    if (!m_temporalBuffer[plane]) {
        return;
    }

    m_temporalBuffer[plane]->update(m_temporalBufferDesc[plane]);

    ldcTaskDependencyMet(&m_taskGroup, m_depTemporalBuffer[plane], m_temporalBuffer[plane]);
}

TemporalBuffer* FrameCPU::detachTemporalBuffer(uint32_t plane)
{
    assert(plane < VNArraySize(m_temporalBuffer));

    TemporalBuffer* const tb = m_temporalBuffer[plane];
    if (!tb) {
        return nullptr;
    }

    m_temporalBuffer[plane] = nullptr;
    tb->frame = nullptr;

    // Fill in timestamp
    tb->desc.timestamp = timestamp;
    return tb;
}

bool FrameCPU::isPlaneEnhanced(LdeLOQIndex loq, uint32_t plane) const
{
    return config.frameConfigSet && config.loqEnabled[loq] && plane < globalConfig->numPlanes;
}

LdpColorFormat FrameCPU::getUpscaleColorFormat() const
{
    if ((baseFormat == LdpColorFormatNV12_8 || baseFormat == LdpColorFormatNV21_8) &&
        ((globalConfig->scalingModes[LOQ1] != Scale0D) != (globalConfig->scalingModes[LOQ0] != Scale0D))) {
        return baseFormat;
    }

    switch (globalConfig->chroma) {
        case CTMonochrome:
            switch (globalConfig->enhancedDepth) {
                case Depth8: return LdpColorFormatGRAY_8;
                case Depth10: return LdpColorFormatGRAY_10_LE;
                case Depth12: return LdpColorFormatGRAY_12_LE;
                case Depth14: return LdpColorFormatGRAY_14_LE;
                default: return LdpColorFormatUnknown;
            }
        case CT420:
            switch (globalConfig->enhancedDepth) {
                case Depth8: return LdpColorFormatI420_8;
                case Depth10: return LdpColorFormatI420_10_LE;
                case Depth12: return LdpColorFormatI420_12_LE;
                case Depth14: return LdpColorFormatI420_14_LE;
                default: return LdpColorFormatUnknown;
            }
        case CT422:
            switch (globalConfig->enhancedDepth) {
                case Depth8: return LdpColorFormatI422_8;
                case Depth10: return LdpColorFormatI422_10_LE;
                case Depth12: return LdpColorFormatI422_12_LE;
                case Depth14: return LdpColorFormatI422_14_LE;
                default: return LdpColorFormatUnknown;
            }
        case CT444:
            switch (globalConfig->enhancedDepth) {
                case Depth8: return LdpColorFormatI444_8;
                case Depth10: return LdpColorFormatI444_10_LE;
                case Depth12: return LdpColorFormatI444_12_LE;
                case Depth14: return LdpColorFormatI444_14_LE;
                default: return LdpColorFormatUnknown;
            }
        default: return LdpColorFormatUnknown;
    }
}

// Figure output colour format from frame configuration
LdpColorFormat FrameCPU::getOutputColorFormat() const
{
    if (baseFormat == LdpColorFormatNV12_8 || baseFormat == LdpColorFormatNV21_8) {
        if (globalConfig->enhancedDepth != Depth8) {
            VNLogError("Cannot enhance to > 8bit when using an NV12/21 base");
            return LdpColorFormatUnknown;
        }
        return baseFormat;
    }

    return getUpscaleColorFormat();
}

// Figure base colour format from frame configuration
//
// This does not know about interleaving eg: NV12
//
LdpColorFormat FrameCPU::getBaseColorFormat() const
{
    switch (globalConfig->chroma) {
        case CTMonochrome:
            switch (globalConfig->baseDepth) {
                case Depth8: return LdpColorFormatGRAY_8;
                case Depth10: return LdpColorFormatGRAY_10_LE;
                case Depth12: return LdpColorFormatGRAY_12_LE;
                case Depth14: return LdpColorFormatGRAY_14_LE;
                default: return LdpColorFormatUnknown;
            }
        case CT420:
            switch (globalConfig->baseDepth) {
                case Depth8: return LdpColorFormatI420_8;
                case Depth10: return LdpColorFormatI420_10_LE;
                case Depth12: return LdpColorFormatI420_12_LE;
                case Depth14: return LdpColorFormatI420_14_LE;
                default: return LdpColorFormatUnknown;
            }
        case CT422:
            switch (globalConfig->baseDepth) {
                case Depth8: return LdpColorFormatI422_8;
                case Depth10: return LdpColorFormatI422_10_LE;
                case Depth12: return LdpColorFormatI422_12_LE;
                case Depth14: return LdpColorFormatI422_14_LE;
                default: return LdpColorFormatUnknown;
            }
        case CT444:
            switch (globalConfig->baseDepth) {
                case Depth8: return LdpColorFormatI444_8;
                case Depth10: return LdpColorFormatI444_10_LE;
                case Depth12: return LdpColorFormatI444_12_LE;
                case Depth14: return LdpColorFormatI444_14_LE;
                default: return LdpColorFormatUnknown;
            }
        default: return LdpColorFormatUnknown;
    }
}

LdpPictureDesc FrameCPU::getOutputPictureDesc(PassthroughMode passthroughMode) const
{
    LdpPictureDesc desc;

    assert(baseDataValid());

    if (!m_passthrough) {
        if (globalConfig->initialized) {
            ldpDefaultPictureDesc(&desc, getOutputColorFormat(), globalConfig->width, globalConfig->height);
        } else {
            VNLogWarning("No global configuration");
            ldpDefaultPictureDesc(&desc, baseFormat, baseWidth, baseHeight);
        }
    } else {
        // Pass-through of some sort
        if (passthroughMode == PassthroughMode::Scale && globalConfig->initialized) {
            ldpDefaultPictureDesc(&desc, getOutputColorFormat(), globalConfig->width, globalConfig->height);
        } else {
            ldpDefaultPictureDesc(&desc, baseFormat, baseWidth, baseHeight);
        }
    }

    return desc;
}

uint8_t FrameCPU::numImagePlanes() const
{
    switch (globalConfig->chroma) {
        case CTMonochrome: return 1;
        case CT420:
        case CT422:
        case CT444: return 3;
        default: assert(0); return 0;
    }
}

std::pair<uint32_t, uint32_t> FrameCPU::temporalDimensions(uint32_t plane) const
{
    if (plane == 0) {
        return {globalConfig->width, globalConfig->height};
    }

    switch (globalConfig->chroma) {
        case CT420: return {globalConfig->width >> 1, globalConfig->height >> 1};
        case CT422: return {globalConfig->width >> 1, globalConfig->height};
        case CT444: return {globalConfig->width, globalConfig->height};
        default: assert(0); return {globalConfig->width, globalConfig->height};
    }
}

#ifdef VN_SDK_LOG_ENABLE_DEBUG
// Write description of frame into string buffer
// Return number of characters written to buffer

namespace {
    const char* frameStateName(FrameState state)
    {
        switch (state) {
            case FrameStateUnknown: return "Unkwn";
            case FrameStateReorder: return "ReOrd";
            case FrameStateProcessing: return "Prcss";
            case FrameStateDone: return "Done ";
            case FrameStateFlush:
                return "Flush";
                //    default:
        }
        return "Unknown";
    }
} // namespace

size_t FrameCPU::longDescription(char* buffer, size_t bufferSize) const
{
    return snprintf(buffer, bufferSize,
                    "ts:%" PRIx64 " st:%s pass:%d"
                    " gc:%p base:%p output:%p etc:%d "
                    "esize:%zd tg.tc:%d tg.wt:%d tg.dc:%d tg.met:%" PRIx64 " depB:%d "
                    "depO:%d depT:%d tbd:%" PRIx64 ",%d,%d,%d "
                    "tb:[%p,%p,%p]",
                    timestamp, frameStateName(m_state.load()), m_passthrough, globalConfig,
                    basePicture, outputPicture, enhancementTileCount, m_enhancementData.size,
                    m_taskGroup.tasksCount, m_taskGroup.waitingTasksCount, m_taskGroup.dependenciesCount,
                    m_taskGroup.dependenciesMet ? m_taskGroup.dependenciesMet[0] : 0,
                    m_depBasePicture, m_depOutputPicture, m_depTemporalBuffer[0],
                    m_temporalBufferDesc[0].timestamp, m_temporalBufferDesc[0].clear,
                    m_temporalBufferDesc[0].width, m_temporalBufferDesc[0].height,
                    m_temporalBuffer[0], m_temporalBuffer[1], m_temporalBuffer[2]);
}

void FrameCPU::dumpTasks(LdcTaskPool* taskPool) const { ldcTaskPoolDump(taskPool, &m_taskGroup); }
#endif

} // namespace lcevc_dec::pipeline_cpu
