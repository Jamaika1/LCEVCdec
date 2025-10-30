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

#ifndef VN_LCEVC_PIPELINE_CPU_FRAME_CPU_H
#define VN_LCEVC_PIPELINE_CPU_FRAME_CPU_H

#include "pipeline_cpu.h"

#include <LCEVC/common/class_utils.hpp>
#include <LCEVC/common/return_code.h>
#include <LCEVC/common/task_pool.h>
#include <LCEVC/enhancement/dimensions.h>
#include <LCEVC/pipeline/frame.h>
#include <LCEVC/pixel_processing/dither.h>

#include <atomic>
#include <cstdint>

namespace lcevc_dec::pipeline_cpu {

class PipelineCPU;
class PictureCPU;

enum FrameState
{
    FrameStateUnknown,
    FrameStateReorder,
    FrameStateProcessing,
    FrameStateDone,
    FrameStateFlush,
};

// Extended frame structure for this pipeline
//
class FrameCPU : public LdpFrame
{
public:
    FrameCPU(LdcMemoryAllocator* enhancementAllocator, LdcMemoryAllocator* bufferAllocator,
             uint64_t timestamp);
    ~FrameCPU() = default;

    // Allocate per frame buffers
    bool initialize(const PipelineConfigCPU& configuration, LdcTaskPool* taskPool,
                    LdppDitherGlobal* ditherBuffer);

    // Attach enhancement data
    uint8_t* setEnhancementData(const uint8_t* data, uint32_t byteSize);
    // Parse enhancement data into config state
    bool parseEnhancementData(LdeConfigPool* configPool);

    // Frame state
    void setState(FrameState state) { m_state = state; }
    bool isStateDone() const { return m_state == FrameStateDone; }
    bool isStateProcessing() const { return m_state == FrameStateProcessing; }

    void setPassthrough() { m_passthrough = true; }
    bool isPassthrough() const { return m_passthrough; }

    //
    bool hasGoodConfig() const { return globalConfig && globalConfig->initialized; }

    // Connect output picture to frame
    void setOutputPicture(LdpPicture* picture);

    // Wait for all runnable tasks in group to complete
    void waitForTasks();
    void waitForCompletableTasks();

    // Tidy up
    void release(bool wait);

    bool initializeCommandBuffers();
    void releaseCommandBuffers();

    bool initializeIntermediateBuffers();
    void releaseIntermediateBuffers();

    // Mark a dependency met if it is valid - used for unblocking skipped frames
    void unblockDependency(LdcTaskDependency dep);

    void unblockForSkip();
    void unblockForFlush();

    // Find command buffer given the tile index
    LdpEnhancementTile* getEnhancementTile(uint32_t tileIdx) const
    {
        assert(tileIdx < ldeTotalNumTiles(globalConfig));

        return VNAllocationPtr(m_enhancementTilesAllocation, LdpEnhancementTile) + tileIdx;
    }

    // Attach a base picture to the frame (and mark dependency as met)
    LdcReturnCode setBasePicture(LdpPicture* picture, uint64_t deadline, void* userData);

    // Return true if a base picture has been set for frame, and it's description recorded
    // NB: the base picture itself may have gone by time this data is needed at output time
    bool baseDataValid() const { return baseFormat != LdpColorFormatUnknown; }

    // Return true if frame need an intermediate buffer for given loq/plane
    bool needsIntermediateBuffer(LdeLOQIndex loq, uint8_t plane) const;

    // Setup frame plane's requirements for temporal buffer, given prior frame's timestamp
    LdcTaskDependency needsTemporalBuffer(uint64_t previousTimestamp, uint32_t plane);

    // Try to connect a prior temporal buffer to this frame
    bool tryAttachTemporalBuffer(uint32_t plane, TemporalBuffer* temporalBuffer);

    // Make the allocated temporal buffer match the frame's description
    void updateTemporalBuffer(uint32_t plane);

    // Disconnect any temporal buffer from frame plane, and return it
    TemporalBuffer* detachTemporalBuffer(uint32_t plane);

    // Return true if frame need an upscale buffer for given loq/plane
    bool needsUpscaleBuffer(LdeLOQIndex loq) const;

    // Return true if the frame has everything such that it will complete without further inputs
    bool canComplete() const;

    // Get the color format of base picture
    LdpColorFormat getBaseColorFormat() const;

    // Find the color format for the output picture
    LdpColorFormat getOutputColorFormat() const;

    // Find the color format for upscale planes
    LdpColorFormat getUpscaleColorFormat() const;

    // Construct picture description for output
    LdpPictureDesc getOutputPictureDesc(PassthroughMode passthroughMode) const;

    uint8_t numEnhancedPlanes() const { return globalConfig->numPlanes; }
    uint8_t numImagePlanes() const;
    std::pair<uint32_t, uint32_t> temporalDimensions(uint32_t plane) const;

    // Return true if the frame's plane should have enhancement applied
    bool isPlaneEnhanced(LdeLOQIndex loq, uint32_t plane) const;

    // Get plane descriptions for each of the buffers
    void getBasePlaneDesc(uint32_t plane, LdpPicturePlaneDesc& planeDesc) const;
    void getOutputPlaneDesc(uint32_t plane, LdpPicturePlaneDesc& planeDesc) const;
    void getIntermediatePlaneDesc(uint32_t plane, LdeLOQIndex loq, LdpPicturePlaneDesc& planeDesc) const;
    void getUpscalePlaneDesc(uint32_t plane, LdeLOQIndex loq, LdpPicturePlaneDesc& planeDesc) const;
    void getTemporalBufferPlaneDesc(uint32_t plane, LdpPicturePlaneDesc& planeDesc) const;

    const LdpPictureLayout* getIntermediateLayout(LdeLOQIndex loq) const
    {
        assert(loq < LOQMaxCount);
        return &m_intermediateLayout[loq];
    }

    const LdpPictureLayout* getUpscaleLayout(LdeLOQIndex loq) const
    {
        assert(loq < LOQMaxCount);
        return &m_upscaleLayout[loq];
    }

    //
    LdcTaskDependency depBasePicture() const { return m_depBasePicture; }
    LdcTaskDependency depOutputPicture() const { return m_depOutputPicture; }
    LdcTaskDependency depTemporalBuffer(uint32_t plane)
    {
        assert(plane < RCMaxPlanes);
        return m_depTemporalBuffer[plane];
    }

    const LdppDitherFrame* dither() const
    {
        return m_frameDither.strength ? &m_frameDither : nullptr;
    }

    // Get this frame's task group
    LdcTaskGroup* taskGroup() { return &m_taskGroup; }

    // Add a task to frame's task group with given inputs and return a new output dependency
    LdcTaskDependency taskAdd(const LdcTaskDependency* inputs, uint32_t inputsCount, LdcTaskFunction task,
                              const void* data, size_t dataSize, const char* name);

    // Add a task to frame's task group that takes inputs, but does not generate an output
    void taskAddSink(const LdcTaskDependency* inputs, uint32_t inputsCount, LdcTaskFunction task,
                     const void* data, size_t dataSize, const char* name);

    // Find the dependencies that depend on frame's base picture
    bool findOutputSetFromBase(LdcTaskDependency* outputs, uint32_t outputsMax, uint32_t* outputsCount) const;

#ifdef VN_SDK_LOG_ENABLE_DEBUG
    // Create a debug description of this frame
    size_t longDescription(char* buffer, size_t bufferSize) const;
    void dumpTasks(LdcTaskPool* taskPool) const;
#endif

    VNNoCopyNoMove(FrameCPU);

private:
    // The allocators to use for this frame
    LdcMemoryAllocator* m_enhancementAllocator{};
    LdcMemoryAllocator* m_bufferAllocator{};

    // Current frame state
    // This get peeked at across threads - so it is atomic
    std::atomic<FrameState> m_state{FrameStateUnknown};

    // Task group for this frame
    LdcTaskGroup m_taskGroup{};

    // Allocation for un-encapsulated enhancement data
    LdcMemoryAllocation m_enhancementData{};

    // An array of LdpEnhancementTile
    LdcMemoryAllocation m_enhancementTilesAllocation{};

    // Internal buffers for residual application
    LdcMemoryAllocation m_intermediateBufferAllocation[RCMaxPlanes][LOQMaxCount] = {};
    LdpPictureLayout m_intermediateLayout[LOQMaxCount] = {};

    // Internal buffers for 2D upscale
    LdcMemoryAllocation m_upscaleBufferAllocation[RCMaxPlanes][LOQEnhancedCount] = {};
    LdpPictureLayout m_upscaleLayout[LOQEnhancedCount] = {};

    // Pointers to buffer to use for each LOQ - may share buffers between LoQs depending on scaling modes
    uint8_t* m_intermediateBufferPtr[RCMaxPlanes][LOQMaxCount] = {};

    // True if intermediate buffers are setup
    bool m_intermediateInitialized{false};

    // Dependencies for inputs to task group
    LdcTaskDependency m_depBasePicture{kTaskDependencyInvalid};
    LdcTaskDependency m_depOutputPicture{kTaskDependencyInvalid};
    LdcTaskDependency m_depTemporalBuffer[RCMaxPlanes] = {kTaskDependencyInvalid, kTaskDependencyInvalid,
                                                          kTaskDependencyInvalid};

    // Description of temporal buffer(s) needed for this frame
    TemporalBufferDesc m_temporalBufferDesc[RCMaxPlanes] = {};

    // Temporal buffer(s) assigned to this frame (once dependency is met)
    TemporalBuffer* m_temporalBuffer[RCMaxPlanes] = {};

    // Dithering info for this frame
    LdppDitherFrame m_frameDither{};

    // True if this frame should be copied to output with no processing (but optionally scaled)
    bool m_passthrough{false};
};

} // namespace lcevc_dec::pipeline_cpu

#endif // VN_LCEVC_PIPELINE_CPU_FRAME_CPU_H
