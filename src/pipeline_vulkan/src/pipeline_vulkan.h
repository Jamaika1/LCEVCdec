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

#ifndef VN_LCEVC_PIPELINE_VULKAN_PIPELINE_VULKAN_H
#define VN_LCEVC_PIPELINE_VULKAN_PIPELINE_VULKAN_H

// #define DEBUG_LOGGING

#include "backend_vulkan.h"

#include <LCEVC/common/class_utils.hpp>
#include <LCEVC/common/constants.h>
#include <LCEVC/common/diagnostics.h>
#include <LCEVC/common/ring_buffer.hpp>
#include <LCEVC/common/rolling_arena.h>
#include <LCEVC/common/task_pool.h>
#include <LCEVC/common/threads.h>
#include <LCEVC/common/threads.hpp>
#include <LCEVC/common/vector.hpp>
#include <LCEVC/enhancement/config_pool.h>
#include <LCEVC/pipeline/event_sink.h>
#include <LCEVC/pipeline/frame.h>
#include <LCEVC/pipeline/pipeline.h>
#include <LCEVC/pixel_processing/dither.h>
#include <picture_vulkan.h>
#include <pipeline_builder_vulkan.h>
//
#include <atomic>

// Set to true to debug Vulkan core
const bool enableValidationLayers = false;

namespace lcevc_dec::pipeline_vulkan {

class BufferVulkan;
class FrameVulkan;
class PictureVulkan;
struct VulkanApplyArgs;
struct VulkanBlitArgs;
struct VulkanConversionArgs;
struct VulkanUpscaleArgs;

// Description of a temporal buffer
struct TemporalBufferDesc
{
    uint64_t timestamp;
    bool clear;
    uint32_t plane;
    uint32_t width;
    uint32_t height;
};

// Temporal buffer associated with pipeline
//
struct TemporalBuffer
{
    // Description of this buffer
    TemporalBufferDesc desc;

    // Timestamp upper limit that this buffer could fulfil
    uint64_t timestampLimit;

    // Frame that is using this buffer or null if available
    FrameVulkan* frame;
    // pointer and stride for buffer
    LdpPicturePlaneDesc planeDesc;

    // Allocator to use
    // NB: this should not be part of the frame arena, as the temporal
    // buffers can survive for many frames.
    LdcMemoryAllocator* allocator;

    // Buffer allocation
    LdcMemoryAllocation allocation;

    // Reallocate buffer to match given description
    void update(const TemporalBufferDesc& newDesc)
    {
        VNLogWarning("Temporal buffer update not supported yet on vulkan");
    }
};

// A base picture reference and other arguments from sendBase()
//
// Used for pending base pictures, before association with frames.
//
struct BasePicture
{
    uint64_t timestamp;
    LdpPicture* picture;
    uint64_t deadline;
    void* userData;
};

class PipelineVulkan : public pipeline::Pipeline
{
public:
    PipelineVulkan(const PipelineBuilderVulkan& builder, pipeline::EventSink* eventSink);
    ~PipelineVulkan() override;

    // Send/receive
    LdcReturnCode sendDecoderBase(uint64_t timestamp, LdpPicture* basePicture, uint32_t timeoutUs,
                                  void* userData) override;
    LdcReturnCode sendDecoderEnhancementData(uint64_t timestamp, const uint8_t* data,
                                             uint32_t byteSize) override;
    LdcReturnCode sendDecoderPicture(LdpPicture* outputPicture) override;

    LdpPicture* receiveDecoderPicture(LdpDecodeInformation& decodeInfoOut) override;
    LdpPicture* receiveDecoderBase() override;

    void getCapacity(LdpPipelineCapacity* capacity) override;

    // Skip/flush
    LdcReturnCode skip(uint64_t timestamp) override;
    LdcReturnCode flush(uint64_t timestamp) override;
    LdcReturnCode peekDecoder(uint64_t timestamp, uint32_t& widthOut, uint32_t& heightOut) override;

    LdcReturnCode synchronizeDecoder(uint64_t timestamp, bool flushPending) override;

    // Check frame against current limits
    bool isProcessing(const FrameVulkan* frame) const;
    bool isSkipped(const FrameVulkan* frame) const;
    bool isFlushed(const FrameVulkan* frame) const;

    // Picture-handling
    LdpPicture* allocPicture(const LdpPictureDesc& desc) override;
    LdpPicture* allocPictureExternal(const LdpPictureDesc& desc, const LdpPicturePlaneDesc* planeDescArr,
                                     const LdpPictureBufferDesc* buffer) override;

    void freePicture(LdpPicture* picture) override;

    // Accessors for use by frames
    const PipelineConfigVulkan& configuration() const { return m_configuration; }
    LdcMemoryAllocator* allocator() const { return m_allocator; }
    LdcTaskPool* taskPool() { return &m_taskPool; }
    LdppDitherGlobal* globalDitherBuffer() { return &m_dither; }

    // Buffer allocation
    BufferVulkan* allocateBuffer(uint32_t requiredSize);
    void releaseBuffer(BufferVulkan* buffer);

    // Picture allocation
    PictureVulkan* allocatePicture();
    void releasePicture(PictureVulkan* picture);

    //// Temporal buffer management

    // Look through all temporal buffers, looking for one that matches the given frame and plane's requirements
    TemporalBuffer* findTemporalBuffer(FrameVulkan* frame, uint32_t plane);

    // Mark the frame as having finished with it's temporal buffer and hand off to next frame that needs it
    void transferTemporalBuffer(FrameVulkan* frame, uint32_t plane);

    // End of frame processing
    void baseDone(LdpPicture* picture);
    void outputDone(FrameVulkan* frame);

    void updateTemporalBufferDesc(TemporalBuffer* buffer, const TemporalBufferDesc& desc) const;

#ifdef VN_SDK_LOG_ENABLE_DEBUG
    // Write Debug log of current frame state
    void logFrames();
    void logFrameIndex(const char* indexName, const lcevc_dec::common::Vector<FrameVulkan*>& index) const;
#endif

    // Vulkan core management
    BackendVulkan& getCore() { return m_core; }
    bool isInitialised() const { return m_initialised; }
    void prepareApplyArgs(VulkanApplyArgs& args, PictureVulkan* picture,
                          LdpEnhancementTile* enhancementTile, FrameVulkan* frame, bool applyDirect);

    void getSubsamplingShifts(LdeChroma chroma, int& widthShift, int& heightShift);
    LdpColorFormat chromaToColorFormat(LdeChroma chroma);
    LdeChroma getChroma() const { return m_chroma; }
    void setChroma(LdeChroma chroma) { m_chroma = chroma; }

    VNNoCopyNoMove(PipelineVulkan);

    // TODO - getters
    // Plane for intermediate vertical upscale
    std::unique_ptr<PictureVulkan> m_intermediateUpscalePicture[LOQEnhancedCount] = {};

    // Planes for apply
    std::unique_ptr<PictureVulkan> m_temporalPicture = nullptr;

private:
    friend PipelineBuilderVulkan;

    // Given a timestamp, either find existing frame, or create a new one
    FrameVulkan* allocateFrame(uint64_t timestamp);

    // Find the Frame associated with a timestamp, or NULL if none.
    FrameVulkan* findFrame(uint64_t timestamp);

    // Frame for given timestamp is finished - release resources
    void releaseFrame(uint64_t timestamp);
    void freeFrame(FrameVulkan* frame);

    // Get next frame reference following reorder and flushing rules
    FrameVulkan* getNextReordered();

    // Move frames from reorder table to generated tasks
    void startReadyFrames();

    // Assign incoming output pictures to Frames
    void connectOutputPictures();

    void unblockSkippedFrames(uint64_t fromTimestamp);
    void unblockFlushedFrames(uint64_t fromTimestamp);
    void releaseFlushedFrames();

    // Number of outstanding frames
    uint32_t frameLatency() const;

    // Move any frames before `timestamp` into processing queue
    void process(uint64_t timestamp);

    // Try to match a frame to current temporal buffer(s)
    TemporalBuffer* matchTemporalBuffer(FrameVulkan* frame, uint32_t plane);

    // Configuration from builder
    const PipelineConfigVulkan m_configuration;

    // Interface to event mechanism
    pipeline::EventSink* m_eventSink{};

    // The system allocator to use
    LdcMemoryAllocator* m_allocator{};

    // A rolling memory allocator for per-frame blocks
    LdcMemoryAllocatorRollingArena m_rollingArena{};

    // Enhancement configuration pool
    LdeConfigPool m_configPool{};

    // Task pool
    LdcTaskPool m_taskPool{};

    // Vector of Buffer allocations
    lcevc_dec::common::Vector<LdcMemoryAllocation> m_buffers;

    // Vector of Picture allocations
    lcevc_dec::common::Vector<LdcMemoryAllocation> m_pictures;

    // Vector of Frames allocations
    // These frames are NOT in timestamp order.
    // The `m_...Index` vectors contain timestamp-order pointers to theFrameCPU structures.
    lcevc_dec::common::Vector<LdcMemoryAllocation> m_frames;

    // Vector of pending frames pointers during reorder - sorted by timestamp
    lcevc_dec::common::Vector<FrameVulkan*> m_reorderIndex;

    // Vector of pending frames pointers whilst in progress - sorted by timestamp
    lcevc_dec::common::Vector<FrameVulkan*> m_processingIndex;

    // Vector of pending frames pointers when done - sorted by timestamp
    lcevc_dec::common::Vector<FrameVulkan*> m_doneIndex;

    // Vector of pending frames pointers when flushed - sorted by timestamp
    lcevc_dec::common::Vector<FrameVulkan*> m_flushIndex;

    // Limit for frame reordering - can be dynamically updated as enhancement data comes in
    uint32_t m_maxReorder{};

    // The timestamp of the highest sent enhancement frame
    std::atomic<uint64_t> m_sendLimit{kInvalidTimestamp};

    // Timestamp for frames in processing state
    std::atomic<uint64_t> m_processingLimit{kInvalidTimestamp};

    // Timestamp for frames to be skipped
    std::atomic<uint64_t> m_skipLimit{kInvalidTimestamp};

    // Timestamp for frames to be flushed
    std::atomic<uint64_t> m_flushLimit{kInvalidTimestamp};

    // The timestamp of the last frame to have it's config parsed successfully
    uint64_t m_lastGoodTimestamp{kInvalidTimestamp};

    // The prior frame during initial in-order config parsing - used to negotiate temporal buffers
    uint64_t m_previousTimestamp{kInvalidTimestamp};

    // Vector of temporal buffers
    // A small pool of  (1 or more) temporal buffers is allocated on startup, then transferred
    // between frames.
    lcevc_dec::common::Vector<TemporalBuffer> m_temporalBuffers;

    // Pending base pictures
    lcevc_dec::common::Vector<BasePicture> m_basePicturePending;

    // Base pictures Out - thread safe FIFO
    lcevc_dec::common::RingBuffer<LdpPicture*> m_basePictureOutBuffer;

    // Output pictures available for rendering - thread safe FIFO
    lcevc_dec::common::RingBuffer<LdpPicture*> m_outputPictureAvailableBuffer;

    // Global dither module
    LdppDitherGlobal m_dither{};

    // Lock for interaction between frame tasks and pipeline - when temporal buffers
    // are handed over / negotiated.
    //
    // Protects m_temporalBuffers and m_processingIndex
    common::Mutex m_interTaskMutex;

    // Signalled when frames are done, whilst holding m_interTaskMutex
    common::CondVar m_interTaskFrameDone;

    BackendVulkan m_core;

    bool m_initialised = false;

    LdeChroma m_chroma = LdeChroma::CT420;
};

} // namespace lcevc_dec::pipeline_vulkan

#endif // VN_LCEVC_PIPELINE_VULKAN_PIPELINE_VULKAN_H
