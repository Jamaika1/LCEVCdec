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

#ifndef VN_LCEVC_PIPELINE_CPU_PIPELINE_CPU_H
#define VN_LCEVC_PIPELINE_CPU_PIPELINE_CPU_H

#include "buffer_cpu.h"
#include "pipeline_builder_cpu.h"
#include "temporal_buffer_cpu.h"

#include <LCEVC/common/constants.h>
#include <LCEVC/common/free_pool.hpp>
#include <LCEVC/common/threads.h>
//
#include <LCEVC/common/class_utils.hpp>
#include <LCEVC/common/recycling_allocator.h>
#include <LCEVC/common/ring_buffer.hpp>
#include <LCEVC/common/rolling_arena.h>
#include <LCEVC/common/simple_allocator.h>
#include <LCEVC/common/task_pool.h>
#include <LCEVC/common/threads.hpp>
#include <LCEVC/common/vector.hpp>
#include <LCEVC/enhancement/config_pool.h>
#include <LCEVC/pipeline/event_sink.h>
#include <LCEVC/pipeline/frame.h>
#include <LCEVC/pipeline/pipeline.h>
#include <LCEVC/pixel_processing/dither.h>

#include <atomic>

namespace lcevc_dec::pipeline_cpu {

class BufferCPU;
class FrameCPU;
class PictureCPU;

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

// PipelineCPU
//
class PipelineCPU : public pipeline::Pipeline
{
public:
    PipelineCPU(const PipelineBuilderCPU& builder, pipeline::EventSink* eventSink);
    ~PipelineCPU() override;

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

    // Picture-handling
    LdpPicture* allocPicture(const LdpPictureDesc& desc) override;
    LdpPicture* allocPictureExternal(const LdpPictureDesc& desc, const LdpPicturePlaneDesc* planeDescArr,
                                     const LdpPictureBufferDesc* buffer) override;

    void freePicture(LdpPicture* picture) override;

    // Check frame against current limits
    bool isProcessing(const FrameCPU* frame) const;
    bool isSkipped(const FrameCPU* frame) const;
    bool isFlushed(const FrameCPU* frame) const;

    // Accessors for use by frames
    const PipelineConfigCPU& configuration() const { return m_configuration; }
    LdcMemoryAllocator* staticAllocator() const { return m_allocator; }
    LdcMemoryAllocator* rollingAllocator() { return &m_simpleAllocator.allocator; }

    LdcTaskPool* taskPool() { return &m_taskPool; }
    LdppDitherGlobal* globalDitherBuffer() { return &m_dither; }

    // Buffer allocation
    BufferCPU* allocateBuffer(uint32_t requiredSize);
    void releaseBuffer(BufferCPU* buffer);

    //// Temporal buffer management

    // Look through all temporal buffers, looking for one that matches the given frame and plane's requirements
    TemporalBuffer* findTemporalBuffer(FrameCPU* frame, uint32_t plane);

    // Mark the frame as having finished with it's temporal buffer and hand off to next frame that needs it
    void transferTemporalBuffer(FrameCPU* frame, uint32_t plane);

    // End of frame processing
    void baseDone(LdpPicture* picture);
    void outputDone(FrameCPU* frame);

    void updateTemporalBufferDesc(TemporalBuffer* buffer, const TemporalBufferDesc& desc) const;

#ifdef VN_SDK_LOG_ENABLE_DEBUG
    // Write Debug log of current frame state
    void logFrames();
    void logFrameIndex(const char* indexName, const lcevc_dec::common::Vector<FrameCPU*>& index) const;
#endif

    VNNoCopyNoMove(PipelineCPU);

private:
    friend PipelineBuilderCPU;

    // Picture allocation
    PictureCPU* allocatePicture();
    void releasePicture(PictureCPU* picture);
    uint32_t findAllocatedPicture(const PictureCPU* frame) const;

    // Given a timestamp, either find existing frame, or create a new one
    FrameCPU* allocateFrame(uint64_t timestamp);

    // Find the Frame associated with a timestamp, or NULL if none.
    FrameCPU* findFrame(uint64_t timestamp);

    // Find the index in allocated frames
    uint32_t findAllocatedFrame(const FrameCPU* frame) const;

    // Frame for given timestamp is finished - release resources
    void releaseFrame(uint64_t timestamp);
    void freeFrame(FrameCPU* frame);

    // Get next frame reference following reorder and flushing rules
    FrameCPU* getNextReordered();

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
    TemporalBuffer* matchTemporalBuffer(FrameCPU* frame, uint32_t plane);

    // Configuration from builder
    const PipelineConfigCPU m_configuration;

    // Interface to event mechanism
    pipeline::EventSink* m_eventSink{};

    // The system allocator to use
    LdcMemoryAllocator* m_allocator{};

    // The allocator for enhancement data
    LdcMemoryAllocator* m_enhancementAllocator{};

    // The allocator for image buffer data
    LdcMemoryAllocator* m_bufferAllocator{};

    // A rolling memory allocator for per-frame enhancement data - command buffers, tiles etc.
#if 0
    LdcMemoryAllocatorRollingArena m_rollingArena{};
#else
    LdcMemorySimpleAllocator m_simpleAllocator{};
#endif
    // A recycling buffer allocator for per-frame buffer data
    ldcMemoryRecyclingAllocator m_recyclingAllocator{};

    // Enhancement configuration pool
    LdeConfigPool m_configPool{};

    // Task pool
    LdcTaskPool m_taskPool{};

    // Pool of buffers
    common::FreePool<BufferCPU> m_buffersPool;

    // Pool of pictures
    common::FreePool<PictureCPU> m_picturesPool;

    // Pool of frames
    common::FreePool<FrameCPU> m_framesPool;

    // Vector of Picture allocations
    common::Vector<PictureCPU*> m_allocatedPictures;

    // Vector of Frames allocations
    // These frames are NOT in timestamp order.
    // The `m_...Index` vectors contain timestamp-order pointers to theFrameCPU structures.
    common::Vector<FrameCPU*> m_allocatedFrames;

    // Vector of pending frames pointers during reorder - sorted by timestamp
    common::Vector<FrameCPU*> m_reorderIndex;

    // Vector of pending frames pointers whilst in progress - sorted by timestamp
    common::Vector<FrameCPU*> m_processingIndex;

    // Vector of pending frames pointers when done - sorted by timestamp
    common::Vector<FrameCPU*> m_doneIndex;

    // Vector of pending frames pointers when flushed - sorted by timestamp
    common::Vector<FrameCPU*> m_flushIndex;

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
};

} // namespace lcevc_dec::pipeline_cpu

#endif // VN_LCEVC_PIPELINE_CPU_PIPELINE_CPU_H
