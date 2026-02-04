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

#include "pipeline_cpu.h"
//
#include "frame_cpu.h"
#include "picture_cpu.h"
#include "pipeline_config_cpu.h"
#include "tasks_cpu.h"

#include <LCEVC/common/recycling_allocator.h>
#include <LCEVC/common/simple_allocator.h>
//
#include <LCEVC/common/check.h>
#include <LCEVC/common/constants.h>
#include <LCEVC/common/diagnostics.h>
#include <LCEVC/common/limit.h>
#include <LCEVC/common/log.h>
#include <LCEVC/common/memory.h>
#include <LCEVC/common/return_code.h>
#include <LCEVC/common/task_pool.h>
#include <LCEVC/common/threads.h>
//
#include <cstdint>
#include <cstring>
#include <ctime>

namespace lcevc_dec::pipeline_cpu {

// Utility functions for finding and sorting things in Vectors
//
namespace {
    // Compare two frames in an array of frame pointers
    inline int sortFramePtrTimestamp(const void* lhs, const void* rhs)
    {
        const auto* frameLhs{*static_cast<const FrameCPU* const *>(lhs)};
        const auto* frameRhs{*static_cast<const FrameCPU* const *>(rhs)};

        return pipeline::compareTimestamps(frameLhs->timestamp, frameRhs->timestamp);
    }

    // Check timestamp of an allocated BasePicture
    inline int findBasePictureTimestamp(const void* element, const void* ptr)
    {
        const auto* alloc{static_cast<const LdcMemoryAllocation*>(element)};
        assert(VNIsAllocated(*alloc));
        const uint64_t ets{VNAllocationPtr(*alloc, BasePicture)->timestamp};
        const uint64_t ts{*static_cast<const uint64_t*>(ptr)};

        return pipeline::compareTimestamps(ets, ts);
    }

    inline int compareFramePtr(const void* element, const void* other)
    {
        const auto* frameLhs{*static_cast<const FrameCPU* const *>(element)};
        const auto* frameRhs{static_cast<const FrameCPU*>(other)};

        if (frameLhs < frameRhs) {
            return -1;
        }
        if (frameLhs > frameRhs) {
            return 1;
        }
        return 0;
    }

} // namespace

// PipelineCPU
//
PipelineCPU::PipelineCPU(const PipelineBuilderCPU& builder, pipeline::EventSink* eventSink)
    : m_configuration(builder.configuration())
    , m_eventSink(eventSink ? eventSink : pipeline::EventSink::nullSink())
    , m_allocator(builder.allocator())
    , m_buffersPool(builder.configuration().maxLatency * 2, builder.allocator())
    , m_picturesPool(builder.configuration().maxLatency * 2, builder.allocator())
    , m_framesPool(builder.configuration().maxLatency * 2, builder.allocator())
    , m_allocatedPictures(builder.configuration().maxLatency * 2, builder.allocator())
    , m_allocatedFrames(builder.configuration().maxLatency * 2, builder.allocator())
    , m_reorderIndex(builder.configuration().maxLatency, builder.allocator())
    , m_processingIndex(builder.configuration().maxLatency, builder.allocator())
    , m_doneIndex(builder.configuration().maxLatency, builder.allocator())
    , m_flushIndex(builder.configuration().maxLatency, builder.allocator())
    , m_maxReorder(m_configuration.defaultMaxReorder)
    , m_temporalBuffers(builder.configuration().numTemporalBuffers * RCMaxPlanes, builder.allocator())
    , m_basePicturePending(nextPowerOfTwoU32(builder.configuration().maxLatency + 1), builder.allocator())
    , m_basePictureOutBuffer(nextPowerOfTwoU32(builder.configuration().maxLatency + 1), builder.allocator())
    , m_outputPictureAvailableBuffer(nextPowerOfTwoU32(builder.configuration().maxLatency + 1),
                                     builder.allocator())
{
    // Set up dithering
    ldppDitherGlobalInitialize(m_allocator, &m_dither, m_configuration.ditherSeed);

    if (m_configuration.useSystemAllocator == false) {
        // Special allocator for per frame enhancement data
        m_enhancementAllocator = ldcMemorySimpleAllocatorInitialize(&m_simpleAllocator, m_allocator);

        // Special allocator for per frame image buffer data
        m_bufferAllocator = ldcRecyclingAllocatorInitialize(&m_recyclingAllocator, m_allocator,
                                                            m_configuration.bufferRecycleCount);
    } else {
        // Use system allocator for all allocations
        m_enhancementAllocator = m_allocator;
        m_bufferAllocator = m_allocator;
    }

    // Configuration pool
    LdeBitstreamVersion bitstreamVersion = BitstreamVersionUnspecified;
    if (m_configuration.forceBitstreamVersion >= BitstreamVersionInitial &&
        m_configuration.forceBitstreamVersion <= BitstreamVersionCurrent) {
        bitstreamVersion = static_cast<LdeBitstreamVersion>(m_configuration.forceBitstreamVersion);
    }
    ldeConfigPoolInitialize(m_allocator, m_allocator, &m_configPool, bitstreamVersion);

    ldcTaskPoolInitialize(&m_taskPool, m_allocator, m_allocator, m_configuration.numThreads,
                          m_configuration.numReservedTasks);

    // Fill in empty temporal buffer anchors
    TemporalBuffer buf{};
    buf.allocator = m_bufferAllocator;
    buf.desc.timestamp = kInvalidTimestamp;
    buf.timestampLimit = kInvalidTimestamp;
    for (uint32_t i = 0; i < m_configuration.numTemporalBuffers * RCMaxPlanes; ++i) {
        buf.desc.plane = i;
        m_temporalBuffers.append(buf);
    }

    m_eventSink->generate(pipeline::EventCanSendEnhancement);
    m_eventSink->generate(pipeline::EventCanSendBase);
    m_eventSink->generate(pipeline::EventCanSendPicture);
}

PipelineCPU::~PipelineCPU()
{
    // Flush and wait for any remaining frames
    this->synchronizeDecoder(kInvalidTimestamp, true);

    // Release pictures
    for (uint32_t i = 0; i < m_allocatedPictures.size(); ++i) {
        PictureCPU* picture{m_allocatedPictures[i]};
        m_picturesPool.destroy(picture);
    }

    // Release frames
    for (uint32_t i = 0; i < m_allocatedFrames.size(); ++i) {
        FrameCPU* frame{m_allocatedFrames[i]};
        frame->release(true);
        // Call destructor directly, as we are doing in-place construct/destruct
        m_framesPool.destroy(frame);
    }

    // Release any temporal buffers
    for (uint32_t i = 0; i < m_temporalBuffers.size(); ++i) {
        TemporalBuffer* tb = m_temporalBuffers.at(i);
        if (VNIsAllocated(tb->allocation)) {
            VNFree(m_bufferAllocator, &tb->allocation);
        }
    }
    // Release dither
    ldppDitherGlobalRelease(&m_dither);

    // Release config
    ldeConfigPoolRelease(&m_configPool);

    if (m_configuration.useSystemAllocator == false) {
        // Release buffer cache
        ldcRecyclingAllocatorDestroy(&m_recyclingAllocator);
        // Release frame memory arena
        ldcMemorySimpleAllocatorDestroy(&m_simpleAllocator);
    }

    // Close down task pool
    ldcTaskPoolDestroy(&m_taskPool);

    m_eventSink->generate(pipeline::EventExit);
}

// Send/receive
//
LdcReturnCode PipelineCPU::sendDecoderEnhancementData(uint64_t timestamp, const uint8_t* data, uint32_t byteSize)
{
    VNLogDebug("sendDecoderEnhancementData: ts:%" PRIx64 " %d", timestamp, byteSize);

    // Invalid if this timestamp is already present in decoder.
    //
    // NB: API clients are expected to make distinct timestamps over discontinuities using utility library
    if (findFrame(timestamp) != nullptr) {
        VNLogDebug("sendDecoderEnhancementData: ts:%" PRIx64 " Duplicate Frame", timestamp);
        return LdcReturnCodeInvalidParam;
    }

    if (frameLatency() >= m_configuration.maxLatency) {
        VNLogDebug("sendDecoderEnhancementData: ts:%" PRIx64 " AGAIN", timestamp);
        return LdcReturnCodeAgain;
    }

    // New pending frame
    FrameCPU* const frame{allocateFrame(timestamp)};
    if (!frame) {
        return LdcReturnCodeError;
    }

    // Keep record of highest sent frame timestamp
    if (m_sendLimit == kInvalidTimestamp || pipeline::compareTimestamps(timestamp, m_sendLimit)) {
        m_sendLimit = timestamp;
    }

    // Attach enhancement data to frame
    frame->setEnhancementData(data, byteSize);

    // Frame ready to be reordered into presentation order for correct LCEVC decode
    frame->setState(FrameStateReorder);

    // Add frame to reorder table sorted by timestamp
    m_reorderIndex.insert(sortFramePtrTimestamp, frame);

    // Attach any pending base for matching timestamp
    if (BasePicture* bp = m_basePicturePending.findUnordered(findBasePictureTimestamp, &frame->timestamp);
        bp) {
        frame->setBasePicture(bp->picture, bp->deadline, bp->userData);
        m_basePicturePending.remove(bp);
        m_eventSink->generate(pipeline::EventCanSendBase);
    }

    startReadyFrames();
    return LdcReturnCodeSuccess;
}

LdcReturnCode PipelineCPU::sendDecoderBase(uint64_t timestamp, LdpPicture* basePicture,
                                           uint32_t timeoutUs, void* userData)
{
    VNLogDebug("sendDecoderBase: ts:%" PRIx64 " %p", timestamp, (void*)basePicture);

    // Find the frame associated with PTS
    FrameCPU* frame{findFrame(timestamp)};
    if (frame) {
        // Enhancement exists
        if (LdcReturnCode ret = frame->setBasePicture(
                basePicture, threadTimeMicroseconds(static_cast<int32_t>(timeoutUs)), userData);
            ret != LdcReturnCodeSuccess) {
            return ret;
        }

        // Force pass-through if requested
        if (m_configuration.passthroughMode == PassthroughMode::Force) {
            frame->setPassthrough();
        }
        // Kick off any frames that are at or before the base timestamp
        process(timestamp);
        m_eventSink->generate(pipeline::EventCanSendBase);
        return LdcReturnCodeSuccess;
    }

    BasePicture bp = {timestamp, basePicture,
                      threadTimeMicroseconds(static_cast<int32_t>(timeoutUs)), userData};

    if (m_basePicturePending.size() < m_configuration.enhancementDelay) {
        // There is capacity to buffer base picture
        m_basePicturePending.append(bp);
        return LdcReturnCodeSuccess;
    }

    // Cannot buffer any more pending bases
    if (m_configuration.passthroughMode == PassthroughMode::Disable) {
        // No pass-through
        return LdcReturnCodeAgain;
    }

    // Base frame is going to go through pipeline as some sort of pass-through ...
    if (!m_basePicturePending.isEmpty()) {
        m_basePicturePending.append(bp);
        bp = m_basePicturePending[0];
        m_basePicturePending.removeIndex(0);
        m_eventSink->generate(pipeline::EventCanSendBase);
    }

    // New pass-through frame - no enhancement
    FrameCPU* const passFrame{allocateFrame(timestamp)};

    if (!passFrame) {
        return LdcReturnCodeError;
    }

    // Add frame to reorder table sorted by timestamp
    passFrame->setBasePicture(basePicture, threadTimeMicroseconds(static_cast<int32_t>(timeoutUs)), userData);
    passFrame->setPassthrough();
    passFrame->setState(FrameStateReorder);
    m_reorderIndex.insert(sortFramePtrTimestamp, passFrame);

    process(timestamp);
    return LdcReturnCodeSuccess;
}

LdcReturnCode PipelineCPU::sendDecoderPicture(LdpPicture* outputPicture)
{
    VNLogDebug("sendDecoderPicture: %p", (void*)outputPicture);

    // Add to available queue
    if (m_outputPictureAvailableBuffer.size() > m_configuration.maxLatency ||
        !m_outputPictureAvailableBuffer.tryPush(outputPicture)) {
        VNLogDebug("sendDecoderPicture: AGAIN");
        return LdcReturnCodeAgain;
    }

    connectOutputPictures();

    startReadyFrames();
    return LdcReturnCodeSuccess;
}

LdpPicture* PipelineCPU::receiveDecoderPicture(LdpDecodeInformation& decodeInfoOut)
{
    FrameCPU* frame{};

    releaseFlushedFrames();

    // Pull any done frame from start (lowest timestamp) of 'processing' frame index.
    {
        common::ScopedLock lock(m_interTaskMutex);

        if (!m_doneIndex.isEmpty()) {
            // Something in 'done' index
            frame = m_doneIndex[0];
            m_doneIndex.removeIndex(0);
        } else if (m_processingIndex.size() > m_configuration.minLatency &&
                   m_processingIndex[0]->canComplete() && !isFlushed(m_processingIndex[0])) {
            const FrameCPU* pendingFrame = m_processingIndex[0];

            // Earliest frame will complete, so hang around and wait for it to move to done index
            VNLogDebug("waiting for ts:%" PRIx64, pendingFrame->timestamp);

            if (!m_interTaskFrameDone.waitDeadline(lock, pendingFrame->deadline)) {
                VNLogWarning("wait timed out ts:%" PRIx64, pendingFrame->timestamp);
#if VN_SDK_LOG(DEBUG)
                ldcTaskPoolDump(&m_taskPool, nullptr);
#endif
            } else {
                VNLogDebug("wait done ts:%" PRIx64, pendingFrame->timestamp);
            }

            if (!m_doneIndex.isEmpty()) {
                frame = m_doneIndex[0];
                m_doneIndex.removeIndex(0);
            } else {
                VNLogDebug("no picture ts:%" PRIx64, m_processingIndex[0]->timestamp);
            }
        }
    }

    if (!frame) {
        return nullptr;
    }

    // Copy surviving data from frame
    decodeInfoOut = frame->decodeInformation;
    LdpPicture* pictureOut{frame->outputPicture};

    VNLogDebug("receiveDecoderPicture: ts:%" PRIx64 " %p hb:%d he:%d sk:%d enh:%d",
               decodeInfoOut.timestamp, (void*)pictureOut, decodeInfoOut.hasBase,
               decodeInfoOut.hasEnhancement, decodeInfoOut.skipped, decodeInfoOut.enhanced);

    // Once an output picture has left the building - we can drop the associated frame
    freeFrame(frame);

    return pictureOut;
}

LdpPicture* PipelineCPU::receiveDecoderBase()
{
    // Is there anything in finished base FIFO?
    LdpPicture* basePicture{};
    if (!m_basePictureOutBuffer.tryPop(basePicture)) {
        return nullptr;
    }

    VNLogDebug("receiveDecoderBase: %" PRIx64 " %p", (void*)basePicture);

    return basePicture;
}

void PipelineCPU::getCapacity(LdpPipelineCapacity* capacity)
{
    assert(capacity);
    capacity->enhancementAvailable = m_configuration.maxLatency - frameLatency();
    capacity->baseAvailable = m_configuration.maxLatency - frameLatency();
    capacity->outputAvailable =
        m_outputPictureAvailableBuffer.capacity() - m_outputPictureAvailableBuffer.size();

    capacity->enhancementMaximum = m_configuration.maxLatency;
    capacity->baseMaximum = m_configuration.maxLatency;
    capacity->outputAvailable = m_outputPictureAvailableBuffer.capacity();
}

// Dig out info about a current timestamp
LdcReturnCode PipelineCPU::peekDecoder(uint64_t timestamp, uint32_t& widthOut, uint32_t& heightOut)
{
    // Flush everything up to given timestamp
    process(timestamp);

    // Find the frame associated with PTS
    const FrameCPU* frame{findFrame(timestamp)};
    if (!frame) {
        return LdcReturnCodeNotFound;
    }
    if (!frame->globalConfig) {
        if (m_configuration.passthroughMode == PassthroughMode::Disable) {
            return LdcReturnCodeNotFound;
        }
        return LdcReturnCodeAgain;
    }

    if (frame->isPassthrough()) {
        widthOut = frame->baseWidth;
        heightOut = frame->baseHeight;
    } else {
        widthOut = frame->globalConfig->width;
        heightOut = frame->globalConfig->height;
    }
    return LdcReturnCodeSuccess;
}

// Move any reorder frames at or before timestamp into processing state
void PipelineCPU::process(uint64_t timestamp)
{
    assert(timestamp != kInvalidTimestamp);

    // Move 'processing' point forwards
    if (m_processingLimit != kInvalidTimestamp &&
        pipeline::compareTimestamps(timestamp, m_processingLimit) < 0) {
        VNLogError("Processing timestamp went backwards.");
        return;
    }
    m_processingLimit = timestamp;

    startReadyFrames();
}

LdcReturnCode PipelineCPU::skip(uint64_t timestamp)
{
    const uint64_t fromTimestamp = m_skipLimit;

    VNLogDebug("skip: ts:%" PRIx64 " %p", timestamp);

    // Using kInvalidTimstamp skips all sent frames
    if (timestamp == kInvalidTimestamp) {
        timestamp = m_sendLimit;
    }

    // Skipping beyond highest sent timestamp does nothing
    if (pipeline::compareTimestamps(timestamp, m_sendLimit) > 0) {
        return LdcReturnCodeSuccess;
    }

    // Move 'skip' point forwards
    if (m_skipLimit != kInvalidTimestamp && pipeline::compareTimestamps(timestamp, m_skipLimit) < 0) {
        VNLogError("Skip timestamp went backwards.");
        return LdcReturnCodeError;
    }
    m_skipLimit = timestamp;

    // Bump 'processing' point if necessary
    if (m_processingLimit == kInvalidTimestamp ||
        pipeline::compareTimestamps(timestamp, m_processingLimit) > 0) {
        m_processingLimit = timestamp;
    }

    startReadyFrames();
    unblockSkippedFrames(fromTimestamp);
    return LdcReturnCodeSuccess;
}

// Flush any frames  at or before timestamp into processing state
LdcReturnCode PipelineCPU::flush(uint64_t timestamp)
{
    const uint64_t fromTimestamp = m_flushLimit;

    VNLogDebug("flush: ts:%" PRIx64 " %p", timestamp);

    // Using kInvalidTimstamp flushed all sent frames
    if (timestamp == kInvalidTimestamp) {
        timestamp = m_sendLimit;
    }

    // Move 'flush' point forwards
    if (m_flushLimit != kInvalidTimestamp && pipeline::compareTimestamps(timestamp, m_flushLimit) < 0) {
        VNLogError("Flush timestamp went backwards.");
        return LdcReturnCodeError;
    }
    m_flushLimit = timestamp;

    // Bump 'processing' point if necessary
    if (m_processingLimit == kInvalidTimestamp ||
        pipeline::compareTimestamps(timestamp, m_processingLimit) > 0) {
        m_processingLimit = timestamp;
    }

    // Bump 'skip' point if necessary
    if (m_skipLimit == kInvalidTimestamp || pipeline::compareTimestamps(timestamp, m_skipLimit) > 0) {
        m_skipLimit = timestamp;
    }

    startReadyFrames();
    unblockFlushedFrames(fromTimestamp);
    return LdcReturnCodeSuccess;
}

// Make sure any skipped frames are ready to run
void PipelineCPU::unblockSkippedFrames(uint64_t fromTimestamp)
{
    for (uint32_t i = 0; i < m_allocatedFrames.size(); ++i) {
        FrameCPU* const frame{m_allocatedFrames[i]};
        if (!frame->isStateProcessing()) {
            continue;
        }
        // Already skipped?
        if (fromTimestamp != kInvalidTimestamp &&
            pipeline::compareTimestamps(fromTimestamp, frame->timestamp) >= 0) {
            continue;
        }

        if (isSkipped(frame)) {
            frame->unblockForSkip();
        }
    }
}

// Make sure any flushed frames are ready to run
void PipelineCPU::unblockFlushedFrames(uint64_t fromTimestamp)
{
    for (uint32_t i = 0; i < m_allocatedFrames.size(); ++i) {
        FrameCPU* const frame{m_allocatedFrames[i]};
        if (!frame->isStateProcessing()) {
            continue;
        }

        // Already flushed?
        if (fromTimestamp != kInvalidTimestamp &&
            pipeline::compareTimestamps(fromTimestamp, frame->timestamp) >= 0) {
            continue;
        }

        if (isFlushed(frame)) {
            frame->unblockForFlush();
        }
    }
}

// Wait for all work to be finished - optionally stopping anything in progress
LdcReturnCode PipelineCPU::synchronizeDecoder(uint64_t timestamp, bool flushPending)
{
    VNLogDebug("synchronizeDecoder: %d", flushPending);

    if (flushPending) {
        // Mark current frames as flushed
        flush(timestamp);

        while (m_processingIndex.size() > 0) {
            FrameCPU* const frame = m_processingIndex[0];
            if (!frame->canComplete()) {
                VNLogError("Flushed frame cannot complete: ts:%" PRIx64, frame->timestamp);
            }
            frame->waitForTasks();
        }
    } else {
        // For frames that are not blocked on input - wait in timestamp order
        while (m_processingIndex.size() > 0 && m_processingIndex[0]->canComplete()) {
            m_processingIndex[0]->waitForCompletableTasks();
        }
    }

    releaseFlushedFrames();
    return LdcReturnCodeSuccess;
}

bool PipelineCPU::isProcessing(const FrameCPU* frame) const
{
    assert(frame);

    if (m_processingLimit == kInvalidTimestamp) {
        return false;
    }
    return frame->timestamp <= m_processingLimit;
}

bool PipelineCPU::isSkipped(const FrameCPU* frame) const
{
    assert(frame);
    if (m_skipLimit == kInvalidTimestamp) {
        return false;
    }
    return frame->timestamp <= m_skipLimit;
};

bool PipelineCPU::isFlushed(const FrameCPU* frame) const
{
    assert(frame);
    if (m_flushLimit == kInvalidTimestamp) {
        return false;
    }
    return frame->timestamp <= m_flushLimit;
};

// Buffers
//
BufferCPU* PipelineCPU::allocateBuffer(uint32_t requiredSize)
{
    // Allocate buffer structure
    BufferCPU* const buffer = m_buffersPool.make(m_bufferAllocator, requiredSize);
    if (!buffer) {
        VNLogError("Could not allocate buffer");
        return nullptr;
    }
    return buffer;
}

void PipelineCPU::releaseBuffer(BufferCPU* buffer)
{
    assert(buffer);

    // Release buffer structure
    m_buffersPool.destroy(buffer);
}

// Pictures
//

// Internal allocation
PictureCPU* PipelineCPU::allocatePicture()
{
    // Allocate picture
    PictureCPU* picture{m_picturesPool.make(*this, this->m_allocator)};
    if (!picture) {
        VNLogError("Could not allocate picture");
        return nullptr;
    }

    // Insert into table
    m_allocatedPictures.append(picture);

    return picture;
}

uint32_t PipelineCPU::findAllocatedPicture(const PictureCPU* frame) const
{
    for (uint32_t i = 0; i < m_allocatedPictures.size(); ++i) {
        if (m_allocatedPictures[i] == frame) {
            return i;
        }
    }

    VNLogError("Could not find picture!");
    return UINT32_MAX;
}

void PipelineCPU::releasePicture(PictureCPU* picture)
{
    uint32_t idx = findAllocatedPicture(picture);
    if (idx == UINT32_MAX) {
        return;
    }

    m_allocatedPictures.removeReorderIndex(idx);

    picture->unbindMemory();
    m_picturesPool.destroy(picture);
}

LdpPicture* PipelineCPU::allocPicture(const LdpPictureDesc& desc)
{
    PictureCPU* picture{allocatePicture()};
    picture->setDesc(desc);
    return picture;
}

LdpPicture* PipelineCPU::allocPictureExternal(const LdpPictureDesc& desc,
                                              const LdpPicturePlaneDesc* planeDescArr,
                                              const LdpPictureBufferDesc* buffer)
{
    PictureCPU* picture{allocatePicture()};
    picture->setDesc(desc);
    picture->setExternal(planeDescArr, buffer);
    return picture;
}

void PipelineCPU::freePicture(LdpPicture* ldpPicture)
{
    // Get back to derived Picture class
    PictureCPU* picture{static_cast<PictureCPU*>(ldpPicture)};
    assert(ldpPicture);

    releasePicture(picture);
}

// Frames
//

// Allocate or find working data for a timestamp
//
// Given that there is going to be in the order of 100 or less frames, stick
// with an array and linear searches.
//
// NB: There may be more allocated frames that the configured latency - 'Done' frames
// do not count towards latency limit.
//
// Returns nullptr if there is no capacity for another frame.
//
FrameCPU* PipelineCPU::allocateFrame(uint64_t timestamp)
{
    assert(findFrame(timestamp) == nullptr);

    // Allocate frame
    FrameCPU* frame = m_framesPool.make(this->m_enhancementAllocator, this->m_bufferAllocator, timestamp);
    if (!frame) {
        return nullptr;
    }

    // Append allocation into table
    m_allocatedFrames.append(frame);

    return frame;
}

// Find existing Frame for a timestamp, or return nullptr if it does not exist.
//
FrameCPU* PipelineCPU::findFrame(uint64_t timestamp)
{
    for (uint32_t i = 0; i < m_allocatedFrames.size(); ++i) {
        FrameCPU* const frame{m_allocatedFrames[i]};
        if (isSkipped(frame)) {
            continue;
        }

        if (pipeline::compareTimestamps(frame->timestamp, timestamp) == 0) {
            return frame;
        }
    }

    return nullptr;
}

uint32_t PipelineCPU::findAllocatedFrame(const FrameCPU* frame) const
{
    for (uint32_t i = 0; i < m_allocatedFrames.size(); ++i) {
        if (m_allocatedFrames[i] == frame) {
            return i;
        }
    }

    VNLogError("Cound not find frame!");
    return UINT32_MAX;
}

// Release frame back to pool
//
void PipelineCPU::freeFrame(FrameCPU* frame)
{
    uint32_t idx = findAllocatedFrame(frame);
    if (idx == UINT32_MAX) {
        return;
    }

    m_allocatedFrames.removeReorderIndex(idx);

    // Release task group and allocations
    frame->release(true);
    m_framesPool.destroy(frame);

    // If there are no remaining frames, reset limits so that we can accept 'earlier' timestamp
    // into an empty decoder.
    if (m_allocatedFrames.isEmpty()) {
        VNLogDebug("Reset limits");
        m_sendLimit = kInvalidTimestamp;
        m_processingLimit = kInvalidTimestamp;
        m_skipLimit = kInvalidTimestamp;
        m_flushLimit = kInvalidTimestamp;
    }
}

// Number of outstanding frames
uint32_t PipelineCPU::frameLatency() const
{
    return m_reorderIndex.size() + m_processingIndex.size();
}

//// Frame start
//
// Get the next frame, if any, in timestamp order - taking into account reorder and flushing.
//
FrameCPU* PipelineCPU::getNextReordered()
{
    // Are there any frames at all?
    if (m_reorderIndex.isEmpty()) {
        return nullptr;
    }

    // If exceeded reorder limit, or flushing
    if (m_reorderIndex.size() >= m_maxReorder || isProcessing(m_reorderIndex[0])) {
        FrameCPU* const frame{m_reorderIndex[0]};
        m_reorderIndex.removeIndex(0);
        // Tell API there is enhancement space
        m_eventSink->generate(pipeline::EventCanSendEnhancement);
        return frame;
    }

    return nullptr;
}

// Resolve ready frame configurations in timestamp order, and generate tasks for each one.
//
// Once we are handling frames here, the frame is in flight - async to the API, so no error returns.
//
void PipelineCPU::startReadyFrames()
{
    releaseFlushedFrames();

    // Pull ready frames from reorder table
    while (FrameCPU* frame = getNextReordered()) {
        const uint64_t timestamp{frame->timestamp};
        bool goodConfig = false;

        if (m_previousTimestamp != kInvalidTimestamp &&
            pipeline::compareTimestamps(m_previousTimestamp, timestamp) > 0) {
            // Frame has been flushed out of reorder queue too late - mark as passthrough
            VNLogDebug("startReadyFrames: out of order: ts:%" PRIx64 " prev: %" PRIx64);
            frame->setPassthrough();
        }

        // Try and parse frame configuration
        if (!frame->isPassthrough()) {
            goodConfig = frame->parseEnhancementData(&m_configPool);
            if (!goodConfig) {
                frame->setPassthrough();
            }
        }

        if (frame->isPassthrough()) {
            // Set up enough frame configuration to support pass-through
            ldeConfigPoolFramePassthrough(&m_configPool, &frame->globalConfig, &frame->config);
        }

        VNLogDebug(
            "Start Frame: ts:%" PRIx64 " goodConfig:%d temporalEnabled:%d, temporalPresent:%d "
            "temporalRefresh:%d loqEnabled[0]:%d loqEnabled[1]:%d skip:%d flush:%d passthrough:%d",
            timestamp, goodConfig, frame->globalConfig->temporalEnabled,
            frame->config.temporalSignallingPresent, frame->config.temporalRefresh,
            frame->config.loqEnabled[0], frame->config.loqEnabled[1], isSkipped(frame),
            isFlushed(frame), frame->isPassthrough());

        // Once we have per frame configuration, we can properly initialize and figure out tasks for the frame
        if (!frame->initialize(m_configuration, &m_taskPool, &m_dither)) {
            VNLogError("Could not allocate frame buffers: ts:%" PRIx64, frame->timestamp);
            // Could not allocate buffers - switch to pass-through
            frame->setPassthrough();
        }

        // Unblock frames if they are skipped or flushed
        if (isSkipped(frame)) {
            frame->unblockForSkip();
        }

        if (isFlushed(frame)) {
            frame->unblockForFlush();
        }

        if (isFlushed(frame) && frame->basePicture == nullptr && frame->outputPicture == nullptr) {
            // Frame can just be released now, otherwise let it go through normal processing
            // to allow pictures to be returned.
            VNLogDebug("Freeing flushed frame: ts:%" PRIx64, frame->timestamp);
            freeFrame(frame);
        } else {
            // Decode the frame as normal, add it to the processing index with the previous
            // frame's temporal buffer timestamp (if required for temporal=on)
            {
                common::ScopedLock lock(m_interTaskMutex);
                frame->setState(FrameStateProcessing);
                m_processingIndex.append(frame);
            }

            generateTasks(this, frame, m_lastGoodTimestamp);

            // Remember timestamps for the next frame
            m_previousTimestamp = timestamp;
            if (goodConfig) {
                m_lastGoodTimestamp = timestamp;
            }
        }
    }

    // Connect available output pictures to started pictures
    connectOutputPictures();
}

// Connect any available output pictures to frames that can use them
//
void PipelineCPU::connectOutputPictures()
{
    // While there are available output pictures and pending frames,
    // go through frames in timestamp order, assigning next output picture
    while (true) {
        FrameCPU* frame{};

        if (m_outputPictureAvailableBuffer.isEmpty()) {
            // No output pictures left
            break;
        }

        // Find next in process frame with base data, and without an assigned output picture
        {
            common::ScopedLock lock(m_interTaskMutex);

            for (uint32_t idx = 0; idx < m_processingIndex.size(); ++idx) {
                if (!m_processingIndex[idx]->outputPicture && m_processingIndex[idx]->baseDataValid()) {
                    frame = m_processingIndex[idx];
                    break;
                }
            }
        }

        if (!frame) {
            // No frames without output pictures left
            break;
        }

        //  Get the picture
        LdpPicture* ldpPicture{};
        m_outputPictureAvailableBuffer.pop(ldpPicture);
        assert(ldpPicture);

        // Set the output layout
        const LdpPictureDesc desc{frame->getOutputPictureDesc(m_configuration.passthroughMode)};
        ldpPictureSetDesc(ldpPicture, &desc);
        if (frame->globalConfig->cropEnabled) {
            ldpPicture->margins.left = frame->globalConfig->crop.left;
            ldpPicture->margins.right = frame->globalConfig->crop.right;
            ldpPicture->margins.top = frame->globalConfig->crop.top;
            ldpPicture->margins.bottom = frame->globalConfig->crop.bottom;
        }

        VNLogDebug("connectOutputPicture: ts:%" PRIx64 " %p %ux%u (r:%d p:%d o:%d)", frame->timestamp,
                   (void*)ldpPicture, desc.width, desc.height, m_reorderIndex.size(),
                   m_processingIndex.size(), m_outputPictureAvailableBuffer.size());

        // Poke it into the frame's task group
        frame->setOutputPicture(ldpPicture);

        // Tell API there is output picture space
        m_eventSink->generate(pipeline::EventCanSendPicture);
    }
}

// Clear 'flush' index
//
// This is done on main pipeline thread (and not as part of taskDone) allowing task groups tasks
// to finish cleanly.
//
void PipelineCPU::releaseFlushedFrames()
{
    while (!m_flushIndex.isEmpty()) {
        FrameCPU* frame{nullptr};
        {
            common::ScopedLock lock(m_interTaskMutex);
            frame = m_flushIndex[0];
            assert(frame);
            m_flushIndex.removeIndex(0);
        }
        VNLogDebug("Released flushed frame: ts:%" PRIx64, frame->timestamp);
        freeFrame(frame);
    }
}

//// Temporal
//
// Look through all temporal buffers, looking for one that matches the given frame and plane's requirements
//
// The frame<->temporal buffer search loops are where individual frame tasks can interact with
// each other, so are protected by protected by m_interTaskMutex.
//
TemporalBuffer* PipelineCPU::findTemporalBuffer(FrameCPU* frame, uint32_t plane)
{
    TemporalBuffer* foundTemporalBuffer{};

    {
        common::ScopedLock lock(m_interTaskMutex);

        for (uint32_t i = 0; i < m_temporalBuffers.size(); ++i) {
            TemporalBuffer* tb{m_temporalBuffers.at(i)};
            if (tb->frame) {
                // In use
                continue;
            }

            if (frame->tryAttachTemporalBuffer(plane, tb)) {
                foundTemporalBuffer = tb;
                break;
            }
        }
    }

    if (!foundTemporalBuffer) {
        // Not found - will get resolved later by being transferred from a previous frame
        return nullptr;
    }

    VNLogDebug("  findTemporalBuffer found: plane:%" PRIu32 " ts:%" PRIx64 " found_ts:%" PRIx64,
               plane, frame->timestamp, foundTemporalBuffer->desc.timestamp);

    //
    frame->updateTemporalBuffer(plane);

    return foundTemporalBuffer;
}

// Mark the frame as having finished with it's temporal buffer, and try to transfer buffer on to another frame
//
void PipelineCPU::transferTemporalBuffer(FrameCPU* frame, uint32_t plane)
{
    VNLogDebug("releaseTemporalBuffer: ts:%" PRIx64 " plane: %" PRIu32, frame->timestamp, plane);

    FrameCPU* foundNextFrame{nullptr};
    TemporalBuffer* tb{nullptr};

    {
        common::ScopedLock lock(m_interTaskMutex);

        tb = frame->detachTemporalBuffer(plane);
        if (tb == nullptr) {
            // No temporal buffer to be released
            return;
        }

        // Do any of the pending frames want this buffer?
        for (uint32_t idx = 0; idx < m_processingIndex.size(); ++idx) {
            FrameCPU* nextFrame{m_processingIndex[idx]};
            if (nextFrame->tryAttachTemporalBuffer(plane, tb)) {
                foundNextFrame = nextFrame;
                break;
            }
        }
    }

    if (!foundNextFrame) {
        return;
    }

    VNLogDebug("  CPU::releaseTemporalBuffer found: plane:%" PRIu32 " next_ts:%" PRIx64
               " ts:%" PRIx64,
               plane, foundNextFrame->timestamp, frame->timestamp);
    foundNextFrame->updateTemporalBuffer(plane);
}

// End of frame processing
//
void PipelineCPU::baseDone(LdpPicture* picture)
{
    // Generate event
    m_eventSink->generate(pipeline::EventBasePictureDone, picture);

    // Send base picture back to API
    m_basePictureOutBuffer.push(picture);
}

void PipelineCPU::outputDone(FrameCPU* frame)
{
    common::ScopedLock lock(m_interTaskMutex);

    // Remove from processing index
    const int idx = m_processingIndex.findUnorderedIndex(compareFramePtr, frame);
    assert(idx != -1);
    m_processingIndex.removeIndex(idx);

    if (!frame->outputPicture) {
        // Hand off to 'flush' index
        frame->setState(FrameStateFlush);
        m_flushIndex.insert(sortFramePtrTimestamp, frame);
    } else {
        // Hand off to 'done' index - even if frame was skipped, so that output
        // picture can be returned to integration (marked as skipped)
        frame->setState(FrameStateDone);
        m_doneIndex.insert(sortFramePtrTimestamp, frame);
        m_interTaskFrameDone.signal();

        m_eventSink->generate(pipeline::EventOutputPictureDone, frame->outputPicture,
                              &frame->decodeInformation);
    }
    m_eventSink->generate(pipeline::EventCanReceive);
}

#if VN_SDK_LOG(DEBUG)
// Dump frame and index state
//
void PipelineCPU::logFrames()
{
    char buffer[512];

    VNLogDebug("Frames: %d", m_allocatedFrames.size());
    for (uint32_t i = 0; i < m_allocatedFrames.size(); ++i) {
        FrameCPU* const frame{m_allocatedFrames[i]};
        frame->longDescription(buffer, sizeof(buffer));
        VNLogDebugF("  %4d: %s", i, buffer);
        frame->dumpTasks(&m_taskPool);
    }

    logFrameIndex("Reorder", m_reorderIndex);
    logFrameIndex("Processing", m_processingIndex);
    logFrameIndex("Done", m_doneIndex);
    logFrameIndex("Flush", m_flushIndex);

    VNLogDebug("Bases In: %d (%d)", m_basePicturePending.size(), m_basePicturePending.reserved());
    VNLogDebug("Bases Out: %d (%d)", m_basePictureOutBuffer.size(), m_basePictureOutBuffer.capacity());
    VNLogDebug("Output: %d (%d)", m_outputPictureAvailableBuffer.size(),
               m_outputPictureAvailableBuffer.capacity());
    VNLogDebug("Limits Flush:%" PRIx64 " Skip:%" PRIx64 " Processing:%" PRIx64 " Send:%" PRIx64,
               m_flushLimit.load(), m_skipLimit.load(), m_processingLimit.load(), m_sendLimit.load());
}

void PipelineCPU::logFrameIndex(const char* indexName, const lcevc_dec::common::Vector<FrameCPU*>& index) const
{
    VNLogDebug("Index %s: %d", indexName, index.size());
    for (uint32_t i = 0; i < index.size(); ++i) {
        const FrameCPU* const frame{index[i]};
        uint32_t idx = findAllocatedFrame(frame);
        VNLogDebugF("  %2d: %4d ts:%" PRIx64, i, idx, frame->timestamp);
    }
}

#endif

} // namespace lcevc_dec::pipeline_cpu
