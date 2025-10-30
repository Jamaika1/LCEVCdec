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

#include "pipeline_vulkan.h"
//
#include "frame_vulkan.h"
#include "picture_vulkan.h"
#include "pipeline_config_vulkan.h"
#include "tasks_vulkan.h"

#include <LCEVC/common/check.h>
#include <LCEVC/common/constants.h>
#include <LCEVC/common/diagnostics.h>
#include <LCEVC/common/limit.h>
#include <LCEVC/common/log.h>
#include <LCEVC/common/memory.h>
#include <LCEVC/common/return_code.h>
#include <LCEVC/common/task_pool.h>
#include <LCEVC/common/threads.h>
#include <LCEVC/pipeline_vulkan/types_vulkan.h>
//
#include <cstdint>
#include <cstring>
#include <ctime>

namespace lcevc_dec::pipeline_vulkan {

// Utility functions for finding things in Arrays
//
namespace {
    // Compare two frames in an array of frame pointers
    inline int sortFramePtrTimestamp(const void* lhs, const void* rhs)
    {
        const auto* frameLhs{*static_cast<const FrameVulkan* const *>(lhs)};
        const auto* frameRhs{*static_cast<const FrameVulkan* const *>(rhs)};

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
        const auto* frameLhs{*static_cast<const FrameVulkan* const *>(element)};
        const auto* frameRhs{static_cast<const FrameVulkan*>(other)};

        if (frameLhs < frameRhs) {
            return -1;
        }
        if (frameLhs > frameRhs) {
            return 1;
        }
        return 0;
    }

} // namespace

PipelineVulkan::PipelineVulkan(const PipelineBuilderVulkan& builder, pipeline::EventSink* eventSink)
    : m_configuration(builder.configuration())
    , m_eventSink(eventSink ? eventSink : pipeline::EventSink::nullSink())
    , m_allocator(builder.allocator())
    , m_buffers(builder.configuration().maxLatency, builder.allocator())
    , m_pictures(builder.configuration().maxLatency, builder.allocator())
    , m_frames(builder.configuration().maxLatency, builder.allocator())
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
    , m_core(*this)

{
    // Set up dithering
    ldppDitherGlobalInitialize(m_allocator, &m_dither, m_configuration.ditherSeed);

    // Set up an allocator for per frame data
    ldcRollingArenaInitialize(&m_rollingArena, m_allocator, m_configuration.initialArenaCount,
                              m_configuration.initialArenaSize);

    // Configuration pool
    LdeBitstreamVersion bitstreamVersion = BitstreamVersionUnspecified;
    if (m_configuration.forceBitstreamVersion >= BitstreamVersionInitial &&
        m_configuration.forceBitstreamVersion <= BitstreamVersionCurrent) {
        bitstreamVersion = static_cast<LdeBitstreamVersion>(m_configuration.forceBitstreamVersion);
    }
    ldeConfigPoolInitialize(m_allocator, m_allocator, &m_configPool, bitstreamVersion);

    // Start task pool - pool threads is 1 less than configured threads
    VNCheck(m_configuration.numThreads >= 1);
    ldcTaskPoolInitialize(&m_taskPool, m_allocator, m_allocator, m_configuration.numThreads - 1,
                          m_configuration.numReservedTasks);

    // Fill in empty temporal buffer anchors
    TemporalBuffer buf{};
    buf.allocator = m_allocator;
    buf.desc.timestamp = kInvalidTimestamp;
    buf.timestampLimit = kInvalidTimestamp;
    for (uint32_t i = 0; i < m_configuration.numTemporalBuffers * RCMaxPlanes; ++i) {
        buf.desc.plane = i;
        m_temporalBuffers.append(buf);
    }

    m_eventSink->generate(pipeline::EventCanSendEnhancement);
    m_eventSink->generate(pipeline::EventCanSendBase);
    m_eventSink->generate(pipeline::EventCanSendPicture);

    m_intermediateUpscalePicture[LOQ1] = std::make_unique<PictureVulkan>(*this);
    m_intermediateUpscalePicture[LOQ0] = std::make_unique<PictureVulkan>(*this);
    m_temporalPicture = std::make_unique<PictureVulkan>(*this);

    // Initialise vulkan state
    m_initialised = m_core.init();
}

PipelineVulkan::~PipelineVulkan()
{
    // Flush and wait for any remaining frames
    this->synchronizeDecoder(kInvalidTimestamp, true);

    // Release pictures
    for (uint32_t i = 0; i < m_pictures.size(); ++i) {
        PictureVulkan* picture{VNAllocationPtr(m_pictures[i], PictureVulkan)};
        // Call destructor directly, as we are doing in-place construct/destruct
        picture->~PictureVulkan();
        VNFree(m_allocator, &m_pictures[i]);
    }

    // Release frames
    for (uint32_t i = 0; i < m_frames.size(); ++i) {
        FrameVulkan* frame{VNAllocationPtr(m_frames[i], FrameVulkan)};
        frame->release(true);
        // Call destructor directly, as we are doing in-place construct/destruct
        frame->~FrameVulkan();
        VNFree(m_allocator, &m_frames[i]);
    }

    // Release any temporal buffers
    for (uint32_t i = 0; i < m_temporalBuffers.size(); ++i) {
        TemporalBuffer* tb = m_temporalBuffers.at(i);
        if (VNIsAllocated(tb->allocation)) {
            VNFree(m_allocator, &tb->allocation);
        }
    }
    // Release dither
    ldppDitherGlobalRelease(&m_dither);

    // Release config
    ldeConfigPoolRelease(&m_configPool);

    // Release frame memory arena
    ldcRollingArenaDestroy(&m_rollingArena);

    // Close down task pool
    ldcTaskPoolDestroy(&m_taskPool);

    m_eventSink->generate(pipeline::EventExit);

    // Release vulkan objects
    if (m_initialised) {
        m_core.destroy();
    }
}

// Send/receive
LdcReturnCode PipelineVulkan::sendDecoderEnhancementData(uint64_t timestamp, const uint8_t* data,
                                                         uint32_t byteSize)
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
    FrameVulkan* const frame{allocateFrame(timestamp)};
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

LdcReturnCode PipelineVulkan::sendDecoderBase(uint64_t timestamp, LdpPicture* basePicture,
                                              uint32_t timeoutUs, void* userData)
{
    VNLogDebug("sendDecoderBase: ts:%" PRIx64 " %p", timestamp, (void*)basePicture);

    // Find the frame associated with PTS
    FrameVulkan* frame{findFrame(timestamp)};
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
    FrameVulkan* const passFrame{allocateFrame(timestamp)};

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

LdcReturnCode PipelineVulkan::sendDecoderPicture(LdpPicture* outputPicture)
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

LdpPicture* PipelineVulkan::receiveDecoderPicture(LdpDecodeInformation& decodeInfoOut)
{
    FrameVulkan* frame{};

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
            const FrameVulkan* pendingFrame = m_processingIndex[0];

            // Earliest frame will complete, so hang around and wait for it to move to done index
            VNLogDebug("waiting for ts:%" PRIx64, pendingFrame->timestamp);

            if (!m_interTaskFrameDone.waitDeadline(lock, pendingFrame->deadline)) {
                VNLogWarning("wait timed out ts:%" PRIx64, pendingFrame->timestamp);
#ifdef VN_SDK_LOG_ENABLE_DEBUG
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

LdpPicture* PipelineVulkan::receiveDecoderBase()
{
    // Is there anything in finished base FIFO?
    LdpPicture* basePicture{};
    if (!m_basePictureOutBuffer.tryPop(basePicture)) {
        return nullptr;
    }

    VNLogDebug("receiveDecoderBase: %" PRIx64 " %p", (void*)basePicture);

    return basePicture;
}

void PipelineVulkan::getCapacity(LdpPipelineCapacity* capacity)
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
LdcReturnCode PipelineVulkan::peekDecoder(uint64_t timestamp, uint32_t& widthOut, uint32_t& heightOut)
{
    // Flush everything up to given timestamp
    process(timestamp);

    // Find the frame associated with PTS
    const FrameVulkan* frame{findFrame(timestamp)};
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
void PipelineVulkan::process(uint64_t timestamp)
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

// Mark everything before timestamp as not needing decoding
LdcReturnCode PipelineVulkan::skip(uint64_t timestamp)
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

// Make pending frames get decoded
LdcReturnCode PipelineVulkan::flush(uint64_t timestamp)
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
void PipelineVulkan::unblockSkippedFrames(uint64_t fromTimestamp)
{
    for (uint32_t i = 0; i < m_frames.size(); ++i) {
        FrameVulkan* const frame{VNAllocationPtr(m_frames[i], FrameVulkan)};
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
void PipelineVulkan::unblockFlushedFrames(uint64_t fromTimestamp)
{
    for (uint32_t i = 0; i < m_frames.size(); ++i) {
        FrameVulkan* const frame{VNAllocationPtr(m_frames[i], FrameVulkan)};
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
LdcReturnCode PipelineVulkan::synchronizeDecoder(uint64_t timestamp, bool flushPending)
{
    VNLogDebug("synchronizeDecoder: %d", flushPending);

    if (flushPending) {
        // Mark current frames as flushed
        flush(timestamp);

        while (m_processingIndex.size() > 0) {
            FrameVulkan* const frame = m_processingIndex[0];
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

bool PipelineVulkan::isProcessing(const FrameVulkan* frame) const
{
    assert(frame);

    if (m_processingLimit == kInvalidTimestamp) {
        return false;
    }
    return frame->timestamp <= m_processingLimit;
}

bool PipelineVulkan::isSkipped(const FrameVulkan* frame) const
{
    assert(frame);
    if (m_skipLimit == kInvalidTimestamp) {
        return false;
    }
    return frame->timestamp <= m_skipLimit;
};

bool PipelineVulkan::isFlushed(const FrameVulkan* frame) const
{
    assert(frame);
    if (m_flushLimit == kInvalidTimestamp) {
        return false;
    }
    return frame->timestamp <= m_flushLimit;
};

// Buffers
//
BufferVulkan* PipelineVulkan::allocateBuffer(uint32_t requiredSize)
{
    // Allocate buffer structure
    LdcMemoryAllocation allocation;
    VNAllocateZero(m_allocator, &allocation, BufferVulkan, "VulkanBuffer");
    BufferVulkan* const buffer{VNAllocationPtr(allocation, BufferVulkan)};
    if (!buffer) {
        return nullptr;
    }
    // Insert into table
    m_buffers.append(allocation);

    // In place construction
    return new (buffer) BufferVulkan(this->getCore(), requiredSize); // NOLINT(cppcoreguidelines-owning-memory)
}

void PipelineVulkan::releaseBuffer(BufferVulkan* buffer)
{
    assert(buffer);

    // Release buffer structure
    LdcMemoryAllocation* const pAlloc{m_buffers.findUnordered(ldcVectorCompareAllocationPtr, buffer)};

    if (!pAlloc) {
        // Could not find picture!
        VNLogWarning("Could not find buffer to release: %p", (void*)buffer);
        return;
    }

    // Call destructor directly, as we are doing in-place construct/destruct
    buffer->~BufferVulkan();

    // Release memory
    VNFree(m_allocator, pAlloc);

    m_buffers.removeReorder(pAlloc);
}

// Picture-handling
// Internal allocation

PictureVulkan* PipelineVulkan::allocatePicture()
{
    // Allocate picture
    LdcMemoryAllocation pictureAllocation;
    VNAllocateZero(m_allocator, &pictureAllocation, PictureVulkan, "PictureVulkan");
    PictureVulkan* picture{VNAllocationPtr(pictureAllocation, PictureVulkan)};
    if (!picture) {
        return nullptr;
    }
    // Insert into table
    m_pictures.append(pictureAllocation);

    // In place construction
    return new (picture) PictureVulkan(*this); // NOLINT(cppcoreguidelines-owning-memory)
}

void PipelineVulkan::releasePicture(PictureVulkan* picture)
{
    picture->unbindMemory(); // TODO - check this

    // Find slot
    LdcMemoryAllocation* pAlloc{m_pictures.findUnordered(ldcVectorCompareAllocationPtr, picture)};

    if (!pAlloc) {
        // Could not find picture!
        VNLogWarning("Could not find picture to release: %p", (void*)picture);
        return;
    }

    // Call destructor directly, as we are doing in-place construct/destruct
    picture->~PictureVulkan();

    // Release memory
    VNFree(m_allocator, pAlloc);

    m_pictures.removeReorder(pAlloc);
}

LdpPicture* PipelineVulkan::allocPicture(const LdpPictureDesc& desc)
{
    PictureVulkan* picture{allocatePicture()};
    picture->setDesc(desc);
    return picture;
}

LdpPicture* PipelineVulkan::allocPictureExternal(const LdpPictureDesc& desc,
                                                 const LdpPicturePlaneDesc* planeDescArr,
                                                 const LdpPictureBufferDesc* buffer)
{
    PictureVulkan* picture{allocatePicture()};
    picture->setDesc(desc);
    picture->setExternal(planeDescArr, buffer);
    return picture;
}

void PipelineVulkan::freePicture(LdpPicture* ldpPicture)
{
    // Get back to derived Picture class
    PictureVulkan* picture{static_cast<PictureVulkan*>(ldpPicture)};
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
FrameVulkan* PipelineVulkan::allocateFrame(uint64_t timestamp)
{
    assert(findFrame(timestamp) == nullptr);

    // Allocate frame with in place construction
    LdcMemoryAllocation frameAllocation = {};
    VNAllocateZero(m_allocator, &frameAllocation, FrameVulkan, "FrameVulkan");
    FrameVulkan* const frame{VNAllocationPtr(frameAllocation, FrameVulkan)};
    if (!frame) {
        return nullptr;
    }

    // Append allocation into table
    m_frames.append(frameAllocation);

    // In place construction
    return new (frame) FrameVulkan(this->m_allocator, timestamp); // NOLINT(cppcoreguidelines-owning-memory)
}

// Find existing Frame for a timestamp, or return nullptr if it does not exist.
//
FrameVulkan* PipelineVulkan::findFrame(uint64_t timestamp)
{
    for (uint32_t i = 0; i < m_frames.size(); ++i) {
        FrameVulkan* const frame{VNAllocationPtr(m_frames[i], FrameVulkan)};
        if (isSkipped(frame)) {
            continue;
        }

        if (pipeline::compareTimestamps(frame->timestamp, timestamp) == 0) {
            return frame;
        }
    }

    return nullptr;
}

// Release frame back to pool
//
void PipelineVulkan::freeFrame(FrameVulkan* frame)
{
    // Release task group and allocations
    frame->release(true);

    // Find allocation containing the frame
    LdcMemoryAllocation* frameAllocation{m_frames.findUnordered(ldcVectorCompareAllocationPtr, frame)};

    if (!frameAllocation) {
        // Could not find frame!
        VNLogWarning("Could not find frame allocation: %p", (void*)frame);
        return;
    }

    // Call destructor directly, as we are doing in-place construct/destruct
    frame->~FrameVulkan();

    // Release memory
    VNFree(m_allocator, frameAllocation);

    m_frames.removeReorder(frameAllocation);

    // If there are no remaining frames, reset limits so that we can accept 'earlier' timestamp
    // into an empty decoder.
    if (m_frames.isEmpty()) {
        VNLogDebug("Reset limits");
        m_sendLimit = kInvalidTimestamp;
        m_processingLimit = kInvalidTimestamp;
        m_skipLimit = kInvalidTimestamp;
        m_flushLimit = kInvalidTimestamp;
    }
}

// Number of outstanding frames
uint32_t PipelineVulkan::frameLatency() const
{
    return m_reorderIndex.size() + m_processingIndex.size();
}

//// Frame start
//
// Get the next frame, if any, in timestamp order - taking into account reorder and flushing.
//
FrameVulkan* PipelineVulkan::getNextReordered()
{
    // Are there any frames at all?
    if (m_reorderIndex.isEmpty()) {
        return nullptr;
    }

    // If exceeded reorder limit, or flushing
    if (m_reorderIndex.size() >= m_maxReorder || isProcessing(m_reorderIndex[0])) {
        FrameVulkan* const frame{m_reorderIndex[0]};
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
void PipelineVulkan::startReadyFrames()
{
    releaseFlushedFrames();

    // Pull ready frames from reorder table
    while (FrameVulkan* frame = getNextReordered()) {
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
void PipelineVulkan::connectOutputPictures()
{
    // While there are available output pictures and pending frames,
    // go through frames in timestamp order, assigning next output picture
    while (true) {
        FrameVulkan* frame{};

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
void PipelineVulkan::releaseFlushedFrames()
{
    while (!m_flushIndex.isEmpty()) {
        FrameVulkan* frame{nullptr};
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
TemporalBuffer* PipelineVulkan::findTemporalBuffer(FrameVulkan* frame, uint32_t plane)
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
void PipelineVulkan::transferTemporalBuffer(FrameVulkan* frame, uint32_t plane)
{
    VNLogDebug("releaseTemporalBuffer: ts:%" PRIx64 " plane: %" PRIu32, frame->timestamp, plane);

    FrameVulkan* foundNextFrame{nullptr};
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
            FrameVulkan* nextFrame{m_processingIndex[idx]};
            if (nextFrame->tryAttachTemporalBuffer(plane, tb)) {
                foundNextFrame = nextFrame;
                break;
            }
        }
    }

    if (!foundNextFrame) {
        return;
    }

    VNLogDebug("  Vulkan::releaseTemporalBuffer found: plane:%" PRIu32 " next_ts:%" PRIx64
               " ts:%" PRIx64,
               plane, foundNextFrame->timestamp, frame->timestamp);
    foundNextFrame->updateTemporalBuffer(plane);
}

// End of frame processing
//
void PipelineVulkan::baseDone(LdpPicture* picture)
{
    // Generate event
    m_eventSink->generate(pipeline::EventBasePictureDone, picture);

    // Send base picture back to API
    m_basePictureOutBuffer.push(picture);
}

void PipelineVulkan::outputDone(FrameVulkan* frame)
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

void PipelineVulkan::prepareApplyArgs(VulkanApplyArgs& args, PictureVulkan* picture,
                                      LdpEnhancementTile* enhancementTile, FrameVulkan* frame,
                                      bool applyDirect)
{
    LdpPictureDesc desc{};
    picture->getDesc(desc);
    args.picture = applyDirect ? picture : nullptr;

    int widthShift{};
    int heightShift{};
    if (enhancementTile->plane != 0) {
        getSubsamplingShifts(m_chroma, widthShift, heightShift);
    }
    args.chroma = m_chroma;
    args.planeWidth = desc.width >> widthShift;
    args.planeHeight = desc.height >> heightShift;
    args.plane = enhancementTile->plane;
    args.bufferGpu = enhancementTile->bufferGpu;
    args.tileX = enhancementTile->tileX;
    args.tileY = enhancementTile->tileY;
    args.tileWidth = enhancementTile->tileWidth;
    args.temporalRefresh = applyDirect ? false : frame->config.temporalRefresh;
    args.highlightResiduals = m_configuration.highlightResiduals;
    args.tuRasterOrder =
        !frame->globalConfig->temporalEnabled && frame->globalConfig->tileDimensions == TDTNone;
}

void PipelineVulkan::getSubsamplingShifts(LdeChroma chroma, int& widthShift, int& heightShift)
{
    widthShift = 0;
    heightShift = 0;
    switch (chroma) {
        case LdeChroma::CT420: heightShift = 1; [[fallthrough]];
        case LdeChroma::CT422: widthShift = 1; break;
        default: break;
    }
}

LdpColorFormat PipelineVulkan::chromaToColorFormat(LdeChroma chroma)
{
    switch (chroma) {
        case LdeChroma::CTMonochrome: return LdpColorFormatGRAY_16_LE;
        case LdeChroma::CT420: return LdpColorFormatI420_16_LE;
        case LdeChroma::CT422: return LdpColorFormatI422_16_LE;
        case LdeChroma::CT444: return LdpColorFormatI444_16_LE;
        default: return LdpColorFormatUnknown;
    }
}

#ifdef VN_SDK_LOG_ENABLE_DEBUG
// Dump frame and index state
//
void PipelineVulkan::logFrames()
{
    char buffer[512];

    VNLogDebug("Frames: %d", m_frames.size());
    for (uint32_t i = 0; i < m_frames.size(); ++i) {
        FrameVulkan* const frame{VNAllocationPtr(m_frames[i], FrameVulkan)};
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

void PipelineVulkan::logFrameIndex(const char* indexName,
                                   const lcevc_dec::common::Vector<FrameVulkan*>& index) const
{
    VNLogDebug("Index %s: %d", indexName, index.size());
    for (uint32_t i = 0; i < index.size(); ++i) {
        const FrameVulkan* const frame{index[i]};
        const LdcMemoryAllocation* const ptr =
            m_frames.findUnordered(ldcVectorCompareAllocationPtr, frame);
        const int32_t idx = static_cast<int32_t>(ptr ? (ptr - &m_frames[0]) : -1);
        VNLogDebugF("  %2d: %4d ts:%" PRIx64, i, idx, frame->timestamp);
    }
}

#endif

} // namespace lcevc_dec::pipeline_vulkan
