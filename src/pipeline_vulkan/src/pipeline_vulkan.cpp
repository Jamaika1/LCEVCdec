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
#include <buffer_vulkan.h>
#include <frame_vulkan.h>
#include <LCEVC/common/check.h>
#include <LCEVC/common/constants.h>
#include <LCEVC/common/diagnostics.h>
#include <LCEVC/common/limit.h>
#include <LCEVC/common/log.h>
#include <LCEVC/common/printf_macros.h>
#include <LCEVC/enhancement/decode.h>
#include <LCEVC/pipeline_vulkan/types_vulkan.h>
#include <LCEVC/pixel_processing/blit.h> // TODO - remove
//
#include <memory>

namespace lcevc_dec::pipeline_vulkan {

// Utility functions for finding things in Arrays
//
namespace {
    // Compare 'close' timestamps - allows wrapping around end of uint64_t range
    // (Unlikely when starting at zero - but allows timestamps to start 'before' zero)
    inline int compareTimestamps(uint64_t lhs, uint64_t rhs)
    {
        const int64_t delta = (int64_t)(lhs - rhs);
        if (delta < 0) {
            return -1;
        }
        if (delta > 0) {
            return 1;
        }
        return 0;
    }

    inline int findFrameTimestamp(const void* element, const void* ptr)
    {
        const auto* alloc{static_cast<const LdcMemoryAllocation*>(element)};
        assert(VNIsAllocated(*alloc));
        const uint64_t ets{VNAllocationPtr(*alloc, FrameVulkan)->timestamp};
        const uint64_t ts{*static_cast<const uint64_t*>(ptr)};

        return compareTimestamps(ets, ts);
    }

    inline int sortFramePtrTimestamp(const void* lhs, const void* rhs)
    {
        const auto* frameLhs{*static_cast<const FrameVulkan* const *>(lhs)};
        const auto* frameRhs{*static_cast<const FrameVulkan* const *>(rhs)};

        return compareTimestamps(frameLhs->timestamp, frameRhs->timestamp);
    }

    inline int findBasePictureTimestamp(const void* element, const void* ptr)
    {
        const auto* alloc{static_cast<const LdcMemoryAllocation*>(element)};
        assert(VNIsAllocated(*alloc));
        const uint64_t ets{VNAllocationPtr(*alloc, BasePicture)->timestamp};
        const uint64_t ts{*static_cast<const uint64_t*>(ptr)};

        return compareTimestamps(ets, ts);
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
    ldeConfigPoolInitialize(m_allocator, &m_configPool, bitstreamVersion);

    // Start task pool - pool threads is 1 less than configured threads
    VNCheck(m_configuration.numThreads >= 1);
    ldcTaskPoolInitialize(&m_taskPool, m_allocator, m_allocator, m_configuration.numThreads - 1,
                          m_configuration.numReservedTasks);

    // Fill in empty temporal buffer anchors
    TemporalBuffer buf{};
    buf.desc.timestamp = kInvalidTimestamp;
    buf.timestampLimit = kInvalidTimestamp;
    for (uint32_t i = 0; i < m_configuration.numTemporalBuffers * RCMaxPlanes; ++i) {
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
        frame->release();
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

    ldeConfigPoolRelease(&m_configPool);

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
    VNLogDebug("sendEnhancementData: %" PRIx64 " %d", timestamp, byteSize);
    VNTraceInstant("sendEnhancementData", timestamp);

    // Invalid if this timestamp is already present in decoder.
    //
    // NB: API clients are expected to make distinct timestamps over discontinuities using utility library
    if (findFrame(timestamp)) {
        return LdcReturnCodeInvalidParam;
    }

    if (frameLatency() >= m_configuration.maxLatency) {
        VNLogDebug("sendEnhancementData: %" PRIx64 " AGAIN", timestamp);
        return LdcReturnCodeAgain;
    }

    // New pending frame
    FrameVulkan* const frame{allocateFrame(timestamp)};
    if (!frame) {
        return LdcReturnCodeError;
    }

    LdcMemoryAllocation enhancementDataAllocation{};
    uint8_t* const enhancement{VNAllocateArray(m_allocator, &enhancementDataAllocation, uint8_t, byteSize)};
    memcpy(enhancement, data, byteSize);
    frame->m_enhancementData = enhancementDataAllocation;
    frame->m_state = FrameStateReorder;

    // Add frame to reorder table sorted by timestamp
    m_reorderIndex.insert(sortFramePtrTimestamp, frame);

    // Attach any pending base for matching timestamp
    if (BasePicture* bp = m_basePicturePending.findUnordered(findBasePictureTimestamp, &frame->timestamp);
        bp) {
        frame->setBase(bp->picture, bp->deadline, bp->userData);
        m_basePicturePending.remove(bp);
        m_eventSink->generate(pipeline::EventCanSendBase);
    }

    startReadyFrames();
    return LdcReturnCodeSuccess;
}

LdcReturnCode PipelineVulkan::sendDecoderBase(uint64_t timestamp, LdpPicture* basePicture,
                                              uint32_t timeoutUs, void* userData)
{
    VNLogDebug("sendBasePicture: %" PRIx64 " %p", timestamp, (void*)basePicture);
    VNTraceInstant("sendBasePicture", timestamp);

    // Find the frame associated with PTS
    FrameVulkan* frame{findFrame(timestamp)};
    if (frame) {
        // Enhancement exists
        if (LdcReturnCode ret = frame->setBase(
                basePicture, threadTimeMicroseconds(static_cast<int32_t>(timeoutUs)), userData);
            ret != LdcReturnCodeSuccess) {
            return ret;
        }

        // Kick off any frames that are at or before the base timestamp
        startProcessing(timestamp);
        m_eventSink->generate(pipeline::EventCanSendBase);
        return LdcReturnCodeSuccess;
    }

    BasePicture bp = {timestamp, basePicture,
                      threadTimeMicroseconds(static_cast<int32_t>(timeoutUs)), userData};

    if (m_basePicturePending.size() < m_configuration.enhancementDelay) {
        // Room to buffer picture
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
    passFrame->m_state = FrameStateReorder;
    passFrame->m_ready = true;
    passFrame->m_passthrough = true;
    passFrame->setBase(basePicture, threadTimeMicroseconds(static_cast<int32_t>(timeoutUs)), userData);

    m_reorderIndex.insert(sortFramePtrTimestamp, passFrame);

    startReadyFrames();
    return LdcReturnCodeSuccess;
}

LdcReturnCode PipelineVulkan::sendDecoderPicture(LdpPicture* outputPicture)
{
    VNLogDebug("sendOutputPicture: %p", (void*)outputPicture);
    VNTraceInstant("sendOutputPicture", (void*)outputPicture);

    // Add to available queue
    if (m_outputPictureAvailableBuffer.size() > m_configuration.maxLatency ||
        !m_outputPictureAvailableBuffer.tryPush(outputPicture)) {
        VNLogDebug("sendOutputPicture: AGAIN");
        return LdcReturnCodeAgain;
    }

    connectOutputPictures();

    startReadyFrames();
    return LdcReturnCodeSuccess;
}

LdpPicture* PipelineVulkan::receiveDecoderPicture(LdpDecodeInformation& decodeInfoOut)
{
    FrameVulkan* frame{};

    // Pull any done frame from start (lowest timestamp) of 'processing' frame index.
    while (true) {
        common::ScopedLock lock(m_interTaskMutex);

        if (m_processingIndex.isEmpty()) {
            // No frames in progress
            break;
        }

        if (m_processingIndex[0]->m_state == FrameStateDone) {
            // Earliest frame is finished
            frame = m_processingIndex[0];
            m_processingIndex.removeIndex(0);
            break;
        }

        if (m_processingIndex[0]->canComplete()) {
            // Earliest frame will complete, so hang around and wait for it
            VNLogDebug("receiveOutputPicture waiting for %" PRIx64, m_processingIndex[0]->timestamp);

            if (m_interTaskFrameDone.waitDeadline(lock, m_processingIndex[0]->m_deadline)) {
                continue;
            }
            VNLogWarning("receiveOutputPicture wait timed out");
#ifdef VN_SDK_LOG_ENABLE_DEBUG
            ldcTaskPoolDump(&m_taskPool, nullptr);
#endif
        } else {
            break;
        }
    }

    if (!frame) {
        return nullptr;
    }

    // Copy surviving data from frame
    decodeInfoOut = frame->m_decodeInfo;
    LdpPicture* pictureOut{frame->outputPicture};

    VNLogDebug("receiveOutputPicture: %" PRIx64 " %p hb:%d he:%d sk:%d enh:%d",
               decodeInfoOut.timestamp, (void*)pictureOut, decodeInfoOut.hasBase,
               decodeInfoOut.hasEnhancement, decodeInfoOut.skipped, decodeInfoOut.enhanced);

    VNTraceInstant("receiveOutputPicture", frame->timestamp, (void*)frame->outputPicture);

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

    VNLogDebug("receiveFinishedBasePicture: %" PRIx64 " %p", (void*)basePicture);
    VNTraceInstant("receiveFinishedBasePicture", (void*)basePicture);

    return basePicture;
}

void PipelineVulkan::getCapacity(LdpPipelineCapacity* capacity) { assert(0); }

// Dig out info about a current timestamp
LdcReturnCode PipelineVulkan::peekDecoder(uint64_t timestamp, uint32_t& widthOut, uint32_t& heightOut)
{
    // Flush everything up to given timestamp
    startProcessing(timestamp);

    // Find the frame associated with PTS
    const FrameVulkan* frame{findFrame(timestamp)};
    if (!frame) {
        return LdcReturnCodeNotFound;
    }
    if (!frame->globalConfig) {
        if (m_configuration.passthroughMode == PassthroughMode::Disable) {
            return LdcReturnCodeNotFound;
        } else {
            return LdcReturnCodeAgain;
        }
    }

    widthOut = frame->globalConfig->width;
    heightOut = frame->globalConfig->height;
    return LdcReturnCodeSuccess;
}

// Move any reorder frames at or before timestamp into processing state
void PipelineVulkan::startProcessing(uint64_t timestamp)
{
    // Mark any frames in reorder buffer as 'flush'
    for (uint32_t i = 0; i < m_reorderIndex.size(); ++i) {
        FrameVulkan* const frame = m_reorderIndex[i];
        if (frame->m_state == FrameStateReorder && compareTimestamps(frame->timestamp, timestamp) <= 0) {
            frame->m_ready = true;
        }
    }

    startReadyFrames();
}

// Make pending frames get decoded
LdcReturnCode PipelineVulkan::flush(uint64_t timestamp)
{
    VNLogDebug("flush: %" PRIx64 " %p", timestamp);
    VNTraceInstant("flush", timestamp);

    // Mark any frames in reorder buffer as 'flush'
    for (uint32_t i = 0; i < m_reorderIndex.size(); ++i) {
        FrameVulkan* const frame = m_reorderIndex[i];
        if (compareTimestamps(frame->timestamp, timestamp) <= 0) {
            // Mark frame as flushable
            frame->m_ready = true;
        }
    }

    startReadyFrames();
    return LdcReturnCodeSuccess;
}

// Mark everything before timestamp as not needing decoding
LdcReturnCode PipelineVulkan::skip(uint64_t timestamp)
{
    VNLogDebug("skip: %" PRIx64 " %p", timestamp);
    VNTraceInstant("skip", timestamp);

    // Look at all frames
    for (uint32_t i = 0; i < m_frames.size(); ++i) {
        FrameVulkan* const frame{VNAllocationPtr(m_frames[i], FrameVulkan)};
        if (compareTimestamps(frame->timestamp, timestamp) <= 0) {
            // Mark frame as skippable and flushable
            frame->m_skip = true;
            frame->m_ready = true;
        }
    }

    startReadyFrames();
    return LdcReturnCodeSuccess;
}

// Wait for all work to be finished - optionally stopping anything in progress
LdcReturnCode PipelineVulkan::synchronizeDecoder(uint64_t timestamp, bool dropPending)
{
    VNLogDebug("synchronize: %d", dropPending);
    VNTraceInstant("synchronize", dropPending);

    // Mark current frames as skippable
    for (uint32_t i = 0; i < m_frames.size(); ++i) {
        FrameVulkan* const frame{VNAllocationPtr(m_frames[i], FrameVulkan)};
        frame->m_skip = dropPending;
    }
    startReadyFrames();

    // For all pending frames that are not blocked on input - wait in timestamp order
    for (uint32_t i = 0; i < m_processingIndex.size(); ++i) {
        FrameVulkan* frame = m_processingIndex[i];
        if (!frame->canComplete()) {
            continue;
        }
        ldcTaskGroupWait(&frame->m_taskGroup);
    }

    return LdcReturnCodeSuccess;
}

// Buffers
//
BufferVulkan* PipelineVulkan::allocateBuffer(uint32_t requiredSize)
{
    // Allocate buffer structure
    LdcMemoryAllocation allocation;
    BufferVulkan* const buffer{VNAllocateZero(m_allocator, &allocation, BufferVulkan)};
    if (!buffer) {
        return nullptr;
    }
    // Insert into table
    m_buffers.append(allocation);

    // In place construction
    return new (buffer) BufferVulkan(m_core, requiredSize); // NOLINT(cppcoreguidelines-owning-memory)
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
    PictureVulkan* picture{VNAllocateZero(m_allocator, &pictureAllocation, PictureVulkan)};
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
    if (picture) {
        picture->setDesc(desc);
        return picture;
    }
    return nullptr;
}

LdpPicture* PipelineVulkan::allocPictureExternal(const LdpPictureDesc& desc,
                                                 const LdpPicturePlaneDesc* planeDescArr,
                                                 const LdpPictureBufferDesc* buffer)
{
    PictureVulkan* picture{allocatePicture()};
    if (picture) {
        picture->setDesc(desc);
        picture->setExternal(planeDescArr, buffer);
        return picture;
    }
    return nullptr;
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
// Returns nullptr if there is no capacity for another frame.
//
FrameVulkan* PipelineVulkan::allocateFrame(uint64_t timestamp)
{
    assert(findFrame(timestamp) == nullptr);
    assert(m_frames.size() < m_configuration.maxLatency);

    // Allocate frame with in place construction
    LdcMemoryAllocation frameAllocation = {};
    FrameVulkan* const frame{VNAllocateZero(m_allocator, &frameAllocation, FrameVulkan)};
    if (!frame) {
        return nullptr;
    }

    // Append allocation into table
    m_frames.append(frameAllocation);

    // In place construction
    return new (frame) FrameVulkan(this, timestamp); // NOLINT(cppcoreguidelines-owning-memory)
}

// Find existing Frame for a timestamp, or return nullptr if it does not exist.
//
FrameVulkan* PipelineVulkan::findFrame(uint64_t timestamp)
{
    // Look for frame
    if (const LdcMemoryAllocation * pAlloc{m_frames.findUnordered(findFrameTimestamp, &timestamp)}) {
        return VNAllocationPtr(*pAlloc, FrameVulkan);
    }

    return nullptr;
}

// Release frame back to pool
//
void PipelineVulkan::freeFrame(FrameVulkan* frame)
{
    // Release task group and allocations
    frame->release();

    // Find slot
    LdcMemoryAllocation* frameAlloc{m_frames.findUnordered(ldcVectorCompareAllocationPtr, frame)};

    if (!frameAlloc) {
        // Could not find picture!
        VNLogWarning("Could not find frame to release: %p", (void*)frame);
        return;
    }

    // Call destructor directly, as we are doing in-place construct/destruct
    frame->~FrameVulkan();

    // Release memory
    VNFree(m_allocator, frameAlloc);

    m_frames.removeReorder(frameAlloc);
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
    if (m_reorderIndex.size() >= m_maxReorder || m_reorderIndex[0]->m_ready) {
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
    // Pull ready frames from reorder table
    while (FrameVulkan* frame = getNextReordered()) {
        const uint64_t timestamp{frame->timestamp};
        bool goodConfig = false;

        if (m_previousTimestamp != kInvalidTimestamp &&
            compareTimestamps(m_previousTimestamp, timestamp) > 0) {
            // Frame has been flushed out of reorder queue too late - mark as pass- through
            VNLogDebug("startReadyFrames: out of order: ts:%" PRIx64 " prev: %" PRIx64);
            frame->m_passthrough = true;
        }

        if (!frame->m_passthrough) {
            // Parse the LCEVC configuration into distinct per-frame data
            // Switch to pass-through if configuration parse failed.
            goodConfig = ldeConfigPoolFrameInsert(&m_configPool, timestamp,
                                                  VNAllocationPtr(frame->m_enhancementData, uint8_t),
                                                  VNAllocationSize(frame->m_enhancementData, uint8_t),
                                                  &frame->globalConfig, &frame->config);

            if (!goodConfig) {
                frame->m_passthrough = true;
            }
        }

        if (frame->m_passthrough) {
            // Set up enough frame configuration to support pass-through
            ldeConfigPoolFramePassthrough(&m_configPool, &frame->globalConfig, &frame->config);
        }

        VNLogDebug("Start Frame: %" PRIx64 " goodConfig:%d temporalEnabled:%d, temporalPresent:%d "
                   "temporalRefresh:%d loqEnabled[0]:%d loqEnabled[1]:%d passthrough:%d",
                   timestamp, goodConfig, frame->globalConfig->temporalEnabled,
                   frame->config.temporalSignallingPresent, frame->config.temporalRefresh,
                   frame->config.loqEnabled[0], frame->config.loqEnabled[1], frame->m_passthrough);

        // Once we have per frame configuration, we can properly initialize and figure out tasks for the frame
        if (!frame->initialize()) {
            VNLogError("Could not allocate frame buffers: %" PRIx64, frame->timestamp);
            // Could not allocate buffers - switch to pass-through
            frame->m_passthrough = true;
        }

        // All good - make tasks, and add to processing index
        // with frame it should get temporal from if it needs it
        frame->generateTasks(m_lastGoodTimestamp);

        {
            common::ScopedLock lock(m_interTaskMutex);
            frame->m_state = FrameStateProcessing;
            m_processingIndex.append(frame);
        }

        // Remember timestamps for next time
        m_previousTimestamp = timestamp;
        if (goodConfig) {
            m_lastGoodTimestamp = timestamp;
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
        const LdpPictureDesc desc{frame->getOutputPictureDesc()};
        ldpPictureSetDesc(ldpPicture, &desc);
        if (frame->globalConfig->cropEnabled) {
            ldpPicture->margins.left = frame->globalConfig->crop.left;
            ldpPicture->margins.right = frame->globalConfig->crop.right;
            ldpPicture->margins.top = frame->globalConfig->crop.top;
            ldpPicture->margins.bottom = frame->globalConfig->crop.bottom;
        }

        // Poke it into the frame's task group
        frame->outputPicture = ldpPicture;

        VNLogDebug("connectOutputPicture: %" PRIx64 " %p %ux%u (r:%d p:%d o:%d)", frame->timestamp,
                   (void*)ldpPicture, desc.width, desc.height, m_reorderIndex.size(),
                   m_processingIndex.size(), m_outputPictureAvailableBuffer.size());
        ldcTaskDependencyMet(&frame->m_taskGroup, frame->m_depOutputPicture, ldpPicture);

        // Tell API there is output picture space

        m_eventSink->generate(pipeline::EventCanSendPicture);
    }
}

//// Temporal
//
// Mark a frame as needing a temporal buffer of given timestamp and dimensions
//
// This might be resolved immediately if the previous frame is done already.
//
LdcTaskDependency PipelineVulkan::requireTemporalBuffer(FrameVulkan* frame, uint64_t timestamp, uint32_t plane)
{
    LdcTaskDependency dep{ldcTaskDependencyAdd(&frame->m_taskGroup)};
    TemporalBuffer* found{};

    uint32_t width = frame->globalConfig->width;
    uint32_t height = frame->globalConfig->height;
    width >>= ldpColorFormatPlaneWidthShift(frame->baseFormat, plane);
    height >>= ldpColorFormatPlaneHeightShift(frame->baseFormat, plane);

    // Fill in requirements
    frame->m_temporalBufferDesc[plane].timestamp = timestamp;
    frame->m_temporalBufferDesc[plane].clear =
        frame->config.nalType == NTIDR || frame->config.temporalRefresh;
    frame->m_temporalBufferDesc[plane].width = width;
    frame->m_temporalBufferDesc[plane].height = height;
    frame->m_temporalBufferDesc[plane].plane = plane;

    VNLogDebug("requireTemporalBuffer: %" PRIx64 " wants %" PRIx64 " plane %" PRIu32 " (%d %dx%d)",
               frame->timestamp, timestamp, plane, frame->m_temporalBufferDesc[plane].clear, width,
               height);

    {
        common::ScopedLock lock(m_interTaskMutex);

        // Do any of the available temporal buffers meet the requirements?
        for (uint32_t i = 0; i < m_temporalBuffers.size(); ++i) {
            TemporalBuffer* tb{m_temporalBuffers.at(i)};
            if (tb->frame) {
                // In use
                continue;
            }

            if (tb->desc.plane == plane && tb->desc.timestamp == timestamp) {
                // Exact plane index and timestamp match
                found = tb;
                break;
            }

            if (frame->m_temporalBufferDesc[plane].clear && tb->desc.timestamp == kInvalidTimestamp) {
                // An existing unused buffer
                found = tb;
                break;
            }
        }

        // Got one - mark it as in use
        if (found) {
            frame->m_temporalBuffer[plane] = found;
            found->frame = frame;

            if (frame->m_temporalBufferDesc[plane].clear) {
                // Update limit on any other prior buffers
                for (uint32_t i = 0; i < m_temporalBuffers.size(); ++i) {
                    TemporalBuffer* tb = m_temporalBuffers.at(i);
                    if (tb == found) {
                        continue;
                    }
                }
            }
        }

        frame->m_depTemporalBuffer[plane] = dep;
    }

    if (!found) {
        // Not found - will get resolved later by prior frame
        return dep;
    }

    VNLogDebug("  requireTemporalBuffer found: plane=%" PRIu32 " frame=%" PRIx64 " prev=%" PRIx64,
               plane, frame->timestamp, found->desc.timestamp);

    // Make sure found buffer meets requirements
    updateTemporalBufferDesc(found, frame->m_temporalBufferDesc[plane]);

    ldcTaskDependencyMet(&frame->m_taskGroup, dep, found);
    return dep;
}

// Mark the frame as having finished with its temporal buffer, and possibly
// hand buffer on to another frame
//
void PipelineVulkan::releaseTemporalBuffer(FrameVulkan* prevFrame, uint32_t plane)
{
    FrameVulkan* foundFrame{nullptr};
    TemporalBuffer* tb{nullptr};

    VNLogDebug("releaseTemporalBuffer: %" PRIx64 " plane: %" PRIu32, prevFrame->timestamp, plane);

    {
        common::ScopedLock lock(m_interTaskMutex);

        tb = prevFrame->m_temporalBuffer[plane];

        if (!tb) {
            // No temporal buffer to be released
            return;
        }

        tb->desc.timestamp = prevFrame->timestamp;
        // Do any of the pending frames want this buffer?
        for (uint32_t idx = 0; idx < m_processingIndex.size(); ++idx) {
            FrameVulkan* frame{m_processingIndex[idx]};

            if (compareTimestamps(frame->timestamp, prevFrame->timestamp) <= 0) {
                continue;
            }

            if (frame->m_depTemporalBuffer[plane] == kTaskDependencyInvalid ||
                frame->m_temporalBuffer[plane]) {
                // Does not need a buffer
                continue;
            }

            if (tb->desc.timestamp == frame->m_temporalBufferDesc[plane].timestamp) {
                // Matches this frame
                foundFrame = frame;
                break;
            }
            if (tb->desc.timestamp == kInvalidTimestamp) {
                // Unused buffer
                foundFrame = frame;
                break;
            }
        }

        // Detach from previous frame
        prevFrame->m_temporalBuffer[plane] = nullptr;
        tb->frame = nullptr;

        if (foundFrame) {
            foundFrame->m_temporalBuffer[plane] = tb;
            tb->frame = foundFrame;
        }
    }

    if (foundFrame) {
        VNLogDebug("  Vulkan::releaseTemporalBuffer found: plane=%" PRIu32 " frame=%" PRIx64
                   " prev=%" PRIx64,
                   plane, foundFrame->timestamp, prevFrame->timestamp);
        updateTemporalBufferDesc(tb, foundFrame->m_temporalBufferDesc[plane]);
        ldcTaskDependencyMet(&foundFrame->m_taskGroup, foundFrame->m_depTemporalBuffer[plane], tb);
    }
}

// Make a temporal buffer match the given description
void PipelineVulkan::updateTemporalBufferDesc(TemporalBuffer* buffer, const TemporalBufferDesc& desc) const
{
    const size_t byteStride{static_cast<size_t>(desc.width) * sizeof(uint16_t)};
    const size_t bufferSize{byteStride * static_cast<size_t>(desc.height)};

    if (!VNIsAllocated(buffer->allocation) || buffer->desc.width != desc.width ||
        buffer->desc.height != desc.height) {
        // Reallocate buffer
        if (!desc.clear && desc.timestamp != kInvalidTimestamp) {
            // Frame was expecting prior residuals - but dimensions are wrong!?
            VNLogWarning("Temporal buffer does not match: %08d Got %dx%d, Wanted %dx%d", desc.timestamp,
                         buffer->desc.width, buffer->desc.height, desc.width, desc.height);
        }
        buffer->planeDesc.firstSample =
            VNReallocateArray(allocator(), &buffer->allocation, uint8_t, bufferSize);
        buffer->planeDesc.rowByteStride = static_cast<uint32_t>(byteStride);
        memset(buffer->planeDesc.firstSample, 0, bufferSize);
    } else if (desc.clear) {
        memset(buffer->planeDesc.firstSample, 0, bufferSize);
    }

    // Update description
    buffer->desc = desc;
    buffer->desc.clear = false;
}

//// ConvertToInternal
//
// Copy incoming picture plane to internal fixed point surface format
//
// NB: There is likely a good templated C++ class that wraps these tasks up neatly,
// Worth figuring out once this has stabilised.
//
struct TaskConvertToInternalData
{
    PipelineVulkan* pipeline;
    FrameVulkan* frame;
    unsigned baseDepth;
    unsigned enhancementDepth;
};

void* PipelineVulkan::taskConvertToInternal(LdcTask* task, const LdcTaskPart* /*part*/)
{
    assert(task->dataSize == sizeof(TaskConvertToInternalData));
    const TaskConvertToInternalData& data{VNTaskData(task, TaskConvertToInternalData)};
    PipelineVulkan* const pipeline{data.pipeline};
    FrameVulkan* const frame{data.frame};

    if (frame->m_skip) {
        return nullptr;
    }

    auto* srcPicture = static_cast<PictureVulkan*>(frame->basePicture);
    LdpPictureDesc srcDesc;
    srcPicture->getDesc(srcDesc);

    // external base check
    if (LdpPictureBufferDesc exDesc{}; srcPicture->getBufferDesc(exDesc)) {
        auto managedBuffer = static_cast<BufferVulkan*>(srcPicture->buffer);
        if (managedBuffer->size() != exDesc.byteSize) { // padded base
            const bool nv12 = (srcPicture->layout.layoutInfo->format == LdpColorFormatNV12_8 ||
                               srcPicture->layout.layoutInfo->format == LdpColorFormatNV21_8)
                                  ? true
                                  : false;

            const uint32_t byteWidth = (frame->baseBitdepth == 8) ? srcDesc.width : 2 * srcDesc.width;
            const uint32_t planeWidth = nv12 ? byteWidth : byteWidth >> 1;

            auto removePadding = [&](uint32_t width, uint32_t height, uint32_t pixelWidth,
                                     uint8_t planeIndex, uint32_t internalOffset,
                                     uint32_t externalOffsetU, uint32_t externalOffsetV) {
                for (uint32_t y = 0; y < height; ++y) {
                    const auto internalIndex = internalOffset + y * pixelWidth;
                    auto externalIndex = y * srcPicture->layout.rowStrides[planeIndex];
                    if (planeIndex > 0) {
                        externalIndex += externalOffsetU * srcPicture->layout.rowStrides[planeIndex - 1];
                    }
                    if (planeIndex > 1) {
                        externalIndex += externalOffsetV * srcPicture->layout.rowStrides[planeIndex - 2];
                    }
                    std::memcpy(managedBuffer->ptr() + internalIndex, exDesc.data + externalIndex, width);
                }
            };

            removePadding(byteWidth, srcDesc.height, byteWidth, 0, 0, 0, 0); // Y

            if (pipeline->m_chroma != LdeChroma::CTMonochrome) {
                removePadding(planeWidth, srcDesc.height >> 1, planeWidth, 1,
                              byteWidth * srcDesc.height, srcDesc.height, 0); // U

                if (!nv12) {
                    removePadding(byteWidth >> 1, srcDesc.height >> 1, planeWidth, 2,
                                  5 * byteWidth * srcDesc.height >> 2, srcDesc.height >> 1,
                                  srcDesc.height); // V

                    srcPicture->layout.rowStrides[2] = planeWidth;
                    srcPicture->layout.planeOffsets[2] = 5 * byteWidth * srcDesc.height >> 2;
                }
                srcPicture->layout.rowStrides[1] = planeWidth;
                srcPicture->layout.planeOffsets[1] = byteWidth * srcDesc.height;
            }
            srcPicture->layout.rowStrides[0] = byteWidth;
            srcPicture->layout.planeOffsets[0] = 0;
        } else {
            std::memcpy(managedBuffer->ptr(), exDesc.data, exDesc.byteSize);
        }
    }

    srcDesc.colorFormat = pipeline->chromaToColorFormat(pipeline->m_chroma);
    auto* dstPicture = static_cast<PictureVulkan*>(pipeline->allocPicture(srcDesc));

    VulkanConversionArgs args{};
    args.src = srcPicture;
    args.dst = dstPicture;
    args.toInternal = true;
    args.bitDepth = frame->baseBitdepth;
    args.chroma = pipeline->m_chroma;

    if (!pipeline->m_core.conversion(&args)) {
        VNLogError("Conversion to internal failed");
    }

    frame->m_intermediatePicture[LOQ2] = dstPicture;

    return nullptr;
}

LdcTaskDependency PipelineVulkan::addTaskConvertToInternal(FrameVulkan* frame, unsigned baseDepth,
                                                           unsigned enhancementDepth,
                                                           LdcTaskDependency inputDep)
{
    const TaskConvertToInternalData data{this, frame, baseDepth, enhancementDepth};
    const LdcTaskDependency inputs[] = {inputDep};
    const LdcTaskDependency outputDep{ldcTaskDependencyAdd(&frame->m_taskGroup)};

    ldcTaskGroupAdd(&frame->m_taskGroup, inputs, VNArraySize(inputs), outputDep,
                    taskConvertToInternal, nullptr, 1, 1, sizeof(data), &data, "ConvertToInternal");

    return outputDep;
}

//// ConvertFromInternal
//
// Concert a picture plane from internal fixed point to output picture pixel format.
//
struct TaskConvertFromInternalData
{
    PipelineVulkan* pipeline;
    FrameVulkan* frame;
    unsigned baseDepth;
    unsigned enhancementDepth;
    uint8_t intermediatePtr;
};

void* PipelineVulkan::taskConvertFromInternal(LdcTask* task, const LdcTaskPart* /*part*/)
{
    assert(task->dataSize == sizeof(TaskConvertFromInternalData));
    const TaskConvertFromInternalData& data{VNTaskData(task, TaskConvertFromInternalData)};
    PipelineVulkan* const pipeline{data.pipeline};
    FrameVulkan* const frame{data.frame};
    const uint8_t intermediatePtr{data.intermediatePtr};

    if (frame->m_skip) {
        return nullptr;
    }

    auto* srcPicture = frame->m_intermediatePicture[intermediatePtr];

    LdpPictureDesc dstDesc;
    srcPicture->getDesc(dstDesc);
    dstDesc.colorFormat = frame->outputPicture->layout.layoutInfo->format;
    frame->outputPicture->functions->setDesc(frame->outputPicture, &dstDesc);

    VulkanConversionArgs args{};
    args.src = srcPicture;
    args.dst = static_cast<PictureVulkan*>(frame->outputPicture);
    args.toInternal = false;
    args.bitDepth = frame->getEnhancementBitDepth();
    args.chroma = pipeline->m_chroma;

    if (!pipeline->m_core.conversion(&args)) {
        VNLogError("Conversion from internal failed");
    }

    if (frame->globalConfig->cropEnabled) {
        args.dst->margins.left = frame->globalConfig->crop.left;
        args.dst->margins.right = frame->globalConfig->crop.right;
        args.dst->margins.top = frame->globalConfig->crop.top;
        args.dst->margins.bottom = frame->globalConfig->crop.bottom;
    }

    // external output check
    if (LdpPictureBufferDesc exDesc{};
        static_cast<PictureVulkan*>(frame->outputPicture)->getBufferDesc(exDesc)) {
        auto managedBuffer =
            static_cast<BufferVulkan*>(static_cast<PictureVulkan*>(frame->outputPicture)->buffer);
        std::memcpy(exDesc.data, managedBuffer->ptr(), exDesc.byteSize);
    }

    return nullptr;
}

LdcTaskDependency PipelineVulkan::addTaskConvertFromInternal(FrameVulkan* frame, unsigned baseDepth,
                                                             unsigned enhancementDepth,
                                                             LdcTaskDependency dstDep,
                                                             LdcTaskDependency srcDep,
                                                             uint8_t intermediatePtr)
{
    const TaskConvertFromInternalData data{this, frame, baseDepth, enhancementDepth, intermediatePtr};
    const LdcTaskDependency inputs[] = {dstDep, srcDep};
    const LdcTaskDependency output = ldcTaskDependencyAdd(&frame->m_taskGroup);

    ldcTaskGroupAdd(&frame->m_taskGroup, inputs, VNArraySize(inputs), output, taskConvertFromInternal,
                    nullptr, 1, 1, sizeof(data), &data, "ConvertFromInternal");

    return output;
}

//// Upsample
//
// Upscale (1D or 2D) for one plane of picture.
//
// Inputs and outputs may be fixed point or 'external' format if no residuals are being applied.
//
struct TaskUpsampleData
{
    PipelineVulkan* pipeline;
    FrameVulkan* frame;
    LdeLOQIndex loq;
    uint8_t intermediatePtr;
    LdeKernel kernel;
};

void* PipelineVulkan::taskUpsample(LdcTask* task, const LdcTaskPart* /*part*/)
{
    assert(task->dataSize == sizeof(TaskUpsampleData));
    const TaskUpsampleData& data{VNTaskData(task, TaskUpsampleData)};

    PipelineVulkan* const pipeline{data.pipeline};
    FrameVulkan* const frame{data.frame};
    const uint8_t intermediatePtr{data.intermediatePtr};
    const LdeLOQIndex loq{data.loq};

    if (frame->m_skip) {
        return nullptr;
    }

    VulkanUpscaleArgs upscaleArgs{};
    upscaleArgs.src = frame->m_intermediatePicture[intermediatePtr];
    const LdpPictureDesc desc{2, 2, LdpColorFormatI420_8};
    frame->m_intermediatePicture[intermediatePtr - 1] =
        static_cast<PictureVulkan*>(pipeline->allocPicture(desc));
    upscaleArgs.dst = frame->m_intermediatePicture[intermediatePtr - 1];

    upscaleArgs.applyPA = static_cast<uint8_t>(frame->globalConfig->predictedAverageEnabled);
    upscaleArgs.dither = nullptr; // TODO pipeline->m_dither;
    upscaleArgs.mode = frame->globalConfig->scalingModes[data.loq - 1];
    upscaleArgs.vertical = false;
    upscaleArgs.loq1 = (loq == 2) ? true : false;
    upscaleArgs.intermediateUpscalePicture[0] = pipeline->m_intermediateUpscalePicture[LOQ0].get();
    upscaleArgs.intermediateUpscalePicture[1] = pipeline->m_intermediateUpscalePicture[LOQ1].get();
    upscaleArgs.chroma = pipeline->m_chroma;

    assert(upscaleArgs.mode != Scale0D);
    VNLogDebug("taskUpsample timestamp:%" PRIx64 " loq:%d", frame->timestamp, (uint32_t)data.loq);

    if (!pipeline->m_core.upscaleFrame(&frame->globalConfig->kernel, &upscaleArgs)) {
        VNLogError("Upsample failed");
    }

    return nullptr;
}

LdcTaskDependency PipelineVulkan::addTaskUpsample(FrameVulkan* frame, LdeLOQIndex loq,
                                                  LdcTaskDependency inputDep, uint8_t intermediatePtr)
{
    const TaskUpsampleData data{this, frame, loq, intermediatePtr};

    assert(frame->globalConfig->scalingModes[loq - 1] != Scale0D);

    const LdcTaskDependency inputs[] = {inputDep};
    const LdcTaskDependency output{ldcTaskDependencyAdd(&frame->m_taskGroup)};

    ldcTaskGroupAdd(&frame->m_taskGroup, inputs, VNArraySize(inputs), output, taskUpsample, nullptr,
                    1, 1, sizeof(data), &data, "Upsample");

    return output;
}

//// GenerateCmdBuffer
//
// Convert un-encapsulated chunks into a single command buffer.
//
struct TaskGenerateCmdBufferData
{
    PipelineVulkan* pipeline;
    FrameVulkan* frame;
    LdpEnhancementTile* enhancementTile;
};

void* PipelineVulkan::taskGenerateCmdBuffer(LdcTask* task, const LdcTaskPart* /*part*/)
{
    assert(task->dataSize == sizeof(TaskGenerateCmdBufferData));
    const TaskGenerateCmdBufferData& data{VNTaskData(task, TaskGenerateCmdBufferData)};
    FrameVulkan* const frame{data.frame};

    VNLogDebug("taskGenerateCmdBuffer timestamp:%" PRIx64 " tile:%d loq:%d plane:%d",
               data.frame->timestamp, data.enhancementTile->tile,
               (uint32_t)data.enhancementTile->loq, data.enhancementTile->plane);

    if (!ldeDecodeEnhancement(frame->globalConfig, &frame->config, data.enhancementTile->loq,
                              data.enhancementTile->plane, data.enhancementTile->tile, nullptr,
                              &data.enhancementTile->bufferGpu, &data.enhancementTile->bufferGpuBuilder)) {
        VNLogError("ldeDecodeEnhancement failed");
    }

    return nullptr;
}

LdcTaskDependency PipelineVulkan::addTaskGenerateCmdBuffer(FrameVulkan* frame,
                                                           LdpEnhancementTile* enhancementTile)
{
    const TaskGenerateCmdBufferData data{this, frame, enhancementTile};
    const LdcTaskDependency output{ldcTaskDependencyAdd(&frame->m_taskGroup)};

    ldcTaskGroupAdd(&frame->m_taskGroup, nullptr, 0, output, taskGenerateCmdBuffer, nullptr, 1, 1,
                    sizeof(data), &data, "GenerateCmdBuffer");

    return output;
}

//// ApplyCmdBufferDirect
//
// Apply a generated GPU command buffer to directly to output plane. (No Temporal)
//
// NB: The output plane will be in 'internal' fixed' point format
//
struct TaskApplyCmdBufferDirectData
{
    PipelineVulkan* pipeline;
    FrameVulkan* frame;
    LdpEnhancementTile* enhancementTile;
    uint8_t intermediatePtr;
};

void* PipelineVulkan::taskApplyCmdBufferDirect(LdcTask* task, const LdcTaskPart* part)
{
    VNTraceScoped();
    VNUnused(part);

    assert(task->dataSize == sizeof(TaskApplyCmdBufferDirectData));
    const TaskApplyCmdBufferDirectData& data{VNTaskData(task, TaskApplyCmdBufferDirectData)};
    PipelineVulkan* const pipeline{data.pipeline};
    FrameVulkan* const frame{data.frame};
    const uint8_t intermediatePtr{data.intermediatePtr};

    if (frame->m_skip) {
        return nullptr;
    }

    VNLogDebug("taskApplyCmdBufferDirect timestamp:%" PRIx64 " loq:%d plane:%d", data.frame->timestamp,
               (uint32_t)data.enhancementTile->loq, data.enhancementTile->plane);

    auto* picture = frame->m_intermediatePicture[intermediatePtr];

    VulkanApplyArgs args{};
    pipeline->prepareApplyArgs(args, picture, data.enhancementTile, frame, true);

    if (!pipeline->m_core.apply(&args)) {
        VNLogError("Vulkan apply direct failed");
    }

    return nullptr;
}

LdcTaskDependency PipelineVulkan::addTaskApplyCmdBufferDirect(FrameVulkan* frame,
                                                              LdpEnhancementTile* enhancementTile,
                                                              LdcTaskDependency inputDep,
                                                              LdcTaskDependency cmdBufferDep,
                                                              uint8_t intermediatePtr)
{
    const TaskApplyCmdBufferDirectData data{this, frame, enhancementTile, intermediatePtr};
    const LdcTaskDependency inputs[] = {inputDep, cmdBufferDep};
    const LdcTaskDependency output{ldcTaskDependencyAdd(&frame->m_taskGroup)};

    ldcTaskGroupAdd(&frame->m_taskGroup, inputs, VNArraySize(inputs), output, taskApplyCmdBufferDirect,
                    nullptr, 1, 1, sizeof(data), &data, "ApplyCmdBufferDirect");

    return output;
}

//// ApplyCmdBufferTemporal
//
// Apply a generated GPU command buffer to a temporal buffer.
//
struct TaskApplyCmdBufferTemporalData
{
    PipelineVulkan* pipeline;
    FrameVulkan* frame;
    LdpEnhancementTile* enhancementTile;
    uint8_t intermediatePtr;
};

void* PipelineVulkan::taskApplyCmdBufferTemporal(LdcTask* task, const LdcTaskPart* part)
{
    VNTraceScoped();
    VNUnused(part);

    assert(task->dataSize == sizeof(TaskApplyCmdBufferTemporalData));
    const TaskApplyCmdBufferTemporalData& data{VNTaskData(task, TaskApplyCmdBufferTemporalData)};
    PipelineVulkan* const pipeline{data.pipeline};
    FrameVulkan* const frame{data.frame};
    const uint8_t intermediatePtr{data.intermediatePtr};

    VNLogDebug("taskApplyCmdBufferTemporal timestamp:%" PRIx64 " tile:%d loq:%d plane:%d",
               data.frame->timestamp, data.enhancementTile->tile,
               (uint32_t)data.enhancementTile->loq, data.enhancementTile->plane);

    auto* picture = frame->m_intermediatePicture[intermediatePtr];

    VulkanApplyArgs args{};
    pipeline->prepareApplyArgs(args, picture, data.enhancementTile, frame, false);
    args.temporalPicture = pipeline->m_temporalPicture.get();

    if (!pipeline->m_core.apply(&args)) {
        VNLogError("Vulkan apply temporal failed");
    }

    return nullptr;
}

LdcTaskDependency PipelineVulkan::addTaskApplyCmdBufferTemporal(FrameVulkan* frame,
                                                                LdpEnhancementTile* enhancementTile,
                                                                LdcTaskDependency temporalBufferDep,
                                                                LdcTaskDependency cmdBufferDep,
                                                                uint8_t intermediatePtr)
{
    const TaskApplyCmdBufferTemporalData data{this, frame, enhancementTile, intermediatePtr};
    const LdcTaskDependency inputs[] = {temporalBufferDep, cmdBufferDep};
    const LdcTaskDependency output{ldcTaskDependencyAdd(&frame->m_taskGroup)};

    ldcTaskGroupAdd(&frame->m_taskGroup, inputs, VNArraySize(inputs), output, taskApplyCmdBufferTemporal,
                    nullptr, 1, 1, sizeof(data), &data, "ApplyCmdBufferTemporal");

    return output;
}

//// ApplyAddTemporal
//
// Add a temporal buffer to a picture plane.
//
struct TaskApplyAddTemporalData
{
    PipelineVulkan* pipeline;
    FrameVulkan* frame;
    uint8_t intermediatePtr;
};

void* PipelineVulkan::taskApplyAddTemporal(LdcTask* task, const LdcTaskPart* part)
{
    VNTraceScoped();
    VNUnused(part);

    assert(task->dataSize == sizeof(TaskApplyAddTemporalData));
    const TaskApplyAddTemporalData& data{VNTaskData(task, TaskApplyAddTemporalData)};
    PipelineVulkan* const pipeline{data.pipeline};
    FrameVulkan* const frame{data.frame};
    const uint8_t intermediatePtr{data.intermediatePtr};

    if (frame->m_skip) {
        return nullptr;
    }

    VNLogDebug("taskApplyAddTemporal timestamp:%" PRIx64 "", data.frame->timestamp);

    VulkanBlitArgs args{};
    args.src = pipeline->m_temporalPicture.get();
    args.dst = frame->m_intermediatePicture[intermediatePtr];
    args.numEnhancedPlanes = frame->numEnhancedPlanes();
    args.chroma = pipeline->m_chroma;

    if (!pipeline->m_core.blit(&args)) {
        VNLogError("Vulkan blit failed");
    }

    return nullptr;
}

LdcTaskDependency PipelineVulkan::addTaskApplyAddTemporal(FrameVulkan* frame, LdcTaskDependency temporalDep,
                                                          LdcTaskDependency sourceDep,
                                                          uint8_t intermediatePtr)
{
    const TaskApplyAddTemporalData data{this, frame, intermediatePtr};
    const LdcTaskDependency inputs[] = {temporalDep, sourceDep};
    const LdcTaskDependency output{ldcTaskDependencyAdd(&frame->m_taskGroup)};

    ldcTaskGroupAdd(&frame->m_taskGroup, inputs, VNArraySize(inputs), output, taskApplyAddTemporal,
                    nullptr, 1, 1, sizeof(data), &data, "ApplyAddTemporal");

    return output;
}

//// Passthrough
//
// Copy incoming picture plane to output picture
//
struct TaskPassthroughData
{
    PipelineVulkan* pipeline;
    FrameVulkan* frame;
    uint32_t planeIndex;
};

void* PipelineVulkan::taskPassthrough(LdcTask* task, const LdcTaskPart* /*part*/)
{
    VNTraceScoped();
    assert(task->dataSize == sizeof(TaskPassthroughData));

    const TaskPassthroughData& data{VNTaskData(task, TaskPassthroughData)};
    PipelineVulkan* const pipeline{data.pipeline};
    const FrameVulkan* const frame{data.frame};

    if (frame->m_skip) {
        return nullptr;
    }

    LdpPicturePlaneDesc srcPlane;
    frame->getBasePlaneDesc(data.planeIndex, srcPlane);

    LdpPicturePlaneDesc dstPlane;
    frame->getOutputPlaneDesc(data.planeIndex, dstPlane);

    VNLogDebug("taskPassthrough timestamp:%" PRIx64 " plane:%d", data.frame->timestamp, data.planeIndex);

    if (!ldppPlaneBlit(&pipeline->m_taskPool, task, pipeline->m_configuration.forceScalar,
                       data.planeIndex, &frame->basePicture->layout, &frame->outputPicture->layout,
                       &srcPlane, &dstPlane, BMCopy)) {
        VNLogError("ldppPlaneBlit In failed");
    }

    return nullptr;
}

LdcTaskDependency PipelineVulkan::addTaskPassthrough(FrameVulkan* frame, uint32_t planeIndex,
                                                     LdcTaskDependency dest, LdcTaskDependency src)
{
    const TaskPassthroughData data{this, frame, planeIndex};
    const LdcTaskDependency inputs[] = {dest, src};
    const LdcTaskDependency output{ldcTaskDependencyAdd(&frame->m_taskGroup)};

    ldcTaskGroupAdd(&frame->m_taskGroup, inputs, VNArraySize(inputs), output, taskPassthrough,
                    nullptr, 1, 1, sizeof(data), &data, "Passthrough");

    return output;
}

//// WaitForMany
//
// Wait for several input dependencies to be met.
//
// NB: If this appears to be a bottleneck, it could be integrated better into the task pool.
//
struct TaskWaitForManyData
{
    PipelineVulkan* pipeline;
    FrameVulkan* frame;
};

void* PipelineVulkan::taskWaitForMany(LdcTask* task, const LdcTaskPart* part)
{
    VNUnused(part);

    assert(task->dataSize == sizeof(TaskWaitForManyData));
    const TaskWaitForManyData& data{VNTaskData(task, TaskWaitForManyData)};

    VNLogDebug("taskWaitForMany timestamp:%" PRIx64 "", data.frame->timestamp);
    return nullptr;
}

LdcTaskDependency PipelineVulkan::addTaskWaitForMany(FrameVulkan* frame, const LdcTaskDependency* inputDeps,
                                                     uint32_t inputDepsCount)
{
    const TaskWaitForManyData data{this, frame};
    const LdcTaskDependency outputDep{ldcTaskDependencyAdd(&frame->m_taskGroup)};

    ldcTaskGroupAdd(&frame->m_taskGroup, inputDeps, inputDepsCount, outputDep, taskWaitForMany,
                    nullptr, 1, 1, sizeof(data), &data, "WaitForMany");

    return outputDep;
}

//// BaseSend
//
// Wait for base picture planes to be used, then send base picture back to client
//
struct TaskBaseDoneData
{
    PipelineVulkan* pipeline;
    FrameVulkan* frame;
};

void* PipelineVulkan::taskBaseDone(LdcTask* task, const LdcTaskPart* part)
{
    VNTraceScoped();
    assert(task->dataSize == sizeof(TaskBaseDoneData));

    const TaskBaseDoneData& data{VNTaskData(task, TaskBaseDoneData)};

    VNLogDebug("taskBaseDone timestamp:%" PRIx64, data.frame->timestamp);

    assert(data.frame->basePicture);

    // Generate event
    data.pipeline->m_eventSink->generate(pipeline::EventBasePictureDone, data.frame->basePicture);

    // Send base picture back to API
    data.pipeline->m_basePictureOutBuffer.push(data.frame->basePicture);

    // Frame no longer has access to base picture
    data.frame->basePicture = nullptr;
    return nullptr;
}

void PipelineVulkan::addTaskBaseDone(FrameVulkan* frame, const LdcTaskDependency* inputs, uint32_t inputsCount)
{
    const TaskBaseDoneData data{this, frame};

    ldcTaskGroupAdd(&frame->m_taskGroup, inputs, inputsCount, kTaskDependencyInvalid, taskBaseDone,
                    nullptr, 1, 1, sizeof(data), &data, "BaseDone");
}

//// OutputSend
//
// Wait for a bunch of input dependencies to be met, then:
//
// - Send output picture to output queue
// - Release frame
//
struct TaskOutputDoneData
{
    PipelineVulkan* pipeline;
    FrameVulkan* frame;
};

void* PipelineVulkan::taskOutputDone(LdcTask* task, const LdcTaskPart* part)
{
    VNUnused(part);
    assert(task->dataSize == sizeof(TaskOutputDoneData));
    const TaskOutputDoneData& data{VNTaskData(task, TaskOutputDoneData)};
    PipelineVulkan* const pipeline{data.pipeline};
    FrameVulkan* const frame{data.frame};

    VNLogDebug("taskOutputDone timestamp:%" PRIx64, frame->timestamp);

    // Mark as done, and signal pipeline if it is waiting
    {
        common::ScopedLock lock(pipeline->m_interTaskMutex);
        frame->m_state = FrameStateDone;

        // Build the decode info for the frame
        frame->m_decodeInfo.timestamp = frame->timestamp;
        frame->m_decodeInfo.hasBase = true;
        frame->m_decodeInfo.hasEnhancement =
            frame->config.loqEnabled[LOQ1] || frame->config.loqEnabled[LOQ0];
        frame->m_decodeInfo.skipped = frame->m_skip;
        frame->m_decodeInfo.enhanced = frame->config.loqEnabled[LOQ1] || frame->config.loqEnabled[LOQ0];
        frame->m_decodeInfo.baseWidth = frame->baseWidth;
        frame->m_decodeInfo.baseHeight = frame->baseHeight;
        frame->m_decodeInfo.baseBitdepth = frame->baseBitdepth;
        frame->m_decodeInfo.userData = frame->userData;

        pipeline->m_interTaskFrameDone.signal();

        pipeline->m_eventSink->generate(pipeline::EventOutputPictureDone, frame->outputPicture,
                                        &frame->m_decodeInfo);
        pipeline->m_eventSink->generate(pipeline::EventCanReceive);
    }

    return nullptr;
}

void PipelineVulkan::addTaskOutputDone(FrameVulkan* frame, const LdcTaskDependency* inputs, uint32_t inputsCount)
{
    const TaskOutputDoneData data{this, frame};

    ldcTaskGroupAdd(&frame->m_taskGroup, inputs, inputsCount, kTaskDependencyInvalid,
                    taskOutputDone, nullptr, 1, 1, sizeof(data), &data, "OutputDone");
}

//// TemporalRelease
//
// Wait for a bunch of input dependencies to be met, then:
//
// - Release temporal buffer to next frame
//
struct TaskTemporalReleaseData
{
    PipelineVulkan* pipeline;
    FrameVulkan* frame;
};

void* PipelineVulkan::taskTemporalRelease(LdcTask* task, const LdcTaskPart* /*part*/)
{
    VNTraceScoped();
    assert(task->dataSize == sizeof(TaskTemporalReleaseData));

    const TaskTemporalReleaseData& data{VNTaskData(task, TaskTemporalReleaseData)};
    PipelineVulkan* const pipeline{data.pipeline};
    FrameVulkan* const frame{data.frame};

    VNLogDebug("taskTemporalRelease timestamp:%" PRIx64, frame->timestamp);

    pipeline->releaseTemporalBuffer(frame, 0);

    return nullptr;
}

void PipelineVulkan::addTaskTemporalRelease(FrameVulkan* frame, const LdcTaskDependency* deps)
{
    const TaskTemporalReleaseData data{this, frame};
    const LdcTaskDependency inputs[] = {deps[0]};

    ldcTaskGroupAdd(&frame->m_taskGroup, inputs, VNArraySize(inputs), kTaskDependencyInvalid,
                    taskTemporalRelease, nullptr, 1, 1, sizeof(data), &data, "TemporalRelease");
}

// Fill out a task group given a frame configuration
//
void PipelineVulkan::generateTasksEnhancement(FrameVulkan* frame, uint64_t previousTimestamp)
{
    VNTraceScoped();

    // Convenience values for readability
    const LdeFrameConfig& frameConfig{frame->config};
    const LdeGlobalConfig& globalConfig{*frame->globalConfig};
    const uint8_t numImagePlanes{frame->numImagePlanes()};

    auto intermediatePtr = static_cast<uint8_t>(LOQ2);

    m_chroma = frame->globalConfig->chroma;

    uint32_t enhancementTileIdx = 0;

    if (frame->config.sharpenType != STDisabled && frame->config.sharpenStrength != 0.0f) {
        VNLogWarning("S-Filter is configured in stream, but not supported by decoder.");
    }

    //// Input conversion
    LdcTaskDependency basePicture{kTaskDependencyInvalid};
    basePicture = addTaskConvertToInternal(frame, globalConfig.baseDepth,
                                           globalConfig.enhancedDepth, frame->m_depBasePicture);

    //// LoQ 1

    //// Base + Residuals
    //
    // First upsample
    LdcTaskDependency baseUpsampled{kTaskDependencyInvalid};
    if (globalConfig.scalingModes[LOQ1] != Scale0D) {
        baseUpsampled = addTaskUpsample(frame, LOQ2, basePicture, intermediatePtr);
        intermediatePtr--;
    } else {
        baseUpsampled = basePicture;
    }

    // Enhancement LOQ1 decoding
    LdcTaskDependency basePlanes[kLdpPictureMaxNumPlanes] = {};
    for (uint8_t plane = 0; plane < numImagePlanes; ++plane) {
        const bool isEnhanced1 = frame->isEnhanced(LOQ1, plane);
        if (isEnhanced1 && frameConfig.loqEnabled[LOQ1]) {
            const uint32_t numTiles = globalConfig.numTiles[plane][LOQ1];
            if (numTiles > 1) {
                LdcTaskDependency* tiles =
                    static_cast<LdcTaskDependency*>(alloca(numTiles * sizeof(LdcTaskDependency)));

                // Generate and apply each tile's command buffer
                for (unsigned tile = 0; tile < numTiles; ++tile) {
                    LdpEnhancementTile* et{frame->getEnhancementTile(enhancementTileIdx++)};
                    assert(et->loq == LOQ1 && et->tile == tile);

                    LdcTaskDependency commands = addTaskGenerateCmdBuffer(frame, et);
                    tiles[tile] = addTaskApplyCmdBufferDirect(frame, et, baseUpsampled, commands,
                                                              intermediatePtr);
                }
                // Wait for all tiles to finish
                basePlanes[plane] = addTaskWaitForMany(frame, tiles, numTiles);
            } else {
                LdpEnhancementTile* et{frame->getEnhancementTile(enhancementTileIdx++)};
                assert(et->loq == LOQ1 && et->tile == 0);

                LdcTaskDependency commands = addTaskGenerateCmdBuffer(frame, et);
                basePlanes[plane] =
                    addTaskApplyCmdBufferDirect(frame, et, baseUpsampled, commands, intermediatePtr);
            }
        } else {
            basePlanes[plane] = baseUpsampled;
        }
    }

    // Upsample from combined intermediate picture to preliminary output picture
    LdcTaskDependency upsampledPicture{};
    if (globalConfig.scalingModes[LOQ0] != Scale0D) {
        upsampledPicture = addTaskUpsample(
            frame, LOQ1, addTaskWaitForMany(frame, basePlanes, numImagePlanes), intermediatePtr);
        intermediatePtr--;
    } else {
        upsampledPicture = basePicture;
    }

    //// LoQ 0
    //
    LdcTaskDependency reconstructedPlanes[kLdpPictureMaxNumPlanes] = {};

    for (uint8_t plane = 0; plane < numImagePlanes; ++plane) {
        const bool isEnhanced0 = frame->isEnhanced(LOQ0, plane);
        LdcTaskDependency recon{upsampledPicture};

        if (globalConfig.temporalEnabled && !frame->m_passthrough) {
            LdcTaskDependency temporal{};

            if (plane == 0) {
                if (frame->config.temporalRefresh && m_temporalPicture->buffer) {
                    const auto* temporalBuffer = static_cast<BufferVulkan*>(m_temporalPicture->buffer);
                    std::memset(temporalBuffer->ptr(), 0, temporalBuffer->size());
                }
            }

            if (isEnhanced0 && frameConfig.loqEnabled[LOQ0]) {
                // Enhancement residuals
                const uint32_t numPlaneTiles = globalConfig.numTiles[plane][LOQ0];
                if (numPlaneTiles > 1) {
                    LdcTaskDependency* tiles = static_cast<LdcTaskDependency*>(
                        alloca(numPlaneTiles * sizeof(LdcTaskDependency)));

                    // Generate and apply each tile's command buffer
                    for (unsigned tile = 0; tile < numPlaneTiles; ++tile) {
                        LdpEnhancementTile* et{frame->getEnhancementTile(enhancementTileIdx++)};
                        assert(et->loq == LOQ0 && et->tile == tile);
                        LdcTaskDependency commands{addTaskGenerateCmdBuffer(frame, et)};

                        tiles[tile] = addTaskApplyCmdBufferTemporal(frame, et, temporal, commands,
                                                                    intermediatePtr);
                    }
                    // Wait for all tiles to finish
                    temporal = addTaskWaitForMany(frame, tiles, numPlaneTiles);
                } else {
                    LdpEnhancementTile* et = frame->getEnhancementTile(enhancementTileIdx++);
                    assert(et->loq == LOQ0 && et->tile == 0);

                    LdcTaskDependency commands{addTaskGenerateCmdBuffer(frame, et)};

                    temporal =
                        addTaskApplyCmdBufferTemporal(frame, et, temporal, commands, intermediatePtr);
                }
            }

            if (frameConfig.loqEnabled[LOQ0] && plane == numImagePlanes - 1) {
                reconstructedPlanes[plane] =
                    addTaskApplyAddTemporal(frame, temporal, recon, intermediatePtr);
            }
        } else {
            if (isEnhanced0 && frameConfig.loqEnabled[LOQ0]) {
                // Enhancement residuals
                const uint32_t numPlaneTiles = globalConfig.numTiles[plane][LOQ0];
                if (numPlaneTiles > 1) {
                    LdcTaskDependency* tiles = static_cast<LdcTaskDependency*>(
                        alloca(numPlaneTiles * sizeof(LdcTaskDependency)));

                    // Generate and apply each tile's command buffer
                    for (unsigned tile = 0; tile < numPlaneTiles; ++tile) {
                        LdpEnhancementTile* et{frame->getEnhancementTile(enhancementTileIdx++)};
                        assert(et->loq == LOQ0 && et->tile == tile);
                        LdcTaskDependency commands{addTaskGenerateCmdBuffer(frame, et)};
                        tiles[tile] =
                            addTaskApplyCmdBufferDirect(frame, et, recon, commands, intermediatePtr);
                    }
                    // Wait for all tiles to finish
                    recon = addTaskWaitForMany(frame, tiles, numPlaneTiles);
                } else {
                    LdpEnhancementTile* et = frame->getEnhancementTile(enhancementTileIdx++);
                    assert(et->loq == LOQ0 && et->tile == 0);

                    LdcTaskDependency commands{addTaskGenerateCmdBuffer(frame, et)};

                    recon = addTaskApplyCmdBufferDirect(frame, et, recon, commands, intermediatePtr);
                }
            }

            reconstructedPlanes[plane] = recon;
        }
    }

    assert(enhancementTileIdx == frame->enhancementTileCount);

    LdcTaskDependency outputPicture{};
    outputPicture = addTaskConvertFromInternal(
        frame, globalConfig.baseDepth, globalConfig.enhancedDepth, frame->m_depOutputPicture,
        addTaskWaitForMany(frame, reconstructedPlanes, numImagePlanes), intermediatePtr);

    // Send output when all planes are ready
    addTaskOutputDone(frame, &outputPicture, 1);

    // Send base when all tasks that use it have completed
    LdcTaskDependency deps[kLdpPictureMaxNumPlanes] = {};
    uint32_t depsCount = 0;
    ldcTaskGroupFindOutputSetFromInput(&frame->m_taskGroup, frame->m_depBasePicture, deps,
                                       kLdpPictureMaxNumPlanes, &depsCount);
    addTaskBaseDone(frame, deps, depsCount);
}

// Fill out a task group for a simple unscaled passthrough configuration
//
void PipelineVulkan::generateTasksPassthrough(FrameVulkan* frame)
{
    VNTraceScoped();

    uint8_t numImagePlanes{kLdpPictureMaxNumPlanes};
    if (frame->basePicture) {
        VNLogDebugF("No base for passthrough: %" PRIx64, frame->timestamp);
        numImagePlanes = ldpPictureLayoutPlanes(&frame->basePicture->layout);
    }

    LdcTaskDependency outputPlanes[kLdpPictureMaxNumPlanes] = {};

    for (uint8_t plane = 0; plane < numImagePlanes; ++plane) {
        outputPlanes[plane] =
            addTaskPassthrough(frame, plane, frame->m_depOutputPicture, frame->m_depBasePicture);
    }

    // Send output and base when all planes are ready
    addTaskOutputDone(frame, outputPlanes, numImagePlanes);
    addTaskBaseDone(frame, outputPlanes, numImagePlanes);
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
void PipelineVulkan::logFrames() const
{
    char buffer[512];

    VNLogDebugF("Frames: %d", m_frames.size());
    for (uint32_t i = 0; i < m_frames.size(); ++i) {
        FrameVulkan* const frame{VNAllocationPtr(m_frames[i], FrameVulkan)};
        frame->longDescription(buffer, sizeof(buffer));
        VNLogDebugF("  %4d: %s", i, buffer);
    }

    VNLogDebugF("Reorder: %d", m_reorderIndex.size());
    for (uint32_t i = 0; i < m_reorderIndex.size(); ++i) {
        FrameVulkan* const frame{m_reorderIndex[i]};
        const LdcMemoryAllocation* const ptr =
            m_frames.findUnordered(ldcVectorCompareAllocationPtr, frame);
        const int32_t idx = static_cast<int32_t>(ptr ? (ptr - &m_frames[0]) : -1);
        VNLogDebugF("  %2d: %4d", i, idx);
    }

    VNLogDebugF("Processing: %d", m_processingIndex.size());
    for (uint32_t i = 0; i < m_processingIndex.size(); ++i) {
        FrameVulkan* const frame{m_processingIndex[i]};
        const LdcMemoryAllocation* const ptr =
            m_frames.findUnordered(ldcVectorCompareAllocationPtr, frame);
        const int32_t idx = static_cast<int32_t>(ptr ? (ptr - &m_frames[0]) : -1);
        VNLogDebugF("  %2d: %4d", i, idx);
    }

    VNLogDebugF("Bases In: %d (%d)", m_basePicturePending.size(), m_basePicturePending.reserved());
    VNLogDebugF("Bases Out: %d (%d)", m_basePictureOutBuffer.size(), m_basePictureOutBuffer.capacity());
    VNLogDebugF("Output: %d (%d)", m_outputPictureAvailableBuffer.size(),
                m_outputPictureAvailableBuffer.capacity());
}
#endif

} // namespace lcevc_dec::pipeline_vulkan
