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

#include "tasks_vulkan.h"

#include <LCEVC/common/constants.h>
#include <LCEVC/common/diagnostics.h>
#include <LCEVC/common/memory.h>
#include <LCEVC/enhancement/bitstream_types.h>
#include <LCEVC/enhancement/decode.h>
#include <LCEVC/pipeline_vulkan/types_vulkan.h>
#include <LCEVC/pixel_processing/apply_cmdbuffer.h>
#include <LCEVC/pixel_processing/blit.h>
#include <LCEVC/pixel_processing/upscale.h>

namespace lcevc_dec::pipeline_vulkan {

// All tasks fns. are static - only exported function is generateTasks()
//
namespace {

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

    void* taskConvertToInternal(LdcTask* task, const LdcTaskPart* /*part*/)
    {
        assert(task->dataSize == sizeof(TaskConvertToInternalData));
        const TaskConvertToInternalData& data{VNTaskData(task, TaskConvertToInternalData)};
        PipelineVulkan* const pipeline{data.pipeline};
        FrameVulkan* const frame{data.frame};

        if (pipeline->isSkipped(frame)) {
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
                            externalIndex +=
                                externalOffsetU * srcPicture->layout.rowStrides[planeIndex - 1];
                        }
                        if (planeIndex > 1) {
                            externalIndex +=
                                externalOffsetV * srcPicture->layout.rowStrides[planeIndex - 2];
                        }
                        std::memcpy(managedBuffer->ptr() + internalIndex,
                                    exDesc.data + externalIndex, width);
                    }
                };

                removePadding(byteWidth, srcDesc.height, byteWidth, 0, 0, 0, 0); // Y

                if (pipeline->getChroma() != LdeChroma::CTMonochrome) {
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

        srcDesc.colorFormat = pipeline->chromaToColorFormat(pipeline->getChroma());
        auto* dstPicture = static_cast<PictureVulkan*>(pipeline->allocPicture(srcDesc));

        VulkanConversionArgs args{};
        args.src = srcPicture;
        args.dst = dstPicture;
        args.toInternal = true;
        args.bitDepth = frame->baseBitdepth;
        args.chroma = pipeline->getChroma();

        if (!pipeline->getCore().conversion(&args)) {
            VNLogError("Conversion to internal failed");
        }

        frame->m_intermediatePicture[LOQ2] = dstPicture;

        return nullptr;
    }

    LdcTaskDependency addTaskConvertToInternal(PipelineVulkan* pipeline, FrameVulkan* frame,
                                               uint32_t baseDepth, uint32_t enhancementDepth,
                                               LdcTaskDependency inputDep)
    {
        const TaskConvertToInternalData data{pipeline, frame, baseDepth, enhancementDepth};
        const LdcTaskDependency inputs[] = {inputDep};
        return frame->taskAdd(inputs, VNArraySize(inputs), taskConvertToInternal, &data,
                              sizeof(data), "ConvertToInternal");
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

    void* taskConvertFromInternal(LdcTask* task, const LdcTaskPart* /*part*/)
    {
        assert(task->dataSize == sizeof(TaskConvertFromInternalData));
        const TaskConvertFromInternalData& data{VNTaskData(task, TaskConvertFromInternalData)};
        PipelineVulkan* const pipeline{data.pipeline};
        FrameVulkan* const frame{data.frame};
        const uint8_t intermediatePtr{data.intermediatePtr};

        if (pipeline->isSkipped(frame)) {
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
        args.chroma = pipeline->getChroma();

        if (!pipeline->getCore().conversion(&args)) {
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

    LdcTaskDependency addTaskConvertFromInternal(PipelineVulkan* pipeline, FrameVulkan* frame,
                                                 uint32_t baseDepth, uint32_t enhancementDepth,
                                                 LdcTaskDependency dst, LdcTaskDependency src,
                                                 uint8_t intermediatePtr)
    {
        const TaskConvertFromInternalData data{pipeline, frame, baseDepth, enhancementDepth, intermediatePtr};
        const LdcTaskDependency inputs[] = {dst, src};
        return frame->taskAdd(inputs, VNArraySize(inputs), taskConvertFromInternal, &data,
                              sizeof(data), "ConvertFromInternal");
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
    };

    void* taskUpsample(LdcTask* task, const LdcTaskPart* /*part*/)
    {
        assert(task->dataSize == sizeof(TaskUpsampleData));
        const TaskUpsampleData& data{VNTaskData(task, TaskUpsampleData)};

        PipelineVulkan* const pipeline{data.pipeline};
        FrameVulkan* const frame{data.frame};
        const uint8_t intermediatePtr{data.intermediatePtr};
        const LdeLOQIndex loq{data.loq};

        if (pipeline->isSkipped(frame)) {
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
        upscaleArgs.chroma = pipeline->getChroma();

        assert(upscaleArgs.mode != Scale0D);
        VNLogDebug("taskUpsample timestamp:%" PRIx64 " loq:%d", frame->timestamp, (uint32_t)data.loq);

        if (!pipeline->getCore().upscaleFrame(&frame->globalConfig->kernel, &upscaleArgs)) {
            VNLogError("Upsample failed");
        }

        return nullptr;
    }

    LdcTaskDependency addTaskUpscale(PipelineVulkan* pipeline, FrameVulkan* frame, LdeLOQIndex fromLoq,
                                     LdcTaskDependency basePicture, uint8_t intermediatePtr)
    {
        assert(fromLoq > LOQ0);

        const TaskUpsampleData data{pipeline, frame, fromLoq, intermediatePtr};
        const LdcTaskDependency inputs[] = {basePicture};
        return frame->taskAdd(inputs, VNArraySize(inputs), taskUpsample, &data, sizeof(data), "Upscale");
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

    void* taskGenerateCmdBuffer(LdcTask* task, const LdcTaskPart* /*part*/)
    {
        assert(task->dataSize == sizeof(TaskGenerateCmdBufferData));
        const TaskGenerateCmdBufferData& data{VNTaskData(task, TaskGenerateCmdBufferData)};
        FrameVulkan* const frame{data.frame};

        VNLogDebug("taskGenerateCmdBuffer timestamp:%" PRIx64 " tile:%d loq:%d plane:%d",
                   data.frame->timestamp, data.enhancementTile->tile,
                   (uint32_t)data.enhancementTile->loq, data.enhancementTile->plane);

        if (!ldeDecodeEnhancement(frame->globalConfig, &frame->config, data.enhancementTile->loq,
                                  data.enhancementTile->plane, data.enhancementTile->tile, nullptr,
                                  &data.enhancementTile->bufferGpu,
                                  &data.enhancementTile->bufferGpuBuilder)) {
            VNLogError("ldeDecodeEnhancement failed");
        }

        return nullptr;
    }

    LdcTaskDependency addTaskGenerateCmdBuffer(PipelineVulkan* pipeline, FrameVulkan* frame,
                                               LdpEnhancementTile* enhancementTile)
    {
        const TaskGenerateCmdBufferData data{pipeline, frame, enhancementTile};
        return frame->taskAdd(nullptr, 0, taskGenerateCmdBuffer, &data, sizeof(data), "GenerateCmdBuffer");
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

    void* taskApplyCmdBufferDirect(LdcTask* task, const LdcTaskPart* part)
    {
        VNTraceScoped();
        VNUnused(part);

        assert(task->dataSize == sizeof(TaskApplyCmdBufferDirectData));
        const TaskApplyCmdBufferDirectData& data{VNTaskData(task, TaskApplyCmdBufferDirectData)};
        PipelineVulkan* const pipeline{data.pipeline};
        FrameVulkan* const frame{data.frame};
        const uint8_t intermediatePtr{data.intermediatePtr};

        if (pipeline->isSkipped(frame)) {
            return nullptr;
        }

        VNLogDebug("taskApplyCmdBufferDirect timestamp:%" PRIx64 " loq:%d plane:%d", data.frame->timestamp,
                   (uint32_t)data.enhancementTile->loq, data.enhancementTile->plane);

        auto* picture = frame->m_intermediatePicture[intermediatePtr];

        VulkanApplyArgs args{};
        pipeline->prepareApplyArgs(args, picture, data.enhancementTile, frame, true);

        if (!pipeline->getCore().apply(&args)) {
            VNLogError("Vulkan apply direct failed");
        }

        return nullptr;
    }

    LdcTaskDependency addTaskApplyCmdBufferDirect(PipelineVulkan* pipeline, FrameVulkan* frame,
                                                  LdpEnhancementTile* enhancementTile,
                                                  LdcTaskDependency imageBuffer,
                                                  LdcTaskDependency cmdBuffer, uint8_t intermediatePtr)
    {
        const TaskApplyCmdBufferDirectData data{pipeline, frame, enhancementTile, intermediatePtr};

        const LdcTaskDependency inputs[] = {imageBuffer, cmdBuffer};
        return frame->taskAdd(inputs, VNArraySize(inputs), taskApplyCmdBufferDirect, &data,
                              sizeof(data), "ApplyCmdBufferDirect");
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

    void* taskApplyCmdBufferTemporal(LdcTask* task, const LdcTaskPart* part)
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

        if (!pipeline->getCore().apply(&args)) {
            VNLogError("Vulkan apply temporal failed");
        }

        return nullptr;
    }

    LdcTaskDependency addTaskApplyCmdBufferTemporal(PipelineVulkan* pipeline, FrameVulkan* frame,
                                                    LdpEnhancementTile* enhancementTile,
                                                    LdcTaskDependency temporalBuffer,
                                                    LdcTaskDependency cmdBuffer, uint8_t intermediatePtr)
    {
        const TaskApplyCmdBufferTemporalData data{pipeline, frame, enhancementTile, intermediatePtr};
        const LdcTaskDependency inputs[] = {temporalBuffer, cmdBuffer};
        return frame->taskAdd(inputs, VNArraySize(inputs), taskApplyCmdBufferTemporal, &data,
                              sizeof(data), "ApplyCmdBufferTemporal");
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

    void* taskApplyAddTemporal(LdcTask* task, const LdcTaskPart* part)
    {
        VNTraceScoped();
        VNUnused(part);

        assert(task->dataSize == sizeof(TaskApplyAddTemporalData));
        const TaskApplyAddTemporalData& data{VNTaskData(task, TaskApplyAddTemporalData)};
        PipelineVulkan* const pipeline{data.pipeline};
        FrameVulkan* const frame{data.frame};
        const uint8_t intermediatePtr{data.intermediatePtr};

        if (pipeline->isFlushed(frame) || frame->isPassthrough()) {
            // Just move temporal buffer along pipline
            pipeline->transferTemporalBuffer(frame, 0); // TODO - check this
            return nullptr;
        }

        VNLogDebug("taskApplyAddTemporal timestamp:%" PRIx64 "", data.frame->timestamp);

        VulkanBlitArgs args{};
        args.src = pipeline->m_temporalPicture.get();
        args.dst = frame->m_intermediatePicture[intermediatePtr];
        args.numEnhancedPlanes = frame->numEnhancedPlanes();
        args.chroma = pipeline->getChroma();

        if (!pipeline->getCore().blit(&args)) {
            VNLogError("Vulkan blit failed");
        }

        return nullptr;
    }

    LdcTaskDependency addTaskApplyAddTemporal(PipelineVulkan* pipeline, FrameVulkan* frame,
                                              LdcTaskDependency temporalDep,
                                              LdcTaskDependency sourceDep, uint8_t intermediatePtr)
    {
        const TaskApplyAddTemporalData data{pipeline, frame, intermediatePtr};
        const LdcTaskDependency inputs[] = {temporalDep, sourceDep};
        return frame->taskAdd(inputs, VNArraySize(inputs), taskApplyAddTemporal, &data,
                              sizeof(data), "ApplyAddTemporal");
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

    void* taskPassthrough(LdcTask* task, const LdcTaskPart* /*part*/)
    {
        VNTraceScoped();
        assert(task->dataSize == sizeof(TaskPassthroughData));

        const TaskPassthroughData& data{VNTaskData(task, TaskPassthroughData)};
        PipelineVulkan* const pipeline{data.pipeline};
        const FrameVulkan* const frame{data.frame};

        if (pipeline->isSkipped(frame)) {
            return nullptr;
        }

        LdpPicturePlaneDesc srcPlane;
        frame->getBasePlaneDesc(data.planeIndex, srcPlane);

        LdpPicturePlaneDesc dstPlane;
        frame->getOutputPlaneDesc(data.planeIndex, dstPlane);

        VNLogDebug("taskPassthrough timestamp:%" PRIx64 " plane:%d", data.frame->timestamp, data.planeIndex);

        if (!ldppPlaneBlit(pipeline->taskPool(), task, pipeline->configuration().forceScalar,
                           data.planeIndex, &frame->basePicture->layout,
                           &frame->outputPicture->layout, &srcPlane, &dstPlane, BMCopy)) {
            VNLogError("ldppPlaneBlit In failed");
        }

        return nullptr;
    }

    LdcTaskDependency addTaskPassthrough(PipelineVulkan* pipeline, FrameVulkan* frame, uint32_t planeIndex,
                                         LdcTaskDependency dest, LdcTaskDependency src)
    {
        const TaskPassthroughData data{pipeline, frame, planeIndex};
        const LdcTaskDependency inputs[] = {dest, src};

        return frame->taskAdd(inputs, VNArraySize(inputs), taskPassthrough, &data, sizeof(data),
                              "Passthrough");
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

    void* taskWaitForMany(LdcTask* task, const LdcTaskPart* /*part*/)
    {
        VNTraceScoped();
        assert(task->dataSize == sizeof(TaskWaitForManyData));

        VNLogDebug("taskWaitForMany ts:%" PRIx64 "", VNTaskData(task, TaskWaitForManyData).frame->timestamp);
        return nullptr;
    }

    LdcTaskDependency addTaskWaitForMany(PipelineVulkan* pipeline, FrameVulkan* frame,
                                         const LdcTaskDependency* inputs, uint32_t inputsCount)
    {
        const TaskWaitForManyData data{pipeline, frame};

        return frame->taskAdd(inputs, inputsCount, taskWaitForMany, &data, sizeof(data), "WaitForMany");
    }

    //// BaseDone
    //
    // Wait for base picture planes to be used, then send base picture back to client
    //
    struct TaskBaseDoneData
    {
        PipelineVulkan* pipeline;
        FrameVulkan* frame;
    };

    void* taskBaseDone(LdcTask* task, const LdcTaskPart* /*part*/)
    {
        VNTraceScoped();
        assert(task->dataSize == sizeof(TaskBaseDoneData));

        const TaskBaseDoneData& data{VNTaskData(task, TaskBaseDoneData)};
        const FrameVulkan* const frame{data.frame};

        VNLogDebug("taskBaseDone ts:%" PRIx64, data.frame->timestamp);

        if (frame->basePicture == nullptr) {
            return nullptr;
        }

        assert(data.frame->basePicture);

        // Generate event and return picture
        data.pipeline->baseDone(data.frame->basePicture);

        // Frame no longer has access to base picture
        data.frame->basePicture = nullptr;
        return nullptr;
    }

    void addTaskBaseDone(PipelineVulkan* pipeline, FrameVulkan* frame,
                         const LdcTaskDependency* inputs, uint32_t inputsCount)
    {
        const TaskBaseDoneData data{pipeline, frame};

        frame->taskAddSink(inputs, inputsCount, taskBaseDone, &data, sizeof(data), "BaseDone");
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

    void* taskOutputDone(LdcTask* task, const LdcTaskPart* /*part*/)
    {
        VNTraceScoped();
        assert(task->dataSize == sizeof(TaskOutputDoneData));

        const TaskOutputDoneData& data{VNTaskData(task, TaskOutputDoneData)};
        PipelineVulkan* const pipeline{data.pipeline};
        FrameVulkan* const frame{data.frame};

        VNLogDebug("taskOutputDone ts:%" PRIx64, frame->timestamp);

        // Build the decode info for the frame
        frame->decodeInformation.timestamp = frame->timestamp;
        frame->decodeInformation.hasBase = true;
        frame->decodeInformation.hasEnhancement =
            frame->config.loqEnabled[LOQ1] || frame->config.loqEnabled[LOQ0];
        frame->decodeInformation.skipped = pipeline->isSkipped(frame);
        frame->decodeInformation.enhanced =
            frame->config.loqEnabled[LOQ1] || frame->config.loqEnabled[LOQ0];
        frame->decodeInformation.baseWidth = frame->baseWidth;
        frame->decodeInformation.baseHeight = frame->baseHeight;
        frame->decodeInformation.baseBitdepth = frame->baseBitdepth;
        frame->decodeInformation.userData = frame->userData;

        // Mark frame as done
        pipeline->outputDone(frame);

        return nullptr;
    }

    void addTaskOutputDone(PipelineVulkan* pipeline, FrameVulkan* frame,
                           const LdcTaskDependency* inputs, uint32_t inputsCount)
    {
        const TaskOutputDoneData data{pipeline, frame};

        frame->taskAddSink(inputs, inputsCount, taskOutputDone, &data, sizeof(data), "OutputDone");
    }

    // Fill out a task group given a frame configuration
    //
    void generateTasksEnhancement(PipelineVulkan* pipeline, FrameVulkan* frame, uint64_t previousTimestamp)
    {
        VNTraceScoped();

        // Convenience values for readability
        const LdeFrameConfig& frameConfig{frame->config};
        const LdeGlobalConfig& globalConfig{*frame->globalConfig};
        const uint8_t numImagePlanes{frame->numImagePlanes()};

        auto intermediatePtr = static_cast<uint8_t>(LOQ2);

        pipeline->setChroma(frame->globalConfig->chroma);

        uint32_t enhancementTileIdx = 0;

        if (frame->config.sharpenType != STDisabled && frame->config.sharpenStrength != 0.0f) {
            VNLogWarning("S-Filter is configured in stream, but not supported by decoder.");
        }

        //// Input conversion
        LdcTaskDependency basePicture{kTaskDependencyInvalid};
        basePicture = addTaskConvertToInternal(pipeline, frame, frame->getEnhancementBitDepth(),
                                               globalConfig.baseDepth, frame->depBasePicture());

        //// LoQ 1

        //// Base + Residuals
        //
        // First upsample
        LdcTaskDependency baseUpsampled{kTaskDependencyInvalid};
        if (globalConfig.scalingModes[LOQ1] != Scale0D) {
            baseUpsampled = addTaskUpscale(pipeline, frame, LOQ2, basePicture, intermediatePtr);
            intermediatePtr--;
        } else {
            baseUpsampled = basePicture;
        }

        // Enhancement LOQ1 decoding
        LdcTaskDependency basePlanes[kLdpPictureMaxNumPlanes] = {};
        for (uint8_t plane = 0; plane < numImagePlanes; ++plane) {
            const bool isEnhanced1 = frame->isPlaneEnhanced(LOQ1, plane);
            if (isEnhanced1 && frameConfig.loqEnabled[LOQ1]) {
                const uint32_t numTiles = globalConfig.numTiles[plane][LOQ1];
                if (numTiles > 1) {
                    LdcTaskDependency* tiles =
                        static_cast<LdcTaskDependency*>(alloca(numTiles * sizeof(LdcTaskDependency)));

                    // Generate and apply each tile's command buffer
                    for (unsigned tile = 0; tile < numTiles; ++tile) {
                        LdpEnhancementTile* et{frame->getEnhancementTile(enhancementTileIdx++)};
                        assert(et->loq == LOQ1 && et->tile == tile);

                        LdcTaskDependency commands = addTaskGenerateCmdBuffer(pipeline, frame, et);
                        tiles[tile] = addTaskApplyCmdBufferDirect(pipeline, frame, et, baseUpsampled,
                                                                  commands, intermediatePtr);
                    }
                    // Wait for all tiles to finish
                    basePlanes[plane] = addTaskWaitForMany(pipeline, frame, tiles, numTiles);
                } else {
                    LdpEnhancementTile* et{frame->getEnhancementTile(enhancementTileIdx++)};
                    assert(et->loq == LOQ1 && et->tile == 0);

                    LdcTaskDependency commands = addTaskGenerateCmdBuffer(pipeline, frame, et);
                    basePlanes[plane] = addTaskApplyCmdBufferDirect(pipeline, frame, et, baseUpsampled,
                                                                    commands, intermediatePtr);
                }
            } else {
                basePlanes[plane] = baseUpsampled;
            }
        }

        // Upsample from combined intermediate picture to preliminary output picture
        LdcTaskDependency upsampledPicture{};
        if (globalConfig.scalingModes[LOQ0] != Scale0D) {
            upsampledPicture = addTaskUpscale(
                pipeline, frame, LOQ1,
                addTaskWaitForMany(pipeline, frame, basePlanes, numImagePlanes), intermediatePtr);
            intermediatePtr--;
        } else {
            upsampledPicture = basePicture;
        }

        //// LoQ 0
        //
        LdcTaskDependency reconstructedPlanes[kLdpPictureMaxNumPlanes] = {};

        for (uint8_t plane = 0; plane < numImagePlanes; ++plane) {
            const bool isEnhanced0 = frame->isPlaneEnhanced(LOQ0, plane);
            LdcTaskDependency recon{upsampledPicture};

            if (globalConfig.temporalEnabled && !frame->isPassthrough()) {
                LdcTaskDependency temporal{};

                if (plane == 0) {
                    if (frame->config.temporalRefresh && pipeline->m_temporalPicture->buffer) {
                        const auto* temporalBuffer =
                            static_cast<BufferVulkan*>(pipeline->m_temporalPicture->buffer);
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
                            LdcTaskDependency commands{addTaskGenerateCmdBuffer(pipeline, frame, et)};

                            tiles[tile] = addTaskApplyCmdBufferTemporal(pipeline, frame, et, temporal,
                                                                        commands, intermediatePtr);
                        }
                        // Wait for all tiles to finish
                        temporal = addTaskWaitForMany(pipeline, frame, tiles, numPlaneTiles);
                    } else {
                        LdpEnhancementTile* et = frame->getEnhancementTile(enhancementTileIdx++);
                        assert(et->loq == LOQ0 && et->tile == 0);

                        LdcTaskDependency commands{addTaskGenerateCmdBuffer(pipeline, frame, et)};

                        temporal = addTaskApplyCmdBufferTemporal(pipeline, frame, et, temporal,
                                                                 commands, intermediatePtr);
                    }
                }

                if (frameConfig.loqEnabled[LOQ0] && plane == numImagePlanes - 1) {
                    reconstructedPlanes[plane] =
                        addTaskApplyAddTemporal(pipeline, frame, temporal, recon, intermediatePtr);
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
                            LdcTaskDependency commands{addTaskGenerateCmdBuffer(pipeline, frame, et)};
                            tiles[tile] = addTaskApplyCmdBufferDirect(pipeline, frame, et, recon,
                                                                      commands, intermediatePtr);
                        }
                        // Wait for all tiles to finish
                        recon = addTaskWaitForMany(pipeline, frame, tiles, numPlaneTiles);
                    } else {
                        LdpEnhancementTile* et = frame->getEnhancementTile(enhancementTileIdx++);
                        assert(et->loq == LOQ0 && et->tile == 0);

                        LdcTaskDependency commands{addTaskGenerateCmdBuffer(pipeline, frame, et)};

                        recon = addTaskApplyCmdBufferDirect(pipeline, frame, et, recon, commands,
                                                            intermediatePtr);
                    }
                }

                reconstructedPlanes[plane] = recon;
            }
        }

        assert(enhancementTileIdx == frame->enhancementTileCount);

        LdcTaskDependency outputPicture{};
        outputPicture = addTaskConvertFromInternal(
            pipeline, frame, globalConfig.baseDepth, globalConfig.enhancedDepth,
            frame->depOutputPicture(),
            addTaskWaitForMany(pipeline, frame, reconstructedPlanes, numImagePlanes), intermediatePtr);

        // Send output when all planes are ready
        addTaskOutputDone(pipeline, frame, &outputPicture, 1);

        // Send base when all tasks that use it have completed
        LdcTaskDependency deps[kLdpPictureMaxNumPlanes] = {};
        uint32_t depsCount = 0;
        ldcTaskGroupFindOutputSetFromInput(frame->taskGroup(), frame->depBasePicture(), deps,
                                           kLdpPictureMaxNumPlanes, &depsCount);
        addTaskBaseDone(pipeline, frame, deps, depsCount);
    }

    // Fill out a task group for a simple unscaled passthrough configuration
    //
    void generateTasksPassthrough(PipelineVulkan* pipeline, FrameVulkan* frame)
    {
        VNTraceScoped();

        uint8_t numImagePlanes{kLdpPictureMaxNumPlanes};
        if (frame->basePicture) {
            VNLogDebugF("No base for passthrough: ts:%" PRIx64, frame->timestamp);
            numImagePlanes = ldpPictureLayoutPlanes(&frame->basePicture->layout);
        }

        LdcTaskDependency outputPlanes[kLdpPictureMaxNumPlanes] = {};

        for (uint8_t plane = 0; plane < numImagePlanes; ++plane) {
            outputPlanes[plane] = addTaskPassthrough(pipeline, frame, plane, frame->depOutputPicture(),
                                                     frame->depBasePicture());
        }

        // Send output and base when all planes are ready
        addTaskOutputDone(pipeline, frame, outputPlanes, numImagePlanes);
        addTaskBaseDone(pipeline, frame, outputPlanes, numImagePlanes);
    }

} // anonymous namespace

// Generate task graph for a frame
//
void generateTasks(PipelineVulkan* pipeline, FrameVulkan* frame, uint64_t previousTimestamp)
{
    // Fill out tasks for this frame
    if (pipeline->configuration().showTasks) {
        // Don't consume tasks whilst group is generated
        ldcTaskGroupBlock(frame->taskGroup());
    }

    // Choose pass-through or enhancement task graph generation.
    //
    // If the pass through is 'Scaled', then use the enhancement graph, which
    // will just end up doing scaling as there is no enhancement data.
    if (frame->isPassthrough() && (pipeline->configuration().passthroughMode != PassthroughMode::Scale ||
                                   !frame->hasGoodConfig())) {
        generateTasksPassthrough(pipeline, frame);
    } else if (pipeline->isFlushed(frame)) {
        generateTasksPassthrough(pipeline, frame);
    } else {
        generateTasksEnhancement(pipeline, frame, previousTimestamp);
    }

    if (pipeline->configuration().showTasks) {
#ifdef VN_SDK_LOG_ENABLE_DEBUG
        ldcTaskPoolDump(pipeline->taskPool(), frame->taskGroup());
#endif
        ldcTaskGroupUnblock(frame->taskGroup());
    }
}

} // namespace lcevc_dec::pipeline_vulkan