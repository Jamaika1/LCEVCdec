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

#include "tasks_cpu.h"

#include "frame_cpu.h"
#include "pipeline_config_cpu.h"
#include "pipeline_cpu.h"

#include <LCEVC/common/constants.h>
#include <LCEVC/common/diagnostics.h>
#include <LCEVC/common/memory.h>
#include <LCEVC/enhancement/bitstream_types.h>
#include <LCEVC/enhancement/decode.h>
#include <LCEVC/pixel_processing/apply_cmdbuffer.h>
#include <LCEVC/pixel_processing/blit.h>
#include <LCEVC/pixel_processing/upscale.h>

namespace lcevc_dec::pipeline_cpu {

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
        PipelineCPU* pipeline;
        FrameCPU* frame;
        uint32_t planeIndex;
        uint32_t baseDepth;
        uint32_t enhancementDepth;
    };

    void* taskConvertToInternal(LdcTask* task, const LdcTaskPart* /*part*/)
    {
        VNTraceScoped();
        assert(task->dataSize == sizeof(TaskConvertToInternalData));

        const TaskConvertToInternalData& data{VNTaskData(task, TaskConvertToInternalData)};
        PipelineCPU* const pipeline{data.pipeline};
        const FrameCPU* const frame{data.frame};

        if (pipeline->isSkipped(frame)) {
            return nullptr;
        }

        bool isNV12 = frame->basePicture->layout.layoutInfo->format == LdpColorFormatNV12_8;
        uint32_t srcPlaneIndex = (isNV12 && data.planeIndex == 2) ? 1 : data.planeIndex;
        LdpPicturePlaneDesc srcPlane;
        frame->getBasePlaneDesc(srcPlaneIndex, srcPlane);

        // Intermediate buffers are set up so that unused ones point to higher LoQs - so requesting
        // LOQ2 will pick up the correct 'input' buffer
        LdpPicturePlaneDesc dstPlane;
        frame->getIntermediatePlaneDesc(data.planeIndex, LOQ2, dstPlane);

        VNLogDebug("taskConvertToInternal ts:%" PRIx64 " plane:%d enhanced:%d",
                   data.frame->timestamp, data.planeIndex);

        if (!ldppPlaneBlit(pipeline->taskPool(), task, pipeline->configuration().forceScalar,
                           data.planeIndex, &frame->basePicture->layout,
                           frame->getIntermediateLayout(LOQ2), &srcPlane, &dstPlane, BMCopy)) {
            VNLogError("ldppPlaneBlit In failed");
        }
        return nullptr;
    }

    LdcTaskDependency addTaskConvertToInternal(PipelineCPU* pipeline, FrameCPU* frame,
                                               uint32_t planeIndex, uint32_t baseDepth,
                                               uint32_t enhancementDepth, LdcTaskDependency inputDep)
    {
        const TaskConvertToInternalData data{pipeline, frame, planeIndex, baseDepth, enhancementDepth};
        const LdcTaskDependency inputs[] = {inputDep};
        return frame->taskAdd(inputs, VNArraySize(inputs), taskConvertToInternal, &data,
                              sizeof(data), "ConvertToInternal");
    }

    //// ConvertFromInternal
    //
    // Convert a picture plane from internal fixed point to output picture pixel format.
    //
    struct TaskConvertFromInternalData
    {
        PipelineCPU* pipeline;
        FrameCPU* frame;
        uint32_t planeIndex;
        uint32_t baseDepth;
        uint32_t enhancementDepth;
    };

    void* taskConvertFromInternal(LdcTask* task, const LdcTaskPart* /*part*/)
    {
        VNTraceScoped();
        assert(task->dataSize == sizeof(TaskConvertFromInternalData));

        const TaskConvertFromInternalData& data{VNTaskData(task, TaskConvertFromInternalData)};
        PipelineCPU* const pipeline{data.pipeline};
        const FrameCPU* const frame{data.frame};

        if (pipeline->isSkipped(frame)) {
            return nullptr;
        }

        LdpPicturePlaneDesc srcPlane;
        frame->getIntermediatePlaneDesc(data.planeIndex, LOQ0, srcPlane);

        bool isNV12 = frame->outputPicture->layout.layoutInfo->format == LdpColorFormatNV12_8;
        uint32_t dstPlaneIndex = (isNV12 && data.planeIndex == 2) ? 1 : data.planeIndex;
        LdpPicturePlaneDesc dstPlane;
        frame->getOutputPlaneDesc(dstPlaneIndex, dstPlane);

        VNLogDebug("taskConvertFromInternal ts:%" PRIx64 " plane:%d", data.frame->timestamp, data.planeIndex);

        if (!ldppPlaneBlit(pipeline->taskPool(), task, pipeline->configuration().forceScalar,
                           data.planeIndex, frame->getIntermediateLayout(LOQ0),
                           &frame->outputPicture->layout, &srcPlane, &dstPlane, BMCopy)) {
            VNLogError("ldppPlaneBlit out failed");
        }

        return nullptr;
    }

    LdcTaskDependency addTaskConvertFromInternal(PipelineCPU* pipeline, FrameCPU* frame, uint32_t planeIndex,
                                                 uint32_t baseDepth, uint32_t enhancementDepth,
                                                 LdcTaskDependency dst, LdcTaskDependency src)
    {
        const TaskConvertFromInternalData data{pipeline, frame, planeIndex, baseDepth, enhancementDepth};
        const LdcTaskDependency inputs[] = {dst, src};
        return frame->taskAdd(inputs, VNArraySize(inputs), taskConvertFromInternal, &data,
                              sizeof(data), "ConvertFromInternal");
    }

    //// Upscale
    //
    // Upscale (1D or 2D) between 16-bit intermediate buffers.
    //
    // Inputs and outputs may be fixed point or 'external' format if no residuals are being applied.
    //
    struct TaskUpscaleData
    {
        PipelineCPU* pipeline;
        FrameCPU* frame;
        LdeLOQIndex fromLoq;
        LdeScalingMode scalingMode;
        uint32_t plane;
    };

    void* taskUpscale(LdcTask* task, const LdcTaskPart* /*part*/)
    {
        VNTraceScoped();
        assert(task->dataSize == sizeof(TaskUpscaleData));

        const TaskUpscaleData& data{VNTaskData(task, TaskUpscaleData)};
        PipelineCPU* const pipeline{data.pipeline};
        const FrameCPU* const frame{data.frame};

        if (pipeline->isSkipped(frame)) {
            return nullptr;
        }

        LdppUpscaleArgs upscaleArgs{};

        const LdeLOQIndex fromLoq = data.fromLoq;
        const LdeLOQIndex toLoq = static_cast<LdeLOQIndex>(fromLoq - 1);
        upscaleArgs.srcLayout = frame->getIntermediateLayout(fromLoq);
        frame->getIntermediatePlaneDesc(data.plane, fromLoq, upscaleArgs.srcPlane);
        upscaleArgs.intermediateLayout = frame->getUpscaleLayout(toLoq);
        frame->getUpscalePlaneDesc(data.plane, toLoq, upscaleArgs.intermediatePlane);
        upscaleArgs.dstLayout = frame->getIntermediateLayout(toLoq);
        frame->getIntermediatePlaneDesc(data.plane, toLoq, upscaleArgs.dstPlane);

        upscaleArgs.planeIndex = data.plane;
        upscaleArgs.applyPA = frame->globalConfig->predictedAverageEnabled;
        upscaleArgs.frameDither = frame->dither();
        upscaleArgs.mode = data.scalingMode;
        upscaleArgs.forceScalar = pipeline->configuration().forceScalar;

        VNLogDebug("taskUpscale timestamp:%" PRIx64 " loq:%d plane:%d", frame->timestamp,
                   (uint32_t)data.fromLoq, data.plane);

        if (!ldppUpscale(pipeline->taskPool(), task, &frame->globalConfig->kernel, &upscaleArgs)) {
            VNLogError("Upscale failed");
        }

        return nullptr;
    }

    LdcTaskDependency addTaskUpscale(PipelineCPU* pipeline, FrameCPU* frame, LdeLOQIndex fromLoq,
                                     uint32_t plane, LdcTaskDependency basePicture)
    {
        assert(fromLoq > LOQ0);
        const LdeScalingMode scalingMode = frame->globalConfig->scalingModes[fromLoq - 1];
        assert(scalingMode != Scale0D);

        const TaskUpscaleData data{pipeline, frame, fromLoq, scalingMode, plane};
        const LdcTaskDependency inputs[] = {basePicture};
        return frame->taskAdd(inputs, VNArraySize(inputs), taskUpscale, &data, sizeof(data), "Upscale");
    }

    //// Upscale Direct
    //
    // Upscale (1D or 2D) between directly from source picture to output in native format. Only for
    // un-enhanced planes without residuals when there's a single upscale.
    void* taskUpscaleDirect(LdcTask* task, const LdcTaskPart* /*part*/)
    {
        VNTraceScoped();
        assert(task->dataSize == sizeof(TaskUpscaleData));

        const TaskUpscaleData& data{VNTaskData(task, TaskUpscaleData)};
        PipelineCPU* const pipeline{data.pipeline};
        const FrameCPU* const frame{data.frame};

        // Exit early if the frame is skipped or for NV12 plane 2 (NV12 chroma completed as one op)
        if (pipeline->isSkipped(frame) ||
            data.plane >= ldpPictureLayoutPlanes(&frame->basePicture->layout)) {
            return nullptr;
        }

        LdppUpscaleArgs upscaleArgs{};

        const LdeLOQIndex toLoq = static_cast<LdeLOQIndex>(data.fromLoq - 1);
        upscaleArgs.srcLayout = &frame->basePicture->layout;
        frame->getBasePlaneDesc(data.plane, upscaleArgs.srcPlane);
        if (data.scalingMode == Scale2D) {
            upscaleArgs.intermediateLayout = frame->getUpscaleLayout(toLoq);
            frame->getUpscalePlaneDesc(data.plane, toLoq, upscaleArgs.intermediatePlane);
        }
        upscaleArgs.dstLayout = &frame->outputPicture->layout;
        frame->getOutputPlaneDesc(data.plane, upscaleArgs.dstPlane);

        upscaleArgs.planeIndex = data.plane;
        upscaleArgs.applyPA = frame->globalConfig->predictedAverageEnabled;
        upscaleArgs.frameDither = frame->dither();
        upscaleArgs.mode = data.scalingMode;
        upscaleArgs.forceScalar = pipeline->configuration().forceScalar;

        VNLogDebug("taskUpscaleDirect timestamp:%" PRIx64 " loq:%d plane:%d", frame->timestamp,
                   (uint32_t)data.fromLoq, data.plane);

        if (!ldppUpscale(pipeline->taskPool(), task, &frame->globalConfig->kernel, &upscaleArgs)) {
            VNLogError("UpscaleDirect failed");
        }

        return nullptr;
    }

    LdcTaskDependency addTaskUpscaleDirect(PipelineCPU* pipeline, FrameCPU* frame, LdeLOQIndex fromLoq,
                                           uint32_t plane, LdcTaskDependency basePicture,
                                           LdcTaskDependency outputPicture)
    {
        assert(fromLoq > LOQ0);
        const LdeScalingMode scalingMode = frame->globalConfig->scalingModes[fromLoq - 1];
        assert(scalingMode != Scale0D);

        const TaskUpscaleData data{pipeline, frame, fromLoq, scalingMode, plane};
        const LdcTaskDependency inputs[] = {basePicture, outputPicture};

        return frame->taskAdd(inputs, VNArraySize(inputs), taskUpscaleDirect, &data, sizeof(data),
                              "UpscaleDirect");
    }

    //// GenerateCmdBuffer
    //
    // Convert un-encapsulated chunks into a single command buffer.
    //
    struct TaskGenerateCmdBufferData
    {
        PipelineCPU* pipeline;
        FrameCPU* frame;
        LdpEnhancementTile* enhancementTile;
    };

    void* taskGenerateCmdBuffer(LdcTask* task, const LdcTaskPart* /*part*/)
    {
        VNTraceScoped();
        assert(task->dataSize == sizeof(TaskGenerateCmdBufferData));

        const TaskGenerateCmdBufferData& data{VNTaskData(task, TaskGenerateCmdBufferData)};
        const PipelineCPU* const pipeline{data.pipeline};
        const FrameCPU* const frame{data.frame};

        VNLogDebug("taskGenerateCmdBuffer ts:%" PRIx64 " tile:%d loq:%d plane:%d",
                   data.frame->timestamp, data.enhancementTile->tile,
                   (uint32_t)data.enhancementTile->loq, data.enhancementTile->plane);

        if (pipeline->isFlushed(frame)) {
            return nullptr;
        }

        if (!ldeDecodeEnhancement(frame->globalConfig, &frame->config, data.enhancementTile->loq,
                                  data.enhancementTile->plane, data.enhancementTile->tile,
                                  &data.enhancementTile->buffer, nullptr, nullptr)) {
            VNLogError("ldeDecodeEnhancement failed");
        }

        return nullptr;
    }

    LdcTaskDependency addTaskGenerateCmdBuffer(PipelineCPU* pipeline, FrameCPU* frame,
                                               LdpEnhancementTile* enhancementTile)
    {
        const TaskGenerateCmdBufferData data{pipeline, frame, enhancementTile};
        return frame->taskAdd(nullptr, 0, taskGenerateCmdBuffer, &data, sizeof(data), "GenerateCmdBuffer");
    }

    //// ApplyCmdBufferDirect
    //
    // Apply a generated CPU command buffer to directly to output plane. (No Temporal)
    //
    // NB: The output plane will be in 'internal' fixed point format
    //
    struct TaskApplyCmdBufferDirectData
    {
        PipelineCPU* pipeline;
        FrameCPU* frame;
        LdpEnhancementTile* enhancementTile;
    };

    void* taskApplyCmdBufferDirect(LdcTask* task, const LdcTaskPart* /*part*/)
    {
        VNTraceScoped();
        assert(task->dataSize == sizeof(TaskApplyCmdBufferDirectData));

        const TaskApplyCmdBufferDirectData& data{VNTaskData(task, TaskApplyCmdBufferDirectData)};
        PipelineCPU* const pipeline{data.pipeline};
        const FrameCPU* const frame{data.frame};

        if (pipeline->isSkipped(frame)) {
            return nullptr;
        }

        VNLogDebug("taskApplyCmdBufferDirect ts:%" PRIx64 " loq:%d plane:%d", data.frame->timestamp,
                   (uint32_t)data.enhancementTile->loq, data.enhancementTile->plane);

        LdpPicturePlaneDesc ppDesc{};

        frame->getIntermediatePlaneDesc(data.enhancementTile->plane, data.enhancementTile->loq, ppDesc);

        const bool tuRasterOrder =
            !frame->globalConfig->temporalEnabled && frame->globalConfig->tileDimensions == TDTNone;

        if (!ldppApplyCmdBuffer(pipeline->taskPool(), NULL, data.enhancementTile, LdpFPS14, &ppDesc,
                                tuRasterOrder, pipeline->configuration().forceScalar,
                                pipeline->configuration().highlightResiduals)) {
            VNLogError("taskApplyCmdBufferDirect failed");
        }

        return nullptr;
    }

    LdcTaskDependency addTaskApplyCmdBufferDirect(PipelineCPU* pipeline, FrameCPU* frame,
                                                  LdpEnhancementTile* enhancementTile,
                                                  LdcTaskDependency imageBuffer, LdcTaskDependency cmdBuffer)
    {
        const TaskApplyCmdBufferDirectData data{pipeline, frame, enhancementTile};

        const LdcTaskDependency inputs[] = {imageBuffer, cmdBuffer};
        return frame->taskAdd(inputs, VNArraySize(inputs), taskApplyCmdBufferDirect, &data,
                              sizeof(data), "ApplyCmdBufferDirect");
    }

    //// ApplyCmdBufferTemporal
    //
    // Apply a generated CPU command buffer to a temporal buffer.
    //
    struct TaskApplyCmdBufferTemporalData
    {
        PipelineCPU* pipeline;
        FrameCPU* frame;
        LdpEnhancementTile* enhancementTile;
    };

    void* taskApplyCmdBufferTemporal(LdcTask* task, const LdcTaskPart* /*part*/)
    {
        VNTraceScoped();
        assert(task->dataSize == sizeof(TaskApplyCmdBufferTemporalData));

        const TaskApplyCmdBufferTemporalData& data{VNTaskData(task, TaskApplyCmdBufferTemporalData)};
        PipelineCPU* const pipeline{data.pipeline};
        const FrameCPU* const frame{data.frame};

        VNLogDebug("taskApplyCmdBufferTemporal ts:%" PRIx64 " tile:%d loq:%d plane:%d",
                   data.frame->timestamp, data.enhancementTile->tile,
                   (uint32_t)data.enhancementTile->loq, data.enhancementTile->plane);

        if (pipeline->isFlushed(frame)) {
            return nullptr;
        }

        LdpPicturePlaneDesc ppDesc{};
        frame->getTemporalBufferPlaneDesc(data.enhancementTile->plane, ppDesc);

        if (!ldppApplyCmdBuffer(pipeline->taskPool(), NULL, data.enhancementTile, LdpFPS14, &ppDesc,
                                false, pipeline->configuration().forceScalar,
                                pipeline->configuration().highlightResiduals)) {
            VNLogError("ldppApplyCmdBufferTemporal failed");
        }
        return nullptr;
    }

    LdcTaskDependency addTaskApplyCmdBufferTemporal(PipelineCPU* pipeline, FrameCPU* frame,
                                                    LdpEnhancementTile* enhancementTile,
                                                    LdcTaskDependency temporalBuffer,
                                                    LdcTaskDependency cmdBuffer)
    {
        const TaskApplyCmdBufferTemporalData data{pipeline, frame, enhancementTile};
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
        PipelineCPU* pipeline;
        FrameCPU* frame;
        uint32_t planeIndex;
    };

    void* taskApplyAddTemporal(LdcTask* task, const LdcTaskPart* /*part*/)
    {
        VNTraceScoped();
        assert(task->dataSize == sizeof(TaskApplyAddTemporalData));

        const TaskApplyAddTemporalData& data{VNTaskData(task, TaskApplyAddTemporalData)};
        PipelineCPU* const pipeline{data.pipeline};
        FrameCPU* const frame{data.frame};

        if (pipeline->isSkipped(frame) || frame->isPassthrough()) {
            // Just move temporal buffer along pipline
            pipeline->transferTemporalBuffer(frame, data.planeIndex);
            return nullptr;
        }

        VNLogDebug("taskApplyAddTemporal ts:%" PRIx64 " plane:%d", data.frame->timestamp, data.planeIndex);

        LdpPicturePlaneDesc dstPlane{};
        frame->getIntermediatePlaneDesc(data.planeIndex, LOQ0, dstPlane);

        LdpPicturePlaneDesc tbDesc{};
        frame->getTemporalBufferPlaneDesc(data.planeIndex, tbDesc);

        if (!ldppPlaneBlit(pipeline->taskPool(), task, pipeline->configuration().forceScalar,
                           data.planeIndex, frame->getIntermediateLayout(LOQ0),
                           frame->getIntermediateLayout(LOQ0), &tbDesc, &dstPlane, BMAdd)) {
            VNLogError("ldppPlaneBlit out failed");
        }

        return nullptr;
    }

    LdcTaskDependency addTaskApplyAddTemporal(PipelineCPU* pipeline, FrameCPU* frame, uint32_t planeIndex,
                                              LdcTaskDependency temporalDep, LdcTaskDependency sourceDep)
    {
        const TaskApplyAddTemporalData data{pipeline, frame, planeIndex};
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
        PipelineCPU* pipeline;
        FrameCPU* frame;
        uint32_t planeIndex;
    };

    void* taskPassthrough(LdcTask* task, const LdcTaskPart* /*part*/)
    {
        VNTraceScoped();
        assert(task->dataSize == sizeof(TaskPassthroughData));

        const TaskPassthroughData& data{VNTaskData(task, TaskPassthroughData)};
        PipelineCPU* const pipeline{data.pipeline};
        const FrameCPU* const frame{data.frame};

        if (pipeline->isSkipped(frame)) {
            return nullptr;
        }

        // Check if this plane is valid
        if (data.planeIndex >= ldpPictureLayoutPlanes(&frame->basePicture->layout)) {
            return nullptr;
        }

        LdpPicturePlaneDesc srcPlane;
        frame->getBasePlaneDesc(data.planeIndex, srcPlane);

        LdpPicturePlaneDesc dstPlane;
        frame->getOutputPlaneDesc(data.planeIndex, dstPlane);

        VNLogDebug("taskPassthrough ts:%" PRIx64 " plane:%d", data.frame->timestamp, data.planeIndex);

        if (!ldppPlaneBlit(pipeline->taskPool(), task, pipeline->configuration().forceScalar,
                           data.planeIndex, &frame->basePicture->layout,
                           &frame->outputPicture->layout, &srcPlane, &dstPlane, BMCopy)) {
            VNLogError("ldppPlaneBlit In failed");
        }

        return nullptr;
    }

    LdcTaskDependency addTaskPassthrough(PipelineCPU* pipeline, FrameCPU* frame, uint32_t planeIndex,
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
        PipelineCPU* pipeline;
        FrameCPU* frame;
    };

    void* taskWaitForMany(LdcTask* task, const LdcTaskPart* /*part*/)
    {
        VNTraceScoped();
        assert(task->dataSize == sizeof(TaskWaitForManyData));

        VNLogDebug("taskWaitForMany ts:%" PRIx64 "", VNTaskData(task, TaskWaitForManyData).frame->timestamp);
        return nullptr;
    }

    LdcTaskDependency addTaskWaitForMany(PipelineCPU* pipeline, FrameCPU* frame,
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
        PipelineCPU* pipeline;
        FrameCPU* frame;
    };

    void* taskBaseDone(LdcTask* task, const LdcTaskPart* /*part*/)
    {
        VNTraceScoped();
        assert(task->dataSize == sizeof(TaskBaseDoneData));

        const TaskBaseDoneData& data{VNTaskData(task, TaskBaseDoneData)};
        const FrameCPU* const frame{data.frame};

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

    void addTaskBaseDone(PipelineCPU* pipeline, FrameCPU* frame, const LdcTaskDependency* inputs,
                         uint32_t inputsCount)
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
        PipelineCPU* pipeline;
        FrameCPU* frame;
    };

    void* taskOutputDone(LdcTask* task, const LdcTaskPart* /*part*/)
    {
        VNTraceScoped();
        assert(task->dataSize == sizeof(TaskOutputDoneData));

        const TaskOutputDoneData& data{VNTaskData(task, TaskOutputDoneData)};
        PipelineCPU* const pipeline{data.pipeline};
        FrameCPU* const frame{data.frame};

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

    void addTaskOutputDone(PipelineCPU* pipeline, FrameCPU* frame, const LdcTaskDependency* inputs,
                           uint32_t inputsCount)
    {
        const TaskOutputDoneData data{pipeline, frame};

        frame->taskAddSink(inputs, inputsCount, taskOutputDone, &data, sizeof(data), "OutputDone");
    }

    //// TemporalTransfer
    //
    // Wait for a bunch of input dependencies to be met, then transfer temporal buffer to next frame
    //
    struct TaskTemporalTransferData
    {
        PipelineCPU* pipeline;
        FrameCPU* frame;
        uint32_t planeIndex;
    };

    void* taskTemporalTransfer(LdcTask* task, const LdcTaskPart* /*part*/)
    {
        VNTraceScoped();
        assert(task->dataSize == sizeof(TaskTemporalTransferData));

        const TaskTemporalTransferData& data{VNTaskData(task, TaskTemporalTransferData)};
        PipelineCPU* const pipeline{data.pipeline};
        FrameCPU* const frame{data.frame};
        const uint32_t planeIndex{data.planeIndex};

        VNLogDebug("taskTemporalTransfer ts:%" PRIx64, frame->timestamp);

        pipeline->transferTemporalBuffer(frame, planeIndex);

        return nullptr;
    }

    void addTaskTemporalTransfer(PipelineCPU* pipeline, FrameCPU* frame,
                                 const LdcTaskDependency* deps, uint32_t planeIndex)
    {
        const TaskTemporalTransferData data{pipeline, frame, planeIndex};
        const LdcTaskDependency inputs[] = {deps[planeIndex]};

        frame->taskAddSink(inputs, VNArraySize(inputs), taskTemporalTransfer, &data, sizeof(data),
                           "TemporalTransfer");
    }

    // Fill out a task group given a frame configuration
    //
    void generateTasksEnhancement(PipelineCPU* pipeline, FrameCPU* frame, uint64_t previousTimestamp)
    {
        VNTraceScoped();

        // Convenience values for readability
        const LdeFrameConfig& frameConfig{frame->config};
        const LdeGlobalConfig& globalConfig{*frame->globalConfig};
        const uint8_t numImagePlanes{frame->numImagePlanes()};
        const uint8_t numEnhancedPlanes =
            std::min(globalConfig.numPlanes, static_cast<uint8_t>(RCMaxPlanes));
        const LdeScalingMode scalingModes[LOQEnhancedCount] = {globalConfig.scalingModes[LOQ0],
                                                               globalConfig.scalingModes[LOQ1]};

        uint32_t enhancementTileIdx = 0;

        if (frame->config.sharpenType != STDisabled && frame->config.sharpenStrength != 0.0f) {
            VNLogWarning("S-Filter is configured in stream, but not supported by decoder.");
        }

        //// LoQ 1
        //
        LdcTaskDependency basePlanes[kLdpPictureMaxNumPlanes] = {};

        // Input plane dependencies - will be filled in via sendBase()
        for (uint8_t plane = 0; plane < numImagePlanes; ++plane) {
            basePlanes[plane] = frame->depBasePicture();
        }

        // Upsscale and residuals
        for (uint8_t plane = 0; plane < numEnhancedPlanes; ++plane) {
            const bool planeEnhanced = frame->isPlaneEnhanced(LOQ1, plane);

            //// Input conversion
            //
            LdcTaskDependency basePlane{};

            // Convert between base and enhancement bit depth
            basePlane = addTaskConvertToInternal(pipeline, frame, plane, globalConfig.baseDepth,
                                                 globalConfig.enhancedDepth, basePlanes[plane]);

            //// Base + Residuals
            //
            // First upscale
            LdcTaskDependency baseUpscaled{kTaskDependencyInvalid};
            if (scalingModes[LOQ1] != Scale0D) {
                baseUpscaled = addTaskUpscale(pipeline, frame, LOQ2, plane, basePlane);
            } else {
                baseUpscaled = basePlane;
            }

            // Enhancement LOQ1 decoding
            if (planeEnhanced) {
                const uint32_t numPlaneTiles = globalConfig.numTiles[plane][LOQ1];
                if (numPlaneTiles > 1) {
                    LdcTaskDependency* tiles = static_cast<LdcTaskDependency*>(
                        alloca(numPlaneTiles * sizeof(LdcTaskDependency)));

                    // Generate and apply each tile's command buffer
                    for (uint32_t tile = 0; tile < numPlaneTiles; ++tile) {
                        LdpEnhancementTile* et{frame->getEnhancementTile(enhancementTileIdx++)};
                        assert(et->plane == plane && et->loq == LOQ1 && et->tile == tile);

                        LdcTaskDependency commands = addTaskGenerateCmdBuffer(pipeline, frame, et);
                        tiles[tile] =
                            addTaskApplyCmdBufferDirect(pipeline, frame, et, baseUpscaled, commands);
                    }
                    // Wait for all tiles to finish
                    basePlanes[plane] = addTaskWaitForMany(pipeline, frame, tiles, numPlaneTiles);
                } else {
                    LdpEnhancementTile* et{frame->getEnhancementTile(enhancementTileIdx++)};
                    assert(et->plane == plane && et->loq == LOQ1 && et->tile == 0);

                    LdcTaskDependency commands = addTaskGenerateCmdBuffer(pipeline, frame, et);
                    basePlanes[plane] =
                        addTaskApplyCmdBufferDirect(pipeline, frame, et, baseUpscaled, commands);
                }
            } else {
                basePlanes[plane] = baseUpscaled;
            }
        }

        // Upscale from combined intermediate picture to preliminary output picture
        LdcTaskDependency upscaledPlanes[kLdpPictureMaxNumPlanes] = {};
        LdcTaskDependency outputPlanes[kLdpPictureMaxNumPlanes] = {};

        for (uint8_t plane = 0; plane < numEnhancedPlanes; ++plane) {
            if (scalingModes[LOQ0] != Scale0D) {
                upscaledPlanes[plane] = addTaskUpscale(pipeline, frame, LOQ1, plane, basePlanes[plane]);
            } else {
                upscaledPlanes[plane] = basePlanes[plane];
            }
        }

        // Upscaling and passthrough for planes with no residuals
        for (uint8_t plane = numEnhancedPlanes; plane < numImagePlanes; ++plane) {
            if (scalingModes[LOQ0] == Scale0D && scalingModes[LOQ1] == Scale0D) {
                // 0D 0D - no upscaling, passthrough chroma planes to output
                outputPlanes[plane] = addTaskPassthrough(pipeline, frame, plane,
                                                         frame->depOutputPicture(), basePlanes[plane]);
            } else if ((scalingModes[LOQ0] != Scale0D && scalingModes[LOQ1] != Scale0D) ||
                       globalConfig.baseDepth != globalConfig.enhancedDepth) {
                // Double upscale or bitdepth conversion - convert to internal, run both upscales in
                // S16 and convert back to output
                LdcTaskDependency baseConverted =
                    addTaskConvertToInternal(pipeline, frame, plane, globalConfig.baseDepth,
                                             globalConfig.enhancedDepth, basePlanes[plane]);
                LdcTaskDependency upscaledLOQ1 = baseConverted;
                LdcTaskDependency upscaledLOQ0 = upscaledLOQ1;
                if (scalingModes[LOQ1] != Scale0D) {
                    upscaledLOQ1 = addTaskUpscale(pipeline, frame, LOQ2, plane, baseConverted);
                }
                if (scalingModes[LOQ0] != Scale0D) {
                    upscaledLOQ0 = addTaskUpscale(pipeline, frame, LOQ1, plane, upscaledLOQ1);
                }
                outputPlanes[plane] = addTaskConvertFromInternal(
                    pipeline, frame, plane, globalConfig.baseDepth, globalConfig.enhancedDepth,
                    frame->depOutputPicture(), upscaledLOQ0);
            } else {
                // Only one LOQ requires upscaling (most common case). Limit base planes for NV12
                const LdeLOQIndex fromLoq = (scalingModes[LOQ0] != Scale0D) ? LOQ1 : LOQ2;
                outputPlanes[plane] = addTaskUpscaleDirect(
                    pipeline, frame, fromLoq, plane, basePlanes[plane], frame->depOutputPicture());
            }
        }

        //// LoQ 0
        //
        LdcTaskDependency reconstructedPlanes[kLdpPictureMaxNumPlanes] = {};

        for (uint8_t plane = 0; plane < numImagePlanes; ++plane) {
            const bool planeEnhanced = frame->isPlaneEnhanced(LOQ0, plane);

            LdcTaskDependency recon{upscaledPlanes[plane]};

            if (globalConfig.temporalEnabled && !frame->isPassthrough()) {
                LdcTaskDependency temporal{kTaskDependencyInvalid};

                if (plane < numEnhancedPlanes) {
                    // Still need a temporal buffer, even if the particular frame is not enhanced
                    // winds up getting passed through and applied
                    temporal = frame->needsTemporalBuffer(previousTimestamp, plane);
                    pipeline->findTemporalBuffer(frame, plane);
                }

                if (planeEnhanced) {
                    // Enhancement residuals
                    const uint32_t numPlaneTiles = globalConfig.numTiles[plane][LOQ0];
                    if (numPlaneTiles > 1) {
                        LdcTaskDependency* tiles = static_cast<LdcTaskDependency*>(
                            alloca(numPlaneTiles * sizeof(LdcTaskDependency)));

                        // Generate and apply each tile's command buffer
                        for (uint32_t tile = 0; tile < numPlaneTiles; ++tile) {
                            LdpEnhancementTile* et{frame->getEnhancementTile(enhancementTileIdx++)};
                            assert(et->plane == plane && et->loq == LOQ0 && et->tile == tile);
                            LdcTaskDependency commands{addTaskGenerateCmdBuffer(pipeline, frame, et)};

                            tiles[tile] =
                                addTaskApplyCmdBufferTemporal(pipeline, frame, et, temporal, commands);
                        }
                        // Wait for all tiles to finish
                        temporal = addTaskWaitForMany(pipeline, frame, tiles, numPlaneTiles);
                    } else {
                        LdpEnhancementTile* et = frame->getEnhancementTile(enhancementTileIdx++);
                        assert(et && et->plane == plane && et->loq == LOQ0 && et->tile == 0);

                        LdcTaskDependency commands{addTaskGenerateCmdBuffer(pipeline, frame, et)};

                        temporal = addTaskApplyCmdBufferTemporal(pipeline, frame, et, temporal, commands);
                    }
                }

                // Always add temporal buffer, even if no enhancement this frame
                if (plane < numEnhancedPlanes) {
                    reconstructedPlanes[plane] =
                        addTaskApplyAddTemporal(pipeline, frame, plane, temporal, recon);
                    addTaskTemporalTransfer(pipeline, frame, reconstructedPlanes, plane);
                } else {
                    reconstructedPlanes[plane] = recon;
                }
            } else {
                if (planeEnhanced && frameConfig.loqEnabled[LOQ0]) {
                    // Enhancement residuals
                    const uint32_t numPlaneTiles = globalConfig.numTiles[plane][LOQ0];
                    if (numPlaneTiles > 1) {
                        LdcTaskDependency* tiles = static_cast<LdcTaskDependency*>(
                            alloca(numPlaneTiles * sizeof(LdcTaskDependency)));

                        // Generate and apply each tile's command buffer
                        for (uint32_t tile = 0; tile < numPlaneTiles; ++tile) {
                            LdpEnhancementTile* et{frame->getEnhancementTile(enhancementTileIdx++)};
                            assert(et->plane == plane && et->loq == LOQ0 && et->tile == tile);
                            LdcTaskDependency commands{addTaskGenerateCmdBuffer(pipeline, frame, et)};
                            tiles[tile] =
                                addTaskApplyCmdBufferDirect(pipeline, frame, et, recon, commands);
                        }
                        // Wait for all tiles to finish
                        recon = addTaskWaitForMany(pipeline, frame, tiles, numPlaneTiles);
                    } else {
                        LdpEnhancementTile* et = frame->getEnhancementTile(enhancementTileIdx++);
                        assert(et->plane == plane && et->loq == LOQ0 && et->tile == 0);

                        LdcTaskDependency commands{addTaskGenerateCmdBuffer(pipeline, frame, et)};

                        recon = addTaskApplyCmdBufferDirect(pipeline, frame, et, recon, commands);
                    }
                }

                reconstructedPlanes[plane] = recon;
            }
        }

        assert(enhancementTileIdx == frame->enhancementTileCount);

        // Convert any enhanced planes back to output
        for (uint8_t plane = 0; plane < numEnhancedPlanes; ++plane) {
            outputPlanes[plane] = addTaskConvertFromInternal(
                pipeline, frame, plane, globalConfig.baseDepth, globalConfig.enhancedDepth,
                frame->depOutputPicture(), reconstructedPlanes[plane]);
        }

        // Send output when all planes are ready
        addTaskOutputDone(pipeline, frame, outputPlanes, numImagePlanes);

        // Send base when all tasks that use it have completed
        LdcTaskDependency deps[kLdpPictureMaxNumPlanes] = {};
        uint32_t depsCount = 0;
        ldcTaskGroupFindOutputSetFromInput(frame->taskGroup(), frame->depBasePicture(), deps,
                                           kLdpPictureMaxNumPlanes, &depsCount);
        addTaskBaseDone(pipeline, frame, deps, depsCount);
    }

    // Fill out a task group for a simple unscaled passthrough configuration
    //
    void generateTasksPassthrough(PipelineCPU* pipeline, FrameCPU* frame)
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
void generateTasks(PipelineCPU* pipeline, FrameCPU* frame, uint64_t previousTimestamp)
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

} // namespace lcevc_dec::pipeline_cpu