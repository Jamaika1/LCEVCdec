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

#ifndef VN_LCEVC_PIPELINE_CPU_FROM_BASE_H
#define VN_LCEVC_PIPELINE_CPU_FROM_BASE_H

#include "buffer_cpu.h"
#include "frame_cpu.h"
#include "picture_cpu.h"
#include "picture_lock_cpu.h"
#include "pipeline_builder_cpu.h"
#include "pipeline_config_cpu.h"
#include "pipeline_cpu.h"
#include "temporal_buffer_cpu.h"

namespace lcevc_dec::pipeline_cpu {

// Trait to describe what concrete classes are derived from bases
//
template <typename B>
struct Derived;

template <>
struct Derived<pipeline::BufferBase>
{
    typedef BufferCPU Type;
};
template <>
struct Derived<pipeline::PictureBase>
{
    typedef PictureCPU Type;
};
template <>
struct Derived<pipeline::FrameBase>
{
    typedef FrameCPU Type;
};
template <>
struct Derived<pipeline::PipelineBase>
{
    typedef PipelineCPU Type;
};
template <>
struct Derived<pipeline::PictureLockBase>
{
    typedef PictureLockCPU Type;
};
template <>
struct Derived<pipeline::PipelineBuilderBase>
{
    typedef PipelineBuilderCPU Type;
};
template <>
struct Derived<pipeline::PipelineConfigBase>
{
    typedef PipelineConfigCPU Type;
};
template <>
struct Derived<pipeline::TemporalBufferBase>
{
    typedef TemporalBufferCPU Type;
};

// Ditto for pipeline types
template <>
struct Derived<LdpPicture>
{
    typedef PictureCPU Type;
};
template <>
struct Derived<LdpFrame>
{
    typedef FrameCPU Type;
};
template <>
struct Derived<LdpBuffer>
{
    typedef BufferCPU Type;
};
template <>
struct Derived<LdpPictureLock>
{
    typedef PictureLockCPU Type;
};

// Given pointer to a xxxBase type - return the xxxVPU type.
//
// Debug builds use dynamic_cast<> to check that we really have the right type
//
template <typename B>
typename Derived<B>::Type* fromBase(B* bp)
{
#if !defined(NDEBUG)
    auto* dp = dynamic_cast<typename Derived<B>::Type*>(bp);
    assert(dp);
    return dp;
#else
    return static_cast<typename Derived<B>::Type*>(bp);
#endif
}

// Given pointer to a Ldpxxx type - return the xxxCPU type.
//
// The underlying pipeline types are not polymorphic, can only static_cast<>
//
template <typename B>
typename Derived<B>::Type* fromPipeline(B* bp)
{
    return static_cast<typename Derived<B>::Type*>(bp);
}

} // namespace lcevc_dec::pipeline_cpu

#endif // VN_LCEVC_PIPELINE_CPU_FROM_BASE_H
