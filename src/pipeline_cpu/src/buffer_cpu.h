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

#ifndef VN_LCEVC_PIPELINE_CPU_BUFFER_CPU_H
#define VN_LCEVC_PIPELINE_CPU_BUFFER_CPU_H

#include "picture_cpu.h"

#include <LCEVC/pipeline/buffer_base.h>

namespace lcevc_dec::pipeline_cpu {

class BufferCPU : public pipeline::BufferBase
{
public:
    BufferCPU(LdcMemoryAllocator* allocator, uint32_t size, pipeline::BufferUsage usage);
    ~BufferCPU();

    uint32_t size() const override;
    pipeline::BufferUsage usage() const override;

    bool resize(uint32_t size) override;

    bool map(LdpBufferMapping* mapping, int32_t offset, uint32_t size, LdpAccess access) override;
    void unmap(const LdpBufferMapping* mapping) override;

    void clear() override;
    void copyIn(uint32_t dstOffset, const void* src, uint32_t size) override;
    void copyOut(void* dst, uint32_t srcOffset, uint32_t size) override;

    VNNoCopyNoMove(BufferCPU);

private:
    friend PictureCPU;

    LdcMemoryAllocator* m_allocator;
    pipeline::BufferUsage m_usage{};

    LdcMemoryAllocation m_allocation = {0};

    bool m_mapped = false;
};

} // namespace lcevc_dec::pipeline_cpu

#endif // VN_LCEVC_PIPELINE_CPU_BUFFER_CPU_H
