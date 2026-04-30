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

#include "buffer_cpu.h"
//
#include <LCEVC/pipeline/buffer_alignment.h>

#include <cassert>

namespace lcevc_dec::pipeline_cpu {

BufferCPU::BufferCPU(LdcMemoryAllocator* allocator, uint32_t size, pipeline::BufferUsage usage)
    : m_allocator(allocator)
    , m_usage(usage)
{
    // Allocate the bytes
    if (size > 0) {
        VNAllocateAlignedArray(m_allocator, &m_allocation, uint8_t, kBufferRowAlignment, size,
                               "BufferCPU_Data");
    }
}

BufferCPU::~BufferCPU()
{
    if (VNIsAllocated(m_allocation)) {
        VNFree(m_allocator, &m_allocation);
    }
}

bool BufferCPU::map(LdpBufferMapping* mapping, int32_t offset, uint32_t mapSize, LdpAccess access)
{
    // Is this mapped already?
    if (m_mapped) {
        return false;
    }

    // Does requested window fit in buffer?
    if (offset < 0 || (offset + mapSize) > size()) {
        return false;
    }

    // Record details
    mapping->buffer = this;
    mapping->offset = offset;
    mapping->size = mapSize;
    mapping->ptr = VNAllocationPtr(m_allocation, uint8_t) + offset;
    mapping->access = access;
    mapping->userData = (void*)this;

    m_mapped = true;
    return true;
}

void BufferCPU::unmap(const LdpBufferMapping* mapping)
{
    VNUnused(mapping);

    assert(mapping->userData == (void*)this);
    m_mapped = false;
}

void BufferCPU::clear() // NOLINT(readability-make-member-function-const)
{
    if (!VNIsAllocated(m_allocation)) {
        return;
    }

    memset(VNAllocationPtr(m_allocation, uint8_t), 0, VNAllocationSize(m_allocation, uint8_t));
}

void BufferCPU::copyIn(uint32_t dstOffset, const void* src, uint32_t size)
{
    assert(VNIsAllocated(m_allocation));
    assert(dstOffset + size <= VNAllocationSize(m_allocation, uint8_t));
    memcpy(VNAllocationPtr(m_allocation, uint8_t) + dstOffset, src, size);
}

void BufferCPU::copyOut(void* dst, uint32_t srcOffset, uint32_t size)
{
    assert(VNIsAllocated(m_allocation));
    assert(srcOffset + size <= VNAllocationSize(m_allocation, uint8_t));
    memcpy(dst, VNAllocationPtr(m_allocation, uint8_t) + srcOffset, size);
}

pipeline::BufferUsage BufferCPU::usage() const { return m_usage; }

uint32_t BufferCPU::size() const
{
    return static_cast<uint32_t>(VNAllocationSize(m_allocation, uint8_t));
}

bool BufferCPU::resize(uint32_t size)
{
    VNReallocateArray(m_allocator, &m_allocation, uint8_t, size, "BufferCPU_Data");
    return VNIsAllocated(m_allocation);
}

} // namespace lcevc_dec::pipeline_cpu
