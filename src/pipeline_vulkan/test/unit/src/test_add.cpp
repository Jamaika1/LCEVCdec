/* Copyright (c) V-Nova International Limited 2024-2026. All rights reserved.
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

#include "buffer_vulkan.h"
#include "picture_vulkan.h"
#include "pipeline_vulkan.h"
#include "test_utility.h"
//
#include <LCEVC/pipeline/buffer_alignment.h>
#include <LCEVC/pipeline/event_sink.h>
#include <LCEVC/pipeline/picture_layout.h>
#include <LCEVC/pipeline/pipeline.h>
#include <LCEVC/pipeline/types.h>
#include <LCEVC/pipeline_vulkan/create_pipeline.h>
#include <LCEVC/pipeline_vulkan/types_vulkan.h>
//
#include <fmt/core.h>
#include <gtest/gtest.h>
//
#include <algorithm>
#include <cstring>
#include <memory>
#include <random>
#include <sstream>
#include <vector>

using namespace lcevc_dec::pipeline;
using namespace lcevc_dec::pipeline_vulkan;

// -----------------------------------------------------------------------------

namespace {

// Resolutions to test. Both dimensions must be even for 420 chroma.
struct Resolution
{
    uint32_t width;
    uint32_t height;

    std::string toString() const { return std::to_string(width) + "x" + std::to_string(height); }
};

constexpr Resolution kResolutions[] = {
    {960, 540},   // Standard qHD
    {1920, 1080}, // Full HD
    {128, 128},   // Small power-of-2
    {66, 34},     // Small non-aligned, even
    {254, 130},   // Non-power-of-2, not workgroup-aligned
    {2, 2},       // Minimum valid 420 size
};

// Fill a buffer with random signed 16-bit values in the S14 internal range [-16384, 16383].
void fillInternalNoise(int16_t* data, uint32_t count, std::mt19937& rng)
{
    std::uniform_int_distribution<int> dist(-16384, 16383);
    for (uint32_t i = 0; i < count; ++i) {
        data[i] = static_cast<int16_t>(dist(rng));
    }
}

// CPU scalar reference for the GPU add operation: dst = saturateS16(dst + src).
// This matches the GPU add.comp shader exactly.
void cpuScalarAdd(const int16_t* src, int16_t* dst, uint32_t count)
{
    for (uint32_t i = 0; i < count; ++i) {
        int32_t sum = static_cast<int32_t>(dst[i]) + static_cast<int32_t>(src[i]);
        dst[i] = static_cast<int16_t>(std::clamp(sum, -32768, 32767));
    }
}

} // namespace

// -----------------------------------------------------------------------------

struct AddTestParams
{
    Resolution resolution;
    LdeChroma chroma;
    uint8_t numEnhancedPlanes;

    std::string name() const
    {
        std::stringstream ss;
        ss << resolution.toString();
        switch (chroma) {
            case LdeChroma::CT420: ss << "_420"; break;
            case LdeChroma::CT422: ss << "_422"; break;
            case LdeChroma::CT444: ss << "_444"; break;
            case LdeChroma::CTMonochrome: ss << "_Mono"; break;
            default: break;
        }
        ss << "_" << static_cast<int>(numEnhancedPlanes) << "planes";
        return ss.str();
    }
};

std::string AddTestName(const testing::TestParamInfo<AddTestParams>& info)
{
    return info.param.name();
}

// -----------------------------------------------------------------------------

class PipelineVulkanAddTest : public testing::TestWithParam<AddTestParams>
{
public:
    void SetUp() override
    {
        auto* pipelineBuilder =
            new lcevc_dec::pipeline_vulkan::PipelineBuilderVulkan(ldcMemoryAllocatorMalloc());
        ASSERT_TRUE(pipelineBuilder);

        EventSink* eventSink = nullptr;
        m_pipeline = pipelineBuilder->finish(eventSink);
        delete pipelineBuilder;

        m_vulkanPipeline = static_cast<PipelineVulkan*>(m_pipeline.get());
        if (!m_vulkanPipeline) {
            GTEST_SKIP() << "Skipping test due to lack of Vulkan support";
        }
    }

    void TearDown() override {}

protected:
    std::unique_ptr<Pipeline> m_pipeline;
    PipelineVulkan* m_vulkanPipeline{};
};

// -----------------------------------------------------------------------------

TEST_P(PipelineVulkanAddTest, CompareGPUvsCPUScalar)
{
    const auto& params = GetParam();
    const uint32_t width = params.resolution.width;
    const uint32_t height = params.resolution.height;

    // The GPU add operates on internal-format pictures (I420_16_LE = S14 stored as int16_t).
    const LdpColorFormat internalFormat = LdpColorFormatI420_16_LE;

    // Get the layout to know strides and offsets (must match GPU picture alignment).
    LdpPictureLayout layout{};
    ldpPictureLayoutInitialize(&layout, internalFormat, width, height, kBufferRowAlignment);
    const uint32_t bufferSize = ldpPictureLayoutSize(&layout);
    const uint32_t sampleCount = bufferSize / sizeof(int16_t);

    // Generate random source and destination data in internal S14 range.
    std::mt19937 rng(123456);
    std::vector<int16_t> srcData(sampleCount);
    std::vector<int16_t> dstData(sampleCount);
    fillInternalNoise(srcData.data(), sampleCount, rng);
    fillInternalNoise(dstData.data(), sampleCount, rng);

    // --- CPU scalar reference ---
    std::vector<int16_t> cpuDst(dstData); // copy dst before GPU modifies it

    const uint8_t numPlanes = ldpPictureLayoutPlanes(&layout);
    for (uint8_t plane = 0; plane < params.numEnhancedPlanes && plane < numPlanes; ++plane) {
        const uint32_t planeWidth = ldpPictureLayoutPlaneWidth(&layout, plane);
        const uint32_t planeHeight = ldpPictureLayoutPlaneHeight(&layout, plane);
        const uint32_t stride = layout.rowStrides[plane];
        const uint32_t offset = layout.planeOffsets[plane];

        // Process row by row to respect stride (which may include padding).
        for (uint32_t y = 0; y < planeHeight; ++y) {
            const uint32_t rowByteOffset = offset + y * stride;
            auto* cpuSrcRow = reinterpret_cast<const int16_t*>(
                reinterpret_cast<const uint8_t*>(srcData.data()) + rowByteOffset);
            auto* cpuDstRow =
                reinterpret_cast<int16_t*>(reinterpret_cast<uint8_t*>(cpuDst.data()) + rowByteOffset);
            cpuScalarAdd(cpuSrcRow, cpuDstRow, planeWidth);
        }
    }

    // --- GPU execution ---
    auto* src =
        vulkan_test_util::allocateTestPictureAndBuffer(m_vulkanPipeline, internalFormat, width, height);
    auto* srcBuffer = static_cast<BufferVulkan*>(src->buffer);
    srcBuffer->copyIn(0, srcData.data(), bufferSize);

    auto* dst =
        vulkan_test_util::allocateTestPictureAndBuffer(m_vulkanPipeline, internalFormat, width, height);
    auto* dstBuffer = static_cast<BufferVulkan*>(dst->buffer);
    dstBuffer->copyIn(0, dstData.data(), bufferSize);

    VulkanAddArgs args{};
    args.src = src;
    args.dst = dst;
    args.numEnhancedPlanes = params.numEnhancedPlanes;
    args.chroma = params.chroma;

    auto& compute = m_vulkanPipeline->backend().compute();
    VulkanFrameContext* frameContext = compute.acquireFrameContext(nullptr);
    ASSERT_NE(frameContext, nullptr);
    args.context = frameContext;
    compute.beginCompute(frameContext);
    EXPECT_TRUE(m_vulkanPipeline->backend().add(&args));
    compute.endCompute(frameContext);
    m_vulkanPipeline->addCompletionContext(frameContext);
    m_vulkanPipeline->waitCompletionContextIdle();

    // Read back GPU result.
    std::vector<uint8_t> gpuResult(bufferSize);
    dstBuffer->copyOut(gpuResult.data(), 0, bufferSize);

    // --- Compare per-plane active region ---
    for (uint8_t plane = 0; plane < params.numEnhancedPlanes && plane < numPlanes; ++plane) {
        const uint32_t planeWidth = ldpPictureLayoutPlaneWidth(&layout, plane);
        const uint32_t planeHeight = ldpPictureLayoutPlaneHeight(&layout, plane);
        const uint32_t rowBytes = planeWidth * sizeof(int16_t);
        const uint32_t stride = layout.rowStrides[plane];
        const uint32_t offset = layout.planeOffsets[plane];

        for (uint32_t y = 0; y < planeHeight; ++y) {
            const uint8_t* gpuRow = gpuResult.data() + offset + y * stride;
            const uint8_t* cpuRow = reinterpret_cast<const uint8_t*>(cpuDst.data()) + offset + y * stride;
            if (memcmp(gpuRow, cpuRow, rowBytes) != 0) {
                // Find first mismatch for diagnostics.
                const auto* gpuSamples = reinterpret_cast<const int16_t*>(gpuRow);
                const auto* cpuSamples = reinterpret_cast<const int16_t*>(cpuRow);
                for (uint32_t x = 0; x < planeWidth; ++x) {
                    if (gpuSamples[x] != cpuSamples[x]) {
                        FAIL() << fmt::format(
                            "Mismatch at plane {}, row {}, sample {} ({}x{}): GPU={} CPU={}", plane,
                            y, x, width, height, gpuSamples[x], cpuSamples[x]);
                    }
                }
            }
        }
    }

    m_vulkanPipeline->freePicture(src);
    m_vulkanPipeline->freePicture(dst);
}

// -----------------------------------------------------------------------------

namespace {

std::vector<AddTestParams> generateAddParams()
{
    std::vector<AddTestParams> params;
    for (const auto& res : kResolutions) {
        // 3 planes (Y, U, V) with 420 chroma
        params.push_back({res, LdeChroma::CT420, 3});
        // 1 plane (Y only) with 420 chroma
        params.push_back({res, LdeChroma::CT420, 1});
    }
    return params;
}

} // namespace

INSTANTIATE_TEST_SUITE_P(AddTests, PipelineVulkanAddTest, testing::ValuesIn(generateAddParams()),
                         AddTestName);
