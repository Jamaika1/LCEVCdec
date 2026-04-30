/* Copyright (c) V-Nova International Limited 2026. All rights reserved.
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

#include <gtest/gtest.h>
#include <LCEVC/pixel_processing/dither.h>
#include <render_vulkan.h>

#include <cstdint>
#include <vector>

using namespace lcevc_dec::pipeline_vulkan;

namespace lcevc_dec::pipeline_vulkan {

class RenderVulkanDitherTestAccess
{
public:
    static bool populateDitherBuffers(const LdppDitherGlobal* global, const LdppDitherFrame* frame,
                                      uint32_t width, uint32_t height, uint32_t ditherBufferSize,
                                      uint32_t* entropyDst, uint32_t* rowOffsets)
    {
        return RenderVulkan::populateDitherBuffers(global, frame, width, height, ditherBufferSize,
                                                   entropyDst, rowOffsets);
    }
};

} // namespace lcevc_dec::pipeline_vulkan

class VulkanDitherFixture : public testing::Test
{
public:
    static constexpr uint32_t kDitherBufferSize = 16384;

    LdcMemoryAllocator* mAllocator = nullptr;
    LdppDitherGlobal mDitherGlobal{};
    std::vector<uint32_t> mEntropyStorage;
    std::vector<uint32_t> mRowOffsetStorage;

    void SetUp() override
    {
        mAllocator = ldcMemoryAllocatorMalloc();
        ASSERT_NE(mAllocator, nullptr);
        ASSERT_TRUE(ldppDitherGlobalInitialize(mAllocator, &mDitherGlobal, 0x0123456789ABCDEFULL));
    }

    void TearDown() override { ldppDitherGlobalRelease(&mDitherGlobal); }

protected:
    LdppDitherGlobal* ditherGlobal() { return &mDitherGlobal; }

    void prepareBuffers(uint32_t height)
    {
        mEntropyStorage.assign(kDitherBufferSize, 0);
        mRowOffsetStorage.assign(height * RenderVulkan::NUM_PLANES, 0);
    }

    static uint32_t expectedRowOffset(const LdppDitherFrame& frame, uint32_t y, uint32_t plane,
                                      uint32_t requiredLength)
    {
        LdppDitherSlice slice{};
        ldppDitherSliceInitialise(&slice, &frame, y, plane);
        const uint16_t* buffer = ldppDitherGetBuffer(&slice, requiredLength);
        EXPECT_NE(buffer, nullptr);
        return static_cast<uint32_t>(buffer - frame.global->buffer);
    }
};

TEST_F(VulkanDitherFixture, PopulateDitherBuffersUploadsEntropyAndRowOffsets)
{
    constexpr uint32_t width = 256;
    constexpr uint32_t height = 8;
    prepareBuffers(height);

    LdppDitherFrame frame{};
    ASSERT_TRUE(ldppDitherFrameInitialise(&frame, ditherGlobal(), 0x12345678ULL, 7));

    ASSERT_TRUE(RenderVulkanDitherTestAccess::populateDitherBuffers(
        ditherGlobal(), &frame, width, height, kDitherBufferSize, mEntropyStorage.data(),
        mRowOffsetStorage.data()));

    for (uint32_t i = 0; i < kDitherBufferSize; ++i) {
        EXPECT_EQ(mEntropyStorage[i], static_cast<uint32_t>(ditherGlobal()->buffer[i]));
    }

    const uint32_t requiredLength = width << 2;
    for (uint32_t plane = 0; plane < RenderVulkan::NUM_PLANES; ++plane) {
        for (uint32_t y = 0; y < height; ++y) {
            EXPECT_EQ(mRowOffsetStorage[plane * height + y],
                      expectedRowOffset(frame, y, plane, requiredLength));
        }
    }
}

TEST_F(VulkanDitherFixture, PopulateDitherBuffersRejectsNullOrZeroStrengthFrame)
{
    constexpr uint32_t width = 256;
    constexpr uint32_t height = 8;
    prepareBuffers(height);

    EXPECT_FALSE(RenderVulkanDitherTestAccess::populateDitherBuffers(
        ditherGlobal(), nullptr, width, height, kDitherBufferSize, mEntropyStorage.data(),
        mRowOffsetStorage.data()));

    LdppDitherFrame zeroStrengthFrame{};
    ASSERT_TRUE(ldppDitherFrameInitialise(&zeroStrengthFrame, ditherGlobal(), 0xCAFEBABEULL, 0));

    EXPECT_FALSE(RenderVulkanDitherTestAccess::populateDitherBuffers(
        ditherGlobal(), &zeroStrengthFrame, width, height, kDitherBufferSize,
        mEntropyStorage.data(), mRowOffsetStorage.data()));
}

TEST_F(VulkanDitherFixture, PopulateDitherBuffersRejectsOversizedRows)
{
    constexpr uint32_t height = 4;
    prepareBuffers(height);

    LdppDitherFrame frame{};
    ASSERT_TRUE(ldppDitherFrameInitialise(&frame, ditherGlobal(), 0xABCDEF01ULL, 9));

    EXPECT_FALSE(RenderVulkanDitherTestAccess::populateDitherBuffers(
        ditherGlobal(), &frame, (kDitherBufferSize >> 2) + 1, height, kDitherBufferSize,
        mEntropyStorage.data(), mRowOffsetStorage.data()));
}
