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

#include "test_plane.h"

#include <find_assets_dir.h>
#include <gtest/gtest.h>
#include <LCEVC/common/acceleration.h>
#include <LCEVC/common/diagnostics.h>
#include <LCEVC/pipeline/picture_layout.h>
#include <LCEVC/pixel_processing/sharpen.h>
#include <range/v3/view.hpp>
#include <range/v3/view/cartesian_product.hpp>

#include <cassert>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

namespace rg = ranges;
namespace rv = ranges::views;
using namespace lcevc_dec::utility;

// -----------------------------------------------------------------------------

static constexpr uint32_t kWidth = 180;
static constexpr uint32_t kHeight = 100;
static constexpr float kStrengthLow = 0.35f;

const static std::filesystem::path kTestAssets = findAssetsDir("src/pixel_processing/test/assets");
static constexpr std::string_view kFile8Bit = "ElfuenteTunnel_180x100_8bit_400p_lf.yuv";
static constexpr std::string_view kFile16Bit = "ElfuenteTunnel_180x100_16bit_400p_lf.yuv";

// -----------------------------------------------------------------------------

struct SharpenHashParams
{
    LdpFixedPoint fixedPoint;
    float strength;
    std::string hash;
};

struct SharpenTestParams
{
    LdpFixedPoint fixedPoint;
    float strength;
    bool disableSIMD;
    std::string hash;
};

void PrintTo(const SharpenTestParams& params, std::ostream* os)
{
    *os << "{fixedPoint=" << fixedPointToString(params.fixedPoint) << ", strength=" << params.strength
        << ", disableSIMD=" << (params.disableSIMD ? "true" : "false") << ", hash=\"" << params.hash
        << "\"}";
}

// -----------------------------------------------------------------------------

static LdpColorFormat getGrayFormat(LdpFixedPoint fixedPoint)
{
    switch (fixedPoint) {
        case LdpFPU8: return LdpColorFormatGRAY_8;
        case LdpFPU10: return LdpColorFormatGRAY_10_LE;
        case LdpFPU12: return LdpColorFormatGRAY_12_LE;
        case LdpFPU14: return LdpColorFormatGRAY_14_LE;
        default: assert(false); return LdpColorFormatUnknown;
    }
}

static void remapInputToFixedPoint(TestPlane& plane)
{
    if (plane.fixedPoint == LdpFPU8) {
        return;
    }

    const uint32_t shift = 16 - bitdepthFromFixedPoint(plane.fixedPoint);
    uint16_t* pixel = reinterpret_cast<uint16_t*>(plane.planeDesc.firstSample);
    const uint32_t count = (plane.planeDesc.rowByteStride * plane.height) / sizeof(uint16_t);

    for (uint32_t i = 0; i < count; ++i) {
        pixel[i] = (uint16_t)(pixel[i] >> shift);
    }
}

std::string testNames(const testing::TestParamInfo<SharpenTestParams>& value)
{
    const SharpenTestParams params = value.param;

    std::stringstream ss;
    ss << fixedPointToString(params.fixedPoint) << "_str_0"
       << static_cast<int32_t>(params.strength * 100.0f)
       << (params.disableSIMD ? "_simdOff" : "_simdOn");
    return ss.str();
}

const std::vector<SharpenHashParams> kBaseHashes = {
    {LdpFPU8, kStrengthLow, "92d4af1895843bc69613ff92de00a452"},
    {LdpFPU10, kStrengthLow, "d3d2acc7cdb930126f9bc346615fe348"},
    {LdpFPU12, kStrengthLow, "841054d1b2acea29487e15c568cfe784"},
    {LdpFPU14, kStrengthLow, "c18ea64923ec0834192d92b1e6a6507e"},
};

const std::vector<bool> kForceScalar = {true, false};

const auto kSharpenTestParams =
    rv::cartesian_product(kBaseHashes, kForceScalar) | rv::transform([](auto value) {
        const SharpenHashParams hashParams = std::get<0>(value);
        const bool disableSIMD = std::get<1>(value);
        return SharpenTestParams{hashParams.fixedPoint, hashParams.strength, disableSIMD, hashParams.hash};
    }) |
    rg::to_vector;

// -----------------------------------------------------------------------------

class SharpenTest : public testing::TestWithParam<SharpenTestParams>
{
protected:
    void SetUp() override
    {
        const SharpenTestParams params = GetParam();

        ldcAccelerationInitialize(!params.disableSIMD);

        const LdpColorFormat colorFormat = getGrayFormat(params.fixedPoint);
        ldpPictureLayoutInitialize(&m_layout, colorFormat, kWidth, kHeight, 0);
        m_plane.initialize(kWidth, kHeight, m_layout.rowStrides[0] / fixedPointByteSize(params.fixedPoint),
                           params.fixedPoint);

        const std::string srcFilePath =
            (kTestAssets / ((params.fixedPoint == LdpFPU8) ? kFile8Bit : kFile16Bit)).string();
        readBinaryFile(m_plane, srcFilePath);
        remapInputToFixedPoint(m_plane);
    }

    TestPlane m_plane = {};
    LdpPictureLayout m_layout = {};
};

TEST_P(SharpenTest, HashPlane)
{
    const SharpenTestParams params = GetParam();

    VNDiagInfo(diagInfo, "SharpenTest", 0, 0, 0);
    EXPECT_TRUE(ldppSharpen(&m_layout, &m_plane.planeDesc, 0, params.strength, nullptr,
                            VNDiagInfoPtr(diagInfo)));
    EXPECT_EQ(params.hash, hashActiveRegion(m_plane));
}

INSTANTIATE_TEST_SUITE_P(SharpenTests, SharpenTest, testing::ValuesIn(kSharpenTestParams), testNames);

// -----------------------------------------------------------------------------
