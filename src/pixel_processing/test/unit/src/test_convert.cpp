/* Copyright (c) V-Nova International Limited 2022-2026. All rights reserved.
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

#include "convert_common.h"
#include "fp_types.h"
#include "test_plane.h"

#include <gtest/gtest.h>
#include <LCEVC/pixel_processing/convert.h>
#include <range/v3/view.hpp>
#include <rng.h>

#include <functional>
#include <random>
#include <sstream>

extern "C"
{
PlaneConvertFunction planeConvertGetFunction(LdpFixedPoint srcFP, LdpFixedPoint dstFP,
                                             bool forceScalar, uint32_t planeIndex, bool isNV12);
}

namespace rg = ranges;
namespace rv = ranges::views;

// -----------------------------------------------------------------------------

constexpr uint32_t kWidth = 500;
constexpr uint32_t kHeight = 400;
constexpr uint32_t kStride = 512;

constexpr bool kForceScalar = true;
constexpr bool kSelectSIMD = false;

// -----------------------------------------------------------------------------

struct ConvertTestParams
{
    LdpFixedPoint srcFP;
    LdpFixedPoint dstFP;
};

// -----------------------------------------------------------------------------

class ConvertTest : public testing::TestWithParam<ConvertTestParams>
{
protected:
    void SetUp() override
    {
        const auto& params = GetParam();
        m_scalarFunction = planeConvertGetFunction(params.srcFP, params.dstFP, kForceScalar, 0, false);
        m_simdFunction = planeConvertGetFunction(params.srcFP, params.dstFP, kSelectSIMD, 0, false);

        m_src.initialize(kWidth, kHeight, kStride, params.srcFP);
        m_dstScalar.initialize(kWidth, kHeight, kStride, params.dstFP);
        m_dstSIMD.initialize(kWidth, kHeight, kStride, params.dstFP);
    }

    TestPlane m_src{};
    TestPlane m_dstScalar{};
    TestPlane m_dstSIMD{};

    PlaneConvertFunction m_scalarFunction{};
    PlaneConvertFunction m_simdFunction{};
};

// -----------------------------------------------------------------------------

TEST_P(ConvertTest, CompareSIMD)
{
    fillPlaneWithNoise(m_src);

    LdppConvertArgs args;
    args.src = &m_src.planeDesc;
    args.dst = &m_dstScalar.planeDesc;
    args.minWidth = kWidth;
    args.offset = 0;
    args.count = kHeight;
    assert(m_scalarFunction);
    m_scalarFunction(&args);

    args.dst = &m_dstSIMD.planeDesc;
    m_simdFunction(&args);

    const auto& params = GetParam();
    const auto compareByteSize = fixedPointByteSize(params.dstFP) * kStride * kHeight;
    EXPECT_EQ(memcmp(m_dstScalar.planeDesc.firstSample, m_dstSIMD.planeDesc.firstSample, compareByteSize), 0);
}

// -----------------------------------------------------------------------------

// Helper for printing a meaningful name for the test parameter
std::string CopyToString(const testing::TestParamInfo<ConvertTestParams>& value)
{
    const LdpFixedPoint srcFP = value.param.srcFP;
    const LdpFixedPoint dstFP = value.param.dstFP;

    std::stringstream ss;
    ss << fixedPointToString(srcFP) << "_to_" << fixedPointToString(dstFP);
    return ss.str();
}

// -----------------------------------------------------------------------------

const std::vector<LdpFixedPoint> kFixedPointUnsigned = {LdpFPU8, LdpFPU10, LdpFPU12, LdpFPU14};

const std::vector<LdpFixedPoint> kFixedPointAll = {LdpFPU8, LdpFPU10, LdpFPU12, LdpFPU14,
                                                   LdpFPS8, LdpFPS10, LdpFPS12, LdpFPS14};

// -----------------------------------------------------------------------------

const auto kCopyParams = rv::cartesian_product(kFixedPointAll, kFixedPointAll) |
                         rv::filter([](auto value) {
                             const LdpFixedPoint srcFP = std::get<0>(value);
                             const LdpFixedPoint dstFP = std::get<1>(value);
                             const bool srcSigned = fixedPointIsSigned(srcFP);
                             const bool dstSigned = fixedPointIsSigned(dstFP);
                             const auto srcDepth = bitdepthFromFixedPoint(srcFP);
                             const auto dstDepth = bitdepthFromFixedPoint(dstFP);

                             // Both signed is an identity copy, so omit them from perms
                             // since identities will be checked anyway.
                             const bool areBothSigned = srcSigned && dstSigned;

                             // Only perform tests for copies where the bit-depth is
                             // promoted, this is because we currently do not support
                             // bit-depth demotion.
                             const bool isDepthPromotion = dstDepth >= srcDepth;

                             return isDepthPromotion && !areBothSigned;
                         }) |
                         rv::transform([](auto value) {
                             return ConvertTestParams{std::get<0>(value), std::get<1>(value)};
                         }) |
                         rg::to_vector;

INSTANTIATE_TEST_SUITE_P(ConvertTests, ConvertTest, testing::ValuesIn(kCopyParams), CopyToString);

// -----------------------------------------------------------------------------
