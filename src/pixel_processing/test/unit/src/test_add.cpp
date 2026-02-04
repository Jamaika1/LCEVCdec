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

#include "add_common.h"
#include "fp_types.h"
#include "test_plane.h"

#include <gtest/gtest.h>
#include <LCEVC/common/acceleration.h>
#include <LCEVC/pixel_processing/convert.h>
#include <range/v3/view.hpp>
#include <rng.h>

#include <functional>
#include <random>
#include <sstream>

extern "C"
{
PlaneAddFunction planeAddGetFunction(LdpFixedPoint dstFP);
}

namespace rg = ranges;
namespace rv = ranges::views;

// -----------------------------------------------------------------------------

constexpr uint32_t kWidth = 500;
constexpr uint32_t kHeight = 400;
constexpr uint32_t kStride = 512;

// -----------------------------------------------------------------------------

class AddTest : public testing::TestWithParam<LdpFixedPoint>
{
protected:
    void SetUp() override
    {
        const auto& param = GetParam();
        ldcAccelerationInitialize(false);
        m_scalarFunction = planeAddGetFunction(param);
        ldcAccelerationInitialize(true);
        m_simdFunction = planeAddGetFunction(param);

        m_src1.initialize(kWidth, kHeight, kStride, LdpFPS8);
        m_src2.initialize(kWidth, kHeight, kStride, LdpFPS8);
        m_dstScalar.initialize(kWidth, kHeight, kStride, param);
        m_dstSIMD.initialize(kWidth, kHeight, kStride, param);
    }

    TestPlane m_src1{};
    TestPlane m_src2{};
    TestPlane m_dstScalar{};
    TestPlane m_dstSIMD{};

    PlaneAddFunction m_scalarFunction{};
    PlaneAddFunction m_simdFunction{};
};

// -----------------------------------------------------------------------------

TEST_P(AddTest, CompareSIMD)
{
    fillPlaneWithNoise(m_src1);
    fillPlaneWithNoise(m_src2);

    LdppAddArgs args;
    args.src1 = &m_src1.planeDesc;
    args.src2 = &m_src2.planeDesc;
    args.dst = &m_dstScalar.planeDesc;
    args.minWidth = kWidth;
    args.offset = 0;
    args.count = kHeight;
    m_scalarFunction(&args);

    args.dst = &m_dstSIMD.planeDesc;
    m_simdFunction(&args);

    const auto& param = GetParam();
    const auto compareByteSize = fixedPointByteSize(param) * kStride * kHeight;
    EXPECT_EQ(memcmp(m_dstScalar.planeDesc.firstSample, m_dstSIMD.planeDesc.firstSample, compareByteSize), 0);
}

// -----------------------------------------------------------------------------

std::string AddToString(const testing::TestParamInfo<LdpFixedPoint>& value)
{
    return fixedPointToString(value.param);
}

// -----------------------------------------------------------------------------

const std::vector<LdpFixedPoint> kFixedPointUnsigned = {LdpFPU8, LdpFPU10, LdpFPU12, LdpFPU14};

// -----------------------------------------------------------------------------

const auto kAddParams =
    kFixedPointUnsigned | rv::transform([](auto value) { return value; }) | rg::to_vector;

INSTANTIATE_TEST_SUITE_P(AddTests, AddTest, testing::ValuesIn(kAddParams), AddToString);

// -----------------------------------------------------------------------------
