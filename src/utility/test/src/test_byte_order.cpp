// Copyright (c) V-Nova International Limited 2023. All rights reserved.
//
#include "LCEVC/utility/byte_order.h"

#include <gtest/gtest.h>

TEST(ByteOrder, ReadLittleEndian)
{
    {
        std::istringstream is("xxx");
        uint32_t v = 0;
        EXPECT_FALSE(lcevc_dec::utility::readLittleEndian(is, v));
    }
    {
        std::istringstream is("");
        uint32_t v = 0;
        EXPECT_FALSE(lcevc_dec::utility::readLittleEndian(is, v));
    }

    {
        std::istringstream is("0123");
        uint32_t v = 0;
        EXPECT_TRUE(lcevc_dec::utility::readLittleEndian(is, v) && v == 0x33323130);
    }

    {
        std::istringstream is("01234567");
        uint32_t v1 = 0;
        uint32_t v2 = 0;
        EXPECT_TRUE(lcevc_dec::utility::readLittleEndian(is, v1) && v1 == 0x33323130);
        EXPECT_TRUE(lcevc_dec::utility::readLittleEndian(is, v2) && v2 == 0x37363534);
    }

    {
        std::istringstream is("01234567");
        uint8_t v1 = 0;
        uint16_t v2 = 0;
        uint32_t v3 = 0;
        uint8_t v4 = 0;
        EXPECT_TRUE(lcevc_dec::utility::readLittleEndian(is, v1) && v1 == 0x30);
        EXPECT_TRUE(lcevc_dec::utility::readLittleEndian(is, v2) && v2 == 0x3231);
        EXPECT_TRUE(lcevc_dec::utility::readLittleEndian(is, v3) && v3 == 0x36353433);
        EXPECT_TRUE(lcevc_dec::utility::readLittleEndian(is, v4) && v4 == 0x37);
    }
}

TEST(ByteOrder, ReadBigEndian)
{
    {
        std::istringstream is("xxx");
        uint32_t v = 0;
        EXPECT_FALSE(lcevc_dec::utility::readBigEndian(is, v));
    }
    {
        std::istringstream is("");
        uint32_t v = 0;
        EXPECT_FALSE(lcevc_dec::utility::readBigEndian(is, v));
    }

    {
        std::istringstream is("0123");
        uint32_t v = 0;
        EXPECT_TRUE(lcevc_dec::utility::readBigEndian(is, v) && v == 0x30313233);
    }

    {
        std::istringstream is("01234567");
        uint32_t v1 = 0;
        uint32_t v2 = 0;
        EXPECT_TRUE(lcevc_dec::utility::readBigEndian(is, v1) && v1 == 0x30313233);
        EXPECT_TRUE(lcevc_dec::utility::readBigEndian(is, v2) && v2 == 0x34353637);
    }

    {
        std::istringstream is("01234567");
        uint8_t v1 = 0;
        uint16_t v2 = 0;
        uint32_t v3 = 0;
        uint8_t v4 = 0;
        EXPECT_TRUE(lcevc_dec::utility::readBigEndian(is, v1) && v1 == 0x30);
        EXPECT_TRUE(lcevc_dec::utility::readBigEndian(is, v2) && v2 == 0x3132);
        EXPECT_TRUE(lcevc_dec::utility::readBigEndian(is, v3) && v3 == 0x33343536);
        EXPECT_TRUE(lcevc_dec::utility::readBigEndian(is, v4) && v4 == 0x37);
    }
}

TEST(ByteOrder, WriteLittleEndian)
{
    {
        std::ostringstream os;
        EXPECT_TRUE(lcevc_dec::utility::writeLittleEndian(os, 0x33323130));
        EXPECT_EQ(os.str(), "0123");
    }
    {
        std::ostringstream os;
        EXPECT_TRUE(lcevc_dec::utility::writeLittleEndian(os, 0x33323130));
        EXPECT_TRUE(lcevc_dec::utility::writeLittleEndian(os, 0x37363534));
        EXPECT_EQ(os.str(), "01234567");
    }

    {
        std::ostringstream os;
        EXPECT_TRUE(lcevc_dec::utility::writeLittleEndian(os, static_cast<uint8_t>(0x30)));
        EXPECT_TRUE(lcevc_dec::utility::writeLittleEndian(os, static_cast<uint16_t>(0x3231)));
        EXPECT_TRUE(lcevc_dec::utility::writeLittleEndian(os, static_cast<uint32_t>(0x36353433)));
        EXPECT_TRUE(lcevc_dec::utility::writeLittleEndian(os, static_cast<uint8_t>(0x37)));

        EXPECT_EQ(os.str(), "01234567");
    }
}

TEST(ByteOrder, WriteBigEndian)
{
    {
        std::ostringstream os;
        EXPECT_TRUE(lcevc_dec::utility::writeBigEndian(os, 0x33323130));
        EXPECT_EQ(os.str(), "3210");
    }
    {
        std::ostringstream os;
        EXPECT_TRUE(lcevc_dec::utility::writeBigEndian(os, 0x33323130));
        EXPECT_TRUE(lcevc_dec::utility::writeBigEndian(os, 0x37363534));
        EXPECT_EQ(os.str(), "32107654");
    }

    {
        std::ostringstream os;
        EXPECT_TRUE(lcevc_dec::utility::writeBigEndian(os, (uint8_t)0x30));
        EXPECT_TRUE(lcevc_dec::utility::writeBigEndian(os, (uint16_t)0x3231));
        EXPECT_TRUE(lcevc_dec::utility::writeBigEndian(os, (uint32_t)0x36353433));
        EXPECT_TRUE(lcevc_dec::utility::writeBigEndian(os, (uint8_t)0x37));

        EXPECT_EQ(os.str(), "02165437");
    }
}
