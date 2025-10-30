/* Copyright (c) V-Nova International Limited 2024-2025. All rights reserved.
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

#include <LCEVC/common/diagnostics.h>
#include <LCEVC/common/memory.h>
#include <LCEVC/common/simple_allocator.h>
#include <LCEVC/utility/md5.h>
//
#include <fmt/core.h>
#include <gtest/gtest.h>
//
#include <climits>

// Some specific tests for the SimpleAllocator that check the internal state of the
// allocator for correct chunk splitting and merging.
//
class SimpleAllocatorTest : public testing::Test
{
public:
    SimpleAllocatorTest()
        : runtimeAllocator(ldcMemoryAllocatorMalloc())
    {
        allocator = ldcMemorySimpleAllocatorInitialize(&simpleAllocator, runtimeAllocator);
    }

    ~SimpleAllocatorTest() override {}

    void checkClean() const { EXPECT_TRUE(ldcMemoryAllocatorSimpleDebugCheck(&simpleAllocator)); }

    size_t blockCount() const
    {
        const LdcLinkedList* list = &simpleAllocator.blockList;
        const LdcLinkedNode* tailSentinel = (const LdcLinkedNode*)&list->tail;
        size_t count = 0;
        for (const LdcLinkedNode* node = list->head; node != tailSentinel; node = node->next) {
            ++count;
        }
        return count;
    }

    LdcMemoryAllocator* runtimeAllocator;
    LdcMemorySimpleAllocator simpleAllocator;
    LdcMemoryAllocator* allocator;

    SimpleAllocatorTest(const SimpleAllocatorTest&) = delete;
    SimpleAllocatorTest(SimpleAllocatorTest&&) = delete;
    void operator=(const SimpleAllocatorTest&) = delete;
    void operator=(SimpleAllocatorTest&&) = delete;
};

TEST_F(SimpleAllocatorTest, FreedChunkIsReused)
{
    const size_t initialBlocks = blockCount();

    LdcMemoryAllocation allocA = {};
    LdcMemoryAllocation allocB = {};
    LdcMemoryAllocation allocC = {};

    VNAllocateArray(allocator, &allocA, uint8_t, 512, "");
    VNAllocateArray(allocator, &allocB, uint8_t, 512, "");
    VNAllocateArray(allocator, &allocC, uint8_t, 512, "");

    EXPECT_GE(blockCount(), initialBlocks == 0 ? 1u : initialBlocks);

    void* freedPtr = allocB.ptr;

    VNFree(allocator, &allocB);

    LdcMemoryAllocation allocD = {};
    VNAllocateArray(allocator, &allocD, uint8_t, 512, "");
    EXPECT_EQ(allocD.ptr, freedPtr);

    VNFree(allocator, &allocA);
    VNFree(allocator, &allocC);
    VNFree(allocator, &allocD);

    checkClean();
}

TEST_F(SimpleAllocatorTest, AdjacentFreesCoalesce)
{
    LdcMemoryAllocation allocA = {};
    LdcMemoryAllocation allocB = {};
    LdcMemoryAllocation allocC = {};

    VNAllocateArray(allocator, &allocA, uint8_t, 512, "");
    VNAllocateArray(allocator, &allocB, uint8_t, 512, "");
    VNAllocateArray(allocator, &allocC, uint8_t, 512, "");

    void* frontPtr = allocA.ptr;

    VNFree(allocator, &allocB);
    VNFree(allocator, &allocA);

    LdcMemoryAllocation merged = {};
    VNAllocateArray(allocator, &merged, uint8_t, 1024, "");
    EXPECT_EQ(merged.ptr, frontPtr);

    VNFree(allocator, &merged);
    VNFree(allocator, &allocC);

    checkClean();
}

TEST_F(SimpleAllocatorTest, ReallocateExtendsInPlace)
{
    LdcMemoryAllocation allocA = {};
    LdcMemoryAllocation allocB = {};

    VNAllocateArray(allocator, &allocA, uint8_t, 512, "");
    VNAllocateArray(allocator, &allocB, uint8_t, 512, "");

    void* originalPtr = allocA.ptr;

    VNFree(allocator, &allocB);

    VNReallocateArray(allocator, &allocA, uint8_t, 1024, "");
    EXPECT_EQ(allocA.ptr, originalPtr);
    EXPECT_EQ(allocA.size, 1024u);

    VNFree(allocator, &allocA);

    checkClean();
}

TEST_F(SimpleAllocatorTest, ReallocateMovesWhenBlocked)
{
    LdcMemoryAllocation allocA = {};
    LdcMemoryAllocation allocB = {};

    VNAllocateArray(allocator, &allocA, uint8_t, 512, "");
    uint8_t* originalBytes = static_cast<uint8_t*>(allocA.ptr);
    for (size_t i = 0; i < 512; ++i) {
        originalBytes[i] = static_cast<uint8_t>(i & 0xFF);
    }

    VNAllocateArray(allocator, &allocB, uint8_t, 8192, "");

    void* originalPtr = allocA.ptr;

    VNReallocateArray(allocator, &allocA, uint8_t, 8192, "");
    ASSERT_TRUE(VNIsAllocated(allocA));
    EXPECT_NE(allocA.ptr, originalPtr);
    EXPECT_EQ(allocA.size, 8192u);

    uint8_t* reallocatedBytes = static_cast<uint8_t*>(allocA.ptr);
    for (size_t i = 0; i < 512; ++i) {
        EXPECT_EQ(reallocatedBytes[i], static_cast<uint8_t>(i & 0xFF));
    }

    VNFree(allocator, &allocA);
    VNFree(allocator, &allocB);

    checkClean();
}
