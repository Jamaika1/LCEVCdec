/* Copyright (c) V-Nova International Limited 2025. All rights reserved.
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
#include <LCEVC/common/free_pool.hpp>
//
#include <fmt/core.h>
#include <gtest/gtest.h>
//
#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

using namespace lcevc_dec::common;

TEST(FreePool, CreateDefault)
{
    FreePool<int32_t> pool{42};

    EXPECT_EQ(pool.capacity(), 42);
    EXPECT_EQ(pool.size(), 42);
}

TEST(FreePool, CreateWithAllocator)
{
    FreePool<int32_t> pool{24, ldcMemoryAllocatorMalloc()};

    EXPECT_EQ(pool.capacity(), 24);
    EXPECT_EQ(pool.size(), 24);
}

TEST(FreePool, AllocAndFree)
{
    FreePool<int32_t> pool{10};

    int32_t* p = pool.allocate();
    ASSERT_NE(p, nullptr);

    EXPECT_EQ(pool.size(), 9);

    pool.release(p);
    EXPECT_EQ(pool.size(), 10);
}

// A helper class with non-trivial ctor/dtor to verify construction & destruction.
struct Counter
{
    static inline int ctor_count = 0;
    static inline int dtor_count = 0;
    int value = 0;

    explicit Counter(int v = 0)
        : value(v)
    {
        ++ctor_count;
    }
    ~Counter() { ++dtor_count; }

    Counter(const Counter&) = delete;
    Counter& operator=(const Counter&) = delete;
};

TEST(FreePool, BasicAllocateFreeInt)
{
    FreePool<int> pool(8);
    // Initially we should have 8 free slots.
    EXPECT_EQ(pool.size(), 8u);
    EXPECT_EQ(pool.capacity(), 8u);

    // Allocate one slot; size (free count) should decrease.
    int* p = pool.allocate();
    ASSERT_NE(p, nullptr);
    *p = 42;
    EXPECT_EQ(pool.size(), 7u);
    EXPECT_EQ(*p, 42);

    // Free it; free-list size increases.
    pool.release(p);
    EXPECT_EQ(pool.size(), 8u);
}

TEST(FreePool, LIFOReuse)
{
    FreePool<int> pool(2);
    int* p1 = pool.allocate();
    int* p2 = pool.allocate();

    // Now free-list should be empty; next allocate will grow capacity.
    EXPECT_EQ(pool.size(), 0u);
    EXPECT_EQ(pool.capacity(), 2u);

    // Free p2 then p1, expect LIFO reuse: next allocations return p1, then p2.
    pool.release(p2);
    pool.release(p1);
    EXPECT_EQ(pool.size(), 2u);

    int* r1 = pool.allocate();
    int* r2 = pool.allocate();
    EXPECT_EQ(r1, p1);
    EXPECT_EQ(r2, p2);
}

TEST(FreePool, NonTrivialTypeWithPlacementNew)
{
    Counter::ctor_count = 0;
    Counter::dtor_count = 0;

    FreePool<Counter> pool(4);

    // Allocate and construct with placement new
    Counter* a = pool.allocate();
    new (a) Counter(10);
    Counter* b = pool.allocate();
    new (b) Counter(20);
    Counter* c = pool.allocate();
    new (c) Counter(30);

    EXPECT_EQ(Counter::ctor_count, 3);
    EXPECT_EQ(a->value, 10);
    EXPECT_EQ(b->value, 20);
    EXPECT_EQ(c->value, 30);

    // Free should call destructor
    pool.destroy(b);
    pool.destroy(a);
    pool.destroy(c);

    EXPECT_EQ(Counter::dtor_count, 3);
    EXPECT_EQ(pool.size(), 4u);
}

TEST(FreePool, StressMixedOpsVariousTypes)
{
    // int
    {
        FreePool<int> pool(5000);
        std::vector<int*> alive;
        std::mt19937 rng(123);
        for (int i = 0; i < 5000; ++i) {
            if (alive.empty() || (rng() % 3 != 0)) {
                int* p = pool.allocate();
                ASSERT_NE(p, nullptr);
                *p = i;
                alive.push_back(p);
            } else {
                size_t idx = rng() % alive.size();
                pool.release(alive[idx]);
                alive.erase(alive.begin() + idx);
            }
        }
        // Clean up
        for (auto* p : alive) {
            pool.release(p);
        }
        EXPECT_EQ(pool.size(), pool.capacity());
    }

    // double
    {
        FreePool<double> pool(100);
        std::vector<double*> v;
        for (int i = 0; i < 100; ++i) {
            double* p = pool.allocate();
            ASSERT_NE(p, nullptr);

            *p = i * 0.5;
            v.push_back(p);
        }
        std::shuffle(v.begin(), v.end(), std::mt19937{321});
        for (double* p : v) {
            pool.release(p);
        }
        EXPECT_EQ(pool.size(), pool.capacity());
    }

    // Non-trivial type using make()
    {
        FreePool<Counter> pool(200);
        std::vector<Counter*> objs;
        for (int i = 0; i < 200; ++i) {
            auto p = pool.make(i);
            ASSERT_NE(p, nullptr);
            objs.push_back(p);
        }
        EXPECT_EQ(Counter::ctor_count >= 200, true);

        // random order free
        std::shuffle(objs.begin(), objs.end(), std::mt19937{999});
        for (auto* p : objs) {
            pool.destroy(p);
        }
        EXPECT_EQ(pool.size(), pool.capacity());
        EXPECT_EQ(Counter::dtor_count >= 200, true);
    }
}
