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

#include <LCEVC/common/cpp_tools.h>
//
#include <fmt/core.h>
#include <gtest/gtest.h>

TEST(CppTools, Concat)
{
#define FirstSecond 42
    int x = VNConcat(First, Second);
    EXPECT_EQ(x, 42);
}

#define OP(idx, arg) "[" #arg "]"
#define SEP() ","
#define PFX() "Prefix:"

TEST(CppTools, ForEach)
{
    const std::string s0 = "" VNForEach(OP, PFX, SEP);
    EXPECT_EQ(s0, "");

    const std::string s1 = VNForEach(OP, PFX, SEP, A);
    EXPECT_EQ(s1, "Prefix:[A]");

    const std::string s2 = VNForEach(OP, PFX, SEP, A, B);
    EXPECT_EQ(s2, "Prefix:[A],[B]");

    const std::string s3 = VNForEach(OP, PFX, SEP, A, B, C);
    EXPECT_EQ(s3, "Prefix:[A],[B],[C]");

    const std::string s4 = VNForEach(OP, PFX, SEP, A, B, C, D);
    EXPECT_EQ(s4, "Prefix:[A],[B],[C],[D]");

    const std::string s5 = VNForEach(OP, PFX, SEP, A, B, C, D, E);
    EXPECT_EQ(s5, "Prefix:[A],[B],[C],[D],[E]");

    const std::string s6 = VNForEach(OP, PFX, SEP, A, B, C, D, E, F);
    EXPECT_EQ(s6, "Prefix:[A],[B],[C],[D],[E],[F]");

    const std::string s7 = VNForEach(OP, PFX, SEP, A, B, C, D, E, F, G);
    EXPECT_EQ(s7, "Prefix:[A],[B],[C],[D],[E],[F],[G]");

    const std::string s8 = VNForEach(OP, PFX, SEP, A, B, C, D, E, F, G, H);
    EXPECT_EQ(s8, "Prefix:[A],[B],[C],[D],[E],[F],[G],[H]");

    const std::string s9 = VNForEach(OP, PFX, SEP, A, B, C, D, E, F, G, H, I);
    EXPECT_EQ(s9, "Prefix:[A],[B],[C],[D],[E],[F],[G],[H],[I]");

    const std::string s10 = VNForEach(OP, PFX, SEP, A, B, C, D, E, F, G, H, I, J);
    EXPECT_EQ(s10, "Prefix:[A],[B],[C],[D],[E],[F],[G],[H],[I],[J]");
}

#define OP1(idx, arg) arg
#define SEP1() ,
#define PFX1()

TEST(CppTools, ForEachComma)
{
    constexpr int a[] = {VNForEach(OP1, PFX1, SEP1, 10, 20)};
    EXPECT_EQ(sizeof(a) / sizeof(a[0]), 2);
    EXPECT_EQ(a[0], 10);
    EXPECT_EQ(a[1], 20);
}

#define OP2(idx, arg0, arg1) "[" #arg0 "," #arg1 "]"
#define SEP2() "+"
#define PFX2() "#"

TEST(CppTools, ForEachPair)
{
    const std::string s0 = "" VNForEachPair(OP2, PFX2, SEP2);
    EXPECT_EQ(s0, "");

    const std::string s1 = VNForEachPair(OP2, PFX2, SEP2, A, B);
    EXPECT_EQ(s1, "#[A,B]");

    const std::string s2 = VNForEachPair(OP2, PFX2, SEP2, A, B, C, D);
    EXPECT_EQ(s2, "#[A,B]+[C,D]");

    const std::string s3 = VNForEachPair(OP2, PFX2, SEP2, A, B, C, D, E, F);
    EXPECT_EQ(s3, "#[A,B]+[C,D]+[E,F]");

    const std::string s4 = VNForEachPair(OP2, PFX2, SEP2, A, B, C, D, E, F, G, H);
    EXPECT_EQ(s4, "#[A,B]+[C,D]+[E,F]+[G,H]");

    const std::string s5 = VNForEachPair(OP2, PFX2, SEP2, A, B, C, D, E, F, G, H, I, J);
    EXPECT_EQ(s5, "#[A,B]+[C,D]+[E,F]+[G,H]+[I,J]");

    const std::string s6 = VNForEachPair(OP2, PFX2, SEP2, A, B, C, D, E, F, G, H, I, J, K, L);
    EXPECT_EQ(s6, "#[A,B]+[C,D]+[E,F]+[G,H]+[I,J]+[K,L]");

    const std::string s7 = VNForEachPair(OP2, PFX2, SEP2, A, B, C, D, E, F, G, H, I, J, K, L, M, N);
    EXPECT_EQ(s7, "#[A,B]+[C,D]+[E,F]+[G,H]+[I,J]+[K,L]+[M,N]");

    const std::string s8 = VNForEachPair(OP2, PFX2, SEP2, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P);
    EXPECT_EQ(s8, "#[A,B]+[C,D]+[E,F]+[G,H]+[I,J]+[K,L]+[M,N]+[O,P]");

    const std::string s9 =
        VNForEachPair(OP2, PFX2, SEP2, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R);
    EXPECT_EQ(s9, "#[A,B]+[C,D]+[E,F]+[G,H]+[I,J]+[K,L]+[M,N]+[O,P]+[Q,R]");

    const std::string s10 =
        VNForEachPair(OP2, PFX2, SEP2, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T);
    EXPECT_EQ(s10, "#[A,B]+[C,D]+[E,F]+[G,H]+[I,J]+[K,L]+[M,N]+[O,P]+[Q,R]+[S,T]");
}
