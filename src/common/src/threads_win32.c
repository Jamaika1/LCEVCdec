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

#include <immintrin.h>
#include <LCEVC/common/diagnostics.h>
#include <LCEVC/common/log.h>
#include <LCEVC/common/platform.h>
#include <LCEVC/common/threads.h>
#include <stdlib.h>

#ifdef __MINGW32__
#ifdef __SIZEOF_INT128__
static uint64_t _div128(uint64_t hi, uint64_t lo, uint64_t divisor, int64_t* remainder)
{
    if (divisor == 0) {
        return 0;
    }
    __uint128_t dividend = ((__uint128_t)hi << 64) | lo;
    uint64_t quotient = dividend / divisor;
    if (remainder) {
        *remainder = (int64_t)(dividend % divisor);
    }
    return quotient;
}

static int64_t mulh(int64_t a, int64_t b)
{
    __int128 result = (__int128)a * (__int128)b;
    return (int64_t)(result >> 64);
}
#else

uint64_t _div128(uint64_t dividend_high, uint64_t dividend_low, uint64_t divisor, int64_t* remainder)
{
    // Handle division by zero
    if (divisor == 0) {
        if (remainder)
            *remainder = 0;
        return UINT64_MAX; // Return max value to indicate error
    }

    // Handle case where dividend < divisor
    if (dividend_high == 0 && dividend_low < divisor) {
        if (remainder)
            *remainder = dividend_low;
        return 0;
    }

    // Handle case where divisor is 1
    if (divisor == 1) {
        if (remainder)
            *remainder = 0;
        return dividend_low; // Note: assumes quotient fits in 64 bits
    }

    // Use long division algorithm for 128-bit / 64-bit division
    uint64_t quotient = 0;
    uint64_t current_remainder_high = dividend_high;
    uint64_t current_remainder_low = dividend_low;

    // Find the highest set bit in the dividend
    int shift = 0;
    if (dividend_high != 0) {
        shift = 63 - __builtin_clzll(dividend_high) + 64;
    } else if (dividend_low != 0) {
        shift = 63 - __builtin_clzll(dividend_low);
    } else {
        // Dividend is zero
        if (remainder)
            *remainder = 0;
        return 0;
    }

    // Perform long division bit by bit
    for (int i = shift; i >= 0; i--) {
        // Shift current remainder left by 1
        current_remainder_high = (current_remainder_high << 1) | (current_remainder_low >> 63);
        current_remainder_low = current_remainder_low << 1;

        // Check if we can subtract the divisor
        if (current_remainder_high > divisor ||
            (current_remainder_high == divisor && current_remainder_low >= 0)) {
            // Subtract divisor from current remainder
            if (current_remainder_low < divisor) {
                current_remainder_high--;
            }
            current_remainder_low -= divisor;
            current_remainder_high -= (current_remainder_high >= divisor) ? divisor : 0;

            // Set the corresponding bit in quotient
            if (i < 64) {
                quotient |= (1ULL << i);
            }
        }
    }

    if (remainder) {
        *remainder = (int64_t)current_remainder_low;
    }

    return quotient;
}

uint64_t mulh(uint64_t a, uint64_t b)
{
    // Handle zero cases
    if (a == 0 || b == 0) {
        return 0;
    }

    // Split each 64-bit number into two 32-bit parts
    uint64_t a_low = a & 0xFFFFFFFFULL;
    uint64_t a_high = a >> 32;
    uint64_t b_low = b & 0xFFFFFFFFULL;
    uint64_t b_high = b >> 32;

    // Compute partial products
    uint64_t low_low = a_low * b_low;
    uint64_t low_high = a_low * b_high;
    uint64_t high_low = a_high * b_low;
    uint64_t high_high = a_high * b_high;

    // Combine partial products
    // The full 128-bit result would be:
    // high_high << 64 + (low_high + high_low) << 32 + low_low

    uint64_t middle = low_high + high_low;
    uint64_t middle_carry = (middle < low_high) ? 1 : 0; // Check for overflow

    uint64_t low_part_carry = (low_low >> 32) + (middle & 0xFFFFFFFFULL);
    uint64_t final_carry = (low_part_carry >> 32);

    uint64_t result = high_high + (middle >> 32) + middle_carry + final_carry;

    return result;
}
#endif
#endif

int32_t threadNumCores(void)
{
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return (int32_t)sysinfo.dwNumberOfProcessors;
}

static DWORD WINAPI threadWrapper(LPVOID arg)
{
    Thread* thread = (Thread*)arg;
    thread->result = 0;
    thread->result = thread->function(thread->argument);
    return 0;
};

int threadCreate(Thread* thread, ThreadFunction function, void* argument)
{
    thread->function = function;
    thread->argument = argument;
    thread->thread = CreateThread(NULL, 0, threadWrapper, thread, 0, NULL);
    return (thread->thread != 0) ? ThreadResultSuccess : ThreadResultError;
}

void threadExit(void) { ExitThread(0); }

int threadJoin(Thread* thread, intptr_t* resultPtr)
{
    if (WaitForSingleObject(thread->thread, INFINITE) != WAIT_OBJECT_0) {
        return ThreadResultError;
    }

    if (resultPtr) {
        *resultPtr = thread->result;
    }

    return ThreadResultSuccess;
}

int threadSleep(uint32_t milliseconds)
{
    Sleep(milliseconds);
    return ThreadResultSuccess;
}

void threadSetPriority(Thread* thread, ThreadPriority priority)
{
    int prio = 0;

    switch (priority) {
        case ThreadPriorityHigh: prio = THREAD_PRIORITY_ABOVE_NORMAL; break;
        case ThreadPriorityNormal: prio = THREAD_PRIORITY_NORMAL; break;
        case ThreadPriorityIdle: prio = THREAD_PRIORITY_IDLE; break;
    }

    SetThreadPriority(thread->thread, prio);
}

void threadSetName(const char* name)
{
    if (!name) {
        return;
    }

    wchar_t wname[256];
    mbstowcs(wname, name, sizeof(wname) / sizeof(wname[1]) - 1);

    // This might not be available on Windows prior to Windows10
    const HRESULT hr = SetThreadDescription(GetCurrentThread(), wname);
    if (FAILED(hr)) {
        VNLogWarning("Cannot set thread name");
    }
}

// Scale from performance counter to microseconds.
//
// microSecondScale = 1000000 * 2^64 / performanceFrequency
//
//  is > 1000000 (Usually something like 10000000 or 24000000)
//
// Counter is then converted to uSecs by taking the high 64 bits of microSecondScale * counter
//
static uint64_t microSecondScale = 0;

uint64_t threadTimeMicroseconds(int32_t offset)
{
    if (offset == UINT32_MAX) {
        return UINT64_MAX;
    }

    if (microSecondScale == 0) {
        // One-time setup of scale
        uint64_t frequency;
        QueryPerformanceFrequency((LARGE_INTEGER*)&frequency);
        int64_t remainder;
        microSecondScale = _div128(1000000, 0, frequency, &remainder);
    }

    uint64_t counter;
    QueryPerformanceCounter((LARGE_INTEGER*)&counter);
#ifdef __MINGW32__
    return mulh(microSecondScale, counter) + offset;
#else
    return __mulh(microSecondScale, counter) + offset;
#endif
}

int threadCondVarWaitDeadline(ThreadCondVar* condVar, ThreadMutex* mutex, uint64_t deadline)
{
    if (deadline == UINT64_MAX) {
        SleepConditionVariableSRW(&condVar->condVar, &mutex->lock, INFINITE, 0);
        return ThreadResultSuccess;
    }

    uint64_t counter = {0};
    QueryPerformanceCounter((LARGE_INTEGER*)&counter);
#ifdef __MINGW32__
    const uint64_t usecs = mulh(microSecondScale, counter);
#else
    const uint64_t usecs = __mulh(microSecondScale, counter);
#endif

    if (usecs >= deadline) {
        // Expired already
        return ThreadResultTimeout;
    }

    // Round up remaining time to deadline to milliseconds
    const DWORD millis = (DWORD)((deadline - usecs + 999) / 1000);

    if (!SleepConditionVariableSRW(&condVar->condVar, &mutex->lock, millis, 0)) {
        return ThreadResultTimeout;
    }

    return ThreadResultSuccess;
}
