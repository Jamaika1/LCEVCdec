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

#include "string_format.h"
//
#include <LCEVC/common/check.h>
#include <LCEVC/common/diagnostics.h>
#include <LCEVC/common/diagnostics_buffer.h>
#include <LCEVC/common/platform.h>
#include <LCEVC/common/printf_macros.h>
#include <LCEVC/common/threads.h>
//
#include <assert.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>

#if !VN_OS(WINDOWS)
#include <time.h>
#endif

#define kCapacity 1024
#define kVarDataSize 1024 * 1024

#define kMaxRecordVarData 1024

// Pointer to state to use - either allocated with this address space, or
// passed in from parent library.
//
DiagnosticState* ldcDiagnosticsState = NULL;

// Local diagnostic state if none provided by parent
static DiagnosticState localState;
static LdcAtomicFlag localStateLock = ATOMIC_FLAG_INIT;

static void localDiagnosticsLock(void)
{
    while (atomic_flag_test_and_set_explicit(&localStateLock, memory_order_acquire)) {
        threadYield();
    }
}

static void localDiagnosticsUnlock(void)
{
    atomic_flag_clear_explicit(&localStateLock, memory_order_release);
}

// Special site used to mark scoped trace ends
const LdcDiagSite ldcDiagnosticsTraceScopedEndSite = {LdcDiagTypeTraceEnd};

// Apply all registered handlers to the given record - starting with most recently registered
//
static void applyDiagnosticsHandlers(const LdcDiagSite* site, const LdcDiagRecord* record,
                                     const LdcDiagValue* values)
{
    for (uint32_t i = ldcDiagnosticsState->handlersCount; i-- != 0;) {
        if (ldcDiagnosticsState->handlers[i].handler(ldcDiagnosticsState->handlers[i].userData,
                                                     site, record, values)) {
            break;
        }
    }
}

#if VN_SDK_FEATURE(DIAGNOSTICS_ASYNC)

// Special site used to mark a flush in buffer
static const LdcDiagSite diagnosticsFlushSite = {LdcDiagTypeFlush};

static intptr_t diagnosticThreadFn(void* argument)
{
    DiagnosticState* state = (DiagnosticState*)argument;
    assert(state);

    while (true) {
        // Pop record from buffer (diagnostic buffer is thread safe)
        LdcDiagRecord rec;
        LdcDiagValue values[kMaxRecordVarData / sizeof(LdcDiagValue)];
        size_t varDataSize = 0;
        ldcDiagnosticsBufferPop(&state->diagnosticsBuffer, &rec, (uint8_t*)&values, sizeof(values),
                                &varDataSize);

        if (rec.site == NULL) {
            // Shutdown from ldcDiagnosticsRelease
            break;
        }

        // All the handler work is now done under the state mutex
        threadMutexLock(&state->mutex);

        applyDiagnosticsHandlers(rec.site, &rec, values);

        if (rec.site == &diagnosticsFlushSite && state->flushCount > 0) {
            state->flushCount--;
            if (state->flushCount == 0) {
                threadCondVarSignal(&state->flushed);
            }
        }

        threadMutexUnlock(&state->mutex);
    }

    return 0;
}
#endif

void ldcDiagnosticsInitialize(void* parentState)
{
    localDiagnosticsLock();

    DiagnosticState* state = ldcDiagnosticsState;
    if (state == NULL) {
        state = parentState ? (DiagnosticState*)parentState : &localState;
    }

    const unsigned int previousRefCount =
        atomic_fetch_add_explicit(&state->refCount, 1, memory_order_acq_rel);
    ldcDiagnosticsState = state;
    localDiagnosticsUnlock();
    if (previousRefCount != 0) {
        return;
    }

#if VN_OS(WINDOWS)
    QueryPerformanceFrequency(&state->performanceCounterFrequency);
#endif

#if VN_SDK_FEATURE(DIAGNOSTICS_ASYNC)
    state->flushCount = 0;
    ldcDiagnosticsBufferInitialize(&state->diagnosticsBuffer, kCapacity, kVarDataSize,
                                   ldcMemoryAllocatorMalloc());
    threadMutexInitialize(&state->mutex);
    threadMutexLock(&state->mutex);
    threadCondVarInitialize(&state->flushed);
    VNCheck(threadCreate(&state->thread, diagnosticThreadFn, (void*)state) == 0);
#endif

    state->handlersCount = 0;
    state->maxLogLevel = LdcLogLevelInfo;
    state->initialized = true;

#if VN_SDK_FEATURE(DIAGNOSTICS_ASYNC)
    threadMutexUnlock(&state->mutex);
#endif
}

void ldcDiagnosticsRelease(void)
{
    localDiagnosticsLock();

    DiagnosticState* state = ldcDiagnosticsState;
    if (state == NULL) {
        localDiagnosticsUnlock();
        return;
    }

    const unsigned int previousRefCount =
        atomic_fetch_sub_explicit(&state->refCount, 1, memory_order_acq_rel);

    if (previousRefCount > 1) {
        localDiagnosticsUnlock();
        return;
    }

    if (previousRefCount == 0) {
        atomic_store_explicit(&state->refCount, 0, memory_order_release);
        ldcDiagnosticsState = NULL;
        localDiagnosticsUnlock();
        return;
    }

    assert(state->initialized);

#if VN_SDK_FEATURE(DIAGNOSTICS_ASYNC)
    threadMutexLock(&state->mutex);
#endif

    // Mark as not initialized so that any output from below does not get generated
    state->initialized = false;

    state->handlersCount = 0;
    for (uint32_t i = 0; i < VNDiagnosticsMaxHandlers; ++i) {
        state->handlers[i].handler = NULL;
        state->handlers[i].userData = NULL;
    }

#if VN_SDK_FEATURE(DIAGNOSTICS_ASYNC)
    // Close down thread by sending a null entry, and waiting for it to finish.
    LdcDiagRecord rec = {0};
    ldcDiagnosticsBufferPush(&state->diagnosticsBuffer, &rec, NULL, 0);
    threadMutexUnlock(&state->mutex);
    threadJoin(&state->thread, NULL);

    // Clear up
    threadCondVarDestroy(&state->flushed);
    threadMutexDestroy(&state->mutex);
    ldcDiagnosticsBufferDestroy(&state->diagnosticsBuffer);
#endif

    ldcDiagnosticsState = NULL;
    localDiagnosticsUnlock();
}

void ldcDiagnosticsFlush(void)
{
    assert(ldcDiagnosticsState);
    assert(ldcDiagnosticsState->initialized);

#if VN_SDK_FEATURE(DIAGNOSTICS_ASYNC)
    threadMutexLock(&ldcDiagnosticsState->mutex);

    // Mark flush as pending
    ldcDiagnosticsState->flushCount++;

    // Push 'flush' record into buffer
    LdcDiagRecord rec = {0};
    rec.site = &diagnosticsFlushSite;
    ldcDiagnosticsBufferPush(&ldcDiagnosticsState->diagnosticsBuffer, &rec, NULL, 0);

    while (ldcDiagnosticsState->flushCount != 0) {
        threadCondVarWait(&ldcDiagnosticsState->flushed, &ldcDiagnosticsState->mutex);
    }

    threadMutexUnlock(&ldcDiagnosticsState->mutex);
#endif
}

void ldcDiagnosticsLogLevel(LdcLogLevel maxLevel)
{
    assert(maxLevel < LdcLogLevelCount);

    ldcDiagnosticsState->maxLogLevel = maxLevel;
}

bool ldcDiagnosticsHandlerPush(LdcDiagHandler* handler, void* userData)
{
    if (!ldcDiagnosticsState || ldcDiagnosticsState->handlersCount >= VNDiagnosticsMaxHandlers) {
        // Too many handlers
        assert(0);
        return false;
    }

    ldcDiagnosticsState->handlers[ldcDiagnosticsState->handlersCount].handler = handler;
    ldcDiagnosticsState->handlers[ldcDiagnosticsState->handlersCount].userData = userData;
    ldcDiagnosticsState->handlersCount++;
    return true;
}

bool ldcDiagnosticsHandlerPop(LdcDiagHandler* handler, void** userData)
{
    if (!ldcDiagnosticsState || ldcDiagnosticsState->handlersCount == 0) {
        // No more handlers
        return false;
    }

    ldcDiagnosticsState->handlersCount--;
    // Check handler matches
    //
    if (handler != ldcDiagnosticsState->handlers[ldcDiagnosticsState->handlersCount].handler) {
        return false;
    }

    // Hand back the userData
    if (userData) {
        *userData = ldcDiagnosticsState->handlers[ldcDiagnosticsState->handlersCount].userData;
    }

    // Clear handler from stack
    ldcDiagnosticsState->handlers[ldcDiagnosticsState->handlersCount].handler = 0;
    ldcDiagnosticsState->handlers[ldcDiagnosticsState->handlersCount].userData = 0;
    return true;
}

// Convert a log record to string
//
int ldcDiagnosticFormatLog(char* dst, uint32_t dstSize, const LdcDiagSite* site,
                           const LdcDiagRecord* record, const LdcDiagValue* values)
{
    return (int)ldcFormat(dst, dstSize, site->str, site->argumentTypes, values, site->argumentCount);
}

// Convert a tracing or metric record to Perfetto JSON
//
int ldcDiagnosticFormatJson(char* dst, uint32_t dstSize, const LdcDiagSite* site,
                            const LdcDiagRecord* record, uint32_t processId)
{
    int numChars = 0;
    char valueBuffer[32];
    double microSeconds = ((double)record->timestamp) / 1000.0;
#define TSFMT ".3f"

    switch (site->type) {
        case LdcDiagTypeNone:
        case LdcDiagTypeFlush: break;

        // Logging - does not appear in trace
        case LdcDiagTypeLog:
        case LdcDiagTypeLogFormatted: break;

        // Tracing
        case LdcDiagTypeTraceBegin:
            numChars = snprintf(dst, dstSize,
                                "{\"ph\":\"B\", \"ts\":%" TSFMT
                                ", \"pid\":%u, \"tid\":%u, \"name\":\"%s\"},\n",
                                microSeconds, processId, record->threadId, site->str);
            break;
        case LdcDiagTypeTraceEnd:
            numChars = snprintf(dst, dstSize, "{\"ph\":\"E\", \"ts\":%" TSFMT ", \"pid\":%u, \"tid\":%u},\n",
                                microSeconds, processId, record->threadId);
            break;

        case LdcDiagTypeTraceInstant:
            numChars = snprintf(dst, dstSize,
                                "{\"ph\":\"i\", \"ts\":%" TSFMT
                                ",\"pid\":%u, \"tid\":%u, \"s\":\"g\", \"name\":\"%s\"},\n",
                                microSeconds, processId, record->threadId, site->str);
            break;

        // Async events
        case LdcDiagTypeTraceAsyncBegin:
            numChars = snprintf(dst, dstSize,
                                "{\"ph\":\"b\", \"ts\":%" TSFMT
                                ", \"pid\":%u, \"tid\":%u, \"name\":\"%s\", \"id\":%" PRIu64 "},\n",
                                microSeconds, processId, record->threadId, site->str, record->value.id);
            break;
        case LdcDiagTypeTraceAsyncEnd:
            numChars = snprintf(dst, dstSize,
                                "{\"ph\":\"e\", \"ts\":%" TSFMT
                                ", \"pid\":%u, \"tid\":%u, \"name\":\"%s\", \"id\":%" PRIu64 "},\n",
                                microSeconds, processId, record->threadId, site->str, record->value.id);
            break;
        case LdcDiagTypeTraceAsyncInstant:
            numChars = snprintf(dst, dstSize,
                                "{\"ph\":\"n\", \"ts\":%" TSFMT
                                ", \"pid\":%u, \"tid\":%u, \"name\":\"%s\", \"id\":%" PRIu64 "},\n",
                                microSeconds, processId, record->threadId, site->str, record->value.id);
            break;

        // Metrics
        case LdcDiagTypeMetric:

            switch (site->valueType) {
                case LdcDiagArgInt32:
                    snprintf(valueBuffer, sizeof(valueBuffer), "%d", record->value.valueInt32);
                    break;
                case LdcDiagArgUInt32:
                    snprintf(valueBuffer, sizeof(valueBuffer), "%u", record->value.valueUInt32);
                    break;
                case LdcDiagArgInt64:
                    snprintf(valueBuffer, sizeof(valueBuffer), "%" PRIi64, record->value.valueInt64);
                    break;
                case LdcDiagArgUInt64:
                    snprintf(valueBuffer, sizeof(valueBuffer), "%" PRIu64, record->value.valueUInt64);
                    break;
                case LdcDiagArgFloat32:
                    snprintf(valueBuffer, sizeof(valueBuffer), "%g", record->value.valueFloat32);
                    break;
                case LdcDiagArgFloat64:
                    snprintf(valueBuffer, sizeof(valueBuffer), "%g", record->value.valueFloat64);
                    break;
                default:
                    // Unsupported metric type
                    break;
            }

            numChars = snprintf(
                dst, dstSize,
                "{\"ph\":\"C\", \"ts\":%" TSFMT
                ", \"pid\":%u, \"tid\":%u, \"name\":\"%s\", \"args\": { \"value\": %s}},\n",
                microSeconds, processId, record->threadId, site->str, valueBuffer);
            break;

        // Memory
        case LdcDiagTypeMemoryAllocate:
        case LdcDiagTypeMemoryReallocate:
        case LdcDiagTypeMemoryFree: break;
    }
#undef TSFMT
    return numChars;
}

// Extract arguments from stack into an array of values
//
void ldcDiagnosticsCopyArguments(const LdcDiagSite* site, LdcDiagValue values[], va_list args)
{
    for (uint32_t i = 0; i < site->argumentCount; ++i) {
        switch (site->argumentTypes[i]) {
            case LdcDiagArgId: {
                values[i].id = va_arg(args, uint64_t);
                break;
            }

            case LdcDiagArgBool: {
                values[i].valueBool = va_arg(args, int);
                break;
            }
            case LdcDiagArgChar: {
                values[i].valueChar = va_arg(args, int);
                break;
            }
            case LdcDiagArgInt8: {
                values[i].valueInt8 = va_arg(args, int);
                break;
            }
            case LdcDiagArgUInt8: {
                values[i].valueUInt8 = va_arg(args, unsigned int);
                break;
            }
            case LdcDiagArgInt16: {
                values[i].valueInt16 = va_arg(args, int);
                break;
            }
            case LdcDiagArgUInt16: {
                values[i].valueUInt16 = va_arg(args, unsigned int);
                break;
            }
            case LdcDiagArgInt32: {
                values[i].valueInt32 = va_arg(args, int);
                break;
            }
            case LdcDiagArgUInt32: {
                values[i].valueUInt32 = va_arg(args, unsigned int);
                break;
            }
            case LdcDiagArgInt64: {
                values[i].valueInt64 = va_arg(args, int64_t);
                break;
            }
            case LdcDiagArgUInt64: {
                values[i].valueUInt64 = va_arg(args, uint64_t);
                break;
            }
            case LdcDiagArgCharPtr: {
                values[i].valueCharPtr = va_arg(args, char*);
                break;
            }
            case LdcDiagArgConstCharPtr: {
                values[i].valueConstCharPtr = va_arg(args, const char*);
                break;
            }
            case LdcDiagArgVoidPtr: {
                values[i].valueVoidPtr = va_arg(args, void*);
                break;
            }
            case LdcDiagArgConstVoidPtr: {
                values[i].valueConstVoidPtr = va_arg(args, const void*);
                break;
            }

            case LdcDiagArgNone:
                // Handles weird case with MSVC generating extra arguments in VNLog() expansion
                break;

            default: assert(0); break;
        }
    }
}

#if !VN_SDK_FEATURE(DIAGNOSTICS_ASYNC)

// All events are synchronous
//
void ldcDiagEvent(const LdcDiagSite* site, size_t valuesSize, ...)
{
    if (!site || site->level > ldcDiagnosticsState->maxLogLevel) {
        return;
    }

    LdcDiagRecord record = {0};
    ldcDiagnosticsRecordSet(&record, site);

    // Extract arguments into a value array
    LdcDiagValue* values = alloca(valuesSize);

    va_list args;
    va_start(args, valuesSize);
    ldcDiagnosticsCopyArguments(site, values, args);
    va_end(args);

    applyDiagnosticsHandlers(site, &record, values);
}

void ldcDiagEventFormatted(const LdcDiagSite* site, const char* fmt, ...)
{
    if (!site || site->level > ldcDiagnosticsState->maxLogLevel) {
        return;
    }

    LdcDiagRecord record = {0};
    ldcDiagnosticsRecordSet(&record, site);

    char buffer[4096];

    va_list args;
    va_start(args, fmt);
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);

    applyDiagnosticsHandlers(site, &record, (LdcDiagValue*)buffer);
}

void ldcTracingScopedBegin(const LdcDiagSite* site)
{
    LdcDiagRecord record = {0};
    ldcDiagnosticsRecordSetAll(&record, site, 1, 0);

    applyDiagnosticsHandlers(site, &record, NULL);
}

void ldcTracingScopedEnd(const LdcDiagSite* site)
{
    LdcDiagRecord record = {0};
    ldcDiagnosticsRecordSetAll(&record, site, 0, 0);

    applyDiagnosticsHandlers(site, &record, NULL);
}

void ldcTracingEvent(const LdcDiagSite* site, size_t valuesSize, ...)
{
    LdcDiagRecord record = {0};
    ldcDiagnosticsRecordSet(&record, site);

    // Extract arguments into a value array
    LdcDiagValue* values = alloca(valuesSize);

    va_list args;
    va_start(args, valuesSize);
    ldcDiagnosticsCopyArguments(site, values, args);
    va_end(args);

    applyDiagnosticsHandlers(site, &record, values);
}

void ldcMetricInt32(const LdcDiagSite* site, int32_t value)
{
    LdcDiagRecord record = {0};
    ldcDiagnosticsRecordSetAll(&record, site, 0, 0);
    record.value.valueInt32 = value;

    applyDiagnosticsHandlers(site, &record, NULL);
}

void ldcMetricUInt32(const LdcDiagSite* site, uint32_t value)
{
    LdcDiagRecord record = {0};
    ldcDiagnosticsRecordSetAll(&record, site, 0, 0);
    record.value.valueUInt32 = value;

    applyDiagnosticsHandlers(site, &record, NULL);
}

void ldcMetricInt64(const LdcDiagSite* site, int64_t value)
{
    LdcDiagRecord record = {0};
    ldcDiagnosticsRecordSetAll(&record, site, 0, 0);
    record.value.valueInt64 = value;

    applyDiagnosticsHandlers(site, &record, NULL);
}

void ldcMetricUInt64(const LdcDiagSite* site, uint64_t value)
{
    LdcDiagRecord record = {0};
    ldcDiagnosticsRecordSetAll(&record, site, 0, 0);
    record.value.valueUInt64 = value;

    applyDiagnosticsHandlers(site, &record, NULL);
}

void ldcMetricFloat32(const LdcDiagSite* site, float value)
{
    LdcDiagRecord record = {0};
    ldcDiagnosticsRecordSetAll(&record, site, 0, 0);
    record.value.valueFloat32 = value;

    applyDiagnosticsHandlers(site, &record, NULL);
}

void ldcMetricFloat64(const LdcDiagSite* site, double value)
{
    LdcDiagRecord record = {0};
    ldcDiagnosticsRecordSetAll(&record, site, 0, 0);
    record.value.valueFloat64 = value;

    applyDiagnosticsHandlers(site, &record, NULL);
}

#endif
