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

#include "handle.h"

#include <gtest/gtest.h>
#include <interface.h>
#include <LCEVC/common/constants.h>
#include <LCEVC/common/memory.h>
#include <LCEVC/lcevc_dec.h>

#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <mutex>
#include <thread>

namespace {

// Shared data for event callback
struct CallbackData
{
    uint32_t magic = 0x5eaf00d;
    std::mutex mutex;
    std::condition_variable condVar;

    uint32_t count = 0;
};

void eventCallback(LCEVC_DecoderHandle decHandle, LCEVC_Event event, LCEVC_PictureHandle picHandle,
                   const LCEVC_DecodeInformation* decodeInformation, const uint8_t* data,
                   uint32_t dataSize, void* userData)
{
    ASSERT_NE(userData, nullptr);
    CallbackData& callbackData = *(reinterpret_cast<CallbackData*>(userData));
    ASSERT_EQ(callbackData.magic, 0x5eaf00d);

    ASSERT_EQ(event, LCEVC_Log);
    ASSERT_EQ(picHandle.hdl, lcevc_dec::decoder::kInvalidHandle);
    ASSERT_EQ(decodeInformation, nullptr);
    ASSERT_NE(data, nullptr);

    ASSERT_EQ(data[0], '4');

    {
        std::unique_lock lock(callbackData.mutex);
        callbackData.count++;
        callbackData.condVar.notify_all();
    }
}
} // namespace

// Check that some log events are generated during decoder startup and shutdown
// The API and pipeline will generate at least some sort of Info events during startup.
//
TEST(EventLog, LogEvents)
{
    CallbackData callbackData;

    LCEVC_DecoderHandle decoderHandle = {};
    const LCEVC_AccelContextHandle dummyHdl = {};
    ASSERT_EQ(LCEVC_CreateDecoder(&decoderHandle, dummyHdl), LCEVC_Success);

    ASSERT_EQ(LCEVC_ConfigureDecoderInt(decoderHandle, "log_level", 4), LCEVC_Success);

    int32_t events[] = {LCEVC_Log};
    ASSERT_EQ(LCEVC_ConfigureDecoderIntArray(decoderHandle, "events", VNArraySize(events), events),
              LCEVC_Success);
    ASSERT_EQ(LCEVC_SetDecoderEventCallback(decoderHandle, eventCallback, static_cast<void*>(&callbackData)),
              LCEVC_Success);

    ASSERT_EQ(LCEVC_InitializeDecoder(decoderHandle), LCEVC_Success);

    LCEVC_PictureDesc desc;
    LCEVC_DefaultPictureDesc(&desc, LCEVC_I420_8, 960, 540);

    LCEVC_PictureHandle pictureHandle = {};
    ASSERT_EQ(LCEVC_AllocPicture(decoderHandle, &desc, &pictureHandle), LCEVC_Success);

    // Wait for events to arrive ...
    while (1) {
        std::unique_lock lock(callbackData.mutex);
        if (callbackData.count == 0) {
            // Block if nothing has arrived yet
            if (callbackData.condVar.wait_until(lock, std::chrono::system_clock::now() +
                                                          std::chrono::milliseconds(1000)) ==
                std::cv_status::timeout) {
                // Timed out - stop waiting
                break;
            }
        } else {
            // Got a log message
            break;
        }
    }

    ASSERT_GT(callbackData.count, 0);

    ASSERT_EQ(LCEVC_FreePicture(decoderHandle, pictureHandle), LCEVC_Success);

    LCEVC_DestroyDecoder(decoderHandle);
}
