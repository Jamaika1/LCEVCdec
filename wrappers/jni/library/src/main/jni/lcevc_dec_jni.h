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

#ifndef LCEVC_DEC_JNI_H
#define LCEVC_DEC_JNI_H

#include <android/log.h>
#include <android/native_window_jni.h>
#include <jni.h>
#include <LCEVC/lcevc_dec.h>
#include <sys/system_properties.h>

#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#define LOG_TAG "lcevc_dec_jni"

// This shall be picked up by the Change Namespace script
#define JAVA_PACKAGE_NAME com_vnova_lcevc_decoder_dec
#define JAVA_CLASS_NAME LcevcNativeAdapter

#define Q(x) #x
#define QUOTE(x) Q(x)

#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__))
#define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__))
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__))
// #ifdef NDEBUG
// #define LOGD(...)
// #else
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
// #endif

// Helper macros for simplifying working with Java classes and fields
#define SET_BOOL_FIELD(o, c, n, v)        \
    fieldId = env->GetFieldID(c, n, "Z"); \
    env->SetBooleanField(o, fieldId, static_cast<jboolean>(v));
#define SET_INT_FIELD(o, c, n, v)         \
    fieldId = env->GetFieldID(c, n, "I"); \
    env->SetIntField(o, fieldId, static_cast<jint>(v));
#define SET_FLOAT_FIELD(o, c, n, v)       \
    fieldId = env->GetFieldID(c, n, "F"); \
    env->SetFloatField(o, fieldId, static_cast<jfloat>(v));
#define SET_LONG_FIELD(o, c, n, v)        \
    fieldId = env->GetFieldID(c, n, "J"); \
    env->SetLongField(o, fieldId, static_cast<jlong>(v));
#define SET_PTR_FIELD(o, c, n, v)         \
    fieldId = env->GetFieldID(c, n, "J"); \
    env->SetLongField(o, fieldId, reinterpret_cast<jlong>(v));

#define GET_BOOL_FIELD(o, c, n, v)        \
    fieldId = env->GetFieldID(c, n, "Z"); \
    v = static_cast<bool>(env->GetBooleanField(o, fieldId));
#define GET_INT_FIELD(o, c, n, v)         \
    fieldId = env->GetFieldID(c, n, "I"); \
    v = static_cast<uint32_t>(env->GetIntField(o, fieldId));
#define GET_FLOAT_FIELD(o, c, n, v)       \
    fieldId = env->GetFieldID(c, n, "F"); \
    v = static_cast<float>(env->GetFloatField(o, fieldId));
#define GET_LONG_FIELD(o, c, n, v)        \
    fieldId = env->GetFieldID(c, n, "J"); \
    v = static_cast<uint64_t>(env->GetLongField(o, fieldId));
#define GET_PTR_FIELD(o, c, n, v)         \
    fieldId = env->GetFieldID(c, n, "J"); \
    v = reinterpret_cast<void*>(env->GetLongField(o, fieldId));

constexpr android_LogPriority logLevelCharToAndroidLogLevel(char level)
{
    switch (level) {
        case '1': return ANDROID_LOG_FATAL;
        case '2': return ANDROID_LOG_ERROR;
        case '3': return ANDROID_LOG_WARN;
        case '4': return ANDROID_LOG_INFO;
        case '5': return ANDROID_LOG_DEBUG;
        case '6': return ANDROID_LOG_VERBOSE;
        default: return ANDROID_LOG_UNKNOWN;
    }
}

/*! Client context for decoder instance and callback metadata */
struct ClientContext
{
    LCEVC_DecoderHandle decoder = {0};
    jobject client = nullptr;
    ANativeWindow* nativeWindow = nullptr;
};

#define NATIVE_FUNC_LONG_DEF(RETURN_TYPE, PACKAGE, CLASS, NAME, ...)                                   \
    extern "C"                                                                                         \
    {                                                                                                  \
    JNIEXPORT RETURN_TYPE Java_##PACKAGE##_##CLASS##_##NAME(JNIEnv* env, jobject thiz, ##__VA_ARGS__); \
    }                                                                                                  \
    JNIEXPORT RETURN_TYPE Java_##PACKAGE##_##CLASS##_##NAME(JNIEnv* env, jobject thiz, ##__VA_ARGS__)

#define NATIVE_FUNC_LONG(RETURN_TYPE, PACKAGE, CLASS, NAME, ...) \
    NATIVE_FUNC_LONG_DEF(RETURN_TYPE, PACKAGE, CLASS, NAME, ##__VA_ARGS__)
#define NATIVE_FUNC(RETURN_TYPE, NAME, ...) \
    NATIVE_FUNC_LONG(RETURN_TYPE, JAVA_PACKAGE_NAME, JAVA_CLASS_NAME, NAME, ##__VA_ARGS__)

extern void logEventCallback(LCEVC_DecoderHandle decHandle, LCEVC_Event event, LCEVC_PictureHandle picHandle,
                             const LCEVC_DecodeInformation* decodeInformation, const uint8_t* data,
                             uint32_t dataSize, void* userData);

extern bool checkJniException(JNIEnv* env);

#endif // LCEVC_DEC_JNI_H
