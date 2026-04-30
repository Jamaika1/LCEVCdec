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

#include "lcevc_dec_jni.h"

#include <android/log.h>
#include <LCEVC/extract/extract.h>
#include <LCEVC/lcevc_dec.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <mutex>
#include <string_view>
#include <thread>
#include <unordered_map>

constexpr const char* TAG = "LCEVCdec";

// save Java VM context
static JavaVM* gJavaVM = nullptr;
static jobject gClassLoader = nullptr;
static jmethodID gFindClassMethod = nullptr;
static jclass gDecoderInfoClass = nullptr;
static jobject gDecoderInfoObject = nullptr;
static jstring gDecoderInfoClassNameJS = nullptr;
static std::string gDecoderInfoClassName = std::string(QUOTE(JAVA_PACKAGE_NAME)) +
                                           std::string("_") + std::string(QUOTE(JAVA_CLASS_NAME)) +
                                           std::string("$") + std::string("DecodeInformation");

static jstring gColorParamsClassNameJS = nullptr;
static std::string gColorParamsClassName = std::string(QUOTE(JAVA_PACKAGE_NAME)) +
                                           std::string("_") + std::string(QUOTE(JAVA_CLASS_NAME)) +
                                           std::string("$") + std::string("ColorParams");
static std::string gColorParamsFieldName = "";

static std::mutex gPictureMapMutex;
static std::unordered_map<uintptr_t, LCEVC_DecoderHandle> gPictureDecoderMap;

static uint32_t greatestCommonDivisor(uint32_t a, uint32_t b)
{
    while (b != 0) {
        uint32_t tmp = a % b;
        a = b;
        b = tmp;
    }
    return a == 0 ? 1 : a;
}

static void setSampleAspectRatioFromPixelAspectRatio(float pixelAspectRatio, uint32_t& num, uint32_t& den)
{
    if (pixelAspectRatio <= 0.0f) {
        num = 1;
        den = 1;
        return;
    }

    constexpr uint32_t kSarScale = 1000000;
    const uint64_t scaled = static_cast<uint64_t>(std::llround(pixelAspectRatio * kSarScale));
    num = static_cast<uint32_t>(std::min<uint64_t>(scaled, std::numeric_limits<uint32_t>::max()));
    den = kSarScale;

    const uint32_t gcd = greatestCommonDivisor(num, den);
    num /= gcd;
    den /= gcd;
}

void logEventCallback(LCEVC_DecoderHandle, LCEVC_Event event, LCEVC_PictureHandle,
                      const LCEVC_DecodeInformation*, const uint8_t* data, uint32_t dataSize, void*)
{
    if (event != LCEVC_Log) {
        return;
    }
    const std::string_view messageWithLevel(reinterpret_cast<const char*>(data), dataSize);
    const android_LogPriority level = logLevelCharToAndroidLogLevel(messageWithLevel.at(0));
    const std::string_view message = messageWithLevel.substr(2);
    __android_log_print(level, TAG, "%s", message.data());
}

bool checkJniException(JNIEnv* env)
{
    if (env->ExceptionCheck()) {
        env->ExceptionDescribe(); // writes to logcat
        env->ExceptionClear();
        return true;
    }
    return false;
}

// we need this to be called in the JNI_Onload function
// because we need to create a class, on a different thread and FindClass only
// seems to work on the main Java thread, not a native thread??? by caching the
// findClass method from the main thread we can call it on any thread :D
static void getClassLoader(JNIEnv* env, const char* className)
{
    auto randomClass = env->FindClass(className);
    jclass classClass = env->GetObjectClass(randomClass);
    auto classLoaderClass = env->FindClass("java/lang/ClassLoader");
    auto getClassLoaderMethod =
        env->GetMethodID(classClass, "getClassLoader", "()Ljava/lang/ClassLoader;");
    gClassLoader = env->NewGlobalRef(env->CallObjectMethod(randomClass, getClassLoaderMethod));
    gFindClassMethod =
        env->GetMethodID(classLoaderClass, "findClass", "(Ljava/lang/String;)Ljava/lang/Class;");
    checkJniException(env);
}

static jclass findClass(JNIEnv* env, jstring name)
{
    if (gFindClassMethod == nullptr)
        return nullptr;
    return static_cast<jclass>(env->CallObjectMethod(gClassLoader, gFindClassMethod, name));
}

static void setJavaDecodeInformationFromLcevc(JNIEnv* env, jclass cls, jobject jInfo,
                                              const LCEVC_DecodeInformation* info)
{
    if (info != nullptr) {
        jfieldID fieldId;
        SET_INT_FIELD(jInfo, cls, "width", info->baseWidth);
        SET_INT_FIELD(jInfo, cls, "height", info->baseHeight);
        SET_FLOAT_FIELD(jInfo, cls, "sarNum", 1.0f);
        SET_FLOAT_FIELD(jInfo, cls, "sarDen", 1.0f);
        SET_INT_FIELD(jInfo, cls, "bitdepth", info->baseBitdepth);
        SET_BOOL_FIELD(jInfo, cls, "skipRequested", info->skipped);
        SET_BOOL_FIELD(jInfo, cls, "lcevcEnhanced", info->enhanced);
        SET_BOOL_FIELD(jInfo, cls, "lcevcAvailable", info->hasEnhancement);
    }
}

static void setJavaDecodeInformationSarFromPicture(JNIEnv* env, jclass cls, jobject jInfo,
                                                   LCEVC_DecoderHandle decoder, LCEVC_PictureHandle picture)
{
    jfieldID fieldId;

    LCEVC_PictureDesc desc = {};
    if (picture.hdl != 0 && LCEVC_GetPictureDesc(decoder, picture, &desc) == LCEVC_Success) {
        SET_FLOAT_FIELD(jInfo, cls, "sarNum", static_cast<float>(desc.sampleAspectRatioNum));
        SET_FLOAT_FIELD(jInfo, cls, "sarDen", static_cast<float>(desc.sampleAspectRatioDen));
        return;
    }

    SET_FLOAT_FIELD(jInfo, cls, "sarNum", 1.0f);
    SET_FLOAT_FIELD(jInfo, cls, "sarDen", 1.0f);
}

static bool createJavaDecodeInformation(JNIEnv* env, jclass& decodeClass, jobject& decodeObject)
{
    decodeClass = findClass(env, gDecoderInfoClassNameJS);
    if (decodeClass != nullptr) {
        jmethodID constr = env->GetMethodID(decodeClass, "<init>", "()V");
        if (constr != nullptr) {
            decodeObject = env->NewObject(decodeClass, constr);
            if (decodeObject == nullptr) {
                LOGE("Failed NewObject");
            }
        } else {
            LOGE("Failed GetMethodID for init");
        }
    } else {
        LOGE("Failed FindClass");
    }
    checkJniException(env);
    return decodeObject != nullptr;
}

static jobject createJavaColorParams(JNIEnv* env)
{
    jobject ret = nullptr;
    jfieldID fID;
    const jclass cls = findClass(env, gColorParamsClassNameJS);
    if (cls != nullptr) {
        jmethodID constr = env->GetMethodID(cls, "<init>", "()V");
        if (constr != nullptr) {
            if ((ret = env->NewObject(cls, constr)) == nullptr) {
                LOGE("Failed NewObject");
            }
        } else {
            LOGE("Failed GetMethodID for init");
        }
    } else {
        LOGE("Failed FindClass");
    }
    checkJniException(env);
    return ret;
}

static int sdkVersion()
{
    char version[PROP_VALUE_MAX + 1];
    __system_property_get("ro.build.version.sdk", version);
    return atoi(version);
}

static uint32_t formatByteSize(LCEVC_ColorFormat format)
{
    switch (format) {
        case LCEVC_I420_10_LE:
        case LCEVC_I420_12_LE:
        case LCEVC_I420_14_LE:
        case LCEVC_I420_16_LE:
        case LCEVC_I422_10_LE:
        case LCEVC_I422_12_LE:
        case LCEVC_I422_14_LE:
        case LCEVC_I422_16_LE:
        case LCEVC_I444_10_LE:
        case LCEVC_I444_12_LE:
        case LCEVC_I444_14_LE:
        case LCEVC_I444_16_LE:
        case LCEVC_GRAY_10_LE:
        case LCEVC_GRAY_12_LE:
        case LCEVC_GRAY_14_LE:
        case LCEVC_GRAY_16_LE: return 2;
        default: return 1;
    }
}

static uint32_t formatPlaneCount(LCEVC_ColorFormat format)
{
    switch (format) {
        case LCEVC_NV12_8:
        case LCEVC_NV21_8: return 2;
        case LCEVC_GRAY_8:
        case LCEVC_GRAY_10_LE:
        case LCEVC_GRAY_12_LE:
        case LCEVC_GRAY_14_LE:
        case LCEVC_GRAY_16_LE: return 1;
        default: return 3;
    }
}

static bool validatePlaneFormat(LCEVC_ColorFormat format, uint32_t width, uint32_t height,
                                uint32_t plane, uint32_t& planeWidth, uint32_t& planeHeight)
{
    const uint32_t halfWidth = width / 2;
    const uint32_t halfHeight = height / 2;

    switch (format) {
        case LCEVC_NV12_8:
        case LCEVC_NV21_8:
            if (plane == 0) {
                planeWidth = width;
                planeHeight = height;
                return true;
            }
            if (plane == 1) {
                planeWidth = width;
                planeHeight = halfHeight;
                return true;
            }
            return false;
        case LCEVC_I420_8:
        case LCEVC_I420_10_LE:
        case LCEVC_I420_12_LE:
        case LCEVC_I420_14_LE:
        case LCEVC_I420_16_LE:
            if (plane == 0) {
                planeWidth = width;
                planeHeight = height;
                return true;
            }
            if (plane < 3) {
                planeWidth = halfWidth;
                planeHeight = halfHeight;
                return true;
            }
            return false;
        case LCEVC_I422_8:
        case LCEVC_I422_10_LE:
        case LCEVC_I422_12_LE:
        case LCEVC_I422_14_LE:
        case LCEVC_I422_16_LE:
            if (plane == 0) {
                planeWidth = width;
                planeHeight = height;
                return true;
            }
            if (plane < 3) {
                planeWidth = halfWidth;
                planeHeight = height;
                return true;
            }
            return false;
        case LCEVC_I444_8:
        case LCEVC_I444_10_LE:
        case LCEVC_I444_12_LE:
        case LCEVC_I444_14_LE:
        case LCEVC_I444_16_LE:
            if (plane < 3) {
                planeWidth = width;
                planeHeight = height;
                return true;
            }
            return false;
        case LCEVC_GRAY_8:
        case LCEVC_GRAY_10_LE:
        case LCEVC_GRAY_12_LE:
        case LCEVC_GRAY_14_LE:
        case LCEVC_GRAY_16_LE:
            if (plane == 0) {
                planeWidth = width;
                planeHeight = height;
                return true;
            }
            return false;
        default: return false;
    }
}

inline int64_t getPtsFromUniqueTimestamp(const uint64_t timestamp)
{
    const int64_t encodedPts = static_cast<int64_t>(timestamp & 0x0000FFFFFFFFFFFFULL);
    return encodedPts - (1LL << 47);
}

jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    JNIEnv* env;
    LOGD("JNI_OnLoad, starting VM: %p", vm);
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
        return -1;
    }

    gJavaVM = vm;

    std::replace(gColorParamsClassName.begin(), gColorParamsClassName.end(), '_', '/');
    gColorParamsClassNameJS =
        reinterpret_cast<jstring>(env->NewGlobalRef(env->NewStringUTF(gColorParamsClassName.c_str())));
    gColorParamsFieldName = "L" + gColorParamsClassName + ";";
    std::replace(gDecoderInfoClassName.begin(), gDecoderInfoClassName.end(), '_', '/');
    gDecoderInfoClassNameJS =
        reinterpret_cast<jstring>(env->NewGlobalRef(env->NewStringUTF(gDecoderInfoClassName.c_str())));
    getClassLoader(env, gDecoderInfoClassName.c_str()); // setup the global class loader

    createJavaDecodeInformation(env, gDecoderInfoClass, gDecoderInfoObject);
    gDecoderInfoClass = reinterpret_cast<jclass>(env->NewGlobalRef(gDecoderInfoClass));
    gDecoderInfoObject = reinterpret_cast<jobject>(env->NewGlobalRef(gDecoderInfoObject));
    LOGD("JNI_OnLoad, running API level %d, decoder info c lass %p object %p", sdkVersion(),
         gDecoderInfoClass, gDecoderInfoObject);

    return JNI_VERSION_1_6;
}

NATIVE_FUNC(jlong, lcevcCreateDecoder)
{
    LOGI("LCEVCdec JNI lcevcCreateDecoder");

    ClientContext* context = new ClientContext(); // Zero initialised

    context->client = reinterpret_cast<jobject>(env->NewWeakGlobalRef(thiz));
    context->nativeWindow = nullptr;

    LCEVC_ReturnCode ret = LCEVC_CreateDecoder(&context->decoder, LCEVC_AccelContextHandle{});
    LCEVC_ConfigureDecoderInt(context->decoder, "threads", 2);
    LCEVC_ConfigureDecoderString(context->decoder, "pipeline", "vulkan");
    LCEVC_ConfigureDecoderBool(context->decoder, "will_render", true);

    // logcat logging via event callback
    constexpr std::array<int32_t, 1> enabledEvents = {LCEVC_Log};
    LCEVC_ConfigureDecoderIntArray(context->decoder, "events", enabledEvents.size(), enabledEvents.data());
    LCEVC_SetDecoderEventCallback(context->decoder, logEventCallback, nullptr);

    if (ret == LCEVC_Success) {
        ret = LCEVC_InitializeDecoder(context->decoder);
    }

    if (ret == LCEVC_Success) {
        LOGD("Created decoder context %p", context);
        LOGI("Using LCEVCdec version %d.%d.%d", LCEVC_DEC_VERSION_MAJOR, LCEVC_DEC_VERSION_MINOR,
             LCEVC_DEC_VERSION_PATCH);
        return reinterpret_cast<jlong>(context);
    }

    return 0;
}

NATIVE_FUNC(jint, lcevcDestroyDecoder, jlong instance)
{
    LOGI("LCEVCdec JNI Destroy");

    ClientContext* context = reinterpret_cast<ClientContext*>(instance);
    if (context == nullptr) {
        return LCEVC_InvalidParam;
    }

    {
        std::lock_guard<std::mutex> lock(gPictureMapMutex);
        for (auto it = gPictureDecoderMap.begin(); it != gPictureDecoderMap.end();) {
            if (it->second.hdl == context->decoder.hdl) {
                it = gPictureDecoderMap.erase(it);
            } else {
                ++it;
            }
        }
    }

    if (context->nativeWindow != nullptr) {
        ANativeWindow_release(context->nativeWindow);
        context->nativeWindow = nullptr;
    }

    if (context->client != nullptr) {
        env->DeleteWeakGlobalRef(context->client);
        context->client = nullptr;
    }

    if (context->decoder.hdl != 0) {
        LCEVC_DestroyDecoder(context->decoder);
        context->decoder = {};
    }

    delete context;
    return LCEVC_Success;
}

NATIVE_FUNC(jint, lcevcSendDecoderBase, jlong instance, jlong timeStamp, jlong timeoutUs,
            jlong inputPictureHandle, jlong outputPictureHandle)
{
    LOGI("LCEVCdec JNI lcevcSendDecoderBase");

    ClientContext* context = reinterpret_cast<ClientContext*>(instance);
    if (context == nullptr) {
        return LCEVC_InvalidParam;
    }
    LCEVC_PictureHandle const inputPicture = {static_cast<uintptr_t>(inputPictureHandle)};
    LCEVC_PictureHandle const outputPicture = {static_cast<uintptr_t>(outputPictureHandle)};

    LOGD("sendBase: timeStamp %" PRId64 " input %lu, output %lu", timeStamp,
         static_cast<unsigned long>(inputPicture.hdl), static_cast<unsigned long>(outputPicture.hdl));

    uint32_t timeout = static_cast<uint32_t>(timeoutUs);
    LCEVC_ReturnCode ret =
        LCEVC_SendDecoderBase(context->decoder, timeStamp, inputPicture, timeout, nullptr);
    if (ret != LCEVC_Success) {
        LOGE("SendBase returned %d", ret);
        return ret;
    }

    ret = LCEVC_SendDecoderPicture(context->decoder, outputPicture);
    if (ret != LCEVC_Success) {
        LOGE("SendOutput returned %d", ret);
        return ret;
    }

    jobject client = env->NewLocalRef(context->client);
    if (client != nullptr) {
        jclass cls = env->GetObjectClass(client);
        std::string methodSig = std::string("(JIJ") + "L" + gDecoderInfoClassName + ";)V";
        jmethodID methodID = env->GetMethodID(cls, "onDecodeCompleted", methodSig.c_str());
        if (methodID != nullptr) {
            LCEVC_DecodeInformation const decodeInformation = {};
            setJavaDecodeInformationFromLcevc(env, gDecoderInfoClass, gDecoderInfoObject, &decodeInformation);
            setJavaDecodeInformationSarFromPicture(env, gDecoderInfoClass, gDecoderInfoObject,
                                                   context->decoder, outputPicture);
            env->CallVoidMethod(client, methodID, static_cast<jlong>(instance),
                                static_cast<jint>(LCEVC_Success), static_cast<jlong>(timeStamp),
                                gDecoderInfoObject);
        } else {
            LOGE("LCEVCdec JNI lcevcSendDecoderBase: failed to find onDecodeCompleted");
        }
        env->DeleteLocalRef(client);
    }

    LCEVC_PictureHandle doneBasePicture;
    if (LCEVC_ReceiveDecoderBase(context->decoder, &doneBasePicture) == LCEVC_Success) {
        LOGD("ReceiveDecoderBase");
    }

    return 0;
}

NATIVE_FUNC(jint, lcevcRenderSendPicture, jlong instance, jlong timeStamp, jlong pictureHandle,
            jobject renderInfo, jlong delayUs)
{
    LOGI("LCEVCdec JNI Render");

    ClientContext* context = reinterpret_cast<ClientContext*>(instance);
    if (context == nullptr) {
        return LCEVC_InvalidParam;
    }

    LCEVC_RenderSendInformation info{};
    uint32_t rotation = 0;
    if (renderInfo != nullptr) {
        jclass renderInfoClass = env->GetObjectClass(renderInfo);
        if (renderInfoClass != nullptr) {
            jfieldID fieldId;
            GET_INT_FIELD(renderInfo, renderInfoClass, "rotation", rotation);
        }
    }
    info.rotation = rotation;
    uint64_t delay = static_cast<uint64_t>(delayUs);
    LCEVC_PictureHandle picture = {static_cast<uintptr_t>(pictureHandle)};
    LCEVC_RenderSendPicture(context->decoder, timeStamp, picture, &info, delay);

    LCEVC_PictureHandle decodedPicture;
    LCEVC_DecodeInformation decodeInformation;
    if (LCEVC_ReceiveDecoderPicture(context->decoder, &decodedPicture, &decodeInformation) == LCEVC_Success) {
        LOGD("ReceiveDecoderPicture");
    }

    jobject client = env->NewLocalRef(context->client);
    if (client != nullptr) {
        jclass cls = env->GetObjectClass(client);
        jmethodID methodID = env->GetMethodID(cls, "onRenderCompleted", "(JIJJJ)V");
        if (methodID != nullptr) {
            env->CallVoidMethod(client, methodID, static_cast<jlong>(instance),
                                static_cast<jint>(LCEVC_Success), static_cast<jlong>(timeStamp),
                                static_cast<jlong>(pictureHandle),
                                static_cast<jlong>(getPtsFromUniqueTimestamp(timeStamp)));
        } else {
            LOGE("LCEVCdec JNI Render: failed to find onRenderCompleted");
        }
        env->DeleteLocalRef(client);
    }
    checkJniException(env);

    return 0;
}

NATIVE_FUNC(jint, lcevcSendDecoderEnhancementData, jlong instance, jlong timestamp,
            jboolean isKeyFrame, jobject inputData, jint syntaxType, jboolean stripLcevc)
{
    if (instance == 0 || inputData == nullptr) {
        return LCEVC_InvalidParam;
    }
    ClientContext* context = reinterpret_cast<ClientContext*>(instance);
    jclass cls = env->GetObjectClass(inputData);
    // Method id for inputData get position
    jmethodID methodId = env->GetMethodID(cls, "position", "()I");
    uint32_t dataOffset = env->CallIntMethod(inputData, methodId);
    uint8_t* data = reinterpret_cast<uint8_t*>(env->GetDirectBufferAddress(inputData));
    data += dataOffset;
    // Method id for inputData get limit
    methodId = env->GetMethodID(cls, "limit", "()I");
    uint32_t dataLimit = env->CallIntMethod(inputData, methodId);
    uint32_t dataSize = dataLimit - dataOffset;

    std::vector<uint8_t> enhancementData(dataSize);
    if (stripLcevc) {
        uint32_t actualSize = 0;
        int32_t const r = LCEVC_extractEnhancementFromNAL(
            data, dataSize, LCEVC_NALFormat_AnnexB, static_cast<LCEVC_CodecType>(syntaxType),
            enhancementData.data(), static_cast<uint32_t>(enhancementData.size()), &actualSize);
        enhancementData.resize(actualSize);
        if (r < 0) {
            LOGW("Couldn't find LCEVC data in input data (%d bytes)", dataSize);
            return r;
        }
        LOGD("Extracted %d LCEVC bytes from NALU (%d bytes)", actualSize, dataSize);
    }

    LCEVC_SendDecoderEnhancementData(context->decoder, timestamp, enhancementData.data(),
                                     enhancementData.size());

    LOGI("LCEVCdec JNI SendDecoderEnhancementData");
    return 0;
}

NATIVE_FUNC(jint, lcevcFlushDecoder, jlong instance)
{
    LOGI("LCEVCdec JNI FlushDecoder");

    ClientContext* context = reinterpret_cast<ClientContext*>(instance);
    if (context == nullptr) {
        return LCEVC_InvalidParam;
    }

    const LCEVC_ReturnCode ret = LCEVC_FlushDecoder(context->decoder);

    return ret;
}

NATIVE_FUNC(jint, lcevcSynchronizeDecoder, jlong instance, jboolean dropPending)
{
    LOGI("LCEVCdec JNI SynchronizeDecoder");

    ClientContext* context = reinterpret_cast<ClientContext*>(instance);
    if (context == nullptr) {
        return LCEVC_InvalidParam;
    }

    const LCEVC_ReturnCode ret =
        LCEVC_SynchronizeDecoder(context->decoder, static_cast<bool>(dropPending));

    return ret;
}

NATIVE_FUNC(jint, lcevcRenderSetWindow, jlong instance, jobject surface, jboolean secure)
{
    LOGI("LCEVCdec JNI RenderSetWindow");
    ClientContext* context = reinterpret_cast<ClientContext*>(instance);
    if (context != nullptr) {
        ANativeWindow* nativeWindow = nullptr;
        if (context->nativeWindow) {
            ANativeWindow_release(context->nativeWindow);
        }
        if (surface != nullptr) {
            nativeWindow = ANativeWindow_fromSurface(env, surface);
        }
        if (nativeWindow) {
            int width = ANativeWindow_getWidth(nativeWindow);
            int height = ANativeWindow_getHeight(nativeWindow);
            int format = ANativeWindow_getFormat(nativeWindow);
            LOGI("ANativeWindow %p size=%dx%d format=%d", nativeWindow, width, height, format);

            if (width <= 0 || height <= 0) {
                LOGI("Surface not ready yet (size=%dx%d), skipping RenderSetWindow", width, height);
                ANativeWindow_release(nativeWindow);
                return 0; // or another "not ready" code you already handle
            }
        }
        context->nativeWindow = nativeWindow;
        return LCEVC_RenderSetWindow(context->decoder, nativeWindow, secure);
    }

    LOGI("LCEVCdec JNI RenderSetWindow");
    return 0;
}

NATIVE_FUNC(jlong, lcevcAllocPicture, jlong instance, jobject pictureDescObj)
{
    LOGI("LCEVCdec JNI AllocPicture");

    ClientContext* context = reinterpret_cast<ClientContext*>(instance);
    if (context == nullptr) {
        return LCEVC_InvalidParam;
    }
    if (pictureDescObj == nullptr) {
        return LCEVC_InvalidParam;
    }
    LCEVC_PictureDesc desc = {0};
    jfieldID fieldId;
    jclass jPictureDesc = env->GetObjectClass(pictureDescObj);

    jobject objectField;
    uint32_t objectValue;

    LCEVC_ColorFormat format = LCEVC_ColorFormat_Unknown;
    uint32_t width = 0;
    uint32_t height = 0;
    float pixelAspectRatio = 1.0f;

    GET_INT_FIELD(pictureDescObj, jPictureDesc, "colorFormat", objectValue);
    format = static_cast<LCEVC_ColorFormat>(objectValue);
    GET_INT_FIELD(pictureDescObj, jPictureDesc, "width", width);
    GET_INT_FIELD(pictureDescObj, jPictureDesc, "height", height);
    GET_FLOAT_FIELD(pictureDescObj, jPictureDesc, "pixelAspectRatio", pixelAspectRatio);

    LOGD("Making a default desc %dx%d, format %d", width, height, format);
    if (LCEVC_DefaultPictureDesc(&desc, format, width, height) != LCEVC_Success) {
        LOGE("Could not get a default picture desc");
        return -1;
    }

    GET_INT_FIELD(pictureDescObj, jPictureDesc, "sampleAspectRatioNum", desc.sampleAspectRatioNum);
    GET_INT_FIELD(pictureDescObj, jPictureDesc, "sampleAspectRatioDen", desc.sampleAspectRatioDen);
    if (desc.sampleAspectRatioNum == 0 && desc.sampleAspectRatioDen == 0) {
        setSampleAspectRatioFromPixelAspectRatio(pixelAspectRatio, desc.sampleAspectRatioNum,
                                                 desc.sampleAspectRatioDen);
    } else {
        if (desc.sampleAspectRatioNum == 0) {
            desc.sampleAspectRatioNum = 1;
        }
        if (desc.sampleAspectRatioDen == 0) {
            desc.sampleAspectRatioDen = 1;
        }
    }

    fieldId = env->GetFieldID(jPictureDesc, "colorParams", gColorParamsFieldName.c_str());
    objectField = env->GetObjectField(pictureDescObj, fieldId);
    LOGD("fieldId %p object %p for <%s>", fieldId, objectField, gColorParamsFieldName.c_str());
    if (objectField != nullptr) {
        jPictureDesc = env->GetObjectClass(objectField);
        uint32_t value = 0;

        GET_INT_FIELD(objectField, jPictureDesc, "colorRange", value);
        desc.colorRange = static_cast<LCEVC_ColorRange>(value);
        GET_INT_FIELD(objectField, jPictureDesc, "colorPrimaries", value);
        desc.colorPrimaries = static_cast<LCEVC_ColorPrimaries>(value);
        GET_INT_FIELD(objectField, jPictureDesc, "transferCharacteristics", value);
        desc.transferCharacteristics = static_cast<LCEVC_TransferCharacteristics>(value);
        GET_INT_FIELD(objectField, jPictureDesc, "matrixCoefficients", value);
        desc.matrixCoefficients = static_cast<LCEVC_MatrixCoefficients>(value);

        fieldId = env->GetFieldID(jPictureDesc, "hdrStaticInfo", "[B");
        jbyteArray hdrInfoField = static_cast<jbyteArray>(env->GetObjectField(objectField, fieldId));
        if (hdrInfoField != nullptr) {
            env->GetByteArrayRegion(hdrInfoField, 0, sizeof(LCEVC_HDRStaticInfo),
                                    reinterpret_cast<jbyte*>(&desc.hdrStaticInfo));
        }
    }

    LOGI("LCEVCdec JNI picture format: %d width: %d height %d", format, width, height);

    LCEVC_PictureHandle handle = {};
    if (LCEVC_AllocPicture(context->decoder, &desc, &handle) == LCEVC_Success) {
        {
            std::lock_guard<std::mutex> const lock(gPictureMapMutex);
            gPictureDecoderMap[handle.hdl] = context->decoder;
        }
        LOGD("Allocated picture %lu (%p)", static_cast<unsigned long>(handle.hdl), &handle);
        return handle.hdl;
    }

    LOGE("Failed to allocate a picture desc");
    return -1;
}

NATIVE_FUNC(jint, lcevcFreePicture, jlong instance, jlong pictureHandle)
{
    LOGI("LCEVCdec JNI FreePicture");
    ClientContext* context = reinterpret_cast<ClientContext*>(instance);
    if (context == nullptr || pictureHandle == 0) {
        return LCEVC_InvalidParam;
    }
    LCEVC_PictureHandle const pictureHdl = {static_cast<uintptr_t>(pictureHandle)};
    {
        std::lock_guard<std::mutex> const lock(gPictureMapMutex);
        gPictureDecoderMap.erase(pictureHdl.hdl);
    }
    return LCEVC_FreePicture(context->decoder, pictureHdl);
}

NATIVE_FUNC(jint, lcevcGetPicturePlaneCount, jlong instance, jlong pictureHandle)
{
    LOGI("LCEVCdec JNI GetPicturePlaneCount");
    ClientContext* context = reinterpret_cast<ClientContext*>(instance);

    uint32_t count = 0;
    if (pictureHandle == 0) {
        return 0;
    }
    LCEVC_PictureHandle const pictureHdl = {static_cast<uintptr_t>(pictureHandle)};
    if (LCEVC_GetPicturePlaneCount(context->decoder, pictureHdl, &count) != LCEVC_Success) {
        LOGE("Failed to get plane count for picture %lu", static_cast<unsigned long>(pictureHdl.hdl));
        return 0;
    }
    LOGD("Found %d planes for picture %lu", count, static_cast<unsigned long>(pictureHdl.hdl));

    return static_cast<jint>(count);
}

NATIVE_FUNC(jint, lcevcGetPictureDesc, jlong instance, jlong pictureHandle, jobject pictureDescObj)
{
    LOGI("LCEVCdec JNI GetPictureDesc");
    ClientContext* context = reinterpret_cast<ClientContext*>(instance);

    LCEVC_PictureHandle pictureHdl = {static_cast<uintptr_t>(pictureHandle)};
    LOGD("Getting picture %lu", static_cast<unsigned long>(pictureHdl.hdl));
    if (pictureDescObj == nullptr) {
        return LCEVC_InvalidParam;
    }
    LCEVC_PictureDesc desc = {};
    LCEVC_ReturnCode ret = LCEVC_GetPictureDesc(context->decoder, pictureHdl, &desc);
    if (ret != LCEVC_Success) {
        return ret;
    }

    jfieldID fieldId;
    jclass cls;
    jobject objField;

    cls = env->GetObjectClass(pictureDescObj);

    SET_INT_FIELD(pictureDescObj, cls, "colorFormat", desc.colorFormat);
    SET_INT_FIELD(pictureDescObj, cls, "width", desc.width);
    SET_INT_FIELD(pictureDescObj, cls, "height", desc.height);
    SET_INT_FIELD(pictureDescObj, cls, "sampleAspectRatioNum", desc.sampleAspectRatioNum);
    SET_INT_FIELD(pictureDescObj, cls, "sampleAspectRatioDen", desc.sampleAspectRatioDen);
    SET_FLOAT_FIELD(pictureDescObj, cls, "pixelAspectRatio",
                    static_cast<float>(desc.sampleAspectRatioNum) /
                        static_cast<float>(desc.sampleAspectRatioDen));

    fieldId = env->GetFieldID(cls, "colorParams", gColorParamsFieldName.c_str());
    LOGD("gColorParamsFieldName: %s", gColorParamsFieldName.c_str());
    objField = env->GetObjectField(pictureDescObj, fieldId);
    LOGD("fieldId %p object %p for <%s>", fieldId, objField, gColorParamsFieldName.c_str());
    if (objField == nullptr) {
        objField = createJavaColorParams(env);
        env->SetObjectField(pictureDescObj, fieldId, objField);
    }
    if (objField != nullptr) {
        cls = env->GetObjectClass(objField);
        SET_INT_FIELD(objField, cls, "colorRange", static_cast<uint32_t>(desc.colorRange));
        SET_INT_FIELD(objField, cls, "colorPrimaries", static_cast<uint32_t>(desc.colorPrimaries));
        SET_INT_FIELD(objField, cls, "transferCharacteristics",
                      static_cast<uint32_t>(desc.transferCharacteristics));
        SET_INT_FIELD(objField, cls, "matrixCoefficients", static_cast<uint32_t>(desc.matrixCoefficients));

        fieldId = env->GetFieldID(cls, "hdrStaticInfo", "[B");
        jbyteArray barr = static_cast<jbyteArray>(env->GetObjectField(objField, fieldId));
        if (barr == nullptr) {
            barr = env->NewByteArray(sizeof(LCEVC_HDRStaticInfo));
            env->SetObjectField(objField, fieldId, barr);
        }
        env->SetByteArrayRegion(barr, 0, sizeof(LCEVC_HDRStaticInfo),
                                reinterpret_cast<const jbyte*>(&desc.hdrStaticInfo));
    }
    return LCEVC_Success;
}

// Special more efficient function to set all the planes for a picture buffer in a single JNI call
NATIVE_FUNC(jint, lcevcPictureSetPlaneBuffers, jlong pictureHandle, jobject inputBuffer,
            jintArray inputOffsets, jintArray inputByteStrides)
{
    LOGI("LCEVCdec JNI PictureSetPlaneBuffers");
    if (pictureHandle == 0 || inputBuffer == nullptr || inputOffsets == nullptr ||
        inputByteStrides == nullptr) {
        return LCEVC_InvalidParam;
    }

    LCEVC_PictureHandle pictureHdl = {static_cast<uintptr_t>(pictureHandle)};
    LCEVC_DecoderHandle decoderHdl = {};
    {
        std::lock_guard<std::mutex> lock(gPictureMapMutex);
        auto it = gPictureDecoderMap.find(pictureHdl.hdl);
        if (it == gPictureDecoderMap.end()) {
            LOGE("PictureSetPlaneBuffers: unknown picture handle %lu",
                 static_cast<unsigned long>(pictureHdl.hdl));
            return LCEVC_InvalidParam;
        }
        decoderHdl = it->second;
    }

    LCEVC_PictureDesc desc = {};
    if (LCEVC_GetPictureDesc(decoderHdl, pictureHdl, &desc) != LCEVC_Success) {
        LOGE("PictureSetPlaneBuffers: failed to get picture desc %lu",
             static_cast<unsigned long>(pictureHdl.hdl));
        return LCEVC_Error;
    }

    const int32_t numPlanes = env->GetArrayLength(inputOffsets);
    const int32_t numStrides = env->GetArrayLength(inputByteStrides);
    if (numPlanes != numStrides) {
        LOGE("PictureSetPlaneBuffers: offsets/strides length mismatch %d vs %d", numPlanes, numStrides);
        return LCEVC_InvalidParam;
    }

    const uint32_t expectedPlanes = formatPlaneCount(desc.colorFormat);
    if (static_cast<uint32_t>(numPlanes) < expectedPlanes) {
        LOGE("PictureSetPlaneBuffers: expected %u planes, got %d", expectedPlanes, numPlanes);
        return LCEVC_InvalidParam;
    }

    auto* buffer = reinterpret_cast<uint8_t*>(env->GetDirectBufferAddress(inputBuffer));
    if (buffer == nullptr) {
        LOGE("PictureSetPlaneBuffers: direct buffer address is null");
        return LCEVC_InvalidParam;
    }

    jint* offsets = env->GetIntArrayElements(inputOffsets, nullptr);
    jint* strides = env->GetIntArrayElements(inputByteStrides, nullptr);
    if (offsets == nullptr || strides == nullptr) {
        if (offsets != nullptr) {
            env->ReleaseIntArrayElements(inputOffsets, offsets, 0);
        }
        if (strides != nullptr) {
            env->ReleaseIntArrayElements(inputByteStrides, strides, 0);
        }
        LOGE("PictureSetPlaneBuffers: failed to read offsets/strides");
        return LCEVC_InvalidParam;
    }

    LCEVC_PictureLockHandle lock = {};
    if (LCEVC_LockPicture(decoderHdl, pictureHdl, LCEVC_Access_Write, &lock) != LCEVC_Success) {
        env->ReleaseIntArrayElements(inputOffsets, offsets, 0);
        env->ReleaseIntArrayElements(inputByteStrides, strides, 0);
        LOGE("PictureSetPlaneBuffers: failed to lock picture %lu",
             static_cast<unsigned long>(pictureHdl.hdl));
        return LCEVC_Error;
    }

    const uint32_t bytesPerSample = formatByteSize(desc.colorFormat);
    for (uint32_t planeIndex = 0; planeIndex < expectedPlanes; ++planeIndex) {
        LCEVC_PicturePlaneDesc plane = {};
        if (LCEVC_GetPictureLockPlaneDesc(decoderHdl, lock, planeIndex, &plane) != LCEVC_Success) {
            LOGE("PictureSetPlaneBuffers: failed to get plane desc %u for %lu", planeIndex,
                 static_cast<unsigned long>(pictureHdl.hdl));
            continue;
        }

        uint32_t planeWidth = 0;
        uint32_t planeHeight = 0;
        if (!validatePlaneFormat(desc.colorFormat, desc.width, desc.height, planeIndex, planeWidth,
                                 planeHeight)) {
            LOGE("PictureSetPlaneBuffers: unsupported format %d for plane %u", desc.colorFormat, planeIndex);
            continue;
        }

        const uint32_t rowBytes = planeWidth * bytesPerSample;
        uint8_t* src = buffer + offsets[planeIndex];
        uint8_t* dst = plane.firstSample;
        const uint32_t srcStride = static_cast<uint32_t>(strides[planeIndex]);
        for (uint32_t row = 0; row < planeHeight; ++row) {
            memcpy(dst + row * plane.rowByteStride, src + row * srcStride, rowBytes);
        }
    }

    LCEVC_UnlockPicture(decoderHdl, lock);

    env->ReleaseIntArrayElements(inputOffsets, offsets, 0);
    env->ReleaseIntArrayElements(inputByteStrides, strides, 0);
    return LCEVC_Success;
}
