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

#ifndef VN_LCEVC_PIPELINE_VULKAN_BACKEND_VULKAN_H
#define VN_LCEVC_PIPELINE_VULKAN_BACKEND_VULKAN_H

#include <vulkan/vulkan.h>

#if defined(ANDROID)
// #define ANDROID_BUFFERS
#include <android/log.h>
#include <vulkan/vulkan_android.h>

#if defined(ANDROID_BUFFERS)
#include <android/hardware_buffer.h>
#include <android/hardware_buffer_jni.h>
#endif

#if defined(DEBUG_LOGGING)
#define COUT_STR(x) __android_log_print(ANDROID_LOG_DEBUG, "vulkan_upscale", "%s", (x).c_str());
#else
#define COUT_STR(x)
#endif

#else

#if defined(DEBUG_LOGGING)
#define COUT_STR(x) std::cout << (x) << std::endl;
#else
#define COUT_STR(x)
#endif

#endif

#include <LCEVC/enhancement/bitstream_types.h>

#include <cstring>
#include <string>
#include <vector>

namespace lcevc_dec::pipeline_vulkan {
class PipelineVulkan;
struct VulkanApplyArgs;
struct VulkanBlitArgs;
struct VulkanConversionArgs;
struct VulkanUpscaleArgs;

class BackendVulkan
{
public:
    BackendVulkan(PipelineVulkan& pipeline)
        : m_pipeline(pipeline){};
    ~BackendVulkan() = default;

    bool init();

    bool apply(VulkanApplyArgs* params);
    bool blit(VulkanBlitArgs* params);
    bool conversion(VulkanConversionArgs* params);
    bool upscaleVertical(const LdeKernel* kernel, VulkanUpscaleArgs* params);
    bool upscaleHorizontal(const LdeKernel* kernel, VulkanUpscaleArgs* params);
    bool upscaleFrame(const LdeKernel* kernel, VulkanUpscaleArgs* params);
    void destroy();

    bool checkValidationLayerSupport();
    std::vector<const char*> getRequiredExtensions();
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
    bool createInstance();
    VkResult CreateDebugUtilsMessengerEXT(VkInstance instance,
                                          const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                          const VkAllocationCallbacks* pAllocator,
                                          VkDebugUtilsMessengerEXT* pDebugMessenger);
    void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger,
                                       const VkAllocationCallbacks* pAllocator);
    bool setupDebugMessenger();
    bool isDeviceSuitable(VkPhysicalDevice device);
    static VkPhysicalDevice pickBestDevice(const std::vector<VkPhysicalDevice>& devices);
    bool pickPhysicalDevice();
    bool createLogicalDeviceAndQueue();
    bool createBindingsAndPipelineLayout(uint32_t numBuffers, uint32_t pushConstantsSize,
                                         VkDescriptorSetLayout& setLayout,
                                         VkPipelineLayout& pipelineLayout);
    bool createComputePipeline(const unsigned char* shaderName, size_t shaderSize,
                               VkPipelineLayout& layout, VkPipeline& pipe, int wgSize);

    bool allocateDescriptorSets();
    void updateComputeDescriptorSets(VkBuffer src, VkBuffer dst, VkDescriptorSet descriptorSet);
    void updateComputeDescriptorSets(VkBuffer src, VkBuffer dst1, VkBuffer dst2,
                                     VkDescriptorSet descriptorSet);
    bool createCommandPoolAndBuffer();
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                        VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                                        void* pUserData);

    void dispatchCompute(int width, int height, VkCommandBuffer& cmdBuf, int wgSize, int packDensity = 2);

    bool mapMemory(VkDeviceMemory& memory, uint8_t* mappedPtr);
    bool unmapMemory(VkDeviceMemory& memory, uint8_t* mappedPtr);

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    VkDevice& getDevice() { return m_device; }
    uint32_t getQueueFamilyIndex() const { return m_queueFamilyIndex; }

private:
    static const int NUM_PLANES = 3;
    const int workGroupSize = 1;
    const int workGroupSizeDebug = 1;
    bool m_realGpu = false;

    const std::vector<const char*> validationLayers = {"VK_LAYER_KHRONOS_validation"};

    const std::vector<const char*> deviceExtensions = {
#if defined(ANDROID)
    //"VK_ANDROID_external_memory_android_hardware_buffer",
    //"VK_EXT_queue_family_foreign"
#endif
    };

    VkInstance m_instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT m_debugMessenger;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    uint32_t m_queueFamilyIndex = 0;
    VkDevice m_device;
    VkQueue m_queue;
    VkQueue m_queueIntermediate;
    VkDescriptorSetLayout m_setLayoutVertical;
    VkDescriptorSetLayout m_setLayoutHorizontal;
    VkDescriptorSetLayout m_setLayoutApply;
    VkDescriptorSetLayout m_setLayoutConversion;
    VkDescriptorSetLayout m_setLayoutBlit;
    VkPipelineLayout m_pipelineLayoutVertical;
    VkPipelineLayout m_pipelineLayoutHorizontal;
    VkPipelineLayout m_pipelineLayoutApply;
    VkPipelineLayout m_pipelineLayoutConversion;
    VkPipelineLayout m_pipelineLayoutBlit;
    VkPipeline m_pipelineVertical;
    VkPipeline m_pipelineHorizontal;
    VkPipeline m_pipelineApply;
    VkPipeline m_pipelineConversion;
    VkPipeline m_pipelineBlit;
    VkDescriptorPool m_descriptorPool;
    VkDescriptorSet m_descriptorSetSrcMid; // 2 buffers: 1.base -> VERTICAL -> 2.output
    VkDescriptorSet m_descriptorSetMidDst; // 3 buffers: 1.vertical + 2.base -> HORIZONTAL -> 3.output
    VkDescriptorSet m_descriptorSetApply;      // 2 buffers: 1.commands -> APPLY -> 2.output plane
    VkDescriptorSet m_descriptorSetConversion; // 2 buffers: 1.input -> CONVERSION -> 2.output plane
    VkDescriptorSet m_descriptorSetBlit;       // 2 buffers: 1.input -> BLIT -> 2.output plane
    VkCommandPool m_commandPool;
    VkCommandBuffer m_commandBuffer;
    VkCommandBuffer m_commandBufferIntermediate;

#if defined(ANDROID_BUFFERS)
    AHardwareBuffer* srcHardwareBuffer[NUM_PLANES];
    AHardwareBuffer* midHardwareBuffer[NUM_PLANES];
    AHardwareBuffer* dstHardwareBuffer[NUM_PLANES];
#endif

    PipelineVulkan& m_pipeline;
};
} // namespace lcevc_dec::pipeline_vulkan

#endif // VN_LCEVC_PIPELINE_VULKAN_BACKEND_VULKAN_H
