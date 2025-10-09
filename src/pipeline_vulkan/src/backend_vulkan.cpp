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

#include "backend_vulkan.h"

#include "apply.h"
#include "blit.h"
#include "buffer_vulkan.h"
#include "conversion.h"
#include "frame_vulkan.h"
#include "picture_vulkan.h"
#include "upscale_horizontal.h"
#include "upscale_vertical.h"

#include <LCEVC/common/log.h>
#include <LCEVC/pipeline_vulkan/types_vulkan.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <sstream>
#include <type_traits>

namespace lcevc_dec::pipeline_vulkan {

struct PushConstants
{
    int kernel[4];
    int srcWidth;
    int srcHeight;
    int pa;
    int containerStrideIn;
    int containerOffsetIn;
    int containerStrideOut;
    int containerOffsetOut;
    int containerStrideBase;
    int containerOffsetBase;
};

struct PushConstantsApply
{
    int srcWidth;
    int srcHeight;
    int residualOffset;
    int containerStride;
    int containerOffset;
    int saturate;
    int layerCount;
    int tuRasterOrder;
    int tileX;
    int tileY;
    int tileWidth;
};

struct PushConstantsConversion
{
    int width;
    int containerStrideIn;
    int containerOffsetIn;
    int containerStrideOut;
    int containerOffsetOut;
    int containerStrideV; // used to check for nv12 and as input or output stride for V-plane
    int containerOffsetV; // used with nv12 as input or output offset for V-plane
    int bit8;
    int toInternal;
    int shift; // 5 for 10bit, 3 for 12bit, and 1 for 14bit
};

struct PushConstantsBlit
{
    int width;
    int height;
    int containerStride;
    int containerOffset;
};

namespace {
    template <typename, typename = std::void_t<>>
    struct HasContainerBase : std::false_type
    {};

    template <typename T>
    struct HasContainerBase<T, std::void_t<decltype(std::declval<T>().containerStrideBase),
                                           decltype(std::declval<T>().containerOffsetBase)>> : std::true_type
    {};

    template <typename T>
    inline void setContainerStrides(T& constants, int index, PictureVulkan* srcPicture,
                                    PictureVulkan* dstPicture, PictureVulkan* basePicture)
    {
        constants.containerStrideIn = srcPicture->layout.rowStrides[index] >> 2;
        constants.containerOffsetIn = srcPicture->layout.planeOffsets[index] >> 2;
        constants.containerStrideOut = dstPicture->layout.rowStrides[index] >> 2;
        constants.containerOffsetOut = dstPicture->layout.planeOffsets[index] >> 2;
        if constexpr (HasContainerBase<T>::value) {
            if (basePicture) {
                constants.containerStrideBase = basePicture->layout.rowStrides[index] >> 2;
                constants.containerOffsetBase = basePicture->layout.planeOffsets[index] >> 2;
            }
        }
    }

    inline bool updateConfig(uint8_t loq, const LdpPictureDesc& desc)
    {
        static LdpPictureDesc previousDesc[LOQEnhancedCount + 1];
        if (desc != previousDesc[loq]) {
            previousDesc[loq] = desc;
            return true;
        }
        return false;
    }
} // namespace

VKAPI_ATTR VkBool32 VKAPI_CALL
BackendVulkan::debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT,
                             const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void*)
{
    VNUnused(pCallbackData);
    COUT_STR(std::string("validation layer: ").append(std::string(pCallbackData->pMessage)));

    return VK_FALSE;
}

bool BackendVulkan::checkValidationLayerSupport()
{
    uint32_t layerCount{};
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> availableLayers(layerCount);

    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char* layerName : validationLayers) {
        bool found = false;
        for (const auto& layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                found = true;
                break;
            }
        }

        if (!found) {
            return false;
        }
    }
    return true;
}

std::vector<const char*> BackendVulkan::getRequiredExtensions()
{
    std::vector<const char*> extensions;
    extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
    if (enableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}

void BackendVulkan::populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
{
    createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
}

bool BackendVulkan::createInstance()
{
    if (enableValidationLayers && !checkValidationLayerSupport()) {
        VNLogError("validation layers requested, but not available!");
        return false;
    }

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vulkan Upscale";
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    auto extensions = getRequiredExtensions();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo;
    if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();

        populateDebugMessengerCreateInfo(debugCreateInfo);
        createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
    } else {
        createInfo.enabledLayerCount = 0;
        createInfo.pNext = nullptr;
    }

    if (vkCreateInstance(&createInfo, nullptr, &m_instance) != VK_SUCCESS) {
        VNLogError("failed to create instance!");
        return false;
    }
    return true;
}

VkResult BackendVulkan::CreateDebugUtilsMessengerEXT(VkInstance instance,
                                                     const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                                     const VkAllocationCallbacks* pAllocator,
                                                     VkDebugUtilsMessengerEXT* pDebugMessenger)
{
    auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }

    return VK_ERROR_EXTENSION_NOT_PRESENT;
}

void BackendVulkan::DestroyDebugUtilsMessengerEXT(VkInstance instance,
                                                  VkDebugUtilsMessengerEXT debugMessenger,
                                                  const VkAllocationCallbacks* pAllocator)
{
    auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

bool BackendVulkan::setupDebugMessenger()
{
    if (!enableValidationLayers) {
        return true;
    }
    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    populateDebugMessengerCreateInfo(createInfo);

    if (CreateDebugUtilsMessengerEXT(m_instance, &createInfo, nullptr, &m_debugMessenger) != VK_SUCCESS) {
        VNLogError("failed to set up debug messenger!");
        return false;
    }
    return true;
}

bool BackendVulkan::isDeviceSuitable(VkPhysicalDevice device)
{
    VkPhysicalDeviceProperties gpuProperties;
    vkGetPhysicalDeviceProperties(device, &gpuProperties);

    const auto realGpu = gpuProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ||
                         gpuProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;

    if (!realGpu) {
        return false;
    }

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    int i = 0;
    for (const auto& queueFamily : queueFamilies) {
        if (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT) {
            m_queueFamilyIndex = i;
            return true;
        }
        i++;
    }

    return false;
}

VkPhysicalDevice BackendVulkan::pickBestDevice(const std::vector<VkPhysicalDevice>& devices)
{
    if (devices.empty()) {
        return VK_NULL_HANDLE;
    }

    std::vector<std::pair<VkPhysicalDevice, uint32_t>> scoredList;
    for (const auto& device : devices) {
        uint32_t score = 0; // TODO - needs to be tuned

        VkPhysicalDeviceProperties deviceProperties{};
        vkGetPhysicalDeviceProperties(device, &deviceProperties);

        VNLogDebug("Suitable compute device: %s", deviceProperties.deviceName);

        score += deviceProperties.limits.maxComputeWorkGroupCount[0];
        VNLogDebug("maxComputeWorkGroupCount: %u %u %u",
                   deviceProperties.limits.maxComputeWorkGroupCount[0],
                   deviceProperties.limits.maxComputeWorkGroupCount[1],
                   deviceProperties.limits.maxComputeWorkGroupCount[2]);

        score += deviceProperties.limits.maxComputeWorkGroupSize[0];
        VNLogDebug("maxComputeWorkGroupSize: %u %u %u",
                   deviceProperties.limits.maxComputeWorkGroupSize[0],
                   deviceProperties.limits.maxComputeWorkGroupSize[1],
                   deviceProperties.limits.maxComputeWorkGroupSize[2]);

        score += deviceProperties.limits.maxComputeWorkGroupInvocations;
        VNLogDebug("maxComputeWorkGroupInvocations: %u",
                   deviceProperties.limits.maxComputeWorkGroupInvocations);

        VkPhysicalDeviceVulkan13Properties vulkan13properties{};
        vulkan13properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES;
        vulkan13properties.pNext = nullptr;

        VkPhysicalDeviceProperties2 deviceProperties2{};
        deviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        deviceProperties2.pNext = &vulkan13properties;
        vkGetPhysicalDeviceProperties2(device, &deviceProperties2);

        score += vulkan13properties.maxSubgroupSize;
        VNLogDebug("subgroupSize: %u", vulkan13properties.maxSubgroupSize);

        score += deviceProperties2.properties.limits.maxMemoryAllocationCount;
        VNLogDebug("maxMemoryAllocationCount: %u",
                   deviceProperties2.properties.limits.maxMemoryAllocationCount);

        scoredList.emplace_back(device, score);
    }

    std::sort(scoredList.begin(), scoredList.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    return scoredList.front().first;
}

bool BackendVulkan::pickPhysicalDevice()
{
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr);

    if (deviceCount == 0) {
        VNLogError("failed to find GPUs with Vulkan support!");
        return false;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(m_instance, &deviceCount, devices.data());

    std::vector<VkPhysicalDevice> suitableDevices;
    for (const auto& device : devices) {
        if (isDeviceSuitable(device)) {
            suitableDevices.push_back(device);
        }
    }

    m_physicalDevice = pickBestDevice(suitableDevices);

    if (m_physicalDevice == VK_NULL_HANDLE) {
        VNLogError("failed to find a suitable GPU from %d devices", deviceCount);
        return false;
    }

    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(m_physicalDevice, &deviceProperties);
    VNLogDebug("Using device: %s", deviceProperties.deviceName);

    return true;
}

bool BackendVulkan::createLogicalDeviceAndQueue()
{
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.queueFamilyIndex = m_queueFamilyIndex;
    const auto queuePriority = 1.0f;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    const VkPhysicalDeviceFeatures deviceFeatures{};
    VkDeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;

    deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();
    deviceCreateInfo.pEnabledFeatures = &deviceFeatures;

    if (enableValidationLayers) {
        deviceCreateInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        deviceCreateInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
        deviceCreateInfo.enabledLayerCount = 0;
    }

    if (vkCreateDevice(m_physicalDevice, &deviceCreateInfo, nullptr, &m_device) != VK_SUCCESS) {
        VNLogError("failed to create logical device!");
        return false;
    }

    vkGetDeviceQueue(m_device, m_queueFamilyIndex, 0, &m_queue);
    vkGetDeviceQueue(m_device, m_queueFamilyIndex, 0, &m_queueIntermediate);
    return true;
}

bool BackendVulkan::createBindingsAndPipelineLayout(uint32_t numBuffers, uint32_t pushConstantsSize,
                                                    VkDescriptorSetLayout& setLayoutFunc,
                                                    VkPipelineLayout& pipelineLayoutFunc)
{
    std::vector<VkDescriptorSetLayoutBinding> layoutBindings;

    for (uint32_t i = 0; i < numBuffers; i++) {
        VkDescriptorSetLayoutBinding layoutBinding{};
        layoutBinding.binding = i;
        layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBinding.descriptorCount = 1;
        layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        layoutBindings.push_back(layoutBinding);
    }

    VkDescriptorSetLayoutCreateInfo setLayoutCreateInfo{};
    setLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    setLayoutCreateInfo.bindingCount = static_cast<uint32_t>(layoutBindings.size());
    setLayoutCreateInfo.pBindings = layoutBindings.data();

    if (vkCreateDescriptorSetLayout(m_device, &setLayoutCreateInfo, nullptr, &setLayoutFunc) != VK_SUCCESS) {
        VNLogError("failed to create descriptor set layout!");
        return false;
    }

    VkPushConstantRange range = {};
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    range.offset = 0;
    range.size = pushConstantsSize;

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = &setLayoutFunc;
    pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
    pipelineLayoutCreateInfo.pPushConstantRanges = &range;

    if (vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayoutFunc)) {
        VNLogError("failed to create pipeline layout");
        return false;
    }

    return true;
}

bool BackendVulkan::createComputePipeline(const unsigned char* shaderName, size_t shaderSize,
                                          VkPipelineLayout& layout, VkPipeline& pipe, int wgSize)
{
    VkShaderModuleCreateInfo shaderModuleCreateInfo{};
    shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;

    shaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t*>(shaderName);
    shaderModuleCreateInfo.codeSize = shaderSize;

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    if (vkCreateShaderModule(m_device, &shaderModuleCreateInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        VNLogError("failed to create shader module");
        return false;
    }

    VkSpecializationMapEntry specializationMapEntry[3]{};
    specializationMapEntry[0].constantID = 0;
    specializationMapEntry[0].offset = 0;
    specializationMapEntry[0].size = 4;
    specializationMapEntry[1].constantID = 1;
    specializationMapEntry[1].offset = 4;
    specializationMapEntry[1].size = 4;
    specializationMapEntry[2].constantID = 2;
    specializationMapEntry[2].offset = 8;
    specializationMapEntry[2].size = 4;

    int32_t specializationData[] = {wgSize, wgSize, 1};

    VkSpecializationInfo specializationInfo{};
    specializationInfo.mapEntryCount = 3;
    specializationInfo.pMapEntries = specializationMapEntry;
    specializationInfo.dataSize = 12;
    specializationInfo.pData = specializationData;

    VkComputePipelineCreateInfo pipelineCreateInfo{};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineCreateInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineCreateInfo.stage.module = shaderModule;
    pipelineCreateInfo.stage.pSpecializationInfo = &specializationInfo;

    pipelineCreateInfo.stage.pName = "main";
    pipelineCreateInfo.layout = layout;

    if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &pipe) !=
        VK_SUCCESS) {
        VNLogError("failed to create compute pipeline");
        return false;
    }
    vkDestroyShaderModule(m_device, shaderModule, nullptr);

    return true;
}

uint32_t BackendVulkan::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    VNLogError("failed to find suitable memory type!");
    return 0;
}

bool BackendVulkan::allocateDescriptorSets()
{
    constexpr int descriptorSetCount = 5; // This is how many VkDescriptorSet data members total
    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo{};
    descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolCreateInfo.maxSets = descriptorSetCount;

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

    /*
    This is how many descriptors (buffers) total
    Vertical shader has 2 (input and output)
    Horizontal has 3 (input, output, and base)
    Apply has 2 (command buffer and output plane)
    Conversion has 2 (input buffer and output buffer)
    Blit has 2 (input buffer and output buffer)
    */
    poolSize.descriptorCount = 11;

    descriptorPoolCreateInfo.poolSizeCount = 1;
    descriptorPoolCreateInfo.pPoolSizes = &poolSize;
    if (vkCreateDescriptorPool(m_device, &descriptorPoolCreateInfo, nullptr, &m_descriptorPool) != VK_SUCCESS) {
        VNLogError("failed to create descriptor pool");
        return false;
    }

    VkDescriptorSetLayout setLayouts[descriptorSetCount]{};
    setLayouts[0] = m_setLayoutVertical;
    setLayouts[1] = m_setLayoutHorizontal;
    setLayouts[2] = m_setLayoutApply;
    setLayouts[3] = m_setLayoutConversion;
    setLayouts[4] = m_setLayoutBlit;

    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo{};
    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.descriptorPool = m_descriptorPool;
    descriptorSetAllocateInfo.descriptorSetCount = descriptorSetCount;
    descriptorSetAllocateInfo.pSetLayouts = setLayouts;

    VkDescriptorSet descriptorSets[descriptorSetCount];
    if (vkAllocateDescriptorSets(m_device, &descriptorSetAllocateInfo, descriptorSets) != VK_SUCCESS) {
        VNLogError("failed to allocate descriptor sets");
        return false;
    }

    m_descriptorSetSrcMid = descriptorSets[0];
    m_descriptorSetMidDst = descriptorSets[1];
    m_descriptorSetApply = descriptorSets[2];
    m_descriptorSetConversion = descriptorSets[3];
    m_descriptorSetBlit = descriptorSets[4];

    return true;
}

void BackendVulkan::updateComputeDescriptorSets(VkBuffer src, VkBuffer dst, VkDescriptorSet descriptorSet)
{
    std::vector<VkWriteDescriptorSet> descriptorSetWrites(2);
    std::vector<VkDescriptorBufferInfo> bufferInfos(2);

    VkWriteDescriptorSet writeDescriptorSet{};
    writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet.dstSet = descriptorSet;
    writeDescriptorSet.dstArrayElement = 0;
    writeDescriptorSet.descriptorCount = 1;
    writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

    VkDescriptorBufferInfo buffInfo{};
    buffInfo.offset = 0;
    buffInfo.range = VK_WHOLE_SIZE;

    buffInfo.buffer = src;
    bufferInfos[0] = buffInfo;
    buffInfo.buffer = dst;
    bufferInfos[1] = buffInfo;

    writeDescriptorSet.dstBinding = 0;
    writeDescriptorSet.pBufferInfo = &bufferInfos[0];
    descriptorSetWrites[0] = writeDescriptorSet;
    writeDescriptorSet.dstBinding = 1;
    writeDescriptorSet.pBufferInfo = &bufferInfos[1];
    descriptorSetWrites[1] = writeDescriptorSet;

    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(descriptorSetWrites.size()),
                           descriptorSetWrites.data(), 0, nullptr);
}

void BackendVulkan::updateComputeDescriptorSets(VkBuffer src, VkBuffer dst1, VkBuffer dst2,
                                                VkDescriptorSet descriptorSet)
{
    std::vector<VkWriteDescriptorSet> descriptorSetWrites(3);
    std::vector<VkDescriptorBufferInfo> bufferInfos(3);

    VkWriteDescriptorSet writeDescriptorSet{};
    writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet.dstSet = descriptorSet;
    writeDescriptorSet.dstArrayElement = 0;
    writeDescriptorSet.descriptorCount = 1;
    writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

    VkDescriptorBufferInfo buffInfo{};
    buffInfo.offset = 0;
    buffInfo.range = VK_WHOLE_SIZE;

    buffInfo.buffer = src;
    bufferInfos[0] = buffInfo;
    buffInfo.buffer = dst1;
    bufferInfos[1] = buffInfo;
    buffInfo.buffer = dst2;
    bufferInfos[2] = buffInfo;

    writeDescriptorSet.dstBinding = 0;
    writeDescriptorSet.pBufferInfo = &bufferInfos[0];
    descriptorSetWrites[0] = writeDescriptorSet;
    writeDescriptorSet.dstBinding = 1;
    writeDescriptorSet.pBufferInfo = &bufferInfos[1];
    descriptorSetWrites[1] = writeDescriptorSet;
    writeDescriptorSet.dstBinding = 2;
    writeDescriptorSet.pBufferInfo = &bufferInfos[2];
    descriptorSetWrites[2] = writeDescriptorSet;

    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(descriptorSetWrites.size()),
                           descriptorSetWrites.data(), 0, nullptr);
}

bool BackendVulkan::createCommandPoolAndBuffer()
{
    VkCommandPoolCreateInfo commandPoolCreateInfo{};
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    commandPoolCreateInfo.queueFamilyIndex = m_queueFamilyIndex;

    if (vkCreateCommandPool(m_device, &commandPoolCreateInfo, nullptr, &m_commandPool) != VK_SUCCESS) {
        VNLogError("failed to create command pool");
        return false;
    }

    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = m_commandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(m_device, &commandBufferAllocateInfo, &m_commandBuffer) != VK_SUCCESS) {
        VNLogError("failed to allocate command buffer");
        return false;
    }

    if (vkAllocateCommandBuffers(m_device, &commandBufferAllocateInfo, &m_commandBufferIntermediate) !=
        VK_SUCCESS) {
        VNLogError("failed to allocate command buffer");
        return false;
    }

    return true;
}

bool BackendVulkan::init()
{
    if (!createInstance()) {
        return false;
    }

    if (!setupDebugMessenger()) {
        return false;
    }

    if (!pickPhysicalDevice()) {
        return false;
    }

    if (!createLogicalDeviceAndQueue()) {
        return false;
    }

    if (!createBindingsAndPipelineLayout(2, sizeof(PushConstants), m_setLayoutVertical,
                                         m_pipelineLayoutVertical)) {
        return false;
    }

    if (!createBindingsAndPipelineLayout(3, sizeof(PushConstants), m_setLayoutHorizontal,
                                         m_pipelineLayoutHorizontal)) {
        return false;
    }

    if (!createBindingsAndPipelineLayout(2, sizeof(PushConstantsApply), m_setLayoutApply,
                                         m_pipelineLayoutApply)) {
        return false;
    }

    if (!createBindingsAndPipelineLayout(2, sizeof(PushConstantsConversion), m_setLayoutConversion,
                                         m_pipelineLayoutConversion)) {
        return false;
    }

    if (!createBindingsAndPipelineLayout(2, sizeof(PushConstantsBlit), m_setLayoutBlit,
                                         m_pipelineLayoutBlit)) {
        return false;
    }

    if (!createComputePipeline(upscale_vertical_spv, sizeof(upscale_vertical_spv),
                               m_pipelineLayoutVertical, m_pipelineVertical, workGroupSize)) {
        return false;
    }

    if (!createComputePipeline(upscale_horizontal_spv, sizeof(upscale_horizontal_spv),
                               m_pipelineLayoutHorizontal, m_pipelineHorizontal, workGroupSize)) {
        return false;
    }

    if (!createComputePipeline(apply_spv, sizeof(apply_spv), m_pipelineLayoutApply, m_pipelineApply,
                               this->workGroupSizeDebug)) {
        return false; // TODO - check this
    }

    if (!createComputePipeline(conversion_spv, sizeof(conversion_spv), m_pipelineLayoutConversion,
                               m_pipelineConversion, workGroupSizeDebug)) {
        return false; // TODO - check this
    }

    if (!createComputePipeline(blit_spv, sizeof(blit_spv), m_pipelineLayoutBlit, m_pipelineBlit,
                               workGroupSizeDebug)) {
        return false; // TODO - check this
    }

    if (!allocateDescriptorSets()) {
        return false;
    }

    if (!createCommandPoolAndBuffer()) {
        return false;
    }

    return true;
}

void BackendVulkan::dispatchCompute(int width, int height, VkCommandBuffer& cmdBuf, int wgSize, int packDensity)
{
    const int modX = width % (packDensity * wgSize);
    const int divX = width / (packDensity * wgSize);
    const int numGroupsX = modX ? divX + 1 : divX;
    const int modY = height % wgSize;
    const int divY = height / wgSize;
    const int numGroupsY = modY ? divY + 1 : divY;
    vkCmdDispatch(cmdBuf, numGroupsX, numGroupsY, 1);
}

bool BackendVulkan::apply(VulkanApplyArgs* params)
{
    const LdeCmdBufferGpu buffer = params->bufferGpu;
    const bool applyTemporal = (params->picture == nullptr) ? true : false;

    if (applyTemporal && params->plane == 0) {
        LdpPictureDesc desc{};
        desc.width = params->planeWidth;
        desc.height = params->planeHeight;
        desc.colorFormat = m_pipeline.chromaToColorFormat(params->chroma);
        if (updateConfig(LOQEnhancedCount, desc)) {
            params->temporalPicture->setDesc(desc);
            auto* temporalBuffer = static_cast<BufferVulkan*>(params->temporalPicture->buffer);
            std::memset(temporalBuffer->ptr(), 0, temporalBuffer->size());
        }
    }

    // TODO temp copy to Vulkan buffer
    const unsigned residualOffset = sizeof(LdeCmdBufferGpuCmd) * buffer.commandCount;
    uint32_t size = residualOffset + buffer.residualCount * sizeof(uint16_t);
    if (size == 0) { // TODO - check this. No commands but possibly still temporal refresh
        return true;
    }

    auto gpuCommandBuffer = std::make_unique<BufferVulkan>(*this, size);
    std::memcpy(gpuCommandBuffer->ptr(), buffer.commands, residualOffset);
    std::memcpy(gpuCommandBuffer->ptr() + residualOffset, buffer.residuals,
                buffer.residualCount * sizeof(uint16_t));

    auto* applyPlaneBuffer = applyTemporal ? static_cast<BufferVulkan*>(params->temporalPicture->buffer)
                                           : static_cast<BufferVulkan*>(params->picture->buffer);

    updateComputeDescriptorSets(gpuCommandBuffer->getVkBuffer(), applyPlaneBuffer->getVkBuffer(),
                                m_descriptorSetApply);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    PushConstantsApply constants{};

    vkBeginCommandBuffer(m_commandBuffer, &beginInfo);

    constants.srcWidth = params->planeWidth;
    constants.srcHeight = params->planeHeight;
    constants.containerStride = applyTemporal
                                    ? params->temporalPicture->layout.rowStrides[params->plane] >> 2
                                    : params->picture->layout.rowStrides[params->plane] >> 2;
    constants.containerOffset = applyTemporal
                                    ? params->temporalPicture->layout.planeOffsets[params->plane] >> 2
                                    : params->picture->layout.planeOffsets[params->plane] >> 2;

    constants.residualOffset = residualOffset;
    constants.saturate = params->highlightResiduals ? 1 : 0;
    constants.layerCount = buffer.layerCount;
    constants.tuRasterOrder = params->tuRasterOrder ? 1 : 0;
    constants.tileX = params->tileX;
    constants.tileY = params->tileY;
    constants.tileWidth = params->tileWidth;

    vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineApply);
    vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutApply,
                            0, 1, &m_descriptorSetApply, 0, nullptr);
    vkCmdPushConstants(m_commandBuffer, m_pipelineLayoutApply, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(PushConstantsApply), &constants);
    vkCmdDispatch(m_commandBuffer, buffer.commandCount, 1, 1); // TODO - check this

    if (vkEndCommandBuffer(m_commandBuffer) != VK_SUCCESS) {
        VNLogError("failed to end command buffer");
        return false;
    }

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &m_commandBuffer;

    const VkResult result = vkQueueSubmit(m_queue, 1, &submitInfo, VK_NULL_HANDLE);
    if (result != VK_SUCCESS) {
        VNLogError("Failed to submit compute apply queue!");
        return false;
    }

    vkDeviceWaitIdle(m_device);

    return true;
}

bool BackendVulkan::blit(VulkanBlitArgs* params)
{
    auto* srcPicture = params->src;
    auto* srcBuffer = static_cast<BufferVulkan*>(srcPicture->buffer);
    VkBuffer& srcVkBuffer = srcBuffer->getVkBuffer();

    auto* dstPicture = params->dst;
    auto* dstBuffer = static_cast<BufferVulkan*>(dstPicture->buffer);
    VkBuffer& dstVkBuffer = dstBuffer->getVkBuffer();

    LdpPictureDesc srcDesc{};
    srcPicture->getDesc(srcDesc);

    updateComputeDescriptorSets(srcVkBuffer, dstVkBuffer, m_descriptorSetBlit);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    vkBeginCommandBuffer(m_commandBuffer, &beginInfo);
    vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineBlit);
    vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutBlit,
                            0, 1, &m_descriptorSetBlit, 0, nullptr);

    PushConstantsBlit constants{};
    for (uint8_t plane = 0; plane < params->numEnhancedPlanes; ++plane) {
        int widthShift = 0;
        int heightShift = 0;
        if (plane != 0) {
            m_pipeline.getSubsamplingShifts(params->chroma, widthShift, heightShift);
        }
        constants.width = srcDesc.width >> widthShift;
        constants.height = srcDesc.height >> heightShift;
        constants.containerStride = srcPicture->layout.rowStrides[plane] >> 2;
        constants.containerOffset = srcPicture->layout.planeOffsets[plane] >> 2;

        vkCmdPushConstants(m_commandBuffer, m_pipelineLayoutBlit, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(PushConstantsBlit), &constants);

        dispatchCompute(constants.width, constants.height, m_commandBuffer, workGroupSizeDebug);
    }

    if (vkEndCommandBuffer(m_commandBuffer) != VK_SUCCESS) {
        VNLogError("failed to end command buffer");
        return false;
    }

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &m_commandBuffer;

    auto result = vkQueueSubmit(m_queue, 1, &submitInfo, VK_NULL_HANDLE);
    if (result != VK_SUCCESS) {
        VNLogError("Failed to submit compute queue!");
        return false;
    }

    vkDeviceWaitIdle(m_device);

    return true;
}

bool BackendVulkan::conversion(VulkanConversionArgs* params)
{
    auto* srcPicture = params->src;
    auto* srcBuffer = static_cast<BufferVulkan*>(srcPicture->buffer);
    VkBuffer& srcVkBuffer = srcBuffer->getVkBuffer();

    auto* dstPicture = params->dst;
    auto* dstBuffer = static_cast<BufferVulkan*>(dstPicture->buffer);
    VkBuffer& dstVkBuffer = dstBuffer->getVkBuffer();

    PushConstantsConversion constants{};
    LdpPictureDesc srcDesc{}, dstDesc{};
    srcPicture->getDesc(srcDesc);
    dstPicture->getDesc(dstDesc);

    auto isBit8 = [](LdpColorFormat colorFormat) {
        return colorFormat == LdpColorFormatI444_8 || colorFormat == LdpColorFormatI422_8 ||
               colorFormat == LdpColorFormatI420_8 || colorFormat == LdpColorFormatGRAY_8 ||
               colorFormat == LdpColorFormatNV12_8 || colorFormat == LdpColorFormatNV21_8;
    };
    const auto srcBit8 = isBit8(srcDesc.colorFormat);
    const auto dstBit8 = isBit8(dstDesc.colorFormat);

    constants.width = srcPicture->layout.width;
    auto height = srcPicture->layout.height;
    constants.bit8 = params->toInternal ? static_cast<int>(srcBit8) : static_cast<int>(dstBit8);
    constants.toInternal = params->toInternal ? 1 : 0;
    constants.shift = 15 - params->bitDepth;

    setContainerStrides(constants, 0, srcPicture, dstPicture, nullptr);

    constants.containerStrideV = 0; // used to signal no nv12
    constants.containerOffsetV = 0;

    updateComputeDescriptorSets(srcVkBuffer, dstVkBuffer, m_descriptorSetConversion);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    vkBeginCommandBuffer(m_commandBuffer, &beginInfo);
    vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineConversion);
    vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            m_pipelineLayoutConversion, 0, 1, &m_descriptorSetConversion, 0, nullptr);
    vkCmdPushConstants(m_commandBuffer, m_pipelineLayoutConversion, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(PushConstantsConversion), &constants);

    const int packDensity = (srcBit8 || dstBit8) ? 4 : 2;
    dispatchCompute(constants.width, height, m_commandBuffer, workGroupSize, packDensity); // Y

    if (params->chroma != LdeChroma::CTMonochrome) {
        int widthShift{};
        int heightShift{};
        m_pipeline.getSubsamplingShifts(params->chroma, widthShift, heightShift);

        const auto srcNv12 = srcDesc.colorFormat == LdpColorFormatNV12_8;
        const auto dstNv12 = dstDesc.colorFormat == LdpColorFormatNV12_8;
        const auto srcNv21 = srcDesc.colorFormat == LdpColorFormatNV21_8;
        const auto dstNv21 = dstDesc.colorFormat == LdpColorFormatNV21_8;
        const auto nv12 = srcNv12 || srcNv21 || dstNv12 || dstNv21;
        if (!nv12) {
            constants.width = srcPicture->layout.width >> widthShift;
        } else {
            // for nv12 src, these will be used as V-plane outputs, otherwise V-plane inputs
            constants.containerStrideV = srcNv12 ? dstPicture->layout.rowStrides[2] >> 2
                                                 : srcPicture->layout.rowStrides[2] >> 2;
            constants.containerOffsetV = srcNv12 ? dstPicture->layout.planeOffsets[2] >> 2
                                                 : srcPicture->layout.planeOffsets[2] >> 2;
        }
        height = srcPicture->layout.height >> heightShift;
        setContainerStrides(constants, 1, srcPicture, dstPicture, nullptr);

        vkCmdPushConstants(m_commandBuffer, m_pipelineLayoutConversion, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(PushConstantsConversion), &constants);
        dispatchCompute(constants.width, height, m_commandBuffer, workGroupSize, packDensity); // U

        if (!nv12) {
            setContainerStrides(constants, 2, srcPicture, dstPicture, nullptr);
            vkCmdPushConstants(m_commandBuffer, m_pipelineLayoutConversion, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(PushConstantsConversion), &constants);
            dispatchCompute(constants.width, height, m_commandBuffer, workGroupSize, packDensity); // V
        }
    }

    if (vkEndCommandBuffer(m_commandBuffer) != VK_SUCCESS) {
        VNLogError("failed to end command buffer");
        return false;
    }

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &m_commandBuffer;

    auto result = vkQueueSubmit(m_queue, 1, &submitInfo, VK_NULL_HANDLE);
    if (result != VK_SUCCESS) {
        VNLogError("Failed to submit compute queue!");
        return false;
    }

    vkDeviceWaitIdle(m_device);

    return true;
}

bool BackendVulkan::upscaleVertical(const LdeKernel* kernel, VulkanUpscaleArgs* params)
{
    PictureVulkan* srcPicture = params->src;
    PictureVulkan* dstPicture = params->dst;

    LdpPictureDesc srcDesc;
    srcPicture->getDesc(srcDesc);

    // Get VkBuffers to update descriptor sets (attach buffers to shaders)
    auto* srcBuffer = static_cast<BufferVulkan*>(srcPicture->buffer);
    VkBuffer& srcVkBuffer = srcBuffer->getVkBuffer();

    auto* dstBuffer = static_cast<BufferVulkan*>(dstPicture->buffer);
    VkBuffer& dstVkBuffer = dstBuffer->getVkBuffer();

    // Vertical set for upscaling from base and writing to intermediate
    updateComputeDescriptorSets(srcVkBuffer, dstVkBuffer, m_descriptorSetSrcMid);

    PushConstants constants{};
    if (kernel->length == 2) {
        constants.kernel[0] = 0;
        constants.kernel[1] = kernel->coeffs[0][0];
        constants.kernel[2] = kernel->coeffs[0][1];
        constants.kernel[3] = 0;
    } else {
        for (int i = 0; i < 4; ++i) {
            constants.kernel[i] = kernel->coeffs[0][i];
        }
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(m_commandBufferIntermediate, &beginInfo);

    VkMemoryBarrier memoryBarrierIntermediate{};
    memoryBarrierIntermediate.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrierIntermediate.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    memoryBarrierIntermediate.dstAccessMask = VK_ACCESS_NONE;
    vkCmdPipelineBarrier(m_commandBufferIntermediate,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, 1, &memoryBarrierIntermediate, 0,
                         nullptr, 0, nullptr);

    constants.srcWidth = srcDesc.width;
    constants.srcHeight = srcDesc.height;
    constants.pa = 0;
    setContainerStrides(constants, 0, srcPicture, dstPicture, nullptr);

    vkCmdBindPipeline(m_commandBufferIntermediate, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineVertical);
    vkCmdBindDescriptorSets(m_commandBufferIntermediate, VK_PIPELINE_BIND_POINT_COMPUTE,
                            m_pipelineLayoutVertical, 0, 1, &m_descriptorSetSrcMid, 0, nullptr);
    vkCmdPushConstants(m_commandBufferIntermediate, m_pipelineLayoutVertical,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &constants);
    dispatchCompute(constants.srcWidth, constants.srcHeight, m_commandBufferIntermediate,
                    workGroupSize); // Y

    if (params->chroma != LdeChroma::CTMonochrome) {
        int widthShift{};
        int heightShift{};
        m_pipeline.getSubsamplingShifts(params->chroma, widthShift, heightShift);

        constants.srcWidth = srcDesc.width >> widthShift;
        constants.srcHeight = srcDesc.height >> heightShift;
        setContainerStrides(constants, 1, srcPicture, dstPicture, nullptr);

        vkCmdPushConstants(m_commandBufferIntermediate, m_pipelineLayoutVertical,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &constants);
        dispatchCompute(constants.srcWidth, constants.srcHeight, m_commandBufferIntermediate,
                        workGroupSize); // U

        setContainerStrides(constants, 2, srcPicture, dstPicture, nullptr);

        vkCmdPushConstants(m_commandBufferIntermediate, m_pipelineLayoutVertical,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &constants);
        dispatchCompute(constants.srcWidth, constants.srcHeight, m_commandBufferIntermediate,
                        workGroupSize); // V
    }

    if (vkEndCommandBuffer(m_commandBufferIntermediate) != VK_SUCCESS) {
        VNLogError("failed to end command buffer");
        return false;
    }

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &m_commandBufferIntermediate;

    auto result = vkQueueSubmit(m_queueIntermediate, 1, &submitInfo, VK_NULL_HANDLE);
    if (result != VK_SUCCESS) {
        VNLogError("Failed to submit compute queue!");
        return false;
    }
    vkDeviceWaitIdle(m_device);

    return true;
}

bool BackendVulkan::upscaleHorizontal(const LdeKernel* kernel, VulkanUpscaleArgs* params)
{
    PictureVulkan* srcPicture = params->src;
    PictureVulkan* dstPicture = params->dst;
    PictureVulkan* basePicture = params->base;

    assert(srcPicture);
    assert(dstPicture);
    assert(basePicture);

    LdpPictureDesc srcDesc;
    srcPicture->getDesc(srcDesc);

    // Get VkBuffers to update descriptor sets (attach buffers to shaders)
    auto* srcBuffer = static_cast<BufferVulkan*>(srcPicture->buffer);
    assert(srcBuffer);
    VkBuffer& srcVkBuffer = srcBuffer->getVkBuffer();

    auto* baseBuffer = static_cast<BufferVulkan*>(basePicture->buffer);
    assert(baseBuffer);
    VkBuffer& baseVkBuffer = baseBuffer->getVkBuffer();

    auto* dstBuffer = static_cast<BufferVulkan*>(dstPicture->buffer);
    assert(dstBuffer);
    VkBuffer& dstVkBuffer = dstBuffer->getVkBuffer();

    // Horizontal set for upscaling from intermediate, applying PA from base, and writing to output
    updateComputeDescriptorSets(srcVkBuffer, dstVkBuffer, baseVkBuffer, m_descriptorSetMidDst);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    PushConstants constants{};
    if (kernel->length == 2) {
        constants.kernel[0] = 0;
        constants.kernel[1] = kernel->coeffs[0][0];
        constants.kernel[2] = kernel->coeffs[0][1];
        constants.kernel[3] = 0;
    } else {
        for (int i = 0; i < 4; ++i) {
            constants.kernel[i] = kernel->coeffs[0][i];
        }
    }

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &m_commandBufferIntermediate;

    vkBeginCommandBuffer(m_commandBuffer, &beginInfo);

    VkMemoryBarrier memoryBarrier{};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0,
                         nullptr);

    constants.srcWidth = srcDesc.width;
    constants.srcHeight = srcDesc.height;
    constants.pa = params->applyPA;

    setContainerStrides(constants, 0, srcPicture, dstPicture, basePicture);

    vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineHorizontal);
    vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            m_pipelineLayoutHorizontal, 0, 1, &m_descriptorSetMidDst, 0, nullptr);
    vkCmdPushConstants(m_commandBuffer, m_pipelineLayoutHorizontal, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(PushConstants), &constants);
    dispatchCompute(constants.srcWidth, constants.srcHeight >> 1, m_commandBuffer,
                    workGroupSize); // Y. constants.srcHeight / 2 because we are doing 2 rows at a time for PA

    if (params->chroma != LdeChroma::CTMonochrome) {
        int widthShift{};
        int heightShift{};
        m_pipeline.getSubsamplingShifts(params->chroma, widthShift, heightShift);

        constants.srcWidth = srcDesc.width >> widthShift;
        constants.srcHeight = srcDesc.height >> heightShift;

        setContainerStrides(constants, 1, srcPicture, dstPicture, basePicture);

        vkCmdPushConstants(m_commandBuffer, m_pipelineLayoutHorizontal, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(PushConstants), &constants);
        dispatchCompute(constants.srcWidth, constants.srcHeight, m_commandBuffer, workGroupSize); // U

        setContainerStrides(constants, 2, srcPicture, dstPicture, basePicture);

        vkCmdPushConstants(m_commandBuffer, m_pipelineLayoutHorizontal, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(PushConstants), &constants);
        dispatchCompute(constants.srcWidth, constants.srcHeight, m_commandBuffer,
                        workGroupSize); // V
    }

    if (vkEndCommandBuffer(m_commandBuffer) != VK_SUCCESS) {
        VNLogError("failed to end command buffer");
        return false;
    }

    submitInfo.pCommandBuffers = &m_commandBuffer;
    auto result = vkQueueSubmit(m_queue, 1, &submitInfo, VK_NULL_HANDLE);
    if (result != VK_SUCCESS) {
        VNLogError("Failed to submit compute queue!");
        return false;
    }

    vkDeviceWaitIdle(m_device);

    return true;
}

bool BackendVulkan::upscaleFrame(const LdeKernel* kernel, VulkanUpscaleArgs* params)
{
    params->base = params->src;
    LdpPictureDesc desc;
    params->src->getDesc(desc);
    if (params->mode == Scale1D) {
        params->vertical ? desc.height *= 2 : desc.width *= 2;
        params->dst->functions->setDesc(params->dst, &desc);
        params->applyPA = params->applyPA ? 1 : 0;
        params->vertical ? upscaleVertical(kernel, params) : upscaleHorizontal(kernel, params);
    } else if (params->mode == Scale2D) {
        PictureVulkan* output = params->dst;
        if (updateConfig(params->loq1 ? 1 : 0, desc)) {
            LdpPictureDesc intDesc{desc};
            intDesc.height *= 2;
            params->intermediateUpscalePicture[params->loq1 ? LOQ1 : LOQ0]->setDesc(intDesc);
        }
        params->dst = params->intermediateUpscalePicture[params->loq1 ? LOQ1 : LOQ0];
        upscaleVertical(kernel, params);

        desc.width *= 2;
        desc.height *= 2;
        output->functions->setDesc(output, &desc);
        params->src = params->dst;
        params->dst = output;
        params->applyPA = params->applyPA ? 2 : 0;
        upscaleHorizontal(kernel, params);
    }
    return true;
}

void BackendVulkan::destroy()
{
    for (int i = 0; i < NUM_PLANES; ++i) {
#if defined(ANDROID_BUFFERS)
        AHardwareBuffer_release(srcHardwareBuffer[i]);
        AHardwareBuffer_release(midHardwareBuffer[i]);
        AHardwareBuffer_release(dstHardwareBuffer[i]);
#endif
    }

    vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
    vkFreeCommandBuffers(m_device, m_commandPool, 1, &m_commandBuffer);
    vkFreeCommandBuffers(m_device, m_commandPool, 1, &m_commandBufferIntermediate);
    vkDestroyCommandPool(m_device, m_commandPool, nullptr);

    vkDestroyPipeline(m_device, m_pipelineVertical, nullptr);
    vkDestroyPipeline(m_device, m_pipelineHorizontal, nullptr);
    vkDestroyPipeline(m_device, m_pipelineApply, nullptr);
    vkDestroyPipeline(m_device, m_pipelineConversion, nullptr);
    vkDestroyPipeline(m_device, m_pipelineBlit, nullptr);

    vkDestroyPipelineLayout(m_device, m_pipelineLayoutVertical, nullptr);
    vkDestroyPipelineLayout(m_device, m_pipelineLayoutHorizontal, nullptr);
    vkDestroyPipelineLayout(m_device, m_pipelineLayoutApply, nullptr);
    vkDestroyPipelineLayout(m_device, m_pipelineLayoutConversion, nullptr);
    vkDestroyPipelineLayout(m_device, m_pipelineLayoutBlit, nullptr);

    vkDestroyDescriptorSetLayout(m_device, m_setLayoutVertical, nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_setLayoutHorizontal, nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_setLayoutApply, nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_setLayoutConversion, nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_setLayoutBlit, nullptr);

    vkDestroyDevice(m_device, nullptr);

    if (enableValidationLayers) {
        DestroyDebugUtilsMessengerEXT(m_instance, m_debugMessenger, nullptr);
    }

    vkDestroyInstance(m_instance, nullptr);
}

} // namespace lcevc_dec::pipeline_vulkan