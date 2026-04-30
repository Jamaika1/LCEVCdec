/* Copyright (c) V-Nova International Limited 2025-2026. All rights reserved.
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

#include <LCEVC/pipeline/buffer_base.h>
#include <LCEVC/pipeline/picture_base.h>
//
#include <LCEVC/common/log.h>
#include <LCEVC/pipeline/buffer_alignment.h>

namespace lcevc_dec::pipeline {

namespace {
    extern const LdpPictureFunctions kPictureFunctions;
}

PictureBase::PictureBase(PipelineBase& pipeline)
    : LdpPicture{&kPictureFunctions}
    , m_pipeline(pipeline)
{
    ldpPictureLayoutInitialize(&layout, LdpColorFormatUnknown, 0, 0, 0);

    // Fill in defaults
    colorRange = LdpColorRangeUnknown;
    colorPrimaries = LdpColorPrimariesUnspecified;
    matrixCoefficients = LdpMatrixCoefficientsUnspecified;
    transferCharacteristics = LdpTransferCharacteristicsUnspecified;
    hdrStaticInfo = {};
    publicFlags = 0;
    sampleAspectRatio = {1, 1};
    margins = {};
    userData = nullptr;
}

PictureBase::~PictureBase()
{
    // Should have already unlocked (and unbound) by now, in the child class.
    assert(!isLocked());
}

bool PictureBase::isValid() const
{
    if (ldpPictureLayoutFormat(&layout) == LdpColorFormatUnknown) {
        return false;
    }

    return true;
}

bool PictureBase::getPublicFlag(uint8_t flag) const
{
    return ((publicFlags & (0x01 << (flag - 1))) != 0);
}

void PictureBase::setPublicFlag(uint8_t flag, bool value)
{
    // flags start at 1 so, subtract 1 to make sure we use all 8 bits.
    if (value) {
        publicFlags = publicFlags | static_cast<uint8_t>((0x1 << (flag - 1)));
    } else {
        publicFlags = publicFlags & static_cast<uint8_t>(~(0x1 << (flag - 1)));
    }
}

void PictureBase::getDesc(LdpPictureDesc& desc) const
{
    desc.colorFormat = ldpPictureLayoutFormat(&layout);
    desc.colorRange = colorRange;
    desc.colorPrimaries = colorPrimaries;
    desc.matrixCoefficients = matrixCoefficients;
    desc.transferCharacteristics = transferCharacteristics;
    desc.hdrStaticInfo = hdrStaticInfo;
    desc.sampleAspectRatioDen = sampleAspectRatio.denominator;
    desc.sampleAspectRatioNum = sampleAspectRatio.numerator;
    desc.width = getActiveWidth();
    desc.height = getActiveHeight();
    desc.cropTop = margins.top;
    desc.cropBottom = margins.bottom;
    desc.cropLeft = margins.left;
    desc.cropRight = margins.right;
}

bool PictureBase::descsMatch(const LdpPictureDesc& desc) const
{
    if (ldpPictureLayoutPlanes(&layout) == 0) {
        return false; // Picture isn't initialized so cannot match
    }
    LdpPictureDesc curDesc;
    getDesc(curDesc);
    return desc == curDesc;
}

bool PictureBase::setDesc(const LdpPictureDesc& newDesc)
{
    if (descsMatch(newDesc)) {
        // Don't need to do anything
        return true;
    }

    if (!initializeDesc(newDesc, nullptr)) {
        return false;
    }

    if (!unbindMemory()) {
        return false;
    }

    return true;
}

bool PictureBase::setDescAndBind(const LdpPictureDesc& newDesc, pipeline::BufferUsage usage)
{
    if (!setDesc(newDesc)) {
        return false;
    }

    if (buffer) {
        return true;
    }

    return bindMemory(usage);
}

bool PictureBase::getBufferDesc(LdpPictureBufferDesc& bufferDescOut) const
{
    if (!m_external) {
        return false;
    }
    bufferDescOut = m_externaBufferDesc;
    return true;
}

bool PictureBase::getPlaneDescArr(LdpPicturePlaneDesc planeDescArrOut[kLdpPictureMaxNumPlanes]) const
{
    if (!m_external) {
        return false;
    }

    for (uint32_t i = 0; i < kLdpPictureMaxNumPlanes; ++i) {
        planeDescArrOut[i] = m_externalPlaneDescs[i];
    }
    return true;
}

bool PictureBase::initializeDesc(const LdpPictureDesc& desc,
                                 const uint32_t rowStridesBytes[kLdpPictureMaxNumPlanes])
{
    // Note that error messages in this function just uses the name, rather than the full debug
    // string. This is because the debug string reports format data that isn't meaningful until
    // AFTER initializeDesc succeeds.

    if (isLocked()) {
        VNLogError("Picture is locked, so cannot set desc.");
        return false;
    }

    if (desc.colorFormat == LdpColorFormatUnknown) {
        VNLogError("Invalid format, cannot set desc.");
        return false;
    }

    colorRange = desc.colorRange;
    colorPrimaries = desc.colorPrimaries;
    matrixCoefficients = desc.matrixCoefficients;
    transferCharacteristics = desc.transferCharacteristics;
    hdrStaticInfo = desc.hdrStaticInfo;
    sampleAspectRatio = {desc.sampleAspectRatioNum, desc.sampleAspectRatioDen};

    if (rowStridesBytes) {
        if (!ldpPictureDescCheckValidStrides(&desc, rowStridesBytes)) {
            VNLogError("Invalid strides given for %dx%d plane", desc.width, desc.height);
            return false;
        }
        ldpPictureLayoutInitializeDescStrides(&layout, &desc, rowStridesBytes);
    } else {
        ldpPictureLayoutInitializeDesc(&layout, &desc, kBufferRowAlignment);
    }

    if (((desc.cropLeft + desc.cropRight) > desc.width) ||
        ((desc.cropTop + desc.cropBottom) > desc.height)) {
        VNLogError("Requested to crop out more than the whole picture. Requested crops are: left "
                   "%u, right %u, top %u, bottom %u. Size is %ux%u.",
                   desc.cropLeft, desc.cropRight, desc.cropTop, desc.cropBottom, desc.width, desc.height);
        return false;
    }
    margins = {desc.cropLeft, desc.cropTop, desc.cropRight, desc.cropBottom};

    return true;
}

void PictureBase::setExternal(const LdpPicturePlaneDesc* planeDescArr, const LdpPictureBufferDesc* bufferDesc)
{
    m_external = true;
    for (uint32_t plane = 0; plane < ldpPictureLayoutPlanes(&layout); ++plane) {
        if (planeDescArr) {
            m_externalPlaneDescs[plane] = planeDescArr[plane];
            layout.rowStrides[plane] = planeDescArr[plane].rowByteStride;
        } else {
            m_externalPlaneDescs[plane].firstSample =
                bufferDesc->data + ldpPictureLayoutPlaneOffset(&layout, plane);
            m_externalPlaneDescs[plane].rowByteStride =
                ldpPictureLayoutDefaultRowStride(&layout, plane, kBufferRowAlignment);
        }
    }

    if (bufferDesc) {
        m_externaBufferDesc = *bufferDesc;
    }
}

uint32_t PictureBase::getRequiredSize() const
{
    uint32_t totalSize = 0;
    for (uint32_t i = 0; i < ldpPictureLayoutPlanes(&layout); i++) {
        totalSize += ldpPictureLayoutPlaneSize(&layout, i);
    }
    return totalSize;
}

void PictureBase::getPlaneDescInternal(uint32_t plane, LdpPicturePlaneDesc& desc) const
{
    assert(plane < kLdpPictureMaxNumPlanes);

    if (!m_external) {
        VNLogFatal("Cannot get PlaneDesc.");
        assert(0);
    } else {
        desc = m_externalPlaneDescs[plane];
    }
}

bool PictureBase::bindMemory(pipeline::BufferUsage usage)
{
    VNLogVerbose("BIND <%p>", (void*)this);

    if (isLocked()) {
        return false;
    }

    // Check for empty
    byteOffset = 0;
    byteSize = getRequiredSize();
    if (byteSize == 0) {
        return false;
    }

    // No buffer - just make a new one
    if (buffer == nullptr) {
        buffer = m_pipeline.allocateBuffer(byteSize, usage);
        return buffer != nullptr;
        return true;
    }

    // Existing buffer
    BufferBase* bp = static_cast<BufferBase*>(buffer);

    // Check usage
    if (bp->usage() != usage) {
        m_pipeline.releaseBuffer(static_cast<BufferBase*>(buffer));
        buffer = m_pipeline.allocateBuffer(byteSize, usage);
        return buffer != nullptr;
    }

    // Resize existing buffer
    if (byteSize > bp->size() && !bp->resize(byteSize)) {
        return false;
    }

    bp->clear();

    return true;
}

bool PictureBase::unbindMemory()
{
    VNLogVerbose("UNBIND <%p>", (void*)this);

    if (isLocked()) {
        return false;
    }
    bool ret = true;
    if (buffer != nullptr) {
        m_pipeline.releaseBuffer(static_cast<BufferBase*>(buffer));
        buffer = nullptr;
    }
    return ret;

    return true;
}

// C function table to connect to C++ class
//
namespace {
    bool setDesc(LdpPicture* picture, const LdpPictureDesc* desc)
    {
        return static_cast<PictureBase*>(picture)->setDesc(*desc);
    }
    void getDesc(const LdpPicture* picture, LdpPictureDesc* desc)
    {
        static_cast<const PictureBase*>(picture)->getDesc(*desc);
    }

    bool getBufferDesc(const LdpPicture* picture, LdpPictureBufferDesc* desc)
    {
        return static_cast<const PictureBase*>(picture)->getBufferDesc(*desc);
    }

    bool setFlag(LdpPicture* picture, uint8_t flag, bool value)
    {
        static_cast<PictureBase*>(picture)->setPublicFlag(flag, value);
        return true;
    }

    bool getFlag(const LdpPicture* picture, uint8_t flag)
    {
        return static_cast<const PictureBase*>(picture)->getPublicFlag(flag);
    }

    bool lock(LdpPicture* picture, LdpAccess access, LdpPictureLock** pictureLock)
    {
        LdpPictureLock* ptr = nullptr;

        if (!static_cast<PictureBase*>(picture)->lock(access, ptr)) {
            return false;
        }

        *pictureLock = ptr;
        return true;
    }

    void unlock(LdpPicture* picture, LdpPictureLock* pictureLock)
    {
        const PictureLockBase* const ptr = static_cast<PictureLockBase*>(pictureLock);
        static_cast<PictureBase*>(picture)->unlock(ptr);
    }

    LdpPictureLock* getLock(const LdpPicture* picture)
    {
        return static_cast<const PictureBase*>(picture)->getLock();
    }

    const LdpPictureFunctions kPictureFunctions = {
        setDesc, getDesc, getBufferDesc, setFlag, getFlag, lock, unlock, getLock,
    };
} // namespace
} // namespace lcevc_dec::pipeline
