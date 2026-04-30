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

#ifndef VN_LCEVC_PIPELINE_PICTURE_BASE_H
#define VN_LCEVC_PIPELINE_PICTURE_BASE_H

#include <LCEVC/pipeline/picture.h>
#include <LCEVC/pipeline/picture_lock_base.h>
#include <LCEVC/pipeline/pipeline_base.h>

namespace lcevc_dec::pipeline {

class PictureBase : public LdpPicture
{
public:
    PictureBase(PipelineBase& m_pipeline);
    virtual ~PictureBase();

    // Access management
    virtual bool lock(LdpAccess access, LdpPictureLock*& lockOut) = 0;
    virtual bool unlock(const LdpPictureLock* lock) = 0;

    // Internal fetch of plane pointer and stride
    virtual void getPlaneDescInternal(uint32_t plane, LdpPicturePlaneDesc& planeDesc) const = 0;

    bool isValid() const;

    bool getPublicFlag(uint8_t flag) const;
    void setPublicFlag(uint8_t flag, bool value);

    void getDesc(LdpPictureDesc& descOut) const;
    bool setDesc(const LdpPictureDesc& newDesc);

    bool getBufferDesc(LdpPictureBufferDesc& bufferDescOut) const;
    bool getPlaneDescArr(LdpPicturePlaneDesc planeDescArrOut[kLdpPictureMaxNumPlanes]) const;

    void* getUserData() const { return userData; }
    void setUserData(void* val) { userData = val; }

    bool isLocked() const { return m_pictureLock != nullptr; }
    PictureLockBase* getLock() const { return m_pictureLock; }

    // Buffer management
    void setExternal(const LdpPicturePlaneDesc* planeDescArr, const LdpPictureBufferDesc* buffer);

    uint32_t getRequiredSize() const;

    bool bindMemory(pipeline::BufferUsage usage);
    bool unbindMemory();

    bool setDescAndBind(const LdpPictureDesc& newDesc, pipeline::BufferUsage usage);

    uint32_t getActiveWidth() const
    {
        return ldpPictureLayoutWidth(&layout) - (margins.left + margins.right);
    }
    uint32_t getActiveHeight() const
    {
        return ldpPictureLayoutHeight(&layout) - (margins.top + margins.bottom);
    }

    VNNoCopyNoMove(PictureBase);

protected:
    bool initializeDesc(const LdpPictureDesc& desc,
                        const uint32_t rowStridesBytes[kLdpPictureMaxNumPlanes]);

    bool descsMatch(const LdpPictureDesc& desc) const;

    // Owning pipeline
    PipelineBase& m_pipeline;

    // Any current lock
    PictureLockBase* m_pictureLock{};

    // Any external buffer and plane description
    bool m_external{false};
    LdpPicturePlaneDesc m_externalPlaneDescs[kLdpPictureMaxNumPlanes] = {};
    LdpPictureBufferDesc m_externaBufferDesc = {};
};

} // namespace lcevc_dec::pipeline

#endif // VN_LCEVC_PIPELINE_PICTURE_BASE_H
