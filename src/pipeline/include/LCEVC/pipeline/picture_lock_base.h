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

#ifndef VN_LCEVC_PIPELINE_PICTURE_LOCK_BASE_H
#define VN_LCEVC_PIPELINE_PICTURE_LOCK_BASE_H

#include <LCEVC/common/class_utils.hpp>
#include <LCEVC/pipeline/picture.h>

namespace lcevc_dec::pipeline {

class PictureBase;

class PictureLockBase : public LdpPictureLock
{
    static constexpr uint8_t arrSize = kLdpPictureMaxColorComponents;

public:
    PictureLockBase(PictureBase* src, LdpAccess access);
    virtual ~PictureLockBase();

    virtual bool getBufferDesc(LdpPictureBufferDesc* desc) const;
    virtual bool getPlaneDesc(uint32_t planeIndex, LdpPicturePlaneDesc* planeDescOut) const;

    VNNoCopyNoMove(PictureLockBase);

private:
};

} // namespace lcevc_dec::pipeline

#endif // VN_LCEVC_PIPELINE_PICTURE_LOCK_BASE_H
