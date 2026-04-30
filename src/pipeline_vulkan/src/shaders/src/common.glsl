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

uvec4 unpack8bit(uint packed)
{
    uvec4 unpacked;
    unpacked.x = packed & 0xFF;
    unpacked.y = (packed >> 8) & 0xFF;
    unpacked.z = (packed >> 16) & 0xFF;
    unpacked.w = (packed >> 24) & 0xFF;
    return unpacked;
}

uvec2 unpack16bit(uint packed)
{
    uvec2 unpacked;
    unpacked.x = packed & 0xFFFF;
    unpacked.y = (packed >> 16) & 0xFFFF;
    return unpacked;
}

ivec2 unpack16bitInternals(uint packed)
{
    const int ab = int(packed);
    const int a = (ab << 16 ) >> 16;
    const int b = (ab >> 16 );
    return ivec2(a,b);
}

uint pack8bit(uvec4 vals) { return vals.x | (vals.y << 8) | (vals.z << 16) | (vals.w << 24); }

uint pack16bit(uvec2 vals) { return vals.x | (vals.y << 16); }

uint pack16bitInternals(ivec2 vals) { return (vals.y << 16) | (vals.x & 0xFFFF); }

ivec4 toInternalFrom8bit(uvec4 vals)
{
    ivec4 internal;
    const int shift = 7;
    internal.x = int((vals.x << shift) - 0x4000);
    internal.y = int((vals.y << shift) - 0x4000);
    internal.z = int((vals.z << shift) - 0x4000);
    internal.w = int((vals.w << shift) - 0x4000);
    return internal;
}

ivec2 toInternalFrom16bit(uvec2 vals, int shift)
{
    ivec2 internal;
    internal.x = int((vals.x << shift) - 0x4000);
    internal.y = int((vals.y << shift) - 0x4000);
    return internal;
}

int clamp_s15(int v)
{
    return clamp(v >> 14, -16384, 16383);
}

int saturateS16(int v)
{
    return clamp(v, -32768, 32767);
}

ivec2 saturateS16(ivec2 v) {
    ivec2 s;
    s.x = saturateS16(v.x);
    s.y = saturateS16(v.y);
    return s;
}

uvec4 fromInternalTo8bit(ivec4 vals)
{
    uvec4 ext;
    const int shift = 7;
    const int halfv = ((1 << shift) / 2);
    ext.x = uint(clamp(((vals.x + 0x4000 + halfv) >> shift), 0, 32767 >> shift));
    ext.y = uint(clamp(((vals.y + 0x4000 + halfv) >> shift), 0, 32767 >> shift));
    ext.z = uint(clamp(((vals.z + 0x4000 + halfv) >> shift), 0, 32767 >> shift));
    ext.w = uint(clamp(((vals.w + 0x4000 + halfv) >> shift), 0, 32767 >> shift));
    return ext;
}

uvec2 fromInternalTo16bit(ivec2 vals, int shift)
{
    uvec2 ext;
    const int halfv = ((1 << shift) / 2);
    ext.x = uint(clamp(((vals.x + 0x4000 + halfv) >> shift), 0, 32767 >> shift));
    ext.y = uint(clamp(((vals.y + 0x4000 + halfv) >> shift), 0, 32767 >> shift));
    return ext;
}

int clamp_int16(int val) { return clamp(val, -32768, 32767); }

int clamp_uint8(int val) { return clamp(val, 0, 255); }
