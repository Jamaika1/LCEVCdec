/* Copyright (c) V-Nova International Limited 2020-2026. All rights reserved.
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

package com.vnova.lcevc.decoder.dec;

import android.graphics.Rect;
import android.media.MediaFormat;

import java.nio.ByteBuffer;
import java.util.Arrays;

/**
 * Provides a bridge between Java and C++, allowing Java code to call C++ functions using JNI
 */
public class LcevcNativeAdapter {
    private static final String TAG = "LcevcNativeAdapter";
    private static final int DEFAULT_SAR_SCALE = 1000000;
    protected static String jniLib = "lcevc_dec_jni";

    static {
        android.util.Log.d(TAG, "Starting, loading LCEVCdec JNI library " + jniLib);
        System.loadLibrary(jniLib);
    }

    /**
     * JNI constants and functions, values shall match corresponding definitions in lcevc_dec.h
     * return values from lcevc calls, shall match LCEVC_ReturnCode
     */
    public static final int LCEVC_Success = 0;
    public static final int LCEVC_Again = -1;
    public static final int LCEVC_NotFound = -2;
    public static final int LCEVC_Error = -3;
    public static final int LCEVC_Uninitialized = -4;
    public static final int LCEVC_Initialized = -5;
    public static final int LCEVC_InvalidParam = -6;
    public static final int LCEVC_NotSupported = -7;
    public static final int LCEVC_Flushed = -8;
    public static final int LCEVC_Timeout = -9;

    // NAL syntax enums, shall match DIL_NALSyntax
    public static final int LCEVC_UnknownNALSyntax = 0;
    public static final int LCEVC_NALSyntaxH264 = 1;
    public static final int LCEVC_NALSyntaxH265 = 2;
    public static final int LCEVC_NALSyntaxH266 = 3;

    // Image color format, shall match LCEVC_ColorFormat
    public static final int LCEVC_ColorFormat_Unknown = 0;
    public static final int LCEVC_I420_8              = 1001;
    public static final int LCEVC_NV12_8              = 2001;
    public static final int LCEVC_RGB_8               = 3001;
    public static final int LCEVC_RGBA_8              = 3003;
    public static final int LCEVC_I420_10_LE          = 1002;
    public static final int LCEVC_RGBA_10_2_LE        = 4001;

    /*! LCEVC_ColorRange enum
     *
     * This enum represents the YUV samples colour range.
     */
    public static final int LCEVC_ColorRange_Unknown  = 0;
    public static final int LCEVC_ColorRange_Full     = 1;   /**< Full range. Y, Cr and Cb component values range from 0 to 255 for 8-bit content */
    public static final int LCEVC_ColorRange_Limited  = 2;   /**< Limited range. Y component values range from 16 to 235 for 8-bit content. Cr, Cy values range from 16 to 240 for 8-bit content */

    /*! LCEVC_ColorPrimaries enum
     *
     * This enum represents the colour primaries with values as defined in Table 2 of ITU-T Rec. H.273 v2 (07/2021) and ISO/IEC TR 23091-4:2021 (twinned doc).
     * https://www.itu.int/ITU-T/recommendations/rec.aspx?id=14661&lang=en
     * Note: these enumerated values can be safely cast to and from integers when interoperating with the above standard.
     * ColourPrimaries indicates the chromaticity coordinates of the source colour primaries in terms of the CIE 1931 definition of x and y as specified by ISO 11664-1
     */
    public static final int LCEVC_ColorPrimaries_Reserved_0    =  0;  /**< Reserved For future use by ITU-T | ISO/IEC */
    public static final int LCEVC_ColorPrimaries_BT709         =  1;  /**< Rec. ITU-R BT.709-6, Rec. ITU-R BT.1361-0, IEC 61966-2-1 sRGB or sYCC, IEC 61966-2-4, SMPTE RP 177 (1993) Annex B */
    public static final int LCEVC_ColorPrimaries_Unspecified   =  2;  /**< Image characteristics are unknown or are determined by the application */
    public static final int LCEVC_ColorPrimaries_Reserved_3    =  3;  /**< Reserved For future use by ITU-T | ISO/IEC */
    public static final int LCEVC_ColorPrimaries_BT470_M       =  4;  /**< Rec. ITU-R BT.470-6 System M (historical), USNTSC 1953 USFCC Title 47 Code of Federal Regulations 73.682 (a) (20) */
    public static final int LCEVC_ColorPrimaries_BT470_BG      =  5;  /**< Rec. ITU-R BT.470-6 System B, G (historical), Rec. ITU-R BT.601-7 625, Rec. ITU-R BT.1358-0 625 (historical), Rec. ITU-R BT.1700-0 625 PAL and 625 SECAM */
    public static final int LCEVC_ColorPrimaries_BT601_NTSC    =  6;  /**< Rec. ITU-R BT.601-7 525, Rec. ITU-R BT.1358-1 525 or 625 (historical), Rec. ITU-R BT.1700-0 NTSC, SMPTE ST 170 (2004), functionally the same as the value 7 */
    public static final int LCEVC_ColorPrimaries_SMPTE240      =  7;  /**< SMPTE ST 240 (1999), functionally the same as the value 6 */
    public static final int LCEVC_ColorPrimaries_GENERIC_FILM  =  8;  /**< Generic film (colour filters using Illuminant C) */
    public static final int LCEVC_ColorPrimaries_BT2020        =  9;  /**< Rec. ITU-R BT.2020-2, Rec. ITU-R BT.2100-2 */
    public static final int LCEVC_ColorPrimaries_XYZ           = 10;  /**< SMPTE ST 428-1 (2019), (CIE 1931 XYZ as in ISO 11664-1) */
    public static final int LCEVC_ColorPrimaries_SMPTE431      = 11;  /**< SMPTE RP 431-2 (2011) */
    public static final int LCEVC_ColorPrimaries_SMPTE432      = 12;  /**< SMPTE EG 432-1 (2010) */
    public static final int LCEVC_ColorPrimaries_Reserved_13   = 13;  /**< Reserved For future use by ITU-T | ISO/IEC */
    public static final int LCEVC_ColorPrimaries_Reserved_14   = 14;  /**< Reserved For future use by ITU-T | ISO/IEC */
    public static final int LCEVC_ColorPrimaries_Reserved_15   = 15;  /**< Reserved For future use by ITU-T | ISO/IEC */
    public static final int LCEVC_ColorPrimaries_Reserved_16   = 16;  /**< Reserved For future use by ITU-T | ISO/IEC */
    public static final int LCEVC_ColorPrimaries_Reserved_17   = 17;  /**< Reserved For future use by ITU-T | ISO/IEC */
    public static final int LCEVC_ColorPrimaries_Reserved_18   = 18;  /**< Reserved For future use by ITU-T | ISO/IEC */
    public static final int LCEVC_ColorPrimaries_Reserved_19   = 19;  /**< Reserved For future use by ITU-T | ISO/IEC */
    public static final int LCEVC_ColorPrimaries_Reserved_20   = 20;  /**< Reserved For future use by ITU-T | ISO/IEC */
    public static final int LCEVC_ColorPrimaries_Reserved_21   = 21;  /**< Reserved For future use by ITU-T | ISO/IEC */
    public static final int LCEVC_ColorPrimaries_P22           = 22;  /**< No corresponding industry specification identified */
    /* 23-255, Reserved For future use by ITU-T | ISO/IEC */

    /*! LCEVC_TransferCharacteristics enum
     *
     * This enum represents the colour transfer characteristics with values as defined in Table 3 of ITU-T Rec. H.273 v2 (07/2021) and ISO/IEC TR 23091-4:2021 (twinned doc).
     * Note: these enumerated values can be safely cast to and from integers when interoperating with the above standard.
     */
    public static final int LCEVC_TransferCharacteristics_RESERVED_0     =  0;  /**< Reserved For future use by ITU-T | ISO/IEC */
    public static final int LCEVC_TransferCharacteristics_BT709          =  1;  /**< Rec. ITU-R BT.709-6, Rec. ITU-R BT.1361-0 conventional colour gamut system (historical), functionally the same as the values 6, 14 and 15 */
    public static final int LCEVC_TransferCharacteristics_UNSPECIFIED    =  2;  /**< Image characteristics are unknown or are determined by the application */
    public static final int LCEVC_TransferCharacteristics_RESERVED_3     =  3;  /**< Reserved For future use by ITU-T | ISO/IEC */
    public static final int LCEVC_TransferCharacteristics_GAMMA22        =  4;  /**< Assumed display gamma 2.2: Rec. ITU-R BT.470-6 System M (historical), USNTSC 1953 USFCC Title 47 Code of Federal Regulations 73.682 (a) (20), Rec. ITU-R BT.1700-0 625 PAL and 625 SECAM */
    public static final int LCEVC_TransferCharacteristics_GAMMA28        =  5;  /**< Assumed display gamma 2.8: Rec. ITU-R BT.470-6 System B, G (historical) */
    public static final int LCEVC_TransferCharacteristics_BT601          =  6;  /**< Rec. ITU-R BT.601-7 525 or 625, Rec. ITU-R BT.1358-1 525 or 625 (historical), Rec. ITU-R BT.1700-0 NTSC, SMPTE ST 170 (2004) */
    public static final int LCEVC_TransferCharacteristics_SMPTE240       =  7;  /**< SMPTE ST 240 (1999) */
    public static final int LCEVC_TransferCharacteristics_LINEAR         =  8;  /**< Linear transfer characteristics */
    public static final int LCEVC_TransferCharacteristics_LOG100         =  9;  /**< Logarithmic transfer characteristic (100:1 range) */
    public static final int LCEVC_TransferCharacteristics_LOG100_SQRT10  = 10;  /**< Logarithmic transfer characteristic (100 * Sqrt( 10 ) : 1 range) */
    public static final int LCEVC_TransferCharacteristics_IEC61966       = 11;  /**< IEC 61966-2-4 */
    public static final int LCEVC_TransferCharacteristics_BT1361         = 12;  /**< Rec. ITU-R BT.1361-0 extended colour gamut system (historical) */
    public static final int LCEVC_TransferCharacteristics_SRGB_SYCC      = 13;  /**< IEC 61966-2-1 sRGB (with MatrixCoefficients equal to 0), IEC 61966-2-1 sYCC (with MatrixCoefficients equal to 5) */
    public static final int LCEVC_TransferCharacteristics_BT2020_10BIT   = 14;  /**< Rec. ITU-R BT.2020-2 (10-bit system), functionally the same as the values 1, 6 and 15 */
    public static final int LCEVC_TransferCharacteristics_BT2020_12BIT   = 15;  /**< Rec. ITU-R BT.2020-2 (12-bit system), functionally the same as the values 1, 6 and 14 */
    public static final int LCEVC_TransferCharacteristics_PQ             = 16;  /**< SMPTE ST 2084 (2014) for 10-, 12-, 14- and 16-bit systems, Rec. ITU-R BT.2100-2 perceptual quantization (PQ) system */
    public static final int LCEVC_TransferCharacteristics_SMPTE428       = 17;  /**< SMPTE ST 428-1 (2019) */
    public static final int LCEVC_TransferCharacteristics_HLG            = 18;  /**< ARIB STD-B67 (2015), Rec. ITU-R BT.2100-2 hybrid log-gamma (HLG) system */
    /* 19-255, Reserved For future use by ITU-T | ISO/IEC */

    /*! LCEVC_MatrixCoefficients enum
     *
     * This enum represents the matrix coefficients with values as defined in Table 4 of ITU-T Rec. H.273 v2 (07/2021) and ISO/IEC TR 23091-4:2021 (twinned doc).
     * https://www.itu.int/ITU-T/recommendations/rec.aspx?id=14661&lang=en
     * Note: these enumerated values can be safely cast to and from integers when interoperating with the above standard.
     * MatrixCoefficients describes the matrix coefficients used in deriving luma and chroma signals from the green, blue and red or X, Y and Z primaries, as specified in Table 4 and equations 11 to 77 of ITU-T Rec. H.273 v2 (07/2021).
     * Equations referenced from enum values are equations found in Section 8.3 of ITU-T Rec. H.273 v2 (07/2021).
     */
    public static final int LCEVC_MatrixCoefficients_IDENTITY            = 0;    /**< Identity;
     The identity matrix. Typically used for GBR (often referred to as RGB); however, may also be used for YZX (often referred to as XYZ); IEC 61966-2-1 sRGB, SMPTE ST 428-1 (2019), See equations 41 to 43 */
    public static final int LCEVC_MatrixCoefficients_BT709               = 1;    /**< KR = 0.2126; KB = 0.0722;
     Rec. ITU-R BT.709-6, Rec. ITU-R BT.1361-0 conventional colour gamut system and extended colour gamut system (historical), IEC 61966-2-4 xvYCC709, SMPTE RP 177 (1993) Annex B, See equations 38 to 40 */
    public static final int LCEVC_MatrixCoefficients_UNSPECIFIED         = 2;    /**< Unspecified;
     Image characteristics are unknown or are determined by the application */
    public static final int LCEVC_MatrixCoefficients_RESERVED_3          = 3;    /**< Reserved;
     For future use by ITU-T | ISO/IEC */
    public static final int LCEVC_MatrixCoefficients_USFCC               = 4;    /**< KR = 0.30; KB = 0.11;
     United States Federal Communications Commission (2003) Title 47 Code of Federal Regulations 73.682 (a) (20), See equations 38 to 40 */
    public static final int LCEVC_MatrixCoefficients_BT470_BG            = 5;    /**< KR = 0.299; KB = 0.114;
     Rec. ITU-R BT.470-6 System B, G (historical), Rec. ITU-R BT.601-7 625, Rec. ITU-R BT.1358-0 625 (historical), Rec. ITU-R BT.1700-0 625 PAL and 625 SECAM, IEC 61966-2-1 sYCC, IEC 61966-2-4 xvYCC601 (functionally the same as the value 6), See equations 38 to 40 */
    public static final int LCEVC_MatrixCoefficients_BT601_NTSC          = 6;    /**< KR = 0.299; KB = 0.114;
     Rec. ITU-R BT.601-7 525, Rec. ITU-R BT.1358-1 525 or 625 (historical), Rec. ITU-R BT.1700-0 NTSC, SMPTE ST 170 (2004), (functionally the same as the value 5), See equations 38 to 40 */
    public static final int LCEVC_MatrixCoefficients_SMPTE240            = 7;    /**< KR = 0.212; KB = 0.087;
     SMPTE ST 240 (1999), See equations 38 to 40 */
    public static final int LCEVC_MatrixCoefficients_YCGCO               = 8;    /**< YCgCo;
     See equations 44 to 58 */
    public static final int LCEVC_MatrixCoefficients_BT2020_NCL          = 9;    /**< KR = 0.2627; KB = 0.0593;
     Rec. ITU-R BT.2020-2 (non-constant luminance), Rec. ITU-R BT.2100-2 Y'CbCr, See equations 38 to 40 */
    public static final int LCEVC_MatrixCoefficients_BT2020_CL           = 10;   /**< KR = 0.2627; KB = 0.0593;
     Rec. ITU-R BT.2020-2 (constant luminance), See equations 59 to 68 */
    public static final int LCEVC_MatrixCoefficients_SMPTE2085           = 11;   /**< Y'D'ZD'X;
     SMPTE ST 2085 (2015), See equations 69 to 71 */
    public static final int LCEVC_MatrixCoefficients_CHROMATICITY_NCL    = 12;   /**< See equations 32 to 37;
     Chromaticity-derived non-constant luminance system, See equations 38 to 40 */
    public static final int LCEVC_MatrixCoefficients_CHROMATICITY_CL     = 13;   /**< See equations 32 to 37;
     Chromaticity-derived non-constant luminance system, See equations 59 to 68 */
    public static final int LCEVC_MatrixCoefficients_ICTCP               = 14;   /**< ICTCP;
     Rec. ITU-R BT.2100-2 ICTCP, See equations 72 to 74 for TransferCharacteristics value 16 (PQ), See equations 75 to 77 for TransferCharacteristics value 18 (HLG) */
    /* 15-255, Reserved For future use by ITU-T | ISO/IEC */
    public native long lcevcCreateDecoder();

    public native int lcevcDestroyDecoder(long instance);

    public native int lcevcSendDecoderEnhancementData(long instance, long timestamp, boolean isKeyFrame, ByteBuffer data, int syntaxType, boolean stripLcevc);

    public native int lcevcSendDecoderBase(long instance, long timestamp, long timeoutUs, long inputPictureHandle, long outputPictureHandle);

    public native int lcevcRenderSendPicture(long instance, long timeStamp, long pictureHandle, DisplayParams renderInfo, long delayUs);

    public native int lcevcRenderSetWindow(long instance, Object surface, boolean secure);

    public native long lcevcAllocPicture(long instance, PictureDesc pictureDesc);

    public native int lcevcFreePicture(long instance, long pictureHandle);

    public native int lcevcGetPicturePlaneCount(long instance, long pictureHandle);

    public native int lcevcGetPictureDesc(long instance, long image, PictureDesc pictureDesc);

    public native int lcevcPictureSetPlaneBuffers(long image, ByteBuffer inBuffer, int inBufferOffsets[], int[] inputByteStrides);

    public native int lcevcFlushDecoder(long instance);

    public native int lcevcSynchronizeDecoder(long instance, boolean dropPending);

    private static int greatestCommonDivisor(int a, int b) {
        a = Math.abs(a);
        b = Math.abs(b);
        while (b != 0) {
            int tmp = a % b;
            a = b;
            b = tmp;
        }
        return a == 0 ? 1 : a;
    }

    // Internal instance reference
    protected long mLcevcDecoderInstance;

    // Java version of output LCEVC_DecodeInformation
    public static class DecodeInformation {
        public int width;
        public int height;
        public float sarNum;
        public float sarDen;
        public int bitdepth;
        public boolean skipRequested;
        public boolean lcevcEnhanced;
        public boolean lcevcAvailable;

        public DecodeInformation() {
            reset();
        }

        public DecodeInformation(int width, int height) {
            this(width, height, 1.0f, 1.0f, 8);
        }

        public DecodeInformation(int width, int height, float sarNum, float sarDen, int bitdepth) {
            this();
            this.width = width;
            this.height = height;
            this.sarNum = sarNum;
            this.sarDen = sarDen;
            this.bitdepth = bitdepth;
        }

        public String toString() {
            return width + "x" + height + " sar=" + sarNum + ":" + sarDen + " bitdepth=" + bitdepth
                    + " skip_requested=" + skipRequested + " enhanced=" + lcevcEnhanced + " available=" + lcevcAvailable;
        }

        public static String getNativeClassName() {
            String nativeClassName = "L" + DecodeInformation.class.getName().replace('.', '/') + ";";
            return nativeClassName;
        }

        @Override
        public boolean equals(final Object obj) {
            if (!(obj instanceof DecodeInformation)) return false;
            DecodeInformation other = (DecodeInformation) obj;
            return other.width == width
                && other.height == height
                && Float.floatToIntBits(other.sarNum) == Float.floatToIntBits(sarNum)
                && Float.floatToIntBits(other.sarDen) == Float.floatToIntBits(sarDen)
                && other.bitdepth == bitdepth
                && other.skipRequested == skipRequested
                && other.lcevcEnhanced == lcevcEnhanced
                && other.lcevcAvailable == lcevcAvailable;
        }

        @Override
        public int hashCode() {
            int result = 17;
            result = 31 * result + width;
            result = 31 * result + height;
            result = 31 * result + Float.floatToIntBits(sarNum);
            result = 31 * result + Float.floatToIntBits(sarDen);
            result = 31 * result + bitdepth;
            result = 31 * result + (skipRequested ? 1 : 0);
            result = 31 * result + (lcevcEnhanced ? 1 : 0);
            result = 31 * result + (lcevcAvailable ? 1 : 0);
            return result;
        }

        public void reset() {
            width = 0;
            height = 0;
            sarNum = 0;
            sarDen = 0;
            bitdepth = 0;
            skipRequested = false;
            lcevcEnhanced = false;
            lcevcAvailable = false;
        }

        public void set(DecodeInformation other) {
            if (other != null) {
                width = other.width;
                height = other.height;
                sarNum = other.sarNum;
                sarDen = other.sarDen;
                bitdepth = other.bitdepth;
                skipRequested = other.skipRequested;
                lcevcEnhanced = other.lcevcEnhanced;
                lcevcAvailable = other.lcevcAvailable;
            }
        }

        public boolean isValid() {
            return width > 0 && height > 0 && sarNum > 0 && sarDen > 0 && bitdepth > 0;
        }
    }

    public static class ColorParams {

        public int colorRange;
        public int colorPrimaries;
        public int colorStandard;
        public int transferCharacteristics;
        public int matrixCoefficients;
        public byte[] hdrStaticInfo;

        public void copy(ColorParams src) {
            colorRange = src.colorRange;
            colorPrimaries = src.colorPrimaries;
            transferCharacteristics = src.transferCharacteristics;
            matrixCoefficients = src.matrixCoefficients;
            if (src.hdrStaticInfo != null) {
                hdrStaticInfo = src.hdrStaticInfo.clone();
            } else {
                hdrStaticInfo = null;
            }
        }

        public void reset() {
            colorRange = LCEVC_ColorRange_Unknown;
            colorPrimaries = LCEVC_ColorPrimaries_Unspecified;
            transferCharacteristics = LCEVC_TransferCharacteristics_UNSPECIFIED;
            matrixCoefficients = LCEVC_MatrixCoefficients_UNSPECIFIED;
            hdrStaticInfo = null;
        }

        public String toString() {
            String hdrStr = "null";
            if (hdrStaticInfo != null) {
                hdrStr = "size=" + hdrStaticInfo.length + " data=" + Arrays.toString(hdrStaticInfo);
            }
            return "colorRange=" + colorRange + " colorPrimaries=" + colorPrimaries + " transferCharacteristics=" + transferCharacteristics + " matrixCoefficients=" + matrixCoefficients /*+ " hdrStaticInfo=" + hdrStr*/;
        }

        public static int toLcevcColorRange(int androidColorRange) {
            switch (androidColorRange) {
                case MediaFormat.COLOR_RANGE_LIMITED:
                    return LCEVC_ColorRange_Limited;
                case MediaFormat.COLOR_RANGE_FULL:
                    return LCEVC_ColorRange_Full;
                default:
                    return LCEVC_ColorRange_Unknown;
            }
        }

        public static int fromLcevcColorRange(int lcevcColorRange) {
            switch (lcevcColorRange) {
                case LCEVC_ColorRange_Limited:
                    return MediaFormat.COLOR_RANGE_LIMITED;
                case LCEVC_ColorRange_Full:
                    return MediaFormat.COLOR_RANGE_FULL;
                default:
                    return 0;
            }
        }

        public static int toLcevcColorPrimaries(int androidColorPrimaries) {
            switch (androidColorPrimaries) {
                case MediaFormat.COLOR_STANDARD_BT709:
                    return LCEVC_ColorPrimaries_BT709;
                case MediaFormat.COLOR_STANDARD_BT601_PAL:
                    return LCEVC_ColorPrimaries_BT470_BG;
                case MediaFormat.COLOR_STANDARD_BT601_NTSC:
                    return LCEVC_ColorPrimaries_BT601_NTSC;
                case MediaFormat.COLOR_STANDARD_BT2020:
                    return LCEVC_ColorPrimaries_BT2020;
                default:
                    return LCEVC_ColorPrimaries_Unspecified;
            }
        }

        public static int fromLcevcColorPrimaries(int lcevcColorPrimaries) {
            switch (lcevcColorPrimaries) {
                case LCEVC_ColorPrimaries_BT709:
                    return MediaFormat.COLOR_STANDARD_BT709;
                case LCEVC_ColorPrimaries_BT470_BG:
                    return MediaFormat.COLOR_STANDARD_BT601_PAL;
                case LCEVC_ColorPrimaries_BT601_NTSC:
                    return MediaFormat.COLOR_STANDARD_BT601_NTSC;
                case LCEVC_ColorPrimaries_BT2020:
                    return MediaFormat.COLOR_STANDARD_BT2020;
                default:
                    return 0;
            }
        }

        public static int toLcevcColorTransfer(int androidColorTransfer) {
            switch (androidColorTransfer) {
                case MediaFormat.COLOR_TRANSFER_LINEAR:
                    return LCEVC_TransferCharacteristics_LINEAR;
                case 2:
                    return LCEVC_TransferCharacteristics_SRGB_SYCC;
                case MediaFormat.COLOR_TRANSFER_SDR_VIDEO:
                    return LCEVC_TransferCharacteristics_BT709;
                case MediaFormat.COLOR_TRANSFER_ST2084:
                    return LCEVC_TransferCharacteristics_PQ;
                case MediaFormat.COLOR_TRANSFER_HLG:
                    return LCEVC_TransferCharacteristics_HLG;
                case 10:
                    return LCEVC_TransferCharacteristics_GAMMA22;
                default:
                    return LCEVC_TransferCharacteristics_UNSPECIFIED;
            }
        }

        public static int fromLcevcColorTransfer(int lcevcColorTransfer) {
            switch (lcevcColorTransfer) {
                case LCEVC_TransferCharacteristics_LINEAR:
                    return MediaFormat.COLOR_TRANSFER_LINEAR;
                case LCEVC_TransferCharacteristics_SRGB_SYCC:
                    return 2;
                case LCEVC_TransferCharacteristics_BT709:
                    return MediaFormat.COLOR_TRANSFER_SDR_VIDEO;
                case LCEVC_TransferCharacteristics_PQ:
                    return MediaFormat.COLOR_TRANSFER_ST2084;
                case LCEVC_TransferCharacteristics_HLG:
                    return MediaFormat.COLOR_TRANSFER_HLG;
                case LCEVC_TransferCharacteristics_GAMMA22:
                    return 10;
                default:
                    return 0;
            }
        }

        public static int toLcevcMatrixCoefficients(int androidMatrixCoefficients) {
            // We have no  MatrixCoefficients available to us so we will use the color standard values to do the mapping
            switch (androidMatrixCoefficients) {
                case MediaFormat.COLOR_STANDARD_BT709:
                    return LCEVC_MatrixCoefficients_BT709;
                case MediaFormat.COLOR_STANDARD_BT601_PAL:
                    return LCEVC_MatrixCoefficients_BT470_BG;
                case MediaFormat.COLOR_STANDARD_BT601_NTSC:
                    return LCEVC_MatrixCoefficients_BT601_NTSC;
                case MediaFormat.COLOR_STANDARD_BT2020:
                    return LCEVC_MatrixCoefficients_BT2020_NCL;
                default:
                    return LCEVC_MatrixCoefficients_UNSPECIFIED;
            }
        }

        public static int fromLcevcMatrixCoefficients(int lcevcMatrixCoefficients) {
            // We have no  MatrixCoefficients available to us so we will use the color standard values to do the mapping
            switch (lcevcMatrixCoefficients) {
                case LCEVC_MatrixCoefficients_BT709:
                    return MediaFormat.COLOR_STANDARD_BT709;
                case LCEVC_MatrixCoefficients_BT470_BG:
                    return MediaFormat.COLOR_STANDARD_BT601_PAL;
                case LCEVC_MatrixCoefficients_BT601_NTSC:
                    return MediaFormat.COLOR_STANDARD_BT601_NTSC;
                case LCEVC_MatrixCoefficients_BT2020_NCL:
                    return MediaFormat.COLOR_STANDARD_BT2020;
                default:
                    return 0;
            }
        }
    }

    public static class DisplayParams {
        public int rotation;
        public int colorTransfer;

        public void copy(DisplayParams src) {
            rotation = src.rotation;
        }

        public void reset() {
            rotation = 0;
        }

        public String toString() {
            return "rotation=" + rotation;
        }
    }

    public static class PictureDesc {
        public int width;
        public int height;
        public int colorFormat;
        public boolean canResize;
        public boolean canModify;
        public ColorParams colorParams;
        public int sampleAspectRatioNum;
        public int sampleAspectRatioDen;
        public float pixelAspectRatio;

        public PictureDesc() {
            reset();
        }

        public void reset() {
            width = 0;
            height = 0;
            colorFormat = 0;
            canResize = false;
            canModify = false;
            colorParams = null;
            sampleAspectRatioNum = 0;
            sampleAspectRatioDen = 0;
            pixelAspectRatio = 1.0f;
        }

        public int getBitdepth() {
            switch (colorFormat) {
                case LCEVC_ColorFormat_Unknown:
                    return 0;
                case LCEVC_I420_8:
                case LCEVC_NV12_8:
                case LCEVC_RGB_8:
                case LCEVC_RGBA_8:
                    return 8;
                case LCEVC_I420_10_LE:
                case LCEVC_RGBA_10_2_LE:
                    return 10;
                default:
                    android.util.Log.e(TAG, "getBitdepth invalid colorFormat=" + colorFormat);
                    return -1;
            }
        }

        public String toString() {
            String cpStr = "null";
            if (colorParams != null) {
                cpStr = colorParams.toString();
            }
            return "colorFormat=" + colorFormat + " resize=" + canResize + " modify=" + canModify + " " + width + "x" + height + " colorParams=" + cpStr + " sampleAspectRatio=" + sampleAspectRatioNum + ":" + sampleAspectRatioDen + " pixelAspectRatio=" + pixelAspectRatio;
        }
    }


    public class LcevcPicture implements Comparable<LcevcPicture> {
        protected int mID;
        protected long mHandle;

        public LcevcPicture(int id) {
            mID = id;
        }
        public int getID() {
            return mID;
        }
        public int compareTo(LcevcPicture other) {
            return (mID - other.mID);
        }
        public long getHandle() {
            return mHandle;
        }
        public boolean isCreated() {
            return mHandle != 0;
        }

        public int getNumPlanes() {
            int planes = lcevcGetPicturePlaneCount(mLcevcDecoderInstance, mHandle);
            android.util.Log.d(TAG, "Got " + planes + " planes from LCEVCdec picture " + mHandle);
            return planes;
        }
        public int getBitdepth() {
            return getPictureDesc().getBitdepth();
        }

        public boolean create(boolean canModify, boolean canResize, int colorFormat, int width, int height, float pixelAspectRatio, ColorParams colorParams) {
            int sampleAspectRatioNum = 1;
            int sampleAspectRatioDen = 1;
            if (pixelAspectRatio > 0.0f) {
                sampleAspectRatioNum = Math.round(pixelAspectRatio * DEFAULT_SAR_SCALE);
                sampleAspectRatioDen = DEFAULT_SAR_SCALE;
                int gcd = greatestCommonDivisor(sampleAspectRatioNum, sampleAspectRatioDen);
                sampleAspectRatioNum /= gcd;
                sampleAspectRatioDen /= gcd;
            }
            return create(canModify, canResize, colorFormat, width, height, sampleAspectRatioNum, sampleAspectRatioDen, colorParams);
        }

        public boolean create(boolean canModify, boolean canResize, int colorFormat, int width, int height, int sampleAspectRatioNum, int sampleAspectRatioDen, ColorParams colorParams) {
            if (isCreated()) {
                android.util.Log.e(TAG, "ID=" + mID + " LcevcPicture is already created (" + mHandle + ")");
                return false;
            }
            PictureDesc desc = new PictureDesc();
            desc.width = width;
            desc.height = height;
            desc.colorFormat = colorFormat;
            desc.canModify = canModify;
            desc.canResize = canResize;
            if (colorParams != null) {
                // Convert the Android values to the LCEVCdec format
                desc.colorParams = new ColorParams();
                desc.colorParams.copy(colorParams);
                desc.colorParams.colorRange = ColorParams.toLcevcColorRange(desc.colorParams.colorRange);
                desc.colorParams.colorPrimaries = ColorParams.toLcevcColorPrimaries(desc.colorParams.colorPrimaries);
                desc.colorParams.transferCharacteristics = ColorParams.toLcevcColorTransfer(desc.colorParams.transferCharacteristics);
                desc.colorParams.matrixCoefficients = ColorParams.toLcevcMatrixCoefficients(desc.colorParams.matrixCoefficients);
            }
            desc.sampleAspectRatioNum = sampleAspectRatioNum > 0 ? sampleAspectRatioNum : 1;
            desc.sampleAspectRatioDen = sampleAspectRatioDen > 0 ? sampleAspectRatioDen : 1;
            desc.pixelAspectRatio = (float) desc.sampleAspectRatioNum / desc.sampleAspectRatioDen;
            mHandle = lcevcAllocPicture(mLcevcDecoderInstance, desc);
            android.util.Log.d(TAG, "lcevcAllocPicture returned handle " + mHandle);
            if (mHandle <= 0) {
                android.util.Log.e(TAG, "Failed to create picture " + mID + ", result=" + mHandle + ", mLcevcDecoderInstance=" + mLcevcDecoderInstance);
                return false;
            }
            return true;
        }

        public PictureDesc getPictureDesc() {
            PictureDesc desc = new PictureDesc();
            if (isCreated()) {  // Gets an empty desc if create() hasn't been called first
                int ret = lcevcGetPictureDesc(mLcevcDecoderInstance, mHandle, desc);
                if (ret != LCEVC_Success) {
                    android.util.Log.e(TAG, "ID=" + mID + " image " + mHandle + " Failed to get image desc ret " + ret);
                    desc.reset();
                }
            }
            // Convert the Android values to the LCEVCdec format
            if (desc.colorParams != null) {
                desc.colorParams.colorRange = ColorParams.fromLcevcColorRange(desc.colorParams.colorRange);
                desc.colorParams.colorPrimaries = ColorParams.fromLcevcColorPrimaries(desc.colorParams.colorPrimaries);
                desc.colorParams.transferCharacteristics = ColorParams.fromLcevcColorTransfer(desc.colorParams.transferCharacteristics);
                desc.colorParams.matrixCoefficients = ColorParams.fromLcevcMatrixCoefficients(desc.colorParams.matrixCoefficients);
            }

            return desc;
        }

        public void reset(String reason) {
            android.util.Log.d(TAG, "reset: " + reason + ": ID=" + mID + " releasing image " + mHandle);
            if (isCreated()) {
                lcevcFreePicture(mLcevcDecoderInstance, mHandle);
            }
            mHandle = 0;
        }

        public String toString() {
            PictureDesc desc = getPictureDesc();
            return "ID=" + mID + " image=" + mHandle + " " + desc.width + "x" + desc.height + ":" + desc.getBitdepth();
        }

        public boolean shouldChange(int newNumPlanes, int newBitdepth, int newWidth, int newHeight, float newPixelAspectRatio) {
            PictureDesc desc = getPictureDesc();
            return (newNumPlanes != getNumPlanes() || newBitdepth != desc.getBitdepth() || newWidth != desc.width || newHeight != desc.height || newPixelAspectRatio != desc.pixelAspectRatio);
        }

        public int setPlanes(ByteBuffer buffer, int[] offsets, int[] rowByteStrides) {
            return lcevcPictureSetPlaneBuffers(mHandle, buffer, offsets, rowByteStrides);
        }
    }
}
