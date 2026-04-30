/* Copyright (c) V-Nova International Limited 2023-2026. All rights reserved.
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

#ifndef VN_LCEVC_PIXEL_PROCESSING_FP_TYPES_H
#define VN_LCEVC_PIXEL_PROCESSING_FP_TYPES_H

#include <LCEVC/pipeline/types.h>

#ifdef __cplusplus
extern "C"
{
#endif

/*! \brief Function signature for promoting unsigned fixed-point values to a signed working range */
typedef int16_t (*FixedPointPromotionFunction)(uint16_t);

/*! \brief Function signature for demoting signed working values back to an unsigned fixed-point range */
typedef uint16_t (*FixedPointDemotionFunction)(int32_t);

/*! \brief Retrieve the promotion function for an unsigned fixed-point type
 *
 * \param[in]  unsignedFP  Fixed-point type describing the unsigned input format.
 *
 * \return Function pointer for promoting values in that format.
 */
FixedPointPromotionFunction fixedPointGetPromotionFunction(LdpFixedPoint unsignedFP);

/*! \brief Retrieve the demotion function for an unsigned fixed-point type
 *
 * \param[in]  unsignedFP  Fixed-point type describing the unsigned output format.
 *
 * \return Function pointer for demoting values to that format.
 */
FixedPointDemotionFunction fixedPointGetDemotionFunction(LdpFixedPoint unsignedFP);

/*! \brief Get the maximum representable value for a fixed-point type
 *
 * \param[in]  fp  Fixed-point type to query.
 *
 * \return Maximum value for the type.
 */
int32_t fixedPointMaxValue(LdpFixedPoint fp);

/*! \brief Get the highlight/white point value for a fixed-point type
 *
 * \param[in]  fp  Fixed-point type to query.
 *
 * \return Highlight value for the type.
 */
int32_t fixedPointHighlightValue(LdpFixedPoint fp);

/*! \brief Get the signed offset used for a fixed-point type
 *
 * \param[in]  fp  Fixed-point type to query.
 *
 * \return Offset value for the type.
 */
int32_t fixedPointOffset(LdpFixedPoint fp);

/*! \brief Check whether a fixed-point type is signed
 *
 * \param[in]  type  Fixed-point type to query.
 *
 * \return True if the type is signed.
 */
bool fixedPointIsSigned(LdpFixedPoint type);

/*! \brief Convert a fixed-point type to a readable string
 *
 * \param[in]  type  Fixed-point type to convert.
 *
 * \return Null-terminated string representation.
 */
const char* fixedPointToString(LdpFixedPoint type);

/*! \brief Validate a fixed-point type value
 *
 * \param[in]  type  Fixed-point type to validate.
 *
 * \return True if the type is supported.
 */
bool fixedPointIsValid(LdpFixedPoint type);

/*! \brief Get the bit depth implied by a fixed-point type
 *
 * \param[in]  type  Fixed-point type to query.
 *
 * \return Bit depth for the type.
 */
uint32_t bitdepthFromFixedPoint(LdpFixedPoint type);

/*! \brief Get the storage size in bytes for a fixed-point type
 *
 * \param[in]  type  Fixed-point type to query.
 *
 * \return Byte size for the type.
 */
uint32_t fixedPointByteSize(LdpFixedPoint type);

/*! \brief Map a fixed-point type to its lower-precision variant
 *
 * \param[in]  type  Fixed-point type to reduce.
 *
 * \return Lower-precision fixed-point type.
 */
LdpFixedPoint fixedPointLowPrecision(LdpFixedPoint type);

/*! \brief Map a fixed-point type to its higher-precision variant
 *
 * \param[in]  type  Fixed-point type to increase.
 *
 * \return Higher-precision fixed-point type.
 */
LdpFixedPoint fixedPointHighPrecision(LdpFixedPoint type);

/*! \brief Check whether a value is a power of two
 *
 * \param[in]  value  Value to test.
 *
 * \return True if the value is a power of two.
 */
bool isPow2(uint32_t value);

/*! \brief Truncate a value down to the nearest alignment
 *
 * \param[in]  value      Value to align.
 * \param[in]  alignment  Alignment in bytes.
 *
 * \return Value truncated down to the alignment boundary.
 */
uint32_t alignTruncU32(uint32_t value, uint32_t alignment);

#ifdef __cplusplus
}
#endif

#endif // VN_LCEVC_PIXEL_PROCESSING_FP_TYPES_H
