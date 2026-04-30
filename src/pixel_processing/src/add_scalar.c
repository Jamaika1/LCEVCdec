/* Copyright (c) V-Nova International Limited 2022-2026. All rights reserved.
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

#include "add_common.h"

#include <LCEVC/common/limit.h>
#include <LCEVC/pipeline/picture_layout.h>
#include <stdint.h>

/*------------------------------------------------------------------------------*/

static void addU8(const LdppAddArgs* args) { VN_ADD_PER_PIXEL_BODY(uint8_t, fpS8ToU8) }

static void addU10(const LdppAddArgs* args) { VN_ADD_PER_PIXEL_BODY(uint16_t, fpS10ToU10) }

static void addU12(const LdppAddArgs* args) { VN_ADD_PER_PIXEL_BODY(uint16_t, fpS12ToU12) }

static void addU14(const LdppAddArgs* args) { VN_ADD_PER_PIXEL_BODY(uint16_t, fpS14ToU14) }

/*------------------------------------------------------------------------------*/

/* clang-format off */
static const PlaneAddFunction kAddTable[LdpFPCount] = {
    &addU8,  /* FP_U8 */
    &addU10, /* FP_U10 */
    &addU12, /* FP_U12 */
    &addU14, /* FP_U14 */
    NULL,    /* FP_S8_7 */
    NULL,    /* FP_S10_5 */
    NULL,    /* FP_S12_3 */
    NULL,    /* FP_S14_1 */
};
/* clang-format on */

PlaneAddFunction planeAddGetFunctionScalar(LdpFixedPoint dstFP) { return kAddTable[dstFP]; }

/*------------------------------------------------------------------------------*/
