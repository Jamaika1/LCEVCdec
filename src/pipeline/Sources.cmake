# Copyright (c) V-Nova International Limited 2025-2026. All rights reserved.
# This software is licensed under the BSD-3-Clause-Clear License by V-Nova Limited.
# No patent licenses are granted under this license. For enquiries about patent licenses,
# please contact legal@v-nova.com.
# The LCEVCdec software is a stand-alone project and is NOT A CONTRIBUTION to any other project.
# If the software is incorporated into another project, THE TERMS OF THE BSD-3-CLAUSE-CLEAR LICENSE
# AND THE ADDITIONAL LICENSING INFORMATION CONTAINED IN THIS FILE MUST BE MAINTAINED, AND THE
# SOFTWARE DOES NOT AND MUST NOT ADOPT THE LICENSE OF THE INCORPORATING PROJECT. However, the
# software may be incorporated into a project under a compatible license provided the requirements
# of the BSD-3-Clause-Clear license are respected, and V-Nova Limited remains
# licensor of the software ONLY UNDER the BSD-3-Clause-Clear license (not the compatible license).
# ANY ONWARD DISTRIBUTION, WHETHER STAND-ALONE OR AS PART OF ANY OTHER PROJECT, REMAINS SUBJECT TO
# THE EXCLUSION OF PATENT LICENSES PROVISION OF THE BSD-3-CLAUSE-CLEAR LICENSE.

list(
    APPEND
    SOURCES
    "src/event_sink.cpp"
    "src/picture_layout.c"
    "src/picture_layout.cpp"
    "src/pipeline.cpp"
    "src/types.cpp"
    "src/buffer_base.cpp"
    "src/frame_base.cpp"
    "src/picture_base.cpp"
    "src/picture_lock_base.cpp"
    "src/pipeline_base.cpp"
    "src/pipeline_builder_base.cpp"
    "src/tasks_base.cpp")

list(APPEND HEADERS)

list(
    APPEND
    INTERFACES
    "include/LCEVC/pipeline/event_sink.h"
    "include/LCEVC/pipeline/buffer.h"
    "include/LCEVC/pipeline/frame.h"
    "include/LCEVC/pipeline/picture.h"
    "include/LCEVC/pipeline/picture_layout.h"
    "include/LCEVC/pipeline/pipeline.h"
    "include/LCEVC/pipeline/tasks_base.h"
    "include/LCEVC/pipeline/types.h"
    "include/LCEVC/pipeline/buffer_base.h"
    "include/LCEVC/pipeline/frame_base.h"
    "include/LCEVC/pipeline/picture_base.h"
    "include/LCEVC/pipeline/pipeline_base.h"
    "include/LCEVC/pipeline/picture_lock_base.h"
    "include/LCEVC/pipeline/pipeline_builder_base.h"
    "include/LCEVC/pipeline/pipeline_config_base.h"
    "include/LCEVC/pipeline/temporal_buffer_base.h"
    "include/LCEVC/pipeline/detail/picture_layout.h"
    "include/LCEVC/pipeline/detail/pipeline_api.h")

set(ALL_FILES ${SOURCES} ${HEADERS} ${INTERFACES} "Sources.cmake")

# IDE groups
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${ALL_FILES})
