###############
Extract Utility
###############

The extract module provides a small utility API for integrations that need to demux LCEVC enhancement data from a muxed base stream, with optional stripping of that enhancement data from the original NAL-unit buffer. Ordinarily only :cpp:func:`LCEVC_extractEnhancementFromNAL` is needed.

Enums
#####

.. doxygenenum:: LCEVC_CodecType

.. doxygenenum:: LCEVC_NALFormat

Functions
#########

.. doxygenfunction:: LCEVC_extractEnhancementFromNAL

.. doxygenfunction:: LCEVC_extractAndRemoveEnhancementFromNAL

.. doxygenfunction:: LCEVC_extractEnhancementFromNALIfKeyframe

.. doxygenfunction:: LCEVC_extractAndRemoveEnhancementFromNALIfKeyframe
