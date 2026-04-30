################
Pixel Processing
################

The Pixel Processing library (LDPP) is a collection of functions to provide the various image manipulation processes required to upscale, apply residuals, add the temporal buffer, dither, sharpen and convert an image to correctly decode and LCEVC stream on the CPU. The library has significant SIMD optimisations and scalar fallbacks for older devices. For an average LCEVC stream the majority of the frame time is spent on these functions so performance is crucial.

Key Concepts
############

Fixed Points
------------

An important concept to grasp before getting into the Pixel processing functions is 'Fixed Points'. LCEVC supports 8, 10, 12 and 14-bit base input as per the MPEG-5 Part 2 LCEVC Specification, various processes within LCEVC must also be completed in signed 16-bit (S16) space. This is represented in the Pipeline library (LDP) as the following:

Raw input unsigned fixed point types (U8 & U16):

.. code-block:: C++

   LdpFPU8,     /**< U8.0  (uint8_t) */
   LdpFPU10,    /**< U10.0 (uint16_t) */
   LdpFPU12,    /**< U12.0 (uint16_t) */
   LdpFPU14,    /**< U14.0 (uint16_t) */

Converted signed fixed point types (S16):

.. code-block:: C++

   LdpFPS8,     /**< S8.7  (int16_t) */
   LdpFPS10,    /**< S10.5 (int16_t) */
   LdpFPS12,    /**< S12.3 (int16_t) */
   LdpFPS14,    /**< S14.1 (int16_t) */

The explicit :cpp:func:`ldppPlaneConvert` function can be used to copy and convert whole planes. Conversion can also happen implicitly by or within other functions.

Library patterns
----------------

Pixel processing operations lend themselves to be easily multithreaded as rows of the image can literally be sliced into many horizontal chunks by offsets and counts of rows. Most functions in the library read from a source plane and write to a separate destination allowing threads to operate completely independently within a pool. LDPP heavily utilizes the task pool threading infrastructure from the common (LDC) library - ``LdcTaskPool`` and an ``LdcTask`` parent are passed through to the ``ldcTaskPoolAddSlicedDeferred`` call in most LDPP functions. This splits the plane into slices and dispatches it across threads while ensuring that a parent task has completed before starting.

An ``LdpPipelineDiagInfo`` object is passed to LDPP functions to debug the timing of each slice call for perfetto tracing. Only compiled in when ``VN_SDK_DIAGNOSTICS_ASYNC`` is ON.

Input and output planes are described with ``LdpPictureLayout`` structs containing plane metadata like width, height, stride information as well as offsets and interleaving information for interleaved planes such as NV12. ``LdpPictureLayout`` contains a ``LdpPictureLayoutInfo`` struct which contains the ``LdpFixedPoint`` as described :ref:`above<pixel_processing:Fixed Points>`

Functions
#########

Convert
-------

.. doxygenfunction:: ldppPlaneConvert

Upscale
-------

.. doxygenfunction:: ldppUpscale

.. doxygenstruct:: ldppUpscaleArgs
   :members:

Apply CmdBuffer
---------------

.. doxygenfunction:: ldppApplyCmdBuffer

Add
---

.. doxygenfunction:: ldppPlaneAdd

Sharpen
-------

.. doxygenfunction:: ldppSharpen
