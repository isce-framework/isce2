===============
Stdproc Modules
===============

Module Descriptions
*******************

Pulsetiming
-----------

.. py:module:: stdproc.orbit.pulsetiming

.. py:class:: pulsetiming

   This pure-Python module resamples the orbital state vectors using a Hermite
   interpolation scheme.  The satellite's position and velocity are evaluated at
   each range line.

   * Input
   
     - frame: an :py:class:`isceobj.Scene.Frame` object

   * Output

     - orbit: The interpolated orbital elements

Setmocomppath
-------------

.. py:module:: stdproc.orbit.setmocomppath

.. py:class:: setmocomppath

   This module selects a peg point for the SCH coordinate system using the
   geometry of the orbits for each satellite.

   * Input

     - foo

   * Output

     - bar


Orbit2sch
---------

.. py:module:: stdproc.orbit.orbit2sch

.. py:class:: orbit2sch

   This module converts orbital state vectors from cartesian to SCH.  The SCH
   coordinate system is defined through the Peg object on input

   * Input

     - orbit: an :py:class:`isceobj.Orbit.Orbit` object in ECEF coordinates
     - planet: an :py:clas::`isceobj.Planet.Planet` object
     - peg: an :py:class:`isceobj.Location.Peg` object

   * Output

     - orbit: an isceobj.Orbit.Orbit object in SCH coordinates

Formslc
-------

.. py:module:: stdproc.stdproc.Formslc

.. py:class:: Formslc

   This module focuses SAR data using a range-doppler algorithm with motion
   compensation.

   * Input

     - foo

   * Output

     - bar

Cpxmag2rg
---------

.. py:module:: stdproc.util.Cpxmag2rg

.. py:class:: cpxmag2rg

   This is a data preparation step in which the amplitudes from two SAR images are
   combined into a single two-band image.  The resulting image is band-interleaved
   by pixel.

   * Input

     - foo

   * Output

     - bar

Rgoffsetprf
-----------

.. py:module:: stdproc.util.Rgoffsetprf

.. py:class:: rgoffsetprf

   This module calculates the offset between two images using a 2-D Fourier
   transform method.  The initial guess for the bulk image offset is derived from
   orbital information.

Offoutliers
-----------

.. py:module:: stdproc.util.Offoutlier

.. py:class:: offoutlier

   This module removes outliers from and offset field.  The offset field is
   approximated by a best fitting plane, and offsets are deemed to be outliers if
   they are greater than a user selected distance.

resamp
------

.. py:module:: stdproc.stdproc.resamp.resamp

.. py:class:: resamp

   This module resamples an interferogram based on the provided offset field.

Mocompbaseline
--------------

.. py:module:: stdproc.orbit.mocompbaseline

.. py:class:: mocompbaseline

   This module utilizes the S-component information from the focusing step to line
   up the master and slave images.  This is done by iterating over the S-component
   of the master image and then linearly interpolating the SCH coordinate at the 
   corresponding S-component in the slave image.  The difference between the SCH
   coordinate of the master and slave is then calculated, providing a 3-D baseline.

Topocorrect
-----------

.. py:module:: stdproc.stdproc.topocorrect.topocorrect

.. py:class:: topocorrect

   This module implements the algorithm outlined in section 9 of [1]_ to
   remove the topographic signal in the interferogram.

shadecpxtorg
------------

.. py:module:: stdproc.util.shade2cpx

.. py:class:: shade2cpx

   Create a single two-band image combining shaded relief from the DEM in radar
   coordinates and a SAR amplitude image.

Rgoffsetprf
-----------

.. py:module:: stdproc.util.rgoffsetprf

.. py:class:: rgoffsetprf

   Estimate the subpixel offset between two interferograms.

Rgoffset
--------

.. py:module:: stdproc.util.rgoffset

.. py:class:: rgoffset

   Estimate the subpixel offset between two images.


Geocode
-------

.. py:module:: stdproc.rectify.geocode

.. py:class:: geocode

   * Input

     - foo

   * Output

     - bar

Citations
*********

.. [1] Zebker, H. A., S. Hensley, P. Shanker, and C. Wortham (2010), Geodetically accurate insar data processor, IEEE T. Geosci. Remote.
