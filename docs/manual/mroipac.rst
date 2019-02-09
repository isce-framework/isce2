===============
MROIPAC Modules
===============

Module Descriptions
*******************

filter
------

.. py:module:: mroipac.filter

.. py:class:: filter

   This module provides access to the Goldstein-Werner power spectral filter from ROI_PAC.  The algorithm
   behind the Goldstein-Werner filtering is explained in [1]_.

   * Input Ports

     - inteferogram: :py:class:`isceobj.Image.IntImage`

   * Output Ports

     - filtered inteferogram: :py:class:`isceobj.Image.IntImage`

   .. py:method:: goldsteinWerner(alpha=0.5)

      Apply the Goldstein-Werner filter with a smoothing value of alpha.

correlation
-----------

.. py:module:: mroipac.correlation.correlation

.. py:class:: Correlation

   This module encapsulates the correlation methods from ROI_PAC and phase gradient correlation methods.

   * Input Ports

     - interferogram: :py:class:`isceobj.Image.IntImage`

     - amplitude: :py:class:`isceobj.Image.AmpImage`

   * Output Ports

     - correlation: :py:class:`isceobj.Image.MhtImage`

   .. py:method:: calculateCorrelation()
     
      Calculate the correlation using the standard correlation formula.

   .. py:method:: calculateEffectiveCorrelation()

      Calculate the effective correlation using the phase gradient

grass
-----

.. py:module:: mroipac.grass.grass

.. py:class:: Grass

   This module encapsulates the grass unwrapping algorithm, an implementation of the branch-cut unwrapping
   outlined in [2]_.

   * Input Ports

     - interferogram: an :py:class:`isceobj.Image.IntImage` object

     - correlation: an :py:class:`isceobj.Image.MhtImage` object

   * Output Ports

     - unwrapped interferogram: an isceobj.Image.FOO

   .. py:method:: unwrap(x=-1,y=-1,threshold=0.1)

      Unwrap an interferogram with a seed location in pixels specified 
      by x (range) and y (azimuth) and an unwrapping correlation threshold (default = 0.1).

Citations
*********

.. [1] Goldstein, R. M., and C. L. Werner (1998), Radar interferogram filtering for geophysical applications, Geophys. Res. Lett., 25(21), 4035–4038.

.. [2] Goldstein, R. M., H. A. Zebker, and C. L. Werner (1988), Satellite radar interferometry: two-dimensional phase unwrapping, Radio Science, 23(4), 713– 720.
