============
ISCE Objects
============

Module Descriptions
*******************

Orbit
-----

.. py:module:: isceobj.Orbit

.. py:class:: StateVector()

   This module provides a basic representation of an orbital element.

   .. py:method:: get/setTime() 

   A Python :py:class:`datetime.datetime` object indicating the time

   .. py:method:: get/setPosition() 

   A three element list indicating the position

   .. py:method:: get/setVelocity() 

   A three element list indicating the velocity

   .. py:method:: getScalarVelocity() 

   Calculate the scalar velocity from the velocity vector

   .. py:method:: calculateHeight(ellipsoid) 

   Calculate the height of the StateVector above the provided ellipsoid object.

   * Notes Comparison of StateVector objects is done with reference to their time attribute.

.. py:class:: Orbit

   This module provides the basic representation of an orbit.

   .. py:method:: set/getOrbitQuality() 

     A string representing the quality of the orbit (e.g. Preliminary, Final).

   .. py:method:: set/getOrbitSource() 

      A string representing the source of the orbital elements (e.g. Header, Delft)

   .. py:method:: set/getReferenceFrame() 
    
      A string representing the reference frame of the orbit (e.g. Earth-centered Earth-Fixed, Earth-centered inertial)

   .. py:method:: addStateVector(stateVector)

      Add an Orbit.StateVector object to the Orbit.

   .. py:method:: interpolateOrbit(time,method) 

      Interpolate the orbit and return an Orbit.StateVector at the specified time using the specified method. The variable
      time must be a datetime.datetime object, and method must be a string.  Currently, the interpolation methods include 
      'linear', 'hermite', and 'legendre'.

   .. py:method:: selectStateVectors(time,before,after) 

      Select a subset of orbital elements before and after the specified time.  The variable time
      must be a datetime.datetime object, and before and after must be integers.

   .. py:method:: trimOrbit(startTime,stopTime)

      Select a subset of orbital elements using the time bounds, startTime and stopTime.  Both startTime
      and stopTime must be datetime.datetime objects.

Attitude
--------

.. py:module:: isceobj.Attitude

.. py:class:: StateVector

   This module provides the basic representation of a spacecraft attitude state vector.

   .. py:method:: get/setTime()

      A Python datetime.datetime object indicating the time

   .. py:method:: get/setPitch()
      
      The pitch

   .. py:method:: get/setRoll() 

      The roll

   .. py:method:: get/setYaw() 

      The yaw

.. py:class:: Attitude

   This module provides the basic representation of the spacecraft attitude.

   .. py:method:: get/setAttitudeQuality()

      A string representing the quality of the spacecraft attitude (e.g. Preliminary, Final)

   .. py:method:: get/setAttitudeSource()

      A string representing the source of the spacecraft attitude (e.g. Header)

   .. py:method:: addStateVector(stateVector)

      Add an Attitude.StateVector object to the Attitude.

   .. py:method:: interpolate(time)

      Interpolate the attitude and return an Attitude.StateVector at the specified time.  The variable
      time must be a datetime.datetime object.  Currently, the interpolation method is 'linear'.

Doppler
-------

.. py:module:: isceobj.Doppler

.. py:class:: Doppler

   This module provides a basic representation of the Doppler variation with range.

   .. py:method:: get/setDopplerCoefficients(inHz=False)

      A list representing the cubic polynomial fit of Doppler with respect to range.  The variable 
      inHz is a boolean indicating whether the coefficients are expressed in Hz, or Hz/PRF.

   .. py:method:: average(doppler)

      Average two sets of Doppler polynomial coefficients.  The variable doppler should be another Doppler object.

Coordinate
----------

.. py:module:: isceobj.Location.Coordinate

.. py:class:: Coordinate(latitude=None,longitude=None,height=None)

   This module provides a basic representation of a geodetic coordinate.

   .. py:method:: get/setLatitude()

   .. py:method:: get/setLongitude()

   .. py:method:: get/setHeight()

Peg
---

.. py:module:: isceobj.Location.Peg

.. py:class:: PegFactory

   .. py:staticmethod:: fromEllipsoid(coordinate=None,heading=None,ellipsoid=None)

      Create an :py:class:`isceobj.Location.Peg` object from an :py:class:`isceobj.Location.Coordinate` object, a 
      heading and an :py:class:`isceobj.Planet.Ellipsoid` object.


.. py:class:: Peg(latitude=None,longitude=None,heading=None,radiusOfCurvature=None)

   A class to hold Peg point data used in the definition of the SCH coordinate system.

   .. py:method:: get/setHeading()

   .. py:method:: get/setRadiusOfCurvature()

Offset
------

.. py:module:: isceobj.Location.Offset

.. py:class:: Offset(x=None,y=None,dx=None,dy=None,snr=0.0)

   A class to represent a two-dimensional offset

   .. py:method:: setCoordinate(x,y)
   .. py:method:: setOffset(dx,dy)
   .. py:method:: setSignalToNoise(snr)
   .. py:method:: getCoordinate()
   .. py:method:: getOffset()
   .. py:method:: getSignalToNoise()

.. py:class:: OffsetField()

   A class to represent a collection of offsets

   .. py:method:: addOffset(offset)

      Add an :py:class:`isceobj.Location.Offset.Offset` object to the offset field.

   .. py:method:: cull(snr=0.0)

      Remove all offsets with a signal to noise lower the `snr`

   .. py:method:: unpackOffsets()

      A convenience method for converting an offset field to a list of lists.  This is 
      useful for interfacing with Fortran and C code.  The order of the elements in 
      the list is: [[x,dx,y,dy,snr],[x,dx,y,dy,snr], ... ]

SCH
---

.. py:module:: isceobj.Location.SCH

.. py:class:: SCH(peg=None)

   A class implementing SCH <-> XYZ coordinate conversions.  The variable peg should be a :py:class:`isceobj.Location.Peg.Peg` object.

   .. py:method:: xyz_to_sch(xyz)

      Convert from XYZ to SCH coordinates.  The variable xyz should be a three-element list of cartesian coordinates.

   .. py:method:: sch_to_xyz(sch)

      Convert from SCH to XYZ coordinates.  The variable sch should be a three-element list of SCH coordinates.

   .. py:method:: vxyz_to_vsch(sch,vxyz)

      Convert from a Cartesian velocity vxyz, to an SCH velocity relative to the point sch.

   .. py:method:: vsch_to_vxyz(sch,vsch)

      Convert from an SCH velocity vsch, to a Cartesian velocity relative to the point sch.

.. py:class:: LocalSCH(peg=None,sch=None)

   A class for converting between SCH coordinate systems with different peg points.

   .. py:method:: xyz_to_localsch(xyz)

   .. py:method:: localsch_to_xyz(sch)

Planet
------

.. py:module:: isceobj.Planet.AstronomicalHandbook

.. py:class::  Const

   A class encapsulating numerous physical constants.

   .. py:data:: pi
   .. py:data:: G
   .. py:data:: AU
   .. py:data:: c

.. py:module:: isceobj.Planet.Ellipsoid

.. py:class:: Ellipsoid(a=1.0,e2=0.0)

   A class for defining a planets ellipsoid

   .. py:method:: get_a() 

      Return the semi-major axis

   .. py:method:: get_e() 

      Return the eccentricity 

   .. py:method:: get_e2() 

      Return the eccentricity squared

   .. py:method:: get_f() 
      
      Return the flattening

   .. py:method:: get_b() 
     
      Return the semi-minor axis

   .. py:method:: get_c() 

      Return the distance from the center to the focus

   .. py:method:: set_a(a)
   .. py:method:: set_e(e)
   .. py:method:: set_e2(e2)
   .. py:method:: set_f(f)
   .. py:method:: set_b(b)
   .. py:method:: set_c(c)

   .. py:method:: xyz_to_llh(xyz)

      Convert from Cartesian XYZ coordinates to latitude, longitude, height.

   .. py:method:: llh_to_xyz(llh)

      Convert from latitude, longitude, height to Cartesian XYZ coordinates

   .. py:method:: geo_dis(llh1,llh2)

      Calculate the distance along the surface of the ellipsoid from llh1 to llh2.

   .. py:method:: geo_hdg(llh1,llh2)

      Calculate the heading from llh1 to llh2.

   .. py:method:: radiusOfCurvature(llh,hdg=0.0)

      Calculate the radius of curvature at a given point in a particular direction.

   .. py:method:: localRadius(llh)

      Compute the equivalent spherical radius at a given coordinate.

.. py:module:: isceobj.Planet.Planet

.. py:class:: Planet(name)

   A class to represent a planet

   .. py:method:: get_elp() 

      Return the :py:class:`isceobj.Planet.Ellipsoid.Ellipsoid` object for the planet.

   .. py:method:: get_GM()
   .. py:method:: get_name()
   .. py:method:: get_spin()

Platform
--------

.. py:module:: isceobj.Platform.Platform

.. py:class:: Platform()

   .. py:attribute:: planet
   .. py:attribute:: mission
   .. py:attribute:: pointingDirection
   .. py:attribute:: antennaLength
   .. py:attribute:: spacecraftName

Radar
-----

.. py:module:: isceobj.Radar

.. py:class:: Radar()

   .. py:attribute:: platform
      
      An :py:class:`isceobj.Platform.Platform.Platform` object

   .. py:attribute:: pulseLength
   .. py:attribute:: rangePixelSize
   .. py:attribute:: PRF
   .. py:attribute:: rangeSamplingRate
   .. py:attribute:: radarWavelength
   .. py:attribute:: radarFrequency
   .. py:attribute:: incidenceAngle
   .. py:attribute:: inPhaseValue
   .. py:attribute:: quadratureValue
   .. py:attribute:: beamNumber 

Scene
-----

.. py:module:: isceobj.Scene.Frame

.. py:class:: Frame()

   A class to represent the smallest SAR image unit.

   .. py:attribute:: instrument

      An :py:class:`isceobj.Radar.Radar.Radar` object.

   .. py:attribute:: orbit

      An :py:class:`isceobj.Orbit.Orbit` object.

   .. py:attribute:: attitude

      An :py:class:`isceobj.Attribute.Attribute` object.

   .. py:attribute:: image

      An object that inherits from :py:class:`isceobj.Image.BaseImage`.

   .. py:attribute:: squint
   .. py:attribute:: polarization
   .. py:attribute:: startingRange
   .. py:attribute:: farRange
   .. py:attribute:: sensingStart
   .. py:attribute:: sensingMid
   .. py:attribute:: sensingStop
   .. py:attribute:: trackNumber
   .. py:attribute:: orbitNumber
   .. py:attribute:: frameNumber
   .. py:attribute:: passDirection
   .. py:attribute:: processingFacility
   .. py:attribute:: processingSystem
   .. py:attribute:: processingLevel
   .. py:attribute:: processingSoftwareVersion

.. py:class:: Track()

   A collection of Frames.

   .. py:method:: combineFrames(output,frames)
   .. py:method:: addFrame(frame)

Image
-----

Image Format Descriptions
*************************

+-----------+--------+-----------+--------------+
| File name |  Bands | Size      | Interleaving |
+===========+========+===========+==============+
| amp       |   2    | real*4    |    BIP       |
+-----------+--------+-----------+--------------+
| int       |   1    | complex*8 |    Single    |
+-----------+--------+-----------+--------------+
| mht       |   2    | real*4    |    BIP       |
+-----------+--------+-----------+--------------+
| slc       |   1    | complex*8 |    Single    |
+-----------+--------+-----------+--------------+
| raw       |   1    | complex*2 |    Single    |
+-----------+--------+-----------+--------------+
| dem       |   1    | int*2     |    Single    |
+-----------+--------+-----------+--------------+
.. py:module:: isceobj.Image.BaseImage

.. py:class:: BaseImage

   The base class for image objects.

   .. py:attribute:: width
   .. py:attribute:: length
   .. py:attribute:: accessMode 
   .. py:attribute:: filename
   .. py:attribute:: byteOrder

.. py:module:: isceobj.Image.AmpImage

.. py:class:: AmpImage

   A band-interleaved-by-pixel file, containing radar amplitude images in each band. 

.. py:module:: isceobj.Image.DemImage

.. py:class:: DemImage

   A single-banded 2-byte integer file, representing a Digital Elevation Model (DEM).

.. py:module:: isceobj.Image.IntImage

.. py:class:: IntImage

   A single-banded, complex-valued interferogram.

.. py:module:: isceobj.Image.MhtImage

.. py:class:: MhtImage

   A band-interleaved-by-pixel Magnitude (M) and height (ht) image.

.. py:module:: isceobj.Image.RawImage

.. py:class:: RawImage

   A single-banded, 2-byte, complex-valued image.  Typically used for unfocussed SAR data.

.. py:module:: isceobj.Image.RgImage

.. py:class:: RgImage

   A band-interleaved-by-pixel Red (r), Green (g) image.

.. py:module:: isceobj.Image.SlcImage

.. py:class:: SlcImage

   A single-banded, 8-byte, complex-valued image.  Typically used for focussed SAR data.
