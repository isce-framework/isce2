#!/usr/bin/env python3

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2010 California Institute of Technology. ALL RIGHTS RESERVED.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# United States Government Sponsorship acknowledged. This software is subject to
# U.S. export control laws and regulations and has been classified as 'EAR99 NLR'
# (No [Export] License Required except when exporting to an embargoed country,
# end user, or in support of a prohibited end use). By downloading this software,
# the user agrees to comply with all applicable U.S. export laws and regulations.
# The user has the responsibility to obtain export licenses, or other export
# authority as may be required before exporting this software to any 'EAR99'
# embargoed foreign country or citizen of those countries.
#
# Author: Walter Szeliga
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




import os
import math
from isce import logging

import isce
from iscesys.Component.FactoryInit import FactoryInit
from iscesys.Component.Component import Component
from iscesys.Component.InitFromXmlFile import InitFromXmlFile
from iscesys.DateTimeUtil.DateTimeUtil import DateTimeUtil as DTU
from contrib.ISSI.FR import FR, ResourceFile
from mroipac.geolocate.Geolocate import Geolocate

# updated 07/24/2012
from iscesys.StdOEL.StdOELPy import _WriterInterface

"""
All instances of method 'wirePort' in this script have been changed to 'wireInputPort'.
"""
# updated 07/24/2012



class Focuser(_WriterInterface):

    def __init__(self,hh=None,hv=None,vh=None,vv=None,fr=None,tec=None,phase=None):
        """
        Constructor

        @param hh (\a isceobj.Sensor) the HH polarity Sensor object
        @param hv (\a isceobj.Sensor) the HV polarity Sensor object
        @param vh (\a isceobj.Sensor) the VH polarity Sensor object
        @param vv (\a isceobj.Sensor) the VV polarity Sensor object
        @param fr (\a string) the output file name for the Faraday rotation
        @param tec (\a string) the output file name for the Total Electron Count (TEC)
        @param phase (\a string) the output file name for the phase delay
        """
        self.hhObj = hh
        self.hvObj = hv
        self.vhObj = vh
        self.vvObj = vv
        self.frOutput = fr
        self.tecOutput = tec
        self.phaseOutput = phase
        self.filter = None
        self.filterSize = ()
        self.width = None
        self.length = None
        self.swap = False # Swap the endianness of the raw ALOS file
        self._fromRaw = True
        self.logger = logging.getLogger('isce.ISSI')

        # updated 07/24/2012
        super(Focuser, self).__init__()
        # updated 07/24/2012

    def focuser(self):
        """
        Create SLCs from unfocused SAR data, or if the input data are SLCs extract them.
        """
        import isceobj

        # updated 07/24/2012
        doppler = isceobj.Doppler.useDOPIQ()
        #doppler = isceobj.Doppler.useCalcDop() #2013-06-03 Kosal: Calc_dop.py seems buggy
        # updated 07/24/2012

        self.hhObj.output = 'hh.raw'
        self.hvObj.output = 'hv.raw'
        self.vhObj.output = 'vh.raw'
        self.vvObj.output = 'vv.raw'

        # Extract the raw, unfocused SAR data
        hhRaw = self.make_raw(self.hhObj,doppler)
        hvRaw = self.make_raw(self.hvObj,doppler)
        vhRaw = self.make_raw(self.vhObj,doppler)
        vvRaw = self.make_raw(self.vvObj,doppler)

        self.length = self.hhObj.getFrame().getNumberOfLines()
        self.width = self.hhObj.getFrame().getNumberOfSamples()

        if (isinstance(hhRaw.getFrame().getImage(),isceobj.Image.RawImage.RawImage)):
            self._fromRaw = True

            # Calculate the average doppler centroid
            fd = 0.0
            for raw in (hhRaw,hvRaw,vhRaw,vvRaw):
                #fd += raw.getDopplerFit().getQuadraticCoefficients()['a']

                # updated 07/24/2012
                fd += raw.dopplerValues.getDopplerCoefficients()[0]
                # updated 07/24/2012

            fd = fd/4.0

            # Focus the SAR images
            self.focus(hhRaw,fd)
            self.focus(hvRaw,fd)
            self.focus(vhRaw,fd)
            self.focus(vvRaw,fd)

            # Resample the VH and VV images
            self.resample(self.vhObj.getFrame(),fd)
            self.resample(self.vvObj.getFrame(),fd)
        else:
            self._fromRaw = False
            os.rename('hh.raw','hh.slc')
            os.rename('hv.raw','hv.slc')
            os.rename('vh.raw','vh.slc')
            os.rename('vv.raw','vv.slc')

        #2013-06-04 Kosal: create PolSARpro config.txt
        f = open('config.txt', 'wb')
        sep = '-' * 9 + '\n'
        txt = 'Nrow\n%d\n' % self.length
        txt += sep
        txt += 'Ncol\n%d\n' % self.width
        txt += sep
        txt += 'PolarCase\nmonostatic\n'
        txt += sep
        txt += 'PolarType\nfull\n'
        f.write(txt)
        f.close()
        #Kosal

        if (hhRaw.getFrame().getImage().byteOrder != self.__getByteOrder()):
            self.logger.info("Will swap bytes")
            self.swap = True
        else:
            self.logger.info("Will not swap bytes")
            self.swap = False

        # Create slc resource files
        self._createResourceFile(self.hhObj.getFrame())

        self.combine()

    def make_raw(self,sensor,doppler):
        """
        Extract the unfocused SAR image and associated data

        @param sensor (\a isceobj.Sensor) the sensor object
        @param doppler (\a isceobj.Doppler) the doppler object
        @return (\a make_raw) a make_raw instance
        """
        from make_raw import make_raw
        import stdproc
        import isceobj

        # Extract raw image
        self.logger.info("Creating Raw Image")
        mr = make_raw()
        mr.wireInputPort(name='sensor',object=sensor)
        mr.wireInputPort(name='doppler',object=doppler)
        mr.make_raw()

        return mr

    def focus(self,mr,fd):
        """
        Focus SAR data

        @param mr (\a make_raw) a make_raw instance
        @param fd (\a float) Doppler centroid for focusing
        """
        import stdproc
        import isceobj

        # Extract some useful variables
        frame = mr.getFrame()
        orbit = frame.getOrbit()
        planet = frame.getInstrument().getPlatform().getPlanet()

        # Calculate Peg Point
        self.logger.info("Calculating Peg Point")
        peg,H,V = self.calculatePegPoint(frame,orbit,planet)

        # Interpolate orbit
        self.logger.info("Interpolating Orbit")
        pt = stdproc.createPulsetiming()
        pt.wireInputPort(name='frame',object=frame)
        pt.pulsetiming()
        orbit = pt.getOrbit()

        # Convert orbit to SCH coordinates
        self.logger.info("Converting orbit reference frame")
        o2s = stdproc.createOrbit2sch()
        o2s.wireInputPort(name='planet',object=planet)
        o2s.wireInputPort(name='orbit',object=orbit)
        o2s.wireInputPort(name='peg',object=peg)
        o2s.setAverageHeight(H)

        # updated 07/24/2012
        o2s.stdWriter = self._writer_set_file_tags(
            "orbit2sch", "log", "err", "out"
            )
        # updated 07/24/2012

        o2s.orbit2sch()

        # Create Raw Image
        rawImage = isceobj.createRawImage()
        filename = frame.getImage().getFilename()
        bytesPerLine = frame.getImage().getXmax()
        goodBytes = bytesPerLine - frame.getImage().getXmin()
        rawImage.setAccessMode('read')
        rawImage.setByteOrder(frame.getImage().byteOrder)
        rawImage.setFilename(filename)
        rawImage.setNumberGoodBytes(goodBytes)
        rawImage.setWidth(bytesPerLine)
        rawImage.setXmin(frame.getImage().getXmin())
        rawImage.setXmax(bytesPerLine)
        rawImage.createImage()

        self.logger.info("Sensing Start: %s" % (frame.getSensingStart()))

        # Focus image
        self.logger.info("Focusing image")
        focus = stdproc.createFormSLC()
        focus.wireInputPort(name='rawImage',object=rawImage)

        #2013-06-03 Kosal: slcImage is not part of ports anymore (see formslc)
        #it is returned by formscl()
        rangeSamplingRate = frame.getInstrument().getRangeSamplingRate()
        rangePulseDuration = frame.getInstrument().getPulseLength()
        chirpSize = int(rangeSamplingRate*rangePulseDuration)
        chirpExtension = 0 #0.5*chirpSize
        numberRangeBin = int(goodBytes/2) - chirpSize + chirpExtension
        focus.setNumberRangeBin(numberRangeBin)
        #Kosal

        focus.wireInputPort(name='orbit',object=o2s.getOrbit())
        focus.wireInputPort(name='frame',object=frame)
        focus.wireInputPort(name='peg',object=peg)
        focus.setBodyFixedVelocity(V)
        focus.setSpacecraftHeight(H)
        focus.setAzimuthPatchSize(8192)
        focus.setNumberValidPulses(2048)
        focus.setSecondaryRangeMigrationFlag('n')
        focus.setNumberAzimuthLooks(1)
        focus.setNumberPatches(12)
        focus.setDopplerCentroidCoefficients([fd,0.0,0.0,0.0])

        # updated 07/24/2012
        focus.stdWriter = self._writer_set_file_tags(
            "formslc", "log", "err", "out"
            )

        # update 07/24/2012

        #2013-06-04 Kosal: slcImage is returned
        slcImage = focus.formslc()
        #Kosal

        rawImage.finalizeImage()

        width = int(slcImage.getWidth())
        length = int(slcImage.getLength())
        self.logger.debug("Width: %s" % (width))
        self.logger.debug("Length: %s" % (length))

        slcImage.finalizeImage()

        self.width = width
        self.length = length

    def resample(self,frame,doppler):
        """
        Resample the VH and VV polarizations by 0.5 pixels in azimuth.

        @param frame (\a isceobj.Scene.Frame) the Frame object for the SAR data
        """
        import isceobj
        import stdproc
        from isceobj import Constants
        from isceobj.Location.Offset import Offset, OffsetField

        instrument = frame.instrument
        fs = instrument.getRangeSamplingRate()
        pixelSpacing = Constants.SPEED_OF_LIGHT/(2.0*fs) #2013-06-03 Kosal: change in constant name
        filename = frame.getImage().getFilename()
        slcFilename = filename.replace('.raw','.slc')
        resampledFilename = filename.replace('.raw','.resampled.slc')

        # Create the SLC image
        slcImage = isceobj.createSlcImage()
        slcImage.setFilename(slcFilename)
        slcImage.setAccessMode('read')
        slcImage.setDataType('CFLOAT')
        slcImage.setWidth(self.width)
        slcImage.createImage()

        # Create the resampled SLC image
        resampledSlcImage = isceobj.createSlcImage()
        resampledSlcImage.setFilename(resampledFilename)
        resampledSlcImage.setAccessMode('write')
        resampledSlcImage.setDataType('CFLOAT')
        resampledSlcImage.setWidth(self.width)
        resampledSlcImage.createImage()

        # Create an offset field with constant 0.5 pixel shifts in azimuth
        offsetField = OffsetField()
        for i in range(0, self.length,100):
            for j in range(0, self.width,100):
                dx = 0.0
                dy = -0.5
                offset = Offset()
                offset.setCoordinate(j,i)
                offset.setOffset(dx,dy)
                offset.setSignalToNoise(10.0)
                offsetField.addOffset(offset)

        self.logger.debug("width: %s" % (self.width))
        self.logger.debug("length: %s" % (self.length))
        self.logger.debug("Pixel Spacing: %s" % (pixelSpacing))
        self.logger.debug("doppler : %s" % (doppler))
        fp = open('offsetField','w')
        fp.write(str(offsetField))
        fp.close()

        #2013-06-03 Kosal: change resamp_only to resamp_slc, which resamples only an SLC
        #(took resamp_only from revision 747)
        resamp = stdproc.createResamp_slc()
        resamp.setNumberLines(self.length)
        resamp.setNumberRangeBin(self.width)
        resamp.setNumberFitCoefficients(1)
        resamp.setSlantRangePixelSpacing(pixelSpacing)
        resamp.setDopplerCentroidCoefficients([doppler, 0.0, 0.0, 0.0])
        resamp.wireInputPort(name='offsets', object=offsetField)
        resamp.wireInputPort(name='instrument', object=instrument)

        # updated 07/24/2012
        resamp.stdWriter = self._writer_set_file_tags(
            "resamp_slc", "log", "err", "out"
            )

        # updated 07/24/2012
        resamp.resamp_slc(slcImage, resampledSlcImage)
        #Kosal

        slcImage.finalizeImage()
        resampledSlcImage.finalizeImage()

        # Rename the resampled slcs
        os.rename(resampledFilename,slcFilename)

    def combine(self):
        """
        Combine each polarization to form the Faraday rotation, Total Electron Count, and ionospheric phase delay
        """
        # Combine each polarization to calculate the Faraday Rotation
        issiObj = FR(hhFile='hh.slc',
                            hvFile='hv.slc',
                            vhFile='vh.slc',
                            vvFile='vv.slc',
                            lines=self.length,
                            samples=self.width,
                            frOutput=self.frOutput,
                            tecOutput=self.tecOutput,
                            phaseOutput=self.phaseOutput)
        # If we started out with an unfocused image, then we need to perform
        # polarimetric correction
        if (self._fromRaw):
            issiObj.polarimetricCorrection(self.hhObj.transmit,self.hhObj.receive)
        issiObj.calculateFaradayRotation(filter=self.filter,filterSize=self.filterSize,swap=self.swap)
        aveFr = issiObj.getAverageFaradayRotation()
        self.logger.info("Image Dimensions: %s x %s" % (self.width,self.length))
        self.logger.info("Average Faraday Rotation: %s rad (%s deg)" % (aveFr,math.degrees(aveFr)))

        # Calculate the geodetic coordinates of the corners of the interferogram
        date = self.hhObj.getFrame().getSensingStart()
        fc = self.hhObj.getFrame().getInstrument().getRadarFrequency()
        lookDirections = self.calculateLookDirections()
        corners,lookAngles = self.calculateCorners()
        self.makeLookIncidenceFiles()
        meankdotb = issiObj.frToTEC(date,corners,lookAngles,lookDirections,fc)
        self.logger.info("Mean k.B value %s" % meankdotb)
        issiObj.tecToPhase(fc)

    def calculatePegPoint(self,frame,orbit,planet):
        """
        Calculate the peg point used as the origin of the SCH coordinate system during focusing.

        @param frame (\a isceobj.Scene.Frame) the Frame object describing the unfocused SAR data
        @param orbit (\a isceobj.Orbit.Orbit) the orbit along which to calculate the peg point
        @param planet (\a isceobj.Planet.Planet) the planet around which the satellite is orbiting
        @return (\a tuple) the peg point, and the height and velocity at mid-orbit
        """
        import math
        from isceobj.Location.Peg import Peg

        # First, get the orbit nadir location at mid-swath and the end of the scene
        midxyz = orbit.interpolateOrbit(frame.getSensingMid())
        endxyz = orbit.interpolateOrbit(frame.getSensingStop())
        # Next, calculate the satellite heading from the mid-point to the end of the scene
        ellipsoid = planet.get_elp()
        midllh = ellipsoid.xyz_to_llh(midxyz.getPosition())
        endllh = ellipsoid.xyz_to_llh(endxyz.getPosition())
        heading = ellipsoid.geo_hdg(midllh,endllh)
        # Then create a peg point from this data
        peg = Peg(latitude=midllh[0],longitude=midllh[1],heading=math.degrees(heading),radiusOfCurvature=ellipsoid.get_a())
        self.logger.debug("Peg Point:\n%s" % peg)
        return peg,midllh[2],midxyz.getScalarVelocity()


    def calculateHeading(self):
        """
        Calculate the satellite heading at mid-orbit

        @return (\a float) the satellite heading in degrees
        """
        orbit = self.hhObj.getFrame().getOrbit()
        ellipsoid = self.hhObj.getFrame().getInstrument().getPlatform().getPlanet().get_elp()

        midsv = orbit.interpolateOrbit(self.hhObj.getFrame().getSensingMid())
        endsv = orbit.interpolateOrbit(self.hhObj.getFrame().getSensingStop())
        midllh = ellipsoid.xyz_to_llh(midsv.getPosition())
        endllh = ellipsoid.xyz_to_llh(endsv.getPosition())
        heading = ellipsoid.geo_hdg(midllh,endllh)
        heading = math.degrees(heading)
        return heading

    def calculateLookDirections(self):
        """
        Calculate the satellite look direction to each corner of the image

        @return (\a list) a list containing the look directions
        @note: currently, only look direction at scene center, duplicated four times is returned.  This is due to the imprecision of
        the yaw data for current satellites.
        """
        # Get the satellite heading
        heading = self.calculateHeading()

        # Get the yaw angle
        attitude = self.hhObj.getFrame().getAttitude()
        yaw = attitude.interpolate(self.hhObj.getFrame().getSensingMid()).getYaw()

        lookDirection = heading+yaw+90.0
        self.logger.info("Heading %f" % (heading))
        self.logger.info("Yaw: %f" % (yaw))
        self.logger.info("Look Direction: %f" % (lookDirection))
        return [lookDirection, lookDirection, lookDirection, lookDirection]

    def calculateCorners(self):
        """
        Calculate the approximate geographic coordinates of corners of the SAR image.

        @return (\a tuple) a list with the corner coordinates and a list with the look angles to these coordinates
        """
        # Extract the planet from the hh object
        planet = self.hhObj.getFrame().getInstrument().getPlatform().getPlanet()
        # Wire up the geolocation object
        geolocate = Geolocate()
        geolocate.wireInputPort(name='planet',object=planet)

        # Get the ranges, squints and state vectors that defined the boundaries of the frame
        orbit = self.hhObj.getFrame().getOrbit()
        nearRange = self.hhObj.getFrame().getStartingRange()
        farRange = self.hhObj.getFrame().getFarRange()
        earlyStateVector = orbit.interpolateOrbit(self.hhObj.getFrame().getSensingStart())
        lateStateVector = orbit.interpolateOrbit(self.hhObj.getFrame().getSensingStop())
        earlySquint = 0.0 # assume a zero squint angle
        nearEarlyCorner,nearEarlyLookAngle,nearEarlyIncAngle = geolocate.geolocate(earlyStateVector.getPosition(),
                                                                                   earlyStateVector.getVelocity(),
                                                                                   nearRange,earlySquint)
        farEarlyCorner,farEarlyLookAngle,farEarlyIncAngle = geolocate.geolocate(earlyStateVector.getPosition(),
                                                                                earlyStateVector.getVelocity(),
                                                                                farRange,earlySquint)
        nearLateCorner,nearLateLookAngle,nearLateIncAngle = geolocate.geolocate(lateStateVector.getPosition(),
                                                                                lateStateVector.getVelocity(),
                                                                                nearRange,earlySquint)
        farLateCorner,farLateLookAngle,farLateIncAngle = geolocate.geolocate(lateStateVector.getPosition(),
                                                                             lateStateVector.getVelocity(),
                                                                             farRange,earlySquint)
        self.logger.debug("Near Early Corner: %s" % nearEarlyCorner)
        self.logger.debug("Near Early Look Angle: %s" % nearEarlyLookAngle)
        self.logger.debug("Near Early Incidence Angle: %s " % nearEarlyIncAngle)

        self.logger.debug("Far Early Corner: %s" % farEarlyCorner)
        self.logger.debug("Far Early Look Angle: %s" % farEarlyLookAngle)
        self.logger.debug("Far Early Incidence Angle: %s" % farEarlyIncAngle)

        self.logger.debug("Near Late Corner: %s" % nearLateCorner)
        self.logger.debug("Near Late Look Angle: %s" % nearLateLookAngle)
        self.logger.debug("Near Late Incidence Angle: %s" % nearLateIncAngle)

        self.logger.debug("Far Late Corner: %s" % farLateCorner)
        self.logger.debug("Far Late Look Angle: %s" % farLateLookAngle)
        self.logger.debug("Far Late Incidence Angle: %s" % farLateIncAngle)

        corners = [nearEarlyCorner,farEarlyCorner,nearLateCorner,farLateCorner]
        lookAngles = [nearEarlyLookAngle,farEarlyLookAngle,nearLateLookAngle,farLateLookAngle]
        return corners,lookAngles

    def makeLookIncidenceFiles(self):
        """
        Make files containing the look and incidence angles to test the antenna pattern calibration
        """
        import array
        import datetime
        # Extract the planet from the hh object
        planet = self.hhObj.getFrame().getInstrument().getPlatform().getPlanet()

        # Wire up the geolocation object
        geolocate = Geolocate()
        geolocate.wireInputPort(name='planet',object=planet)
        # Get the ranges, squints and state vectors that defined the boundaries of the frame
        orbit = self.hhObj.getFrame().getOrbit()
        nearRange = self.hhObj.getFrame().getStartingRange()
        deltaR = self.hhObj.getFrame().getInstrument().getRangePixelSize()
        prf = self.hhObj.getFrame().getInstrument().getPulseRepetitionFrequency()
        pri = 1.0/prf
        squint = 0.0 # assume a zero squint angle

        lookFP = open('look.dat','wb')
        incFP = open('inc.dat','wb')

        # Calculate the variation in look angle and incidence angle for the first range line
        time = self.hhObj.getFrame().getSensingStart()# + datetime.timedelta(microseconds=int(j*pri*1e6))
        sv = orbit.interpolateOrbit(time=time)
        look = array.array('f')
        inc = array.array('f')
        for i in range(self.width):
            rangeDistance = nearRange + i*deltaR
            coordinate,lookAngle,incidenceAngle = geolocate.geolocate(sv.getPosition(),sv.getVelocity(),rangeDistance,squint)
            look.append(lookAngle)
            inc.append(incidenceAngle)

        # Use the first range line as a proxy for the remaining lines
        for j in range(self.length):
            look.tofile(lookFP)
            inc.tofile(incFP)

        lookFP.close()
        incFP.close()

    def _createResourceFile(self,frame):
        pri = 1.0/frame.getInstrument().getPulseRepetitionFrequency()
        startingRange = frame.getStartingRange()
        startTime = DTU.secondsSinceMidnight(frame.getSensingStart())
        rangeSampleSpacing = frame.getInstrument().getRangePixelSize()
        for file in ('hh.slc.rsc','hv.slc.rsc','vh.slc.rsc','vv.slc.rsc'):
            rsc = ResourceFile(file)
            rsc.write('WIDTH',self.width)
            rsc.write('FILE_LENGTH',self.length)
            rsc.write('RANGE_SAMPLE_SPACING',rangeSampleSpacing)
            rsc.write('STARTING_RANGE',startingRange)
            rsc.write('STARTING_TIME',startTime)
            rsc.write('PRI',pri)
            rsc.close()

    def __getByteOrder(self):
        """
        Get the byte order of the current machine.

        @return (\a string) 'b' for big endian, or 'l' for little endian
        """
        import sys

        byteOrder = sys.byteorder
        return byteOrder[0]

def main():
    import sys
    import isceobj

    fi = FactoryInit()
    fi.fileInit = sys.argv[1]
    fi.defaultInitModule = 'InitFromXmlFile'
    fi.initComponentFromFile()

    hh = fi.getComponent('HH')
    hv = fi.getComponent('HV')
    vh = fi.getComponent('VH')
    vv = fi.getComponent('VV')

    #2013-06-03 Kosal: getComponent returns an object which attributes _leaderFileList and _imageFileList are dictionary
    #but in ALOS.py, extractImage() expects lists
    for f in [hh, hv, vh, vv]:
        f._leaderFileList = f._leaderFileList.values()
        f._imageFileList = f._imageFileList.values()
    #Kosal

    xmlFile = InitFromXmlFile(sys.argv[2])
    variables = xmlFile.init()
    filter = variables['FILTER']['value']
    filterSize = ()
    if (filter != 'None'):
        filterSize = (variables['FILTER_SIZE_X']['value'],variables['FILTER_SIZE_Y']['value'])
    frOutput = variables['FARADAY_ROTATION']['value']
    tecOutput = variables['TEC']['value']
    phaseOutput = variables['PHASE']['value']

    focuser = Focuser(hh=hh,hv=hv,vh=vh,vv=vv,fr=frOutput,tec=tecOutput,phase=phaseOutput)
    focuser.filter = filter
    focuser.filterSize = filterSize
    focuser.focuser()

if __name__ == "__main__":
    main()
