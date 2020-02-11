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




import math
from isce import logging
import isceobj
from iscesys.Component.FactoryInit import FactoryInit
from iscesys.DateTimeUtil.DateTimeUtil import DateTimeUtil as DTU

class Focuser(object):

    def __init__(self,rawObj=None):
        self.rawObj = rawObj
        self.logger = logging.getLogger('isce.focus')

    def focuser(self):
        """
        Create a make_raw object and then focus it!
        """
        doppler = isceobj.Doppler.useDOPIQ()
        hhRaw = self.make_raw(self.rawObj,doppler)
        fd = hhRaw.getDopplerValues().getDopplerCoefficients(inHz=False)
        # Hard-wire the doppler for point-target analysis
        # C-band point target Doppler
        fd = [0.0163810952106773,-0.0000382864254695,0.0000000012335234,0.0]
        # L-band point target Doppler
        #fd = [0.0700103587387314, 0.0000030023105646, -0.0000000000629754, 0.0]
        self.focus(hhRaw,fd)

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
        #from isceobj.Sensor.Generic import Generic

        # Extract some useful variables
        frame = mr.getFrame()
        orbit = frame.getOrbit()
        planet = frame.getInstrument().getPlatform().getPlanet()

        # Calculate Peg Point
        self.logger.info("Calculating Peg Point")
        peg = self.calculatePegPoint(frame,orbit,planet)
        V,H = self.calculateProcessingVelocity(frame,peg)

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

        # Create SLC Image
        slcImage = isceobj.createSlcImage()
        rangeSamplingRate = frame.getInstrument().getRangeSamplingRate()
        rangePulseDuration = frame.getInstrument().getPulseLength()
        chirpSize = int(rangeSamplingRate*rangePulseDuration)
        chirpExtension = 0 #0.5*chirpSize
        numberRangeBins = int(goodBytes/2) - chirpSize + chirpExtension
        slcImage.setFilename(filename.replace('.raw','.slc'))
        slcImage.setByteOrder(frame.getImage().byteOrder)
        slcImage.setAccessMode('write')
        slcImage.setDataType('CFLOAT')
        slcImage.setWidth(numberRangeBins)
        slcImage.createImage()

        # Calculate motion compenstation correction for Doppler centroid
        self.logger.info("Correcting Doppler centroid for motion compensation")
        fdmocomp = stdproc.createFdMocomp()
        fdmocomp.wireInputPort(name='frame',object=frame)
        fdmocomp.wireInputPort(name='peg',object=peg)
        fdmocomp.wireInputPort(name='orbit',object=o2s.getOrbit())
        fdmocomp.setWidth(numberRangeBins)
        fdmocomp.setSatelliteHeight(H)
        fdmocomp.setDopplerCoefficients([fd[0],0.0,0.0,0.0])
        fdmocomp.fdmocomp()
        fd[0] = fdmocomp.getDopplerCentroid()
        self.logger.info("Updated Doppler centroid: %s" % (fd))

        # Calculate the motion compensation Doppler centroid correction plus rate
        #self.logger.info("Testing new Doppler code")
        #frate = stdproc.createFRate()
        #frate.wireInputPort(name='frame',object=frame)
        #frate.wireInputPort(name='peg', object=peg)
        #frate.wireInputPort(name='orbit',object=o2s.getOrbit())
        #frate.wireInputPort(name='planet',object=planet)
        #frate.setWidth(numberRangeBins)
        #frate.frate()
        #fd = frate.getDopplerCentroid()
        #fdrate = frate.getDopplerRate()
        #self.logger.info("Updated Doppler centroid and rate: %s %s" % (fd,fdrate))

        synthetic_aperature_length = self._calculateSyntheticAperatureLength(frame,V)

        patchSize = self.nextpow2(2*synthetic_aperature_length)
        valid_az_samples = patchSize - synthetic_aperature_length
        rawFileSize = rawImage.getLength()*rawImage.getWidth()
        linelength = rawImage.getXmax()
        overhead = patchSize - valid_az_samples
        numPatches = (1+int((rawFileSize/float(linelength)-overhead)/valid_az_samples))

        # Focus image
        self.logger.info("Focusing image")
        focus = stdproc.createFormSLC()
        focus.wireInputPort(name='rawImage',object=rawImage)
        focus.wireInputPort(name='slcImage',object=slcImage)
        focus.wireInputPort(name='orbit',object=o2s.getOrbit())
        focus.wireInputPort(name='frame',object=frame)
        focus.wireInputPort(name='peg',object=peg)
        focus.wireInputPort(name='planet',object=planet)
        focus.setDebugFlag(96)
        focus.setBodyFixedVelocity(V)
        focus.setSpacecraftHeight(H)
        focus.setAzimuthPatchSize(patchSize)
        focus.setNumberValidPulses(valid_az_samples)
        focus.setSecondaryRangeMigrationFlag('n')
        focus.setNumberAzimuthLooks(1)
        focus.setNumberPatches(numPatches)
        focus.setDopplerCentroidCoefficients(fd)
        #focus.setDopplerCentroidCoefficients([fd[0], 0.0, 0.0])
        focus.formslc()
        mocompPos = focus.getMocompPosition()
        fp = open('position.sch','w')
        for i in range(len(mocompPos[0])):
            fp.write("%f %f\n" % (mocompPos[0][i],mocompPos[1][i]))
        fp.close()

        slcImage.finalizeImage()
        rawImage.finalizeImage()

        # Recreate the SLC image
        slcImage = isceobj.createSlcImage()
        slcImage.setFilename(filename.replace('.raw','.slc'))
        slcImage.setAccessMode('read')
        slcImage.setDataType('CFLOAT')
        slcImage.setWidth(numberRangeBins)
        slcImage.createImage()
        width = int(slcImage.getWidth())
        length = int(slcImage.getLength())

        # Create a frame object and write it out using the Generic driver
        frame.setImage(slcImage)
        frame.setOrbit(o2s.getOrbit())
        #writer = Generic()
        #writer.frame = frame
        #writer.write('test.h5',compression='gzip')

        slcImage.finalizeImage()

        self.width = width
        self.length = length

    def calculateProcessingVelocity(self,frame,peg):
        """
        Calculate the optimal processing velocity and height from the orbit.

        @param frame (\a isceobj.Scene.Frame) the Frame object describing the unfocused SAR data
        @param peg (\a isceobj.Location.Peg) a Peg point object defining the origin of the SCH coordinate system
        @return (\a tuple) the processing velocity and satellite height
        """
        from isceobj.Location.SCH import SCH

        orbit = frame.getOrbit()
        ellipsoid = frame.getInstrument().getPlatform().getPlanet().get_elp()

        # Get the mid point of the orbit
        midxyz = orbit.interpolateOrbit(frame.getSensingMid())
        midllh = ellipsoid.xyz_to_llh(midxyz.getPosition())
        # Calculate the SCH S-velocity
        sch = SCH(peg=peg)
        midsch = sch.xyz_to_sch(midxyz.getPosition())
        midvsch = sch.vxyz_to_vsch(midsch,midxyz.getVelocity())
        self.logger.debug("XYZ Velocity: %s" % (midxyz.getVelocity()))
        self.logger.debug("SCH Velocity: %s" % (midvsch))
        H = midllh[2] # The height at midswath
        V = midvsch[0] # SCH S-velocity at midswath
        self.logger.debug("Satellite Height: %s" % (H))
        return V,H

    def calculatePegPoint(self,frame,orbit,planet):
        """
        Calculate the peg point used as the origin of the SCH coordinate system during focusing.

        @param frame (\a isceobj.Scene.Frame) the Frame object describing the unfocused SAR data
        @param orbit (\a isceobj.Orbit.Orbit) the orbit along which to calculate the peg point
        @param planet (\a isceobj.Planet.Planet) the planet around which the satellite is orbiting
        @return (\a isceobj.Location.Peg) the peg point
        """
        from isceobj.Location.Peg import PegFactory
        from isceobj.Location.Coordinate import Coordinate

        # First, get the orbit nadir location at mid-swath and the end of the scene
        midxyz = orbit.interpolateOrbit(frame.getSensingMid())
        endxyz = orbit.interpolateOrbit(frame.getSensingStop())
        # Next, calculate the satellite heading from the mid-point to the end of the scene
        ellipsoid = planet.get_elp()
        midllh = ellipsoid.xyz_to_llh(midxyz.getPosition())
        endllh = ellipsoid.xyz_to_llh(endxyz.getPosition())
        heading = math.degrees(ellipsoid.geo_hdg(midllh,endllh))
        # Then create a peg point from this data
        coord = Coordinate(latitude=midllh[0],longitude=midllh[1],height=0.0)
        peg = PegFactory.fromEllipsoid(coordinate=coord,heading=heading,ellipsoid=ellipsoid)
        self.logger.debug("Peg Point: %s" % (peg))
        return peg

    def _calculateSyntheticAperatureLength(self,frame,v):
        """
        Calculate the length of the synthetic aperature in pixels.

        @param frame (\a isceobj.Scene.Frame) the Frame object describing the unfocussed SAR data
        """
        wavelength = frame.getInstrument().getRadarWavelength()
        prf = frame.getInstrument().getPulseRepetitionFrequency()
        L = frame.getInstrument().getPlatform().getAntennaLength()
        farRange = frame.getFarRange()

        syntheticAperatureLength = int(round((wavelength*farRange*prf)/(L*v),0))

        return syntheticAperatureLength

    def nextpow2(self,v):
        v = v-1
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v = v+1
        return v

def main():
    import sys
    import isceobj

    fi = FactoryInit()
    fi.fileInit = sys.argv[1]
    fi.defaultInitModule = 'InitFromXmlFile'
    fi.initComponentFromFile()

    master = fi.getComponent('Master')

    focuser = Focuser(rawObj=master)
    focuser.focuser()

if __name__ == "__main__":
    main()
