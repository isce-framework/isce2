#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2014 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Eric Gurrola
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




from __future__ import (print_function, absolute_import,)
#                        unicode_literals, division)

from isceobj.Util.py2to3 import *
import logging
import datetime
import math

import isceobj
from isceobj.Scene.Frame import Frame
from isceobj.Orbit.Orbit import StateVector as OrbitStateVector
from isceobj.Planet.Planet import Planet
from isceobj.Planet.AstronomicalHandbook import Const
from isceobj.Sensor import cosar
from iscesys import DateTimeUtil as DTU
from iscesys.Component.Component import Component

METADATAFILE = Component.Parameter(
    'metadataFile',
    public_name='annotation file',
    default=None,
    type=str,
    mandatory=True,
    doc="Name of the input annotation file"
)

OUTPUT = Component.Parameter('output',
    public_name='OUTPUT',
    default = '',
    type=str,
    mandatory=False,
    doc="Name of output slc file"
)

from .Sensor import Sensor

class UAVSAR_RPI(Sensor):
    """
    A class representing a UAVSAR SLC.
    """

    family = 'uavsar_rpi'
    logging_name = 'isce.Sensor.UAVSAR_RPI'
    lookMap = {'RIGHT' : -1,
               'LEFT'  : 1}

    parameter_list = (METADATAFILE,) + Sensor.parameter_list

    def __init__(self, name=''):
#        print("UAVSAR_RPI: self.family, name = ", self.family, name)
        super().__init__(family=self.family, name=name)
        self.frame = Frame()
        self.frame.configure()
        return

    def _populatePlatform(self, **kwargs):
#        print("UAVSAR_RPI._populatePlatform")
        platform = self.frame.getInstrument().getPlatform()
        platform.setMission('UAVSAR')
        platform.setPointingDirection(
            self.lookMap[self.metadata['Radar Look Direction'].upper()])
        platform.setPlanet(Planet(pname="Earth"))
        platform.setAntennaLength(1.5)  # Thierry Michel
        return

    def _populateInstrument(self, **kwargs):
#        print("UAVSAR_RPI._populateInstrument")
        instrument = self.frame.getInstrument()
        instrument.setRadarWavelength(
            self.metadata['Center Wavelength'])
        fudgefactor = 1.0/1.0735059946800756
        instrument.setPulseRepetitionFrequency(
            fudgefactor*1.0/self.metadata['Average Pulse Repetition Interval'])
#        print("instrument.getPulseRepetitionFrequency() = ",
#            instrument.getPulseRepetitionFrequency(),
#            type(instrument.getPulseRepetitionFrequency()))
        instrument.setRangePixelSize(
            self.metadata['Single Look Complex Data Range Spacing'])
        instrument.setAzimuthPixelSize(
            self.metadata['Single Look Complex Data Azimuth Spacing'])
        instrument.setPulseLength(self.metadata['Pulse Length'])
        instrument.setChirpSlope(
            -self.metadata['Bandwidth']/self.metadata['Pulse Length'])
        from isceobj.Constants.Constants import SPEED_OF_LIGHT
        instrument.setRangeSamplingRate(
            SPEED_OF_LIGHT/2.0/instrument.getRangePixelSize())
        instrument.setIncidenceAngle(0.5*(
            self.metadata['Average Look Angle in Near Range'] +
            self.metadata['Average Look Angle in Far Range']))

        return

    def _populateFrame(self,**kwargs):
#        print("UAVSAR_RPI._populateFrame")

        if self.metadata['UAVSAR RPI Annotation File Version Number']:
#            print("UAVSAR_RPI._populateFrame, pair = True")
            if self.name.lower() == 'reference':
                sip1 = str(1)
            else:
                sip1 = str(2)
            print("UAVSAR_RPI._populateFrame, 1-based index = ", sip1)
            self._populateFrameFromPair(sip1)
        else:
#            print("UAVSAR_RPI._populateFrame, pair = False")
            self._populateFrameSolo()

        pass

    def _populateFrameFromPair(self, sip1):
#        print("UAVSAR_RPI._populateFrameFromPair: metadatafile = ",
#            self.metadataFile)

        #Get the Start, Mid, and Stop times
        import datetime
        tStart = datetime.datetime.strptime(
            self.metadata['Start Time of Acquisition for Pass '+sip1],
            "%d-%b-%Y %H:%M:%S %Z"
        )
        tStop = datetime.datetime.strptime(
            self.metadata['Stop Time of Acquisition for Pass '+sip1],
            "%d-%b-%Y %H:%M:%S %Z"
        )
        dtMid = DTU.timeDeltaToSeconds(tStop - tStart)/2.
#        print("dtMid = ", dtMid)
        tMid = tStart + datetime.timedelta(microseconds=int(dtMid*1e6))
#        print("tStart = ", tStart)
#        print("tMid   = ", tMid)
#        print("tStop  = ", tStop)
        frame = self.frame
        frame.setSensingStart(tStart)
        frame.setSensingStop(tStop)
        frame.setSensingMid(tMid)
        frame.setNumberOfLines(
            int(self.metadata['Single Look Complex Data Azimuth Lines']))
        frame.setNumberOfSamples(
            int(self.metadata['Single Look Complex Data Range Samples']))
        frame.setPolarization(self.metadata['Polarization'])
        frame.C0 = self.metadata['Single Look Complex Data at Near Range']
        frame.S0 = self.metadata['Single Look Complex Data Starting Azimuth']
        frame.nearLookAngle = self.metadata['Average Look Angle in Near Range']
        frame.farLookAngle = self.metadata['Average Look Angle in Far Range']
#        print("frame.nearLookAngle = ", math.degrees(frame.nearLookAngle))
#        frame.setStartingAzimuth(frame.S0)
        self.extractDoppler()
        frame.setStartingRange(self.startingRange)
        frame.platformHeight = self.platformHeight
#        print("platformHeight, startingRange = ", self.platformHeight, frame.getStartingRange())
        width = frame.getNumberOfSamples()
        deltaRange = frame.instrument.getRangePixelSize()
        nearRange = frame.getStartingRange()
        midRange = nearRange + (width/2.)*deltaRange
        frame.setFarRange(nearRange+width*deltaRange)

        frame.peg = self.peg
#        print("frame.peg = ", frame.peg)
        frame.procVelocity = self.velocity
#        print("frame.procVelocity = ", frame.procVelocity)

        from isceobj.Location.Coordinate import Coordinate
        frame.terrainHeight = self.terrainHeight
        frame.upperLeftCorner = Coordinate()
        frame.upperLeftCorner.setLatitude(
            math.degrees(self.metadata['Approximate Upper Left Latitude']))
        frame.upperLeftCorner.setLongitude(
            math.degrees(self.metadata['Approximate Upper Left Longitude']))
        frame.upperLeftCorner.setHeight(self.terrainHeight)
        frame.upperRightCorner = Coordinate()
        frame.upperRightCorner.setLatitude(
            math.degrees(self.metadata['Approximate Upper Right Latitude']))
        frame.upperRightCorner.setLongitude(
            math.degrees(self.metadata['Approximate Upper Right Longitude']))
        frame.upperRightCorner.setHeight(self.terrainHeight)
        frame.lowerRightCorner = Coordinate()
        frame.lowerRightCorner.setLatitude(
            math.degrees(self.metadata['Approximate Lower Right Latitude']))
        frame.lowerRightCorner.setLongitude(
            math.degrees(self.metadata['Approximate Lower Right Longitude']))
        frame.lowerRightCorner.setHeight(self.terrainHeight)
        frame.lowerLeftCorner = Coordinate()
        frame.lowerLeftCorner.setLatitude(
            math.degrees(self.metadata['Approximate Lower Left Latitude']))
        frame.lowerLeftCorner.setLongitude(
            math.degrees(self.metadata['Approximate Lower Left Longitude']))
        frame.lowerLeftCorner.setHeight(self.terrainHeight)

        frame.nearLookAngle = math.degrees(
            self.metadata['Average Look Angle in Near Range'])
        frame.farLookAngle = math.degrees(
            self.metadata['Average Look Angle in Far Range'])

        return

    def _populateFrameSolo(self):
        print("UAVSAR_RPI._populateFrameSolo")

    def _populateExtras(self):
        pass

    def _populateOrbit(self, **kwargs):
        """
        Create the orbit as the reference orbit defined by the peg
        """
#        print("UAVSAR_RPI._populateOrbit")
        numExtra = 10
        deltaFactor = 200
        dt = deltaFactor*1.0/self.frame.instrument.getPulseRepetitionFrequency()
        t0 = (self.frame.getSensingStart() -
              datetime.timedelta(microseconds=int(numExtra*dt*1e6)))
        ds = deltaFactor*self.frame.instrument.getAzimuthPixelSize()
        s0 = self.platformStartingAzimuth - numExtra*ds
#        print("populateOrbit: t0, startingAzimuth, platformStartingAzimuth, s0, ds = ",
#            t0, self.frame.S0, self.platformStartingAzimuth, s0, ds)
        h = self.platformHeight
        v = [self.velocity, 0., 0.]
#        print("t0, dt = ", t0, dt)
#        print("s0, ds, h = ", s0, ds, h)
#        print("v = ", v[0])
        platform = self.frame.getInstrument().getPlatform()
        elp = platform.getPlanet().get_elp()
        elp.setSCH(self.peg.latitude, self.peg.longitude, self.peg.heading)
        orbit = self.frame.getOrbit()
        orbit.setOrbitSource('Header')
#        print("_populateOrbit: self.frame.numberOfLines, numExtra = ", self.frame.getNumberOfLines(), numExtra)
        for i in range(self.frame.getNumberOfLines()+numExtra):
            vec = OrbitStateVector()
            t = t0 + datetime.timedelta(microseconds=int(i*dt*1e6))
            vec.setTime(t)
            posSCH = [s0 + i*ds*(elp.pegRadCur+h)/elp.pegRadCur, 0., h]
            velSCH = v
            posXYZ, velXYZ = elp.schdot_to_xyzdot(posSCH, velSCH)
            vec.setPosition(posXYZ)
            vec.setVelocity(velXYZ)
            orbit.addStateVector(vec)
#            if i%1000 == 0 or i>self.frame.getNumberOfLines()+numExtra-3 or i < 3:
#                print("vec = ", vec)

        return

    def populateMetadata(self):
        self._populatePlatform()
        self._populateInstrument()
        self._populateFrame()
#        self.extractDoppler()
        self._populateOrbit()

    def extractImage(self):
        from iscesys.Parsers import rdf
        self.metadata = rdf.parse(self.metadataFile)
        self.populateMetadata()

        slcImage = isceobj.createSlcImage()
        if self.name == 'reference' or self.name == 'scene1':
            self.slcname = self.metadata['Single Look Complex Data of Pass 1']
        elif self.name == 'secondary' or self.name == 'scene2':
            self.slcname = self.metadata['Single Look Complex Data of Pass 2']
        else:
            print("Unrecognized sensor.name = ", sensor.name)
            import sys
            sys.exit(0)
        slcImage.setFilename(self.slcname)
        slcImage.setXmin(0)
        slcImage.setXmax(self.frame.getNumberOfSamples())
        slcImage.setWidth(self.frame.getNumberOfSamples())
        slcImage.setAccessMode('r')
        self.frame.setImage(slcImage)
        return

    def extractDoppler(self):
#        print("UAVSAR_RPI._extractDoppler")

        #Recast the Near, Mid, and Far Reskew Doppler values
        #into three RDF records because they were not parsed
        #correctly by the RDF parser; it was parsed as a string.
        #Use the RDF parser on the individual Doppler values to
        #do the unit conversion properly.

        #The units, and values parsed from the metadataFile
        key = "Reskew Doppler Near Mid Far"
        u = self.metadata.data[key].units.split(',')
        v = map(float, self.metadata.data[key].value.split())
        k = ["Reskew Doppler "+x for x in ("Near", "Mid", "Far")]

        #Use the interactive RDF accumulator to create an RDF object
        #for the near, mid, and far Doppler values
        from iscesys.Parsers.rdf import iRDF
        dop = iRDF.RDFAccumulator()
        for z in zip(k,u,v):
            dop("%s (%s) = %f" % z)

        self.dopplerVals = {}
        for r in dop.record_list:
            self.dopplerVals[r.key.split()[-1]] = r.field.value
        self.dopplerVals['Mid'] = self.dopplerVals['Mid']
        self.dopplerVals['Far'] = self.dopplerVals['Far']

#        print("UAVSAR_RPI: dopplerVals = ", self.dopplerVals)

        #quadratic model using Near, Mid, Far range doppler values
        #UAVSAR has a subroutine to compute doppler values at each pixel
        #that should be used instead.
        frame = self.frame
        instrument = frame.getInstrument()
        width = frame.getNumberOfSamples()
        deltaRange = instrument.getRangePixelSize()
        nearRangeBin = 0.
        midRangeBin = float(int((width-1.0)/2.0))
        farRangeBin = width-1.0

        import numpy
        A = numpy.matrix([[1.0, nearRangeBin, nearRangeBin**2],
                          [1.0, midRangeBin,  midRangeBin**2],
                          [1.0, farRangeBin,  farRangeBin**2]])
        d = numpy.matrix([self.dopplerVals['Near'],
                          self.dopplerVals['Mid'],
                          self.dopplerVals['Far']]).transpose()
        coefs = (numpy.linalg.inv(A)*d).transpose().tolist()[0]
        prf = instrument.getPulseRepetitionFrequency()
#        print("UAVSAR_RPI.extractDoppler: self.dopplerVals = ", self.dopplerVals)
#        print("UAVSAR_RPI.extractDoppler: prf = ", prf)
#        print("UAVSAR_RPI.extractDoppler: A, d = ", A, d)
#        print("UAVSAR_RPI.extractDoppler: coefs = ", coefs)
        coefs = {'a':coefs[0]/prf, 'b':coefs[1]/prf, 'c':coefs[2]/prf}
#        print("UAVSAR_RPI.extractDoppler: coefs normalized by prf = ", coefs)

        #Set the coefs in frame._dopplerVsPixel because that is where DefaultDopp looks for them
        self.frame._dopplerVsPixel = coefs

        return coefs


    @property
    def terrainHeight(self):
       return self.metadata['Global Average Terrain Height']

    @property
    def platformHeight(self):
        return self.metadata['Global Average Altitude']

    @property
    def platformStartingAzimuth(self):
#        r, a = self.getStartingRangeAzimuth()
#        return a
        h = self.platformHeight
        peg = self.peg
        platform = self.frame.getInstrument().getPlatform()
        elp = platform.getPlanet().get_elp()
        elp.setSCH(peg.latitude, peg.longitude, peg.heading)
        rc = elp.pegRadCur
        range = self.startingRange
        wavl = self.frame.getInstrument().getRadarWavelength()
        fd = self.dopplerVals['Near']
        v = self.velocity
        tanbeta = (fd*wavl/v)*range*(rc+h)/(range**2-(rc+h)**2-rc**2)
        beta = math.atan(tanbeta)
#        th = self.metadata['Global Average Terrain Height']
#        sinTheta = math.sqrt( 1 - ((h-th)/range)**2 )
#        squint = math.radians(self.squintAngle)
#        c0 = self.startingRange*sinTheta*math.cos(squint)
#        print("platformStartingAzimuth: c0 = ", c0)
#        gamma = c0/rc
#        cosbeta = -(range**2-(rc+h)**2-rc**2)/(2.*rc*(rc+h)*math.cos(gamma))
#        sinbeta = -fd*range*wavl/(2.*rc*v*math.cos(gamma))
#        beta = math.atan2(sinbeta,cosbeta)
        t = beta*(rc+h)/v
        pDS = v*t
        azimuth = self.frame.S0 #- pDS + 473.
        return azimuth

    @property
    def startingRange(self):
#         r, a = self.getStartingRangeAzimuth()
#         return r
        return self.metadata['Single Look Complex Data at Near Range']

    @property
    def squintAngle(self):
        """
        Update this to use the sphere rather than planar approximation.
        """
        startingRange = self.startingRange
        h = self.platformHeight
        v = self.velocity
        prf = self.frame.getInstrument().getPulseRepetitionFrequency()
        wavelength = self.frame.getInstrument().getRadarWavelength()

        if h > startingRange:
            raise ValueError("Spacecraft Height too large (%s>%s)" %
                             (h, startingRange))

        sinTheta = math.sqrt( 1 - (h/startingRange)**2 )
        fd = self.dopplerVals['Near']
        sinSquint = fd/(2.0*v*sinTheta)*wavelength
#        print("calculateSquint: h = ", h)
#        print("calculateSquint: startingRange = ", startingRange)
#        print("calculateSquint: sinTheta = ", sinTheta)
#        print("calculateSquint: self.dopplerVals['Near'] = ", self.dopplerVals['Near'])
#        print("calculateSquint: prf = ", prf)
#        print("calculateSquint: fd = ", fd)
#        print("calculateSquint: v = ", v)
#        print("calculateSquint: wavelength = ", wavelength)
#        print("calculateSquint: sinSquint = ", sinSquint)

        if sinSquint**2 > 1:
            raise ValueError(
                "Error in One or More of the Squint Calculation Values\n"+
                "Doppler Centroid: %s\nVelocity: %s\nWavelength: %s\n" %
                (fd, v, wavelength)
                )
        self.squint = math.degrees(
            math.atan2(sinSquint, math.sqrt(1-sinSquint**2))
            )
        #jng squint is also used later on from the frame, just add it here
        self.frame.squintAngle = math.radians(self.squint)
#        print("UAVSAR_RPI: self.frame.squintAngle = ", self.frame.squintAngle)
        return self.squint

    def getStartingRangeAzimuth(self):
        peg = self.peg
        platform = self.frame.getInstrument().getPlatform()
        elp = platform.getPlanet().get_elp()
        elp.setSCH(peg.latitude, peg.longitude, peg.heading)
        rc = elp.pegRadCur
#        assert(abs(rc-6370285.323386391) < 0.1)
        h = self.platformHeight
#        assert(abs(h-12494.4008) < 0.01)
#        c0 = self.frame.C0
#        assert(abs(c0-13450.0141) < 0.01)
        fd = self.dopplerVals['Near']
#        assert(abs(fd-84.21126622) < 0.01)
        wavl = self.frame.getInstrument().getRadarWavelength()
#        assert(abs((wavl-23.8403545e-2) /wavl) < 0.01)
        gamma = c0/rc
        v = self.velocity
#        assert(abs(v-234.84106135055598) < 0.01)
        A = (fd*wavl/v)**2*(1+h/rc)**2
        B = 1. + (1.+h/rc)**2
        C = 2.0*(1+h/rc)*math.cos(gamma)
#        assert(abs(A-0.0073370197515515235) < 0.00001)
#        assert(abs(B-2.003926560005551) < 0.0001)
#        assert(abs(C-2.0039182464710574) < 0.0001)
        A2B = A/2.-B
        D = (A/2.-B)**2 - (B**2-C**2)
        x2p = -(A/2.-B) + math.sqrt(D)
        x2m = -(A/2.-B) - math.sqrt(D)
#        assert(abs(x2m-8.328781731403723e-06) < 1.e-9)
        range = rc*math.sqrt(x2m)
#        assert(abs(range-18384.406963585432) < 0.1)

        sinbeta = -fd*range*wavl/(2.*rc*v*math.cos(gamma))
        cosbeta = -(range**2-(rc+h)**2-rc**2)/(2.*rc*(rc+h)*math.cos(gamma))
#        assert(abs(sinbeta**2+cosbeta**2 - 1.0) < 0.00001)
        beta = math.atan2(sinbeta, cosbeta)
#        assert(abs(beta+0.00012335892779153295) < 0.000001)
        t = beta*(rc+h)/v
#        assert(abs(t+3.3527904301617375) < 0.001)
        pDS = v*t
#        assert(abs(pDS+787.3728631051696) < 0.01)
        azimuth = self.frame.S0 #self.frame.getStartingAzimuth() #- pDS

        return range, azimuth

    @property
    def heightDt(self):
        """
        Delta(height)/Delta(Time) from frame start-time to mid-time
        """
        return 0.0

    @property
    def velocity(self):
        platform = self.frame.getInstrument().getPlatform()
        elp = platform.getPlanet().get_elp()
        peg = self.peg
        elp.setSCH(peg.latitude, peg.longitude, peg.heading)
        rc = elp.pegRadCur
        scale = (elp.pegRadCur + self.platformHeight)/elp.pegRadCur
        ds_ground = self.frame.instrument.getAzimuthPixelSize()
        dt = 1.0/self.frame.instrument.getPulseRepetitionFrequency()
        v = scale*ds_ground/dt
        return v

    @property
    def peg(self):
        peg = [math.degrees(self.metadata['Peg Latitude']),
               math.degrees(self.metadata['Peg Longitude']),
               math.degrees(self.metadata['Peg Heading'])]

        platform = self.frame.getInstrument().getPlatform()
        elp = platform.getPlanet().get_elp()
        elp.setSCH(*peg)
        rc = elp.pegRadCur

        from isceobj.Location.Peg import Peg
        return Peg(latitude=peg[0], longitude=peg[1], heading=peg[2],
                   radiusOfCurvature=rc)
