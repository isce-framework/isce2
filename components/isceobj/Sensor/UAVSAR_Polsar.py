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
# Authors: Marco Lavalle, Eric Gurrola
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





from __future__ import (print_function, absolute_import,)
import datetime
import math
import numpy
import isceobj
from isceobj.Scene.Frame import Frame
from isceobj.Orbit.Orbit import StateVector as OrbitStateVector
from isceobj.Planet.Planet import Planet
from iscesys import DateTimeUtil as DTU
from iscesys.Component.Component import Component
from isceobj.Constants.Constants import SPEED_OF_LIGHT
from .Sensor import Sensor
from isceobj.Location.Coordinate import Coordinate
import os
from isceobj.Util.py2to3 import *

METADATAFILE = Component.Parameter(
    'metadataFile',
    public_name='annotation file',
    default=None,
    type=str,
    mandatory=True,
    doc="Name of the input annotation file"
)

OUTPUT = Component.Parameter(
    'output',
    public_name='OUTPUT',
    default='',
    type=str,
    mandatory=False,
    doc="Name of output slc file"
)


class UAVSAR_Polsar(Sensor):
    """
    A class representing a UAVSAR Polsar SLC.
    """

    family = 'uavsar_polsar'
    logging_name = 'isce.Sensor.UAVSAR_Polsar'
    lookMap = {'RIGHT': -1,
               'LEFT': 1}

    parameter_list = (METADATAFILE,) + Sensor.parameter_list

    def __init__(self, name=''):
        super().__init__(family=self.family, name=name)
        self.frame = Frame()
        self.frame.configure()
        self._elp = None
        self._peg = None
        
    def _populatePlatform(self, **kwargs):
        platform = self.frame.getInstrument().getPlatform()
        platform.setMission('UAVSAR')
        platform.setPointingDirection(
            self.lookMap[self.metadata['Look Direction'].upper()])
        platform.setPlanet(Planet(pname="Earth"))
        platform.setAntennaLength(1.5)

    def _populateInstrument(self, **kwargs):
        fudgefactor = 1.0  # 1.0/1.0735059946800756
        instrument = self.frame.getInstrument()
        instrument.setRadarWavelength(
            self.metadata['Center Wavelength'])
        instrument.setPulseRepetitionFrequency(
            fudgefactor*1.0/self.metadata['Average Pulse Repetition Interval'])
        instrument.setRangePixelSize(
            self.metadata['slc_mag.col_mult'])
        instrument.setAzimuthPixelSize(
            self.metadata['slc_mag.row_mult'])
        instrument.setPulseLength(
            self.metadata['Pulse Length'])
        instrument.setChirpSlope(
            -self.metadata['Bandwidth'] / self.metadata['Pulse Length'])
        instrument.setRangeSamplingRate(
            SPEED_OF_LIGHT / 2.0 / instrument.getRangePixelSize())

    def _populateFrame(self, **kwargs):
        tStart = datetime.datetime.strptime(
            self.metadata['Start Time of Acquisition'],
            "%d-%b-%Y %H:%M:%S %Z"
        )
        tStop = datetime.datetime.strptime(
            self.metadata['Stop Time of Acquisition'],
            "%d-%b-%Y %H:%M:%S %Z"
        )
        dtMid = DTU.timeDeltaToSeconds(tStop - tStart)/2.
        tMid = tStart + datetime.timedelta(microseconds=int(dtMid*1e6))

        frame = self.frame
        frame.setSensingStart(tStart)
        frame.setSensingStop(tStop)
        frame.setSensingMid(tMid)
        frame.setNumberOfLines(
            int(self.metadata['slc_mag.set_rows']))
        frame.setNumberOfSamples(
            int(self.metadata['slc_mag.set_cols']))
        
        frame.C0 = self.metadata['slc_mag.col_addr']
        frame.S0 = self.metadata['slc_mag.row_addr']

        self.extractDoppler()
        frame.setStartingRange(self.startingRange)
        frame.platformHeight = self.platformHeight

        width = frame.getNumberOfSamples()
        deltaRange = frame.instrument.getRangePixelSize()
        nearRange = frame.getStartingRange()
        
        frame.setFarRange(nearRange+width*deltaRange)

        frame._ellipsoid = self.elp
        frame.peg = self.peg
        frame.procVelocity = self.velocity

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

    def _populateFrameSolo(self):
        self.logger.info("UAVSAR_Polsar._populateFrameSolo")

    def _populateExtras(self):
        pass

    def _populateOrbit(self, **kwargs):
        """
        Create the orbit as the reference orbit defined by the peg
        """

        numgroup = 1000
        prf = self.frame.instrument.getPulseRepetitionFrequency()
        daz = self.frame.instrument.getAzimuthPixelSize()
        vel = daz * prf
        t0 = self.frame.getSensingStart()

        nlines = int((self.frame.getSensingStop() - t0).total_seconds() * prf)

        # make sure the elp property has been called
        elp = self.elp
        orbit = self.frame.getOrbit()
        orbit.setOrbitSource('Header')

        for i in range(-5*numgroup, int(nlines/numgroup)*numgroup+5*numgroup, numgroup):
            delt = int(i * 1.0e6 / prf)
            torb = self.frame.getSensingStart() + datetime.timedelta(microseconds=delt)
            ds = delt*1.0e-6*vel

            vec = OrbitStateVector()

            posSCH = [self.frame.S0 + ds, 0.0, self.platformHeight]
            velSCH = [self.velocity, 0., 0.]
            posXYZ, velXYZ = elp.schdot_to_xyzdot(posSCH, velSCH)

            vec.setTime(torb)
            vec.setPosition(posXYZ)
            vec.setVelocity(velXYZ)
            orbit.addStateVector(vec)
            
    def populateMetadata(self):
        self._populatePlatform()
        self._populateInstrument()
        self._populateFrame()
        self._populateOrbit()

    def extractImage(self):
        from iscesys.Parsers import rdf
        self.metadata = rdf.parse(self.metadataFile)
        self.populateMetadata()
        slcImage = isceobj.createSlcImage()
        self.slcname = os.path.join(
            os.path.dirname(os.path.abspath(self.metadataFile)),
            self.metadata['slc'+self.polarization.upper()])
        slcImage.setFilename(self.slcname)
        slcImage.setXmin(0)
        slcImage.setXmax(self.frame.getNumberOfSamples())
        slcImage.setWidth(self.frame.getNumberOfSamples())
        slcImage.setAccessMode('r')
        slcImage.renderHdr()
        self.frame.setImage(slcImage)

    def extractDoppler(self):
        # Recast the Near, Mid, and Far Reskew Doppler values
        # into three RDF records because they were not parsed
        # correctly by the RDF parser; it was parsed as a string.
        # Use the RDF parser on the individual Doppler values to
        # do the unit conversion properly.

        # The units, and values parsed from the metadataFile
        key = 'Reskew Doppler Near Mid Far'
        u = self.metadata.data[key].units.split(',')
        v = map(float, self.metadata.data[key].value.split())
        k = ["Reskew Doppler "+x for x in ("Near", "Mid", "Far")]

        # Use the interactive RDF accumulator to create an RDF object
        # for the near, mid, and far Doppler values
        from iscesys.Parsers.rdf import iRDF
        dop = iRDF.RDFAccumulator()
        for z in zip(k, u, v):
            dop("%s (%s) = %f" % z)
        self.dopplerVals = {}
        for r in dop.record_list:
            self.dopplerVals[r.key.split()[-1]] = r.field.value

        # Quadratic model using Near, Mid, Far range doppler values
        # UAVSAR has a subroutine to compute doppler values at each pixel
        # that should be used instead.
        frame = self.frame
        instrument = frame.getInstrument()
        width = frame.getNumberOfSamples()
        nearRangeBin = 0.
        midRangeBin = float(int((width-1.0)/2.0))
        farRangeBin = width-1.0

        A = numpy.matrix([[1.0, nearRangeBin, nearRangeBin**2],
                          [1.0, midRangeBin,  midRangeBin**2],
                          [1.0, farRangeBin,  farRangeBin**2]])
        d = numpy.matrix([self.dopplerVals['Near'],
                          self.dopplerVals['Mid'],
                          self.dopplerVals['Far']]).transpose()
        coefs = (numpy.linalg.inv(A)*d).transpose().tolist()[0]
        prf = instrument.getPulseRepetitionFrequency()
        coefs_norm = {'a': coefs[0]/prf,
                      'b': coefs[1]/prf,
                      'c': coefs[2]/prf}

        self.doppler_coeff = coefs
        return coefs_norm
    
    @property
    def terrainHeight(self):
        # The peg point incorporates the actual terrainHeight
        # return self.metadata['Global Average Terrain Height']
        return 0.0
        
    @property
    def platformHeight(self):
        # Reduce the platform height by the terrain height because the
        # peg radius of curvature includes the terrain height
        h = (self.metadata['Global Average Altitude'] -
             self.metadata['Global Average Terrain Height'])
        return h

    @property
    def platformStartingAzimuth(self):
        azimuth = self.frame.S0
        return azimuth

    @property
    def startingRange(self):
        return self.metadata['Image Starting Range']

    @property
    def squintAngle(self):
        """
        Update this to use the sphere rather than planar approximation.
        """
        startingRange = self.startingRange
        h = self.platformHeight
        v = self.velocity
        wavelength = self.frame.getInstrument().getRadarWavelength()

        if h > startingRange:
            raise ValueError("Spacecraft Height too large (%s>%s)" %
                             (h, startingRange))

        sinTheta = math.sqrt(1 - (h/startingRange)**2)
        fd = self.dopplerVals['Near']
        sinSquint = fd/(2.0*v*sinTheta)*wavelength

        if sinSquint**2 > 1:
            raise ValueError(
                "Error in One or More of the Squint Calculation Values\n" +
                "Doppler Centroid: %s\nVelocity: %s\nWavelength: %s\n" %
                (fd, v, wavelength)
                )
        self.squint = math.degrees(
            math.atan2(sinSquint, math.sqrt(1-sinSquint**2))
            )
        # jng squint is also used later on from the frame, just add it here
        self.frame.squintAngle = math.radians(self.squint)

        return self.squint

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
        scale = (elp.pegRadCur + self.platformHeight)/elp.pegRadCur
        ds_ground = self.frame.instrument.getAzimuthPixelSize()
        dt = 1.0/self.frame.instrument.getPulseRepetitionFrequency()
        v = scale*ds_ground/dt
        return v

    @property
    def elp(self):
        if not self._elp:
            planet = Planet(pname="Earth")
            self._elp = planet.get_elp()
        return self._elp

    @property
    def peg(self):
        if not self._peg:
            peg = [math.degrees(self.metadata['Peg Latitude']),
                   math.degrees(self.metadata['Peg Longitude']),
                   math.degrees(self.metadata['Peg Heading'])]
            th = self.metadata['Global Average Terrain Height']

            if self.metadata['Mocomp II Applied'] is 'Y':
                self.elp.setSCH(peg[0], peg[1], peg[2], th)
            else:
                self.elp.setSCH(peg[0], peg[1], peg[2], 0)

            rc = self.elp.pegRadCur

            from isceobj.Location.Peg import Peg
            self._peg = Peg(latitude=peg[0], longitude=peg[1], heading=peg[2],
                            radiusOfCurvature=rc)

            self.logger.info("UAVSAR_Polsar: peg radius of curvature = {}".format(self.elp.pegRadCur))
            self.logger.info("UAVSAR_Polsar: terrain height = {}".format(th))
            self.logger.info("UAVSAR_Polsar: mocomp II applied = {}".format(self.metadata['Mocomp II Applied']))

        return self._peg
