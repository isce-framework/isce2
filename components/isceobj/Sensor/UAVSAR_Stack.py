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
from isceobj.Util.decorators import pickled, logged
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

SEGMENT_INDEX = Component.Parameter(
    'segment_index',
    public_name='segment index',
    default=1,
    type=int,
    mandatory=False,
    doc="The index of the first SLC segment to process"
)

def polyval(coeffs, rho):
    v = 0.0
    for i, c in enumerate(coeffs):
        v += c*rho**i
    return v


from isceobj.Sensor.Sensor import Sensor

@pickled
class UAVSAR_Stack(Component):
    """
    A class representing a UAVSAR SLC.
    """

    family = 'uavsar_stack'
    logging_name = 'isce.Sensor.UAVSAR_Stack'
    lookMap = {'RIGHT' : -1,
               'LEFT'  : 1}

    parameter_list = (METADATAFILE, SEGMENT_INDEX)

    @logged
    def __init__(self, name=''):
        super().__init__(family=self.family, name=name)
        self.frame = Frame()
        self.frame.configure()
        self._elp = None
        self._peg = None
        elp = self.elp
        return

    def _populatePlatform(self, **kwargs):
        platform = self.frame.getInstrument().getPlatform()
        platform.setMission('UAVSAR')
        platform.setPointingDirection(
            self.lookMap[self.metadata['Look Direction'].upper()])
        platform.setPlanet(Planet(pname="Earth"))
        platform.setAntennaLength(self.metadata['Antenna Length'])
        return

    def _populateInstrument(self, **kwargs):
        instrument = self.frame.getInstrument()
        instrument.setRadarWavelength(
            self.metadata['Center Wavelength'])
        instrument.setPulseRepetitionFrequency(
            1.0/self.metadata['Average Pulse Repetition Interval'])
        instrument.setRangePixelSize(
            self.metadata['1x1 SLC Range Pixel Spacing'])
        instrument.setAzimuthPixelSize(
            self.metadata['1x1 SLC Azimuth Pixel Spacing'])
        instrument.setPulseLength(self.metadata['Pulse Length'])
        instrument.setChirpSlope(
            -self.metadata['Bandwidth']/self.metadata['Pulse Length'])
        from isceobj.Constants.Constants import SPEED_OF_LIGHT
        instrument.setRangeSamplingRate(
            SPEED_OF_LIGHT/2.0/instrument.getRangePixelSize())
        instrument.setIncidenceAngle(0.5*(
            self.metadata['Minimum Look Angle'] +
            self.metadata['Maximum Look Angle']))

        return

    def _populateFrame(self):
        #Get the Start, Mid, and Stop times
        import datetime
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
        frame._frameNumber = 1
        frame._trackNumber = 1
        frame.setSensingStart(tStart)
        frame.setSensingStop(tStop)
        frame.setSensingMid(tMid)
        frame.setNumberOfLines(int(self.metadata['slc_{}_1x1_mag.set_rows'.format(self.segment_index)]))
        frame.setNumberOfSamples(int(self.metadata['slc_{}_1x1_mag.set_cols'.format(self.segment_index)]))
        frame.setPolarization(self.metadata['Polarization'])
        frame.C0 = self.metadata['slc_{}_1x1_mag.col_addr'.format(self.segment_index)]
        frame.S0 = self.metadata['Segment {} Data Starting Azimuth'.format(self.segment_index)]
        frame.nearLookAngle = self.metadata['Minimum Look Angle']
        frame.farLookAngle = self.metadata['Maximum Look Angle']
        frame.setStartingRange(self.startingRange)
        frame.platformHeight = self.platformHeight
        width = frame.getNumberOfSamples()
        deltaRange = frame.instrument.getRangePixelSize()
        nearRange = frame.getStartingRange()
        midRange = nearRange + (width/2.)*deltaRange
        frame.setFarRange(nearRange+width*deltaRange)
        self.extractDoppler()
        frame._ellipsoid = self.elp
        frame.peg = self.peg
        frame.procVelocity = self.velocity

        from isceobj.Location.Coordinate import Coordinate
        frame.upperLeftCorner = Coordinate()

        #The corner latitude, longitudes are given as a pair
        #of values in degrees at each corner (without rdf unit specified)
        llC = []
        for ic in range(1,5):
            key = 'Segment {0} Data Approximate Corner {1}'.format(self.segment_index, ic)
            self.logger.info("key = {}".format(key))
            self.logger.info("metadata[key] = {}".format(self.metadata[key], type(self.metadata[key])))
            llC.append(list(map(float, self.metadata[key].split(','))))

        frame.terrainHeight = self.terrainHeight
        frame.upperLeftCorner.setLatitude(llC[0][0])
        frame.upperLeftCorner.setLongitude(llC[0][1])
        frame.upperLeftCorner.setHeight(self.terrainHeight)

        frame.upperRightCorner = Coordinate()
        frame.upperRightCorner.setLatitude(llC[1][0])
        frame.upperRightCorner.setLongitude(llC[1][1])
        frame.upperRightCorner.setHeight(self.terrainHeight)

        frame.lowerRightCorner = Coordinate()
        frame.lowerRightCorner.setLatitude(llC[2][0])
        frame.lowerRightCorner.setLongitude(llC[2][1])
        frame.lowerRightCorner.setHeight(self.terrainHeight)

        frame.lowerLeftCorner = Coordinate()
        frame.lowerLeftCorner.setLatitude(llC[3][0])
        frame.lowerLeftCorner.setLongitude(llC[3][1])
        frame.lowerLeftCorner.setHeight(self.terrainHeight)

        frame.nearLookAngle = math.degrees(self.metadata['Minimum Look Angle'])
        frame.farLookAngle = math.degrees(self.metadata['Maximum Look Angle'])

        return

    def _populateFrameSolo(self):
        self.logger.info("UAVSAR_Stack._populateFrameSolo")

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

        nlines = int(( self.frame.getSensingStop() - t0).total_seconds() * prf)

        #make sure the elp property has been called
        elp = self.elp
        orbit = self.frame.getOrbit()
        orbit.setOrbitSource('Header')


        for i in range(-5*numgroup, int(nlines/numgroup)*numgroup+5*numgroup, numgroup):
            delt = int(i * 1.0e6 /prf)
            torb = self.frame.getSensingStart() + datetime.timedelta(microseconds=delt)
            ###Need to compute offset
            ###While taking into account, rounding off in time
            ds = delt*1.0e-6*vel

            vec = OrbitStateVector()
            vec.setTime( torb )

            posSCH = [self.frame.S0 + ds, 0.0, self.platformHeight]
            velSCH = [self.velocity, 0., 0.]
            posXYZ, velXYZ =  elp.schdot_to_xyzdot(posSCH, velSCH)
            vec.setPosition(posXYZ)
            vec.setVelocity(velXYZ)
            orbit.addStateVector(vec)

        return
        #t0 = (self.frame.getSensingStart() -
              #datetime.timedelta(microseconds=delta))
        #ds = deltaFactor*self.frame.instrument.getAzimuthPixelSize()
        #s0 = self.platformStartingAzimuth - numExtra*ds
        #self.logger.info("populateOrbit: frame.sensingStart, frame.sensingStop  = ", self.frame.getSensingStart(),
            #self.frame.getSensingStop())
        #self.logger.info("populateOrbit: deltaFactor, numExtra, dt = ", deltaFactor, numExtra, dt)
        #self.logger.info("populateOrbit: t0, startingAzimuth, platformStartingAzimuth, s0, ds = ",
            #t0, self.frame.S0, self.platformStartingAzimuth, s0, ds)
        #h = self.platformHeight
        #v = [self.velocity, 0., 0.]
        #self.logger.info("t0, dt = ", t0, dt)
        #self.logger.info("s0, ds, h = ", s0, ds, h)
        #self.logger.info("elp.pegRadCur = ", self.elp.pegRadCur)
        #self.logger.info("v = ", v[0])
        #platform = self.frame.getInstrument().getPlatform()
        #elp = self.elp   #make sure the elp property has been called
        #orbit = self.frame.getOrbit()
        #orbit.setOrbitSource('Header')

        #for i in range(int(self.frame.getNumberOfLines()/deltaFactor)+1000*numExtra+1):
            #vec = OrbitStateVector()
            #t = t0 + datetime.timedelta(microseconds=int(i*dt*1e6))
            #vec.setTime(t)
            #posSCH = [s0 + i*ds , 0., h]
            #velSCH = v
            #posXYZ, velXYZ = self.elp.schdot_to_xyzdot(posSCH, velSCH)
            #sch_pos, sch_vel = elp.xyzdot_to_schdot(posXYZ, velXYZ)

            #vec.setPosition(posXYZ)
            #vec.setVelocity(velXYZ)
            #orbit.addStateVector(vec)
        #return

    def populateMetadata(self):
        self._populatePlatform()
        self._populateInstrument()
        self._populateFrame()
        #self.extractDoppler()
        self._populateOrbit()

    def parse(self):
        from iscesys.Parsers import rdf
        self.metadata = rdf.parse(self.metadataFile)
        self.populateMetadata()

    def extractImage(self):
        self.parse()
        slcImage = isceobj.createSlcImage()
        self.slcname = self.metadata['slc_{}_1x1'.format(self.segment_index)]
        slcImage.setFilename(self.slcname)
        slcImage.setXmin(0)
        slcImage.setXmax(self.frame.getNumberOfSamples())
        slcImage.setWidth(self.frame.getNumberOfSamples())
        slcImage.setAccessMode('r')
        self.frame.setImage(slcImage)
        return

    def extractDoppler(self):
        """
        Read doppler values from the doppler file and fit a polynomial
        """
        frame = self.frame
        instrument = frame.getInstrument()
        rho0 = frame.getStartingRange()
        drho = instrument.getRangePixelSize()  #full res value, not spacing in the dop file
        prf = instrument.getPulseRepetitionFrequency()
        self.logger.info("extractDoppler: rho0, drho, prf = {}, {}, {}".format(rho0, drho, prf))
        dopfile = getattr(self, 'dopplerFile', self.metadata['dop'])
        with open(dopfile,'r') as f:
            x = f.readlines()  #first line is a header

        import numpy
        z = numpy.array(
            [list(map(float, e)) for e in list(map(str.split, x[1:]))]
        )
        rho = z[:,0]
        dop = z[:,1]
        #rho0 = rho[0]
        #drho = (rho[1] - rho[0])/2.0
        rhoi = [(r-rho0)/drho for r in rho]
        polydeg = 6  #2  #Quadratic is built in for now
        fit = numpy.polynomial.polynomial.polyfit(rhoi, dop, polydeg, rcond=1.e-9,
              full=True)

        coefs = fit[0]
        res2 = fit[1][0]  #sum of squared residuals
        self.logger.info("coeffs = {}".format(coefs))
        self.logger.info("rms residual = {}".format(numpy.sqrt(res2/len(dop))))
        with open("dop.txt", 'w') as o:
            for i, d in zip(rhoi, dop):
                val = polyval(coefs,i)
                res = d-val
                o.write("{0} {1} {2} {3}\n".format(i, d, val, res))

        self.dopplerVals = {'Near':polyval(coefs, 0)}  #need this temporarily in this module

        self.logger.info("UAVSAR_Stack.extractDoppler: self.dopplerVals = {}".format(self.dopplerVals))
        self.logger.info("UAVSAR_Stack.extractDoppler: prf = {}".format(prf))

        #The doppler file values are in units rad/m.  divide by 2*pi rad/cycle to convert
        #to cycle/m.  Then multiply by velocity to get Hz and divide by prf for dimensionless
        #doppler coefficients
        dop_scale = self.velocity/2.0/math.pi
        coefs =  [x*dop_scale for x in coefs]
        #Set the coefs in frame._dopplerVsPixel because that is where DefaultDopp looks for them
        self.frame._dopplerVsPixel = coefs

        return coefs

    @property
    def terrainHeight(self):
        #The peg point incorporates the actual terrainHeight
        return 0.0

    @property
    def platformHeight(self):
        h = self.metadata['Global Average Altitude']
        #Reduce the platform height by the terrain height because the
        #peg radius of curvature includes the terrain height
        h -= self.metadata['Global Average Terrain Height']
        return h

    @property
    def platformStartingAzimuth(self):
        azimuth = self.frame.S0
        return azimuth

    @property
    def startingRange(self):
        return self.metadata['Image Starting Slant Range']

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

        if sinSquint**2 > 1:
            raise ValueError(
                "Error in One or More of the Squint Calculation Values\n"+
                "Doppler Centroid: %s\nVelocity: %s\nWavelength: %s\n" %
                (fd, v, wavelength)
                )
        self.squint = math.degrees(
            math.atan2(sinSquint, math.sqrt(1-sinSquint**2))
            )
        #squint is also required in the frame.
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
        v = self.metadata['Average Along Track Velocity']
        platform = self.frame.getInstrument().getPlatform()
        elp = self.elp
        peg = self.peg
        scale = (elp.pegRadCur + self.platformHeight)/elp.pegRadCur
        ds_ground = self.frame.instrument.getAzimuthPixelSize()
        dt = 1.0/self.frame.instrument.getPulseRepetitionFrequency()
        v1 = scale*ds_ground/dt
        return v1

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
            platform = self.frame.getInstrument().getPlatform()
            self.elp.setSCH(peg[0], peg[1], peg[2], th)
            rc = self.elp.pegRadCur

            from isceobj.Location.Peg import Peg
            self._peg =  Peg(latitude=peg[0], longitude=peg[1], heading=peg[2],
                             radiusOfCurvature=rc)
            self.logger.info("UAVSAR_Stack: peg radius of curvature = {}".format(self.elp.pegRadCur))
            self.logger.info("UAVSAR_Stack: terrain height = {}".format(th))

        return self._peg
