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
# Author: Giangi Sacco
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



import logging
import math

from iscesys.Compatibility import Compatibility

from isceobj.Planet import Planet
from isceobj import Constants as CN
from iscesys.Component.Component import Component, Port

RANGE_SAMPLING_RATE = Component.Parameter('rangeSamplingRate',
                                          public_name='range sampling rate',
                                          type=float,
                                          default=None,
                                          units='Hz',
                                          mandatory=True,
                                          doc="Sampling rate in range"
                                          )
PRF = Component.Parameter('prf',
                          public_name='prf',
                          type=float,
                          default=None,
                          units='Hz',
                          mandatory=True,
                          doc="Pulse repetition frequency"
                          )
RANGE_FIRST_SAMPLE = Component.Parameter('rangeFirstSample',
                                         public_name='range to first sample',
                                         type=float,
                                         default=None,
                                         units='meter',
                                         mandatory=True,
                                         doc="Range in meters to the first sample"
                                         )


class CalcSchHeightVel(Component):

    parameter_list = (RANGE_SAMPLING_RATE,
                      PRF,
                      RANGE_FIRST_SAMPLE)


    def calculate(self):
        for port in self.inputPorts:
            port()

        self.b = self.a*math.sqrt(1-self.e2)
        ro = self.rangeFirstSample
        sol = CN.SPEED_OF_LIGHT
        rc = self.b
        ra = self.a

        fs = self.rangeSamplingRate
        dt = 1/self.prf
        dr = 1/2.*sol/fs
        half = len(self.pos)//2 - 1
        xyz = self.pos[half]
        vxyz = self.vel[half]
        rs = math.sqrt(xyz[0]*xyz[0] + xyz[1]*xyz[1] + xyz[2]*xyz[2])
        vs = math.sqrt(vxyz[0]*vxyz[0] + vxyz[1]*vxyz[1] + vxyz[2]*vxyz[2])
        rlat = math.asin(xyz[2]/rs)
        rlatg = math.atan(math.tan(rlat)*ra*ra/(rc*rc))

        st = math.sin(rlatg)
        ct = math.cos(rlatg)
        arg = (ct*ct)/(ra*ra) + (st*st)/(rc*rc)
        re = 1./math.sqrt(arg)
        try:
            re = self.pegRadCur
        except:
            pass
        # compute the vector orthogonal to both the radial vector and velocity vector */

        a = [xyz[0]/rs,xyz[1]/rs,xyz[2]/rs]
        b = [vxyz[0]/vs,vxyz[1]/vs,vxyz[2]/vs]

#     cross product
        c =  [(a[1]*b[2]) - (a[2]*b[1]),(-a[0]*b[2]) + (a[2]*b[0]),(a[0]*b[1]) - (a[1]*b[0])]

#     /*  compute the look angle */
        ct = (rs*rs+ro*ro-(re+self.terrainHeight)**2)/(2.*rs*ro)
        st = math.sin(math.acos(ct))

#     /* add the satellite and LOS vectors to get the new point */
        xe = xyz[0]+ro*(-st*c[0]-ct*a[0])
        ye = xyz[1]+ro*(-st*c[1]-ct*a[1])
        ze = xyz[2]+ro*(-st*c[2]-ct*a[2])
        rlat = math.asin(ze/re)
        rlatg = math.atan(math.tan(rlat)*ra*ra/(rc*rc))


#     /*  compute elipse height in the scene */
        st = math.sin(rlatg)
        ct = math.cos(rlatg)
        arg = (ct*ct)/(ra*ra)+(st*st)/(rc*rc)
        re = 1./(math.sqrt(arg))

        self.height = rs - re

#     /* now check range over time */

#jng the original code claims that it uses the center +- 2 sec to compute the velocity. it skips 10000 lines from the beginning and the end, which is about 8000 lines in the center. this is 2.4 sec. default the self.offset to be close to 2.4 sec.
        offset = int(self.offset*self.prf)

        lo = max(half-offset,0)
        hi = min(half+offset,2*half)
        #lo = 10000
        #hi = len(self.pos) - 10000
        rng = [0]*(hi-lo)
        cnt =  0
        for i in range(lo,hi):
            rng[cnt] = math.sqrt((xe-self.pos[i][0])*(xe-self.pos[i][0]) + (ye-self.pos[i][1])*(ye-self.pos[i][1]) + (ze-self.pos[i][2])*(ze-self.pos[i][2])) - ro
            cnt += 1
        sumdr = 0
        for i  in range(1,len(rng)-1):
            sumdr += rng[i+1] + rng[i-1] -2*rng[i]
        sumdr /= (len(rng)-2)*dt*dt

        self.velocity = math.sqrt(ro*math.fabs(sumdr))
        return None


    ## You need this identity to use Componenet.__call__
    calcschheightvel = calculate


    def setRangeFirstSample(self,rfs):
        self.rangeFirstSample = rfs

    def setRangeSamplingRate(self,rsr):
        self.rangeSamplingRate = rsr

    def setPRF(self,prf):
        self.prf = prf

    def setPosition(self,pos):
        self.pos = pos

    def setVelocity(self,vel):
        self.vel = vel

    def setOffest(self,off):
        self.offset = off

    def setEllipsoidMajorAxis(self,a):
        self.a = a

    def setEllipsoidEccentricitySquared(self,e2):
        self.e2 = e2

    def getHeight(self):
        return self.height

    def getVelocity(self):
        return self.velocity

    def addOrbit(self):
        orbit = self._inputPorts.getPort('orbit').getObject()
        if (orbit):
            try:
                (time,position,velocity,offset) = orbit._unpackOrbit()
                self.pos = position
                self.vel = velocity
            except AttributeError:
                self.logger.error("Object %s requires an _unpackOrbit() method" % (orbit.__class__))
                raise AttributeError
    def addFrame(self):
        frame = self._inputPorts.getPort('frame').getObject()
        if (frame):
            try:
                self.rangeFirstSample = frame.getStartingRange()
                instrument = frame.getInstrument()
                self.rangeSamplingRate = instrument.getRangeSamplingRate()
                self.prf = instrument.getPulseRepetitionFrequency()
            except AttributeError as strerr:
                self.logger.error(strerr)
                raise AttributeError
            try:
                self.terrainHeight = frame.terrainHeight
                self.pegRadCur = frame._ellipsoid.pegRadCur
            except:
                self.terrainHeight = 0.0

    def addPlanet(self):
        planet = self._inputPorts.getPort('planet').getObject()
        if (planet):
            try:
                ellipsoid = planet.get_elp()
                self.a = ellipsoid.get_a()
                self.e2 = ellipsoid.get_e2()
            except AttributeError as strerr:
                self.logger.error(strerr)
                raise AttributeError

    logging_name = "CalcSchHeightVel"

    def __init__(self):
        super(CalcSchHeightVel, self).__init__()
        planet = Planet.Planet(pname='Earth')
        ellipsoid = planet.get_elp()
        self.a = ellipsoid.get_a()
        self.e2 = ellipsoid.get_e2()
        self.b = None
        self.rangeFirstSample = None
        self.rangeSamplingRate = None
        self.prf = None
        self.height = None
        self.velocity = None
        self.prf = None
        self.pos = None
        self.vel = None
        self.offset = 2.3758
        self.terrainHeight = 0.0
#        self.logger = logging.getLogger("CalcSchHeightVel")
#        self.createPorts()
        self.dictionaryOfOutputVariables = {
            'HEIGHT' : 'height' ,
            'VELOCITY' : 'velocity'
            }

# TODO: 'radius' does not exist as an member of this class
#                                            'RADIUS' : 'radius' \
#                                           }
        self.descriptionOfVariables = {}
        self.mandatoryVariables = []
        self.optionalVariables = []
        self.initOptionalAndMandatoryLists()
        return None

    def createPorts(self):
        orbitPort = Port(name='orbit',method=self.addOrbit)
        framePort = Port(name='frame',method=self.addFrame)
        planetPort = Port(name='planet',method=self.addPlanet)
        self._inputPorts.add(orbitPort)
        self._inputPorts.add(framePort)
        self._inputPorts.add(planetPort)
        return None
    pass


def main():
    import pdb
    pdb.set_trace()
    with open(sys.argv[1]) as fp:
        allL = fp.readlines()
        numberOfLines = len(allL)
        position = []
        velocity = []
        for i in range(numberOfLines):
            line = allL[i].split()
            position.append([float(line[2]),float(line[3]),float(line[4])])
            velocity.append([float(line[5]),float(line[6]),float(line[7])])

    ch = CalcSchHeightVel()
    ch.setPosition(position)
    ch.setVelocity(velocity)
    ch.setPRF(1741.71924)
    ch.setRangeFirstSample(955972.779)
    ch.setRangeSamplingRate(19207680.)
    ch.calculate()

if __name__ == '__main__':
    import sys
    sys.exit(main())
