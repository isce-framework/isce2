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



from __future__ import print_function
import math
import datetime
import logging

from isceobj import Constants as CN
from isceobj.Orbit.Orbit import Orbit, StateVector
from iscesys.Component.Component import Component, Port
from iscesys.Compatibility import Compatibility
from stdproc.orbit import sch2orbit
from isceobj.Util.decorators import port, logged, pickled

ORBIT_POSITION = Component.Parameter(
    'orbitPosition',
    public_name='orbit sch position vectors',
    default=[],
    container=list,
    type=float,
    units='m',
    mandatory=True,
    doc="Orbit xyz position vectors"
    )

ORBIT_VELOCITY = Component.Parameter(
    'orbitVelocity',
    public_name='orbit sch velocity vectors',
    default=[],
    container=list,
    type=float,
    units='m/s',
    mandatory=True,
    doc="Orbit xyz velocity vectors"
    )

PLANET_GM = Component.Parameter(
    'planetGM',
    public_name='planet GM (m**3/s**2)',
    type=float,
    default= CN.EarthGM,
    units='m**3/s**2',
    mandatory=True,
    doc="Planet mass times Newton's constant in units m**3/s**2"
    )

ELLIPSOID_MAJOR_SEMIAXIS = Component.Parameter(
    'ellipsoidMajorSemiAxis',
    public_name='ellipsoid semi major axis (m)',
    type=float,
    default=CN.EarthMajorSemiAxis,
    units='m',
    mandatory=True,
    doc="Ellipsoid semi major axis"
    )

ELLIPSOID_ECCENTRICITY_SQUARED = Component.Parameter(
    'ellipsoidEccentricitySquared',
    public_name='ellipsoid eccentricity squared',
    type=float,
    default=CN.EarthEccentricitySquared,
    units=None,
    mandatory=True,
    doc="Ellipsoid eccentricity squared"
    )

PEG_LATITUDE = Component.Parameter(
    'pegLatitude',
    public_name='peg latitude (rad)',
    default=0.,
    units='rad',
    type=float,
    mandatory=False,
    doc="Peg point latitude to use if compute peg flag = -1"
    )

PEG_LONGITUDE = Component.Parameter(
    'pegLongitude',
    public_name='peg longitude (rad)',
    default=0.,
    units='rad',
    type=float,
    mandatory=False,
    doc="Peg longitude to use if compute peg flag = -1"
    )

PEG_HEADING = Component.Parameter(
    'pegHeading',
    public_name='peg heading (rad)',
    default=0.,
    units='rad',
    type=float,
    mandatory=False,
    doc="Peg point heading to use if compute peg flag = -1"
    )

RADIUS_OF_CURVATURE = Component.Parameter(
    'radiusOfCurvature',
    public_name='local radius of curvature for SCH orbit',
    default=0,
    units='m',
    type=float,
    mandatory=False,
    doc="Radius of curvature at peg point used for SCH transform"
    )

class Sch2orbit(Component):

    planetGM = CN.EarthGM
    ellipsoidMajorSemiAxis = CN.EarthMajorSemiAxis
    ellipsoidEccentricitySquared = CN.EarthEccentricitySquared

    def __init__(self,
                 averageHeight=None,
                 planet=None,
                 orbit=None,
                 peg=None):

        super(Sch2orbit, self).__init__()

        self.averageHeight = averageHeight

        if planet is not None: self.wireInputPort(name='planet', object=planet)
        if orbit is not orbit: self.wireInputPort(name='orbit', object=orbit)
        if peg is not None: self.wireInputPort(name='peg', object=peg)

        self._numVectors = None
        self._time = None
        self._orbit = None

        self.position = []
        self.velocity = []
        self.acceleration = []
        self.logger = logging.getLogger('isce.sch2orbit')
        self.dictionaryOfOutputVariables = {'XYZ_POSITION' : 'self.position',

                                            'XYZ_VELOCITY':'self.velocity',

                                            'XYZ_GRAVITATIONAL_ACCELERATION':'self.acceleration'}
        return

    def createPorts(self):
        # Create input ports
        orbitPort = Port(name='orbit', method=self.addOrbit)
        planetPort = Port(name='planet', method=self.addPlanet)
        pegPort = Port(name='peg', method=self.addPeg)
        # Add the ports
        self.inputPorts.add(orbitPort)
        self.inputPorts.add(planetPort)
        self.inputPorts.add(pegPort)
        return None


    def sch2orbit(self):
        for port in self.inputPorts:
            port()

        lens = [len(self.orbitPosition), len(self.orbitVelocity)]
        if min(lens) != max(lens):
            raise Exception('Position and Velocity vector lengths dont match')

        self._numVectors = lens[0]

        self.allocateArrays()
        self.setState()
        sch2orbit.sch2orbit_Py()
        self.getState()
        self.deallocateArrays()
        self._orbit = Orbit(source='XYZ')
        self._orbit.setReferenceFrame('XYZ')
#
        for i in range(len(self.position)):
            sv = StateVector()
            sv.setTime(self._time[i])
            sv.setPosition(self.position[i])
            sv.setVelocity(self.velocity[i])
            self._orbit.addStateVector(sv)
        return

    def setState(self):
        sch2orbit.setStdWriter_Py(int(self.stdWriter))
        sch2orbit.setPegLatitude_Py(float(self.pegLatitude))
        sch2orbit.setPegLongitude_Py(float(self.pegLongitude))
        sch2orbit.setPegHeading_Py(float(self.pegHeading))
        sch2orbit.setRadiusOfCurvature_Py(float(self.radiusOfCurvature))

        sch2orbit.setOrbitPosition_Py(self.orbitPosition,
                                      self._numVectors)
        sch2orbit.setOrbitVelocity_Py(self.orbitVelocity,
                                      self._numVectors)
        sch2orbit.setPlanetGM_Py(float(self.planetGM))
        sch2orbit.setEllipsoidMajorSemiAxis_Py(
            float(self.ellipsoidMajorSemiAxis)
            )
        sch2orbit.setEllipsoidEccentricitySquared_Py(
            float(self.ellipsoidEccentricitySquared)
            )
        return None

    def setOrbitPosition(self, var):
        self.orbitPosition = var
        return

    def setOrbitVelocity(self, var):
        self.orbitVelocity = var
        return

    def setPlanetGM(self, var):
        self.planetGM = float(var)
        return

    def setEllipsoidMajorSemiAxis(self, var):
        self.ellipsoidMajorSemiAxis = float(var)
        return

    def setEllipsoidEccentricitySquared(self, var):
        self.ellipsoidEccentricitySquared = float(var)
        return

    def setPegLatitude(self, var):
        self.pegLatitude = float(var)
        return

    def setPegLongitude(self, var):
        self.pegLongitude = float(var)
        return

    def setPegHeading(self, var):
        self.pegHeading = float(var)
        return

    def setRadiusOfCurvature(self, var):
        self.radiusOfCurvature = float(var)
        return

    def getState(self):
        self.position = sch2orbit.getXYZPosition_Py(self._numVectors)
        self.velocity = sch2orbit.getXYZVelocity_Py(self._numVectors)
        self.acceleration = sch2orbit.getXYZGravitationalAcceleration_Py(self._numVectors)
        return


    def getXYZVelocity(self):
        return self.velocity

    def getXYZGravitationalAcceleration(self):
        return self.acceleration

    def getOrbit(self):
        return self._orbit

    def allocateArrays(self):
        if (not self._numVectors):
            raise ValueError("Error. Trying to allocate zero size array")

        sch2orbit.allocateArrays_Py(self._numVectors)

        return





    def deallocateArrays(self):
        sch2orbit.deallocateArrays_Py()

        return


    @property
    def orbit(self):
        return self._orbit
    @orbit.setter
    def orbit(self, orbit):
        self.orbit = orbit
        return None

    @property
    def time(self):
        return self._time
    @time.setter
    def time(self):
        self.time = time
        return None

    def addOrbit(self):
        orbit = self.inputPorts['orbit']
        if orbit:
            try:
                time, self.orbitPosition, self.orbitVelocity, offset = orbit.to_tuple()
                self._time = []
                for t in time:
                    self._time.append(offset +  datetime.timedelta(seconds=t))
            except AttributeError:
                self.logger.error(
                    "orbit port should look like an orbit, not: %s" %
                    (orbit.__class__)
                    )
                raise AttributeError
            pass
        return None

    def addPlanet(self):
        planet = self._inputPorts.getPort('planet').getObject()
        if(planet):
            try:
                self.planetGM = planet.get_GM()
                self.ellipsoidMajorSemiAxis = planet.get_elp().get_a()
                self.ellipsoidEccentricitySquared = planet.get_elp().get_e2()
            except AttributeError:
                self.logger.error(
                    "Object %s requires get_GM(), get_elp().get_a() and get_elp().get_e2() methods" % (planet.__class__)
                    )
                raise AttributeError

    def addPeg(self):
        peg = self._inputPorts.getPort('peg').getObject()
        if(peg):
            try:
                self.pegLatitude = math.radians(peg.getLatitude())
                self.pegLongitude = math.radians(peg.getLongitude())
                self.pegHeading = math.radians(peg.getHeading())
                self.logger.debug("Peg Object: %s" % (str(peg)))
            except AttributeError:
                self.logger.error(
                    "Object %s requires getLatitude(), getLongitude() and getHeading() methods" %
                    (peg.__class__)
                    )
                raise AttributeError

            pass
        pass
    pass
