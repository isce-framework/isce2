#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2013 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Authors: Piyush Agram, Giangi Sacco
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



from __future__ import print_function
import os
import math
from isceobj.Location.Peg import Peg
from isceobj import Constants as CN
from iscesys.Component.Component import Component, Port
from iscesys.Compatibility import Compatibility
from stdproc.orbit import getpeg

PLANET_GM = Component.Parameter(
    'planetGM',
    public_name='planet GM (m**3/s**2)',
    type=float,
    default= CN.EarthGM,
    units='m**3/s**2',
    mandatory=True,
    doc="Planet mass times Newton's constant in units m**3/s**2"
    )

POSITION = Component.Parameter(
    'position',
    public_name='frame xyz position vectors (m)',
    type=float,
    default=None,
    units='m',
    mandatory=True,
    doc="List of xyz positions for frame."
    )

VELOCITY = Component.Parameter(
    'velocity',
    public_name='frame xyz velocity vectors (m/s)',
    type=float,
    default=None,
    units='m/s',
    mandatory=True,
    doc="List of xyz velocities for frame."
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


class Getpeg(Component):

    def estimatePeg(self):
        for port in self.inputPorts:
            port()
        self.allocateArrays()
        self.setState()
        getpeg.getpeg_Py()
        self.getState()
        self.deallocateArrays()
        self._peg = Peg(latitude=math.degrees(self.pegLatitude),
                        longitude=math.degrees(self.pegLongitude),
                        heading=math.degrees(self.pegHeading),
                        radiusOfCurvature=self.pegRadiusOfCurvature)

        return None


    def setState(self):
        getpeg.setStdWriter_Py(int(self.stdWriter))
        getpeg.setPosition_Py(self.position,
                                          self.dim1_position,
                                          self.dim2_position)
        getpeg.setVelocity_Py(self.velocity,
                                          self.dim1_velocity,
                                          self.dim2_velocity)
        getpeg.setPlanetGM_Py(float(self.planetGM))

        getpeg.setEllipsoidMajorSemiAxis_Py(
            float(self.ellipsoidMajorSemiAxis)
            )
        getpeg.setEllipsoidEccentricitySquared_Py(
            float(self.ellipsoidEccentricitySquared)
            )

        return None

    def setPosition(self,var):
        self.position1 = var
        return None

    def setVelocity(self,var):
        self.velocity1 = var
        return None

    def setPlanetGM(self,var):
        self.planetGM = float(var)
        return None

    def setEllipsoidMajorSemiAxis(self,var):
        self.ellipsoidMajorSemiAxis = float(var)
        return None

    def setEllipsoidEccentricitySquared(self,var):
        self.ellipsoidEccentricitySquared = float(var)
        return None


    def getState(self):
        self.pegLatitude = getpeg.getPegLatitude_Py()
        self.pegLongitude = getpeg.getPegLongitude_Py()
        self.pegHeading = getpeg.getPegHeading_Py()
        self.pegRadiusOfCurvature = getpeg.getPegRadiusOfCurvature_Py()
        self.averageHeight = getpeg.getAverageHeight_Py()
        self.procVelocity = getpeg.getProcVelocity_Py()

        return None

    # added the setter to allow precomputed peg point to be used
    def setPeg(self,peg):
        self._peg = peg

    def getPeg(self):
        return self._peg

    def getPegLatitude(self):
        return self.pegLatitude

    def getPegLongitude(self):
        return self.pegLongitude

    def getPegHeading(self):
        return self.pegHeading

    def getPegRadiusOfCurvature(self):
        return self.pegRadiusOfCurvature

    def getAverageHeight(self):
        return self.averageHeight

    def getProcVelocity(self):
        return self.procVelocity

    def allocateArrays(self):
        if (self.dim1_position == None):
            self.dim1_position = len(self.position)
            self.dim2_position = len(self.position[0])

        if (not self.dim1_position) or (not self.dim2_position):
            print("Error. Trying to allocate zero size array")

            raise Exception

        getpeg.allocate_xyz_Py(self.dim1_position, self.dim2_position)

        if (self.dim1_velocity == None):
            self.dim1_velocity = len(self.velocity)
            self.dim2_velocity = len(self.velocity[0])

        if (not self.dim1_velocity) or (not self.dim2_velocity):
            print("Error. Trying to allocate zero size array")

            raise Exception

        getpeg.allocate_vxyz_Py(self.dim1_velocity, self.dim2_velocity)

        return None

    def addOrbit(self):                
        Orbit = self._inputPorts.getPort('Orbit').getObject()
        if (Orbit):
            try:
                time, self.position, self.velocity, offset = Orbit._unpackOrbit()
            except AttributeError:
                print("Object %s requires private method _unpackOrbit()" % (Orbit.__class__))                 
                raise AttributeError

    def addPlanet(self):        
        planet = self._inputPorts.getPort('planet').getObject()
        if(planet):
            try:
                self.planetGM = planet.get_GM()
                self.ellipsoidMajorSemiAxis = planet.get_elp().get_a()
                self.ellipsoidEccentricitySquared = planet.get_elp().get_e2()
            except AttributeError:
                print("Object %s requires get_GM(), get_elp().get_a() and get_elp().get_e2() methods" % (planet.__class__))

    def deallocateArrays(self):
        getpeg.deallocate_xyz_Py()
        getpeg.deallocate_vxyz_Py()
        return None

    def __init__(self):
        super(Getpeg, self).__init__()
        #some defaults 
        self.planetGM = CN.EarthGM
        self.ellipsoidMajorSemiAxis = CN.EarthMajorSemiAxis
        self.ellipsoidEccentricitySquared = CN.EarthEccentricitySquared
        
        self.position = []
        self.dim1_position = None
        self.dim2_position = None
        self.velocity = []
        self.dim1_velocity = None
        self.dim2_velocity = None
        self.pegLatitude = None
        self.pegLongitude = None
        self.pegHeading = None
        self.pegRadiusOfCurvature = None
        self.averageHeight = None
        self.procVelocity  = None
        self._peg = None
        #Create ports
        self.createPorts()
                
        self.dictionaryOfOutputVariables = {
            'PEG_LATITUDE':'self.pegLatitude',
            'PEG_LONGITUDE':'self.pegLongitude',
            'PEG_HEADING':'self.pegHeading',
            'PEG_RADIUS_OF_CURVATURE':'self.pegRadiusOfCurvature',
            'AVERAGE_HEIGHT':'self.averageHeight',
            'PROC_VELOCITY':'self.procVelocity',
            }
        self.descriptionOfVariables = {}
        self.mandatoryVariables = []
        self.optionalVariables = []
        return None

    def createPorts(self):
        planetPort = Port(name='planet',method=self.addPlanet)
        orbitPort = Port(name='Orbit',method=self.addOrbit)
        # Add the ports
        self._inputPorts.add(planetPort)
        self._inputPorts.add(orbitPort)
        return None

    pass
