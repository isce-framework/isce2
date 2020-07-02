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
from isceobj.Location.Peg import Peg
from isceobj import Constants as CN
from iscesys.Component.Component import Component, Port
from iscesys.Compatibility import Compatibility
from stdproc.orbit import setmocomppath

PLANET_GM = Component.Parameter(
    'planetGM',
    public_name='planet GM (m**3/s**2)',
    type=float,
    default= CN.EarthGM,
    units='m**3/s**2',
    mandatory=True,
    doc="Planet mass times Newton's constant in units m**3/s**2"
    )

FIRST_POSITION = Component.Parameter(
    'position1',
    public_name='first frame xyz position vectors (m)',
    type=float,
    default=None,
    units='m',
    mandatory=True,
    doc="List of xyz positions for first (reference) frame."
    )

SECOND_POSITION = Component.Parameter(
    'position2',
    public_name='second frame xyz position vectors (m)',
    type=float,
    default=None,
    units='m',
    mandatory=True,
    doc="List of xyz positions for second (secondary) frame."
    )

FIRST_VELOCITY = Component.Parameter(
    'velocity1',
    public_name='first frame xyz velocity vectors (m/s)',
    type=float,
    default=None,
    units='m/s',
    mandatory=True,
    doc="List of xyz velocities for first (reference) frame."
    )

SECOND_POSITION = Component.Parameter(
    'velocity2',
    public_name='second frame xyz velocity vectors (m/s)',
    type=float,
    default=None,
    units='m/s',
    mandatory=True,
    doc="List of xyz velocities for second (secondary) frame."
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


class Setmocomppath(Component):
    
    def setmocomppath(self):
        for port in self.inputPorts:
            port()

        self.allocateArrays()
        self.setState()
        setmocomppath.setmocomppath_Py()
        self.getState()
        self.deallocateArrays()
        self._peg = Peg(latitude=math.degrees(self.pegLatitude),
                        longitude=math.degrees(self.pegLongitude),
                        heading=math.degrees(self.pegHeading),
                        radiusOfCurvature=self.pegRadiusOfCurvature)

        return None


    def setState(self):
        setmocomppath.setStdWriter_Py(int(self.stdWriter))
        setmocomppath.setFirstPosition_Py(self.position1,
                                          self.dim1_position1,
                                          self.dim2_position1)
        setmocomppath.setFirstVelocity_Py(self.velocity1,
                                          self.dim1_velocity1,
                                          self.dim2_velocity1)
        setmocomppath.setSecondPosition_Py(self.position2,
                                           self.dim1_position2,
                                           self.dim2_position2)
        setmocomppath.setSecondVelocity_Py(self.velocity2,
                                           self.dim1_velocity2,
                                           self.dim2_velocity2)
        setmocomppath.setPlanetGM_Py(float(self.planetGM))

        setmocomppath.setEllipsoidMajorSemiAxis_Py(
            float(self.ellipsoidMajorSemiAxis)
            )
        setmocomppath.setEllipsoidEccentricitySquared_Py(
            float(self.ellipsoidEccentricitySquared)
            )

        return None

    def setFirstPosition(self,var):
        self.position1 = var
        return None

    def setFirstVelocity(self,var):
        self.velocity1 = var
        return None

    def setSecondPosition(self,var):
        self.position2 = var
        return None

    def setSecondVelocity(self,var):
        self.velocity2 = var
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
        self.pegLatitude = setmocomppath.getPegLatitude_Py()
        self.pegLongitude = setmocomppath.getPegLongitude_Py()
        self.pegHeading = setmocomppath.getPegHeading_Py()
        self.pegRadiusOfCurvature = setmocomppath.getPegRadiusOfCurvature_Py()
        self.averageHeight1 = setmocomppath.getFirstAverageHeight_Py()
        self.averageHeight2 = setmocomppath.getSecondAverageHeight_Py()
        self.procVelocity1 = setmocomppath.getFirstProcVelocity_Py()
        self.procVelocity2 = setmocomppath.getSecondProcVelocity_Py()

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

    def getFirstAverageHeight(self):
        return self.averageHeight1

    def getSecondAverageHeight(self):
        return self.averageHeight2

    def getFirstProcVelocity(self):
        return self.procVelocity1

    def getSecondProcVelocity(self):
        return self.procVelocity2

    def allocateArrays(self):
        if (self.dim1_position1 == None):
            self.dim1_position1 = len(self.position1)
            self.dim2_position1 = len(self.position1[0])

        if (not self.dim1_position1) or (not self.dim2_position1):
            print("Error. Trying to allocate zero size array")

            raise Exception

        setmocomppath.allocate_xyz1_Py(self.dim1_position1, self.dim2_position1)

        if (self.dim1_velocity1 == None):
            self.dim1_velocity1 = len(self.velocity1)
            self.dim2_velocity1 = len(self.velocity1[0])

        if (not self.dim1_velocity1) or (not self.dim2_velocity1):
            print("Error. Trying to allocate zero size array")

            raise Exception

        setmocomppath.allocate_vxyz1_Py(self.dim1_velocity1, self.dim2_velocity1)

        if (self.dim1_position2 == None):
            self.dim1_position2 = len(self.position2)
            self.dim2_position2 = len(self.position2[0])

        if (not self.dim1_position2) or (not self.dim2_position2):
            print("Error. Trying to allocate zero size array")

            raise Exception

        setmocomppath.allocate_xyz2_Py(self.dim1_position2, self.dim2_position2)

        if (self.dim1_velocity2 == None):
            self.dim1_velocity2 = len(self.velocity2)
            self.dim2_velocity2 = len(self.velocity2[0])

        if (not self.dim1_velocity2) or (not self.dim2_velocity2):
            print("Error. Trying to allocate zero size array")

            raise Exception

        setmocomppath.allocate_vxyz2_Py(self.dim1_velocity2, self.dim2_velocity2)
        return None

    def addReferenceOrbit(self):                
        referenceOrbit = self._inputPorts.getPort('referenceOrbit').getObject()
        if referenceOrbit:
            try:
                time, self.position1, self.velocity1, offset = referenceOrbit._unpackOrbit()
            except AttributeError:
                print("Object %s requires private method _unpackOrbit()" % (referenceOrbit.__class__))                 
                raise AttributeError


    def addSecondaryOrbit(self):                
        secondaryOrbit = self._inputPorts.getPort('secondaryOrbit').getObject()
        if secondaryOrbit:
            try:
                time, self.position2, self.velocity2, offset = secondaryOrbit._unpackOrbit()
            except AttributeError:
                print("Object %s requires private method _unpackOrbit()" % (secondaryOrbit.__class__))                 
                raise AttributeError


    def addPlanet(self):        
        planet = self._inputPorts.getPort('planet').getObject()
        if planet:
            try:
                self.planetGM = planet.get_GM()
                self.ellipsoidMajorSemiAxis = planet.get_elp().get_a()
                self.ellipsoidEccentricitySquared = planet.get_elp().get_e2()
            except AttributeError:
                print("Object %s requires get_GM(), get_elp().get_a() and get_elp().get_e2() methods" % (planet.__class__))


    def deallocateArrays(self):
        setmocomppath.deallocate_xyz1_Py()
        setmocomppath.deallocate_vxyz1_Py()
        setmocomppath.deallocate_xyz2_Py()
        setmocomppath.deallocate_vxyz2_Py()

        return None

    def __init__(self):
        super(Setmocomppath, self).__init__()
        #some defaults 
        self.planetGM = CN.EarthGM
        self.ellipsoidMajorSemiAxis = CN.EarthMajorSemiAxis
        self.ellipsoidEccentricitySquared = CN.EarthEccentricitySquared
        
        self.position1 = []
        self.dim1_position1 = None
        self.dim2_position1 = None
        self.velocity1 = []
        self.dim1_velocity1 = None
        self.dim2_velocity1 = None
        self.position2 = []
        self.dim1_position2 = None
        self.dim2_position2 = None
        self.velocity2 = []
        self.dim1_velocity2 = None
        self.dim2_velocity2 = None
        self.pegLatitude = None
        self.pegLongitude = None
        self.pegHeading = None
        self.pegRadiusOfCurvature = None
        self.averageHeight1 = None
        self.averageHeight2 = None
        self.procVelocity1 = None
        self.procVelocity2 = None
        self._peg = None
#        self.createPorts()
        self.dictionaryOfOutputVariables = {
            'PEG_LATITUDE':'self.pegLatitude',
            'PEG_LONGITUDE':'self.pegLongitude',
            'PEG_HEADING':'self.pegHeading',
            'PEG_RADIUS_OF_CURVATURE':'self.pegRadiusOfCurvature',
            'FIRST_AVERAGE_HEIGHT':'self.averageHeight1',
            'SECOND_AVERAGE_HEIGHT':'self.averageHeight2',
            'FIRST_PROC_VELOCITY':'self.procVelocity1',
            'SECOND_PROC_VELOCITY':'self.procVelocity2',\
                }
        self.descriptionOfVariables = {}
        self.mandatoryVariables = []
        self.optionalVariables = []
        typePos = 2
        for key , val in self.dictionaryOfVariables.items():
            if val[typePos] == 'mandatory':
                self.mandatoryVariables.append(key)
            elif val[typePos] == 'optional':
                self.optionalVariables.append(key)
            else:
                print('Error. Variable can only be optional or mandatory')
                raise Exception
        return

    def createPorts(self):
        #Create ports
        planetPort = Port(name='planet',method=self.addPlanet)
        referenceOrbitPort = Port(name='referenceOrbit',method=self.addReferenceOrbit)
        secondaryOrbitPort = Port(name='secondaryOrbit',method=self.addSecondaryOrbit)
        # Add the ports
        self._inputPorts.add(planetPort)
        self._inputPorts.add(referenceOrbitPort)
        self._inputPorts.add(secondaryOrbitPort)
        return None


    pass
