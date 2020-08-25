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
from isceobj import Constants as CN
from iscesys.Component.Component import Component, Port
from iscesys.Compatibility import Compatibility
from stdproc.orbit import mocompbaseline


DIM = 3

ELLIPSOID_ECCENTRICITY_SQUARED = Component.Parameter(
    'ellipsoidEccentricitySquared',
    public_name='ELLIPSOID_ECCENTRICITY_SQUARED',
    default=CN.EarthEccentricitySquared,
    type=float,
    mandatory=False,
    intent='input',
    doc=''
)


ELLIPSOID_MAJOR_SEMIAXIS = Component.Parameter(
    'ellipsoidMajorSemiAxis',
    public_name='ELLIPSOID_MAJOR_SEMIAXIS',
    default=CN.EarthMajorSemiAxis,
    type=float,
    mandatory=False,
    intent='input',
    doc=''
)


HEIGHT = Component.Parameter(
    'height',
    public_name='HEIGHT',
    default=None,
    type=float,
    mandatory=True,
    intent='input',
    doc=''
)

POSITION1 = Component.Parameter(
    'position1',
    public_name='POSITION1',
    default=[],
    container=list,
    type=float,
    mandatory=True,
    intent='input',
    doc=''
)

POSITION2 = Component.Parameter(
    'position2',
    public_name='POSITION2',
    default=[],
    container=list,
    type=float,
    mandatory=True,
    intent='input',
    doc=''
)

MOCOMP_POSITION1 = Component.Parameter(
    'mocompPosition1',
    public_name='MOCOMP_POSITION1',
    default=[],
    container=list,
    type=float,
    mandatory=True,
    intent='input',
    doc=''
)


MOCOMP_POSITION2 = Component.Parameter(
    'mocompPosition2',
    public_name='MOCOMP_POSITION2',
    default=[],
    container=list,
    type=float,
    mandatory=True,
    intent='input',
    doc=''
)


MOCOMP_POSITION_INDEX1 = Component.Parameter(
    'mocompPositionIndex1',
    public_name='MOCOMP_POSITION_INDEX1',
    default=[],
    container=list,
    type=int,
    mandatory=True,
    intent='input',
    doc=''
)


MOCOMP_POSITION_INDEX2 = Component.Parameter(
    'mocompPositionIndex2',
    public_name='MOCOMP_POSITION_INDEX2',
    default=[],
    container=list,
    type=int,
    mandatory=True,
    intent='input',
    doc=''
)


PEG_HEADING = Component.Parameter(
    'pegHeading',
    public_name='PEG_HEADING',
    default=None,
    type=float,
    mandatory=True,
    intent='input',
    doc=''
)


PEG_LATITUDE = Component.Parameter(
    'pegLatitude',
    public_name='PEG_LATITUDE',
    default=None,
    type=float,
    mandatory=True,
    intent='input',
    doc=''
)


PEG_LONGITUDE = Component.Parameter(
    'pegLongitude',
    public_name='PEG_LONGITUDE',
    default=None,
    type=float,
    mandatory=True,
    intent='input',
    doc=''
)


PLANET_LOCAL_RADIUS = Component.Parameter(
    'planetLocalRadius',
    public_name='PLANET_LOCAL_RADIUS',
    default=None,
    type=float,
    mandatory=True,
    intent='input',
    doc=''
)


BASE1 = Component.Parameter(
    'base1',
    public_name='BASE1',
    default=[],
    container=list,
    type=float,
    mandatory=False,
    intent='output',
    doc=''
)


BASE2 = Component.Parameter(
    'base2',
    public_name='BASE2',
    default=[],
    container=list,
    type=float,
    mandatory=False,
    intent='output',
    doc=''
)


MIDPOINT = Component.Parameter(
    'midpoint',
    public_name='MIDPOINT',
    default=[],
    container=list,
    type=float,
    mandatory=False,
    intent='output',
    doc=''
)


MIDPOINT1 = Component.Parameter(
    'midpoint1',
    public_name='MIDPOINT1',
    default=[],
    container=list,
    type=float,
    mandatory=False,
    intent='output',
    doc=''
)


MIDPOINT2 = Component.Parameter(
    'midpoint2',
    public_name='MIDPOINT2',
    default=[],
    container=list,
    type=float,
    mandatory=False,
    intent='output',
    doc=''
)


MOCOMP_BASELINE = Component.Parameter(
    'baselineArray',
    public_name='MOCOMP_BASELINE',
    default=[],
    container=list,
    type=float,
    mandatory=False,
    intent='output',
    doc=''
)


SC = Component.Parameter(
    'sc',
    public_name='SC',
    default=[],
    container=list,
    type=float,
    mandatory=False,
    intent='output',
    doc=''
)


SCH = Component.Parameter(
    'sch',
    public_name='SCH',
    default=[],
    container=list,
    type=float,
    mandatory=False,
    intent='output',
    doc=''
)


class Mocompbaseline(Component):


    parameter_list = (
                      HEIGHT,
                      ELLIPSOID_ECCENTRICITY_SQUARED,
                      PEG_LATITUDE,
                      PEG_LONGITUDE,
                      PLANET_LOCAL_RADIUS,
                      MOCOMP_POSITION_INDEX1,
                      ELLIPSOID_MAJOR_SEMIAXIS,
                      MOCOMP_POSITION_INDEX2,
                      POSITION1,
                      POSITION2,
                      MOCOMP_POSITION1,
                      MOCOMP_POSITION2,
                      PEG_HEADING,
                      SCH,
                      SC,
                      BASE2,
                      MIDPOINT1,
                      MIDPOINT2,
                      MIDPOINT,
                      BASE1,
                      MOCOMP_BASELINE
                     )


    logging_name = 'isce.stdproc.orbit.mocompbaseline'
    family = 'mocompbaseline'

    def __init__(self,family='',name=''):
        super(Mocompbaseline, self).__init__(family if family else  self.__class__.family, name=name)
        self.dim1_midpoint = None
        self.dim2_midpoint = None
        self.dim1_midpoint1 = None
        self.dim2_midpoint1 = None
        self.dim1_midpoint2 = None
        self.dim2_midpoint2 = None
        self.dim1_base1 = None
        self.dim2_base1 = None
        self.dim1_base2 = None
        self.dim2_base2 = None
        self.dim1_sch = None
        self.dim2_sch = None
        self.dim1_sc = None
        self.dim2_sc = None
        # Planet information
        # Peg information
        # Orbit2SCH information
        self.dim1_position1 = None
        self.dim2_position1 = None
        self.dim1_position2 = None
        self.dim2_position2 = None
        # FormSLC information
        self.dim1_mocompPosition1 = None
        self.dim1_mocompPositionIndex1 = None
        self.dim1_mocompPosition2 = None
        self.dim1_mocompPositionIndex2 = None
        # Output
        self.dim1_baselineArray = None
        self.dim2_baselineArray = None
#        self.createPorts()

        self.initOptionalAndMandatoryLists()
        return None

    def createPorts(self):

        referenceOrbitPort = Port(name='referenceOrbit', method=self.addReferenceOrbit)
        secondaryOrbitPort = Port(name='secondaryOrbit', method=self.addSecondaryOrbit)
        pegPort = Port(name='peg', method=self.addPeg)
        ellipsoidPort = Port(name='ellipsoid', method=self.addEllipsoid)

        self._inputPorts.add(referenceOrbitPort)
        self._inputPorts.add(secondaryOrbitPort)
        self._inputPorts.add(pegPort)
        self._inputPorts.add(ellipsoidPort)
        return None


    def mocompbaseline(self):
        for port in self.inputPorts:
            port()

        self.prepareArraySizes()
        self.allocateArrays()
        self.setState()
        mocompbaseline.mocompbaseline_Py()
        self.getState()
        self.deallocateArrays()

    def prepareArraySizes(self):
        self.dim1_baselineArray = len(self.mocompPosition1)
        self.dim2_baselineArray = DIM
        self.dim1_base1 = len(self.mocompPosition1)
        self.dim2_base1 = DIM
        self.dim1_base2 = len(self.mocompPosition1)
        self.dim2_base2 = DIM
        self.dim1_sch = len(self.mocompPosition1)
        self.dim2_sch = DIM
        self.dim1_sc = len(self.mocompPosition1)
        self.dim2_sc = DIM
        self.dim1_midpoint = len(self.mocompPosition1)
        self.dim2_midpoint = DIM
        self.dim1_midpoint1 = len(self.mocompPosition1)
        self.dim2_midpoint1 = DIM
        self.dim1_midpoint2 = len(self.mocompPosition1)
        self.dim2_midpoint2 = DIM

    def setState(self):
        mocompbaseline.setStdWriter_Py(int(self.stdWriter))
        mocompbaseline.setSchPosition1_Py(self.position1,
                                          self.dim1_position1,
                                          self.dim2_position1)
        mocompbaseline.setSchPosition2_Py(self.position2,
                                          self.dim1_position2,
                                          self.dim2_position2)
        mocompbaseline.setMocompPosition1_Py(self.mocompPosition1,
                                             self.dim1_mocompPosition1)
        mocompbaseline.setMocompPositionIndex1_Py(
            self.mocompPositionIndex1,
            self.dim1_mocompPositionIndex1)
        mocompbaseline.setMocompPosition2_Py(self.mocompPosition2,
                                             self.dim1_mocompPosition2)
        mocompbaseline.setMocompPositionIndex2_Py(
            self.mocompPositionIndex2,
            self.dim1_mocompPositionIndex2)
        mocompbaseline.setEllipsoidMajorSemiAxis_Py(
            float(self.ellipsoidMajorSemiAxis)
            )
        mocompbaseline.setEllipsoidEccentricitySquared_Py(
            float(self.ellipsoidEccentricitySquared)
            )
        mocompbaseline.setPlanetLocalRadius_Py(float(self.planetLocalRadius))
        mocompbaseline.setPegLatitude_Py(float(self.pegLatitude))
        mocompbaseline.setPegLongitude_Py(float(self.pegLongitude))
        mocompbaseline.setPegHeading_Py(float(self.pegHeading))
        mocompbaseline.setHeight_Py(float(self.height))

    def setSchPosition1(self, var):
        self.position1 = var

    def setSchPosition2(self, var):
        self.position2 = var

    def setHeight(self, var):
        self.height = var
    def setMocompPosition1(self, var):
        self.mocompPosition1 = var

    def setMocompPositionIndex1(self, var):
        self.mocompPositionIndex1 = var

    def setMocompPosition2(self, var):
        self.mocompPosition2 = var

    def setMocompPositionIndex2(self, var):
        self.mocompPositionIndex2 = var

    def setEllipsoidMajorSemiAxis(self, var):
        self.ellipsoidMajorSemiAxis = float(var)

    def setEllipsoidEccentricitySquared(self, var):
        self.ellipsoidEccentricitySquared = float(var)

    def setPegLatitude(self, var):
        self.pegLatitude = float(var)

    def setPegLongitude(self, var):
        self.pegLongitude = float(var)

    def setPegHeading(self, var):
        self.pegHeading = float(var)

    def getState(self):
        dim1 = mocompbaseline.get_dim1_s1_Py()
        if dim1 != self.dim1_baselineArray:
            self.logger.info("dim1_baselineArray changed to %d" % (dim1))
            self.dim1_baselineArray = dim1
            self.dim1_midpoint = dim1
            self.dim1_midpoint1 = dim1
            self.dim1_midpoint2 = dim1
            self.dim1_base1 = dim1
            self.dim1_base2 = dim1
            self.dim1_sch = dim1
            self.dim1_sc = dim1

        self.baselineArray = mocompbaseline.getBaseline_Py(
            self.dim1_baselineArray, self.dim2_baselineArray
            )
        self.midpoint = mocompbaseline.getMidpoint_Py(self.dim1_midpoint,
                                                      self.dim2_midpoint)
        self.midpoint1 = mocompbaseline.getMidpoint1_Py(self.dim1_midpoint1,
                                                        self.dim2_midpoint1)
        self.midpoint2 = mocompbaseline.getMidpoint2_Py(self.dim1_midpoint2,
                                                        self.dim2_midpoint2)
        self.base1 = mocompbaseline.getBaseline1_Py(self.dim1_base1,
                                                    self.dim2_base1)
        self.base2 = mocompbaseline.getBaseline2_Py(self.dim1_base2,
                                                    self.dim2_base2)
        self.sch = mocompbaseline.getSch_Py(self.dim1_sch, self.dim2_sch)
        self.sc = mocompbaseline.getSc_Py(self.dim1_sc, self.dim2_sc)

    def getBaseline(self):
        return self.baselineArray
    @property
    def baseline(self):
        return self.baselineArray

    def getMidpoint(self):
        return self.midpoint

    def getMidpoint1(self):
        return self.midpoint1

    def getMidpoint2(self):
        return self.midpoint2

    def getBaseline1(self):
        return self.base1

    def getBaseline2(self):
        return self.base2

    def getSchs(self):
        return self.position1, self.sch

    def getSc(self):
        return self.sc

    def allocateArrays(self):
        if self.dim1_position1 is None:
            self.dim1_position1 = len(self.position1)
            self.dim2_position1 = len(self.position1[0])

        if (not self.dim1_position1) or (not self.dim2_position1):
            print("Error. Trying to allocate zero size array")

            raise Exception

        mocompbaseline.allocate_sch1_Py(self.dim1_position1,
                                        self.dim2_position1)

        if self.dim1_position2 is None:
            self.dim1_position2 = len(self.position2)
            self.dim2_position2 = len(self.position2[0])

        if (not self.dim1_position2) or (not self.dim2_position2):
            print("Error. Trying to allocate zero size array")

            raise Exception

        mocompbaseline.allocate_sch2_Py(self.dim1_position2,
                                        self.dim2_position2)

        if self.dim1_mocompPosition1 is None:
            self.dim1_mocompPosition1 = len(self.mocompPosition1)

        if (not self.dim1_mocompPosition1):
            print("Error. Trying to allocate zero size array")

            raise Exception

        mocompbaseline.allocate_s1_Py(self.dim1_mocompPosition1)

        if self.dim1_mocompPositionIndex1 is None:
            self.dim1_mocompPositionIndex1 = len(self.mocompPositionIndex1)

        if (not self.dim1_mocompPositionIndex1):
            print("Error. Trying to allocate zero size array")

            raise Exception

        mocompbaseline.allocate_is1_Py(self.dim1_mocompPositionIndex1)

        if self.dim1_mocompPosition2 is None:
            self.dim1_mocompPosition2 = len(self.mocompPosition2)

        if not self.dim1_mocompPosition2:
            print("Error. Trying to allocate zero size array")

            raise Exception

        mocompbaseline.allocate_s2_Py(self.dim1_mocompPosition2)

        if self.dim1_mocompPositionIndex2 is None:
            self.dim1_mocompPositionIndex2 = len(self.mocompPositionIndex2)

        if not self.dim1_mocompPositionIndex2:
            print("Error. Trying to allocate zero size array")

            raise Exception

        mocompbaseline.allocate_is2_Py(self.dim1_mocompPositionIndex2)

        if self.dim1_baselineArray is None:
            self.dim1_baselineArray = len(self.baselineArray)
            self.dim2_baselineArray = len(self.baselineArray[0])

        if (not self.dim1_baselineArray) or (not self.dim2_baselineArray):
            print("Error. Trying to allocate zero size array")

            raise Exception

        mocompbaseline.allocate_baselineArray_Py(self.dim1_baselineArray,
                                                 self.dim2_baselineArray)

        if self.dim1_midpoint is None:
            self.dim1_midpoint = len(self.midpoint)
            self.dim2_midpoint = len(self.midpoint[0])

        if (not self.dim1_midpoint) or (not self.dim2_midpoint):
            print("Error. Trying to allocate zero size array")

            raise Exception

        mocompbaseline.allocate_midPointArray_Py(self.dim1_midpoint,
                                                 self.dim2_midpoint)

        if self.dim1_midpoint1 is None:
            self.dim1_midpoint1 = len(self.midpoint1)
            self.dim2_midpoint1 = len(self.midpoint1[0])

        if (not self.dim1_midpoint1) or (not self.dim2_midpoint1):
            print("Error. Trying to allocate zero size array")

            raise Exception

        mocompbaseline.allocate_midPointArray1_Py(self.dim1_midpoint1,
                                                  self.dim2_midpoint1)

        if self.dim1_midpoint2 is None:
            self.dim1_midpoint2 = len(self.midpoint2)
            self.dim2_midpoint2 = len(self.midpoint2[0])

        if (not self.dim1_midpoint2) or (not self.dim2_midpoint2):
            print("Error. Trying to allocate zero size array")

            raise Exception

        mocompbaseline.allocate_midPointArray2_Py(self.dim1_midpoint2,
                                                  self.dim2_midpoint2)

        if self.dim1_base1 is None:
            self.dim1_base1 = len(self.base1)
            self.dim2_base1 = len(self.base1[0])

        if (not self.dim1_base1) or (not self.dim2_base1):
            print("Error. Trying to allocate zero size array")

            raise Exception

        mocompbaseline.allocate_baselineArray1_Py(self.dim1_base1,
                                                  self.dim2_base1)

        if self.dim1_base2 is None:
            self.dim1_base2 = len(self.base2)
            self.dim2_base2 = len(self.base2[0])

        if (not self.dim1_base2) or (not self.dim2_base2):
            print("Error. Trying to allocate zero size array")

            raise Exception

        mocompbaseline.allocate_baselineArray2_Py(self.dim1_base2,
                                                  self.dim2_base2)

        if self.dim1_sch is None:
            self.dim1_sch = len(self.sch)
            self.dim2_sch = len(self.sch[0])

        if (not self.dim1_sch) or (not self.dim2_sch):
            print("Error. Trying to allocate zero size array")

            raise Exception

        mocompbaseline.allocate_schArray_Py(self.dim1_sch,
                                            self.dim2_sch)

        if self.dim1_sc is None:
            self.dim1_sc = len(self.sc)
            self.dim2_sc = len(self.sc[0])

        if (not self.dim1_sc) or (not self.dim2_sc):
            print("Error. Trying to allocate zero size array")

            raise Exception

        mocompbaseline.allocate_scArray_Py(self.dim1_sc, self.dim2_sc)


    def deallocateArrays(self):
        mocompbaseline.deallocate_sch1_Py()
        mocompbaseline.deallocate_sch2_Py()
        mocompbaseline.deallocate_s1_Py()
        mocompbaseline.deallocate_is1_Py()
        mocompbaseline.deallocate_s2_Py()
        mocompbaseline.deallocate_is2_Py()
        mocompbaseline.deallocate_baselineArray_Py()
        mocompbaseline.deallocate_midPointArray_Py()
        mocompbaseline.deallocate_midPointArray1_Py()
        mocompbaseline.deallocate_midPointArray2_Py()
        mocompbaseline.deallocate_baselineArray1_Py()
        mocompbaseline.deallocate_baselineArray2_Py()
        mocompbaseline.deallocate_schArray_Py()
        mocompbaseline.deallocate_scArray_Py()

    def addPeg(self):
        import math
        peg = self._inputPorts.getPort(name='peg').getObject()
        if peg:
            try:
                self.planetLocalRadius = peg.getRadiusOfCurvature()
                self.pegLatitude = math.radians(peg.getLatitude())
                self.pegLongitude = math.radians(peg.getLongitude())
                self.pegHeading = math.radians(peg.getHeading())
            except AttributeError:
                self.logger.error("Object %s requires getLatitude(), getLongitude() and getHeading() methods" % (peg.__class__))

    def addEllipsoid(self):
        ellipsoid = self._inputPorts.getPort(name='ellipsoid').getObject()
        if(ellipsoid):
            try:
                self.ellipsoidEccentricitySquared = ellipsoid.get_e2()
                self.ellipsoidMajorSemiAxis = ellipsoid.get_a()
            except AttributeError:
                self.logger.error("Object %s requires get_e2() and get_a() methods" % (ellipsoid.__class__))

    def addReferenceOrbit(self):
        orbit = self._inputPorts.getPort(name='referenceOrbit').getObject()
        if (orbit):
            try:
                (time,position,velocity,offset) = orbit._unpackOrbit()
                self.time = time
                self.position1 = position
            except AttributeError:
                self.logger.error("Object %s requires an _unpackOrbit() method" % (orbit.__class__))
                raise AttributeError

    def addSecondaryOrbit(self):
        orbit = self._inputPorts.getPort(name='secondaryOrbit').getObject()
        if (orbit):
            try:
                (time,position,velocity,offset) = orbit._unpackOrbit()
                self.time = time
                self.position2 = position
            except AttributeError:
                self.logger.error("Object %s requires an _unpackOrbit() method" % (orbit.__class__))
                raise AttributeError



    pass
