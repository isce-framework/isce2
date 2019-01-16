#!/usr/bin/env python3

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
# Author: Giangi Sacco
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





from __future__ import print_function
import sys
import os
import math
import logging
from iscesys.Component.Component import Component,Port
from iscesys.Compatibility import Compatibility
Compatibility.checkPythonVersion()
from stdproc.orbit.fdmocomp import fdmocomp

class FdMocomp(Component):

    logging_name = 'isce.stdproc.orbit.FdMocomp'
    def __init__(self):
        super(FdMocomp, self).__init__()
        self.startingRange = None
        self.prf = None
        self.radarWavelength = None
        self.width = None
        self.heigth = None
        self.rangeSamplingRate = None
        self.planetRadiusOfCurvature = None
        self.dopplerCoefficients = []
        self.dim1_dopplerCoefficients = None
        self.schVelocity = []
        self.dim1_schVelocity = None
        self.dim2_schVelocity = None
        self.fd = None
        self.lookSide = -1    #Right side by default
        self.dictionaryOfVariables = { \
                                      'STARTING_RANGE' : ['startingRange', 'float','mandatory'], \
                                      'PRF' : ['prf', 'float','mandatory'], \
                                      'RADAR_WAVELENGTH' : ['radarWavelength', 'float','mandatory'], \
                                      'WIDTH' : ['width', 'int','mandatory'], \
                                      'HEIGTH' : ['heigth', 'int','mandatory'], \
                                      'PLATFORM_HEIGTH' : ['platformHeigth', 'int','mandatory'], \
                                      'RANGE_SAMPLING_RATE' : ['rangeSamplingRate', 'float','mandatory'], \
                                      'RADIUS_OF_CURVATURE' : ['planetRadiusOfCurvature', 'float','mandatory'], \
                                      'DOPPLER_COEFFICIENTS' : ['dopplerCoefficients', 'float','mandatory'], \
                                      'SCH_VELOCITY' : ['schVelocity', '','mandatory'] \
                                      }
        self.dictionaryOfOutputVariables = { \
                                            'CORRECTED_DOPPLER' : 'correctedDoppler' \
                                            }
        self.descriptionOfVariables = {}
        self.mandatoryVariables = []
        self.optionalVariables = []
        self.initOptionalAndMandatoryLists()
        return
    def fdmocomp(self):
        self.activateInputPorts()
        self.allocateArrays()
        self.setState()
        fdmocomp.fdmocomp_Py()
        self.getState()
        self.deallocateArrays()

        return





    def setState(self):
        fdmocomp.setStartingRange_Py(float(self.startingRange))
        fdmocomp.setPRF_Py(float(self.prf))
        fdmocomp.setRadarWavelength_Py(float(self.radarWavelength))
        fdmocomp.setWidth_Py(int(self.width))
        fdmocomp.setHeigth_Py(int(self.heigth))
        fdmocomp.setPlatformHeigth_Py(int(self.platformHeigth))
        fdmocomp.setRangeSamplingRate_Py(float(self.rangeSamplingRate))
        fdmocomp.setRadiusOfCurvature_Py(float(self.planetRadiusOfCurvature))
        fdmocomp.setDopplerCoefficients_Py(self.dopplerCoefficients, self.dim1_dopplerCoefficients)
        fdmocomp.setSchVelocity_Py(self.schVelocity, self.dim1_schVelocity, self.dim2_schVelocity)
        fdmocomp.setLookSide_Py(self.lookSide)

        return





    def setStartingRange(self,var):
        self.startingRange = float(var)
        return

    def setPRF(self,var):
        self.prf = float(var)
        return

    def setRadarWavelength(self,var):
        self.radarWavelength = float(var)
        return

    def setWidth(self,var):
        self.width = int(var)
        return

    def setHeigth(self,var):
        self.heigth = int(var)
        return

    def setSatelliteHeight(self,var):
        self.platformHeigth = int(var)
        return

    def setRangeSamplingRate(self,var):
        self.rangeSamplingRate = float(var)
        return

    def setRadiusOfCurvature(self,var):
        self.planetRadiusOfCurvature = float(var)
        return

    def setDopplerCoefficients(self,var):
        self.dopplerCoefficients = var
        return

    def setSchVelocity(self,var):
        self.schVelocity = var
        return

    def setLookSide(self,var):
        self.lookSide = int(var)
        return


    def createPorts(self):
        pegPort = Port(name='peg', method=self.addPeg)
        orbitPort = Port(name='orbit', method=self.addOrbit)
        framePort = Port(name='frame',method=self.addFrame)

        self._inputPorts.add(pegPort)
        self._inputPorts.add(orbitPort)
        self._inputPorts.add(framePort)
        return None

    def addPeg(self):
        peg = self.inputPorts['peg']
        if peg:
            try:
                self.planetRadiusOfCurvature = peg.getRadiusOfCurvature()
                self.logger.debug("Rcurv %s" %(self.planetRadiusOfCurvature))
            except AttributeError:
                self.logger.error(
                    "Object %s require a getRadiusOfCurvature method" %
                    (peg.__class__)
                    )
                raise AttributeError

    def addFrame(self):
        frame = self.inputPorts['frame']
        if frame:
            try:
                self.startingRange = frame.getStartingRange()
                self.radarWavelength = frame.getInstrument().getRadarWavelength()
                self.rangeSamplingRate = frame.getInstrument().getRangeSamplingRate()
                self.prf = frame.getInstrument().getPulseRepetitionFrequency()
            except AttributeError as err:
                self.logger.error(err)
                raise AttributeError
            pass
        return None

    def addOrbit(self):
        orbit = self.inputPorts['orbit']
        if  orbit:
            try:
                time, position, self.schVelocity, offset = orbit.to_tuple()
                self.heigth = len(self.schVelocity)
            except (TypeError, ValueError) as err:
                self.logger.error("orbit could not be unpacked")
                raise err
            pass
        return None





    def getState(self):
        self.correctedDoppler = fdmocomp.getCorrectedDoppler_Py()

        return





    def getDopplerCentroid(self):
        return self.correctedDoppler
    @property
    def dopplerCentroid(self):
        return self.correctedDoppler






    def allocateArrays(self):
        if (self.dim1_dopplerCoefficients == None):
            self.dim1_dopplerCoefficients = len(self.dopplerCoefficients)

        if (not self.dim1_dopplerCoefficients):
            print("Error. Trying to allocate zero size array")

            raise Exception

        fdmocomp.allocate_fdArray_Py(self.dim1_dopplerCoefficients)

        if (self.dim1_schVelocity == None):
            self.dim1_schVelocity = len(self.schVelocity)
            self.dim2_schVelocity = len(self.schVelocity[0])

        if (not self.dim1_schVelocity) or (not self.dim2_schVelocity):
            print("Error. Trying to allocate zero size array")

            raise Exception

        fdmocomp.allocate_vsch_Py(self.dim1_schVelocity, self.dim2_schVelocity)


        return





    def deallocateArrays(self):
        fdmocomp.deallocate_fdArray_Py()
        fdmocomp.deallocate_vsch_Py()

        return










#end class




if __name__ == "__main__":
    sys.exit(main())
