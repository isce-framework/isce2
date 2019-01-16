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



from __future__ import print_function
import sys
import os
import math
from iscesys.Component.Component import Component
from isceobj import Constants as CN
from iscesys.Compatibility import Compatibility
from mroipac.getPegInfo import get_peg_info

class Get_peg_info(Component):

    def get_peg_info(self):
        self.dim1_posVect = len(self.posVect)
        self.dim2_posVect = len(self.posVect[0])
        self.dim1_velVect = len(self.posVect)
        self.dim2_velVect = len(self.posVect[0])
        self.dim1_intPosition = self.numLinesInt
        self.dim2_intPosition = 3
        self.dim1_intVelocity = self.numLinesInt
        self.dim2_intVelocity = 3
        self.numObs = self.dim1_posVect
        self.dim1_pegLat =  self.numObs
        self.dim1_pegLon =  self.numObs
        self.dim1_pegHgt =  self.numObs
        self.dim1_pegHead =  self.numObs
        self.dim1_verticalFit = 3 
        self.dim1_horizontalFit =  3
        self.dim1_verticalVelocityFit = 2 
        self.dim1_horizontalVelocityFit = 2
        self.dim1_crossTrackVelocityFit = 2 
        self.dim1_alongTrackVelocityFit = 2
        self.dim1_transVect =  3
        self.dim1_transfMat =  3
        self.dim2_transfMat =  3
        self.dim1_pegVelocity = 3 
        self.dim1_platVel = 3 
        self.dim1_platAcc = 3 
        #set the dimension of the other arrays
        
        self.allocateArrays()
        self.setState()
        get_peg_info.get_peg_info_Py()
        self.getState()
        self.deallocateArrays()

        return





    def setState(self):
        get_peg_info.setNumObservations_Py(int(self.numObs))
        get_peg_info.setStartLineSlc_Py(int(self.startLineSlc))
        get_peg_info.setNumLinesInt_Py(int(self.numLinesInt))
        get_peg_info.setNumLinesSlc_Py(int(self.numLinesSlc))
        get_peg_info.setNumAzimuthLooksInt_Py(int(self.numAzimuthLooksInt))
        get_peg_info.setPrfSlc_Py(float(self.prfSlc))
        get_peg_info.setTimeSlc_Py(float(self.timeSlc))
        get_peg_info.setTime_Py(self.time, self.dim1_time)
        get_peg_info.setPositionVector_Py(self.posVect, self.dim1_posVect, self.dim2_posVect)
        get_peg_info.setVelocityVector_Py(self.velVect, self.dim1_velVect, self.dim2_velVect)
        #not supported at the moment
        #get_peg_info.setAccelerationVector_Py(self.accVect, self.dim1_accVect, self.dim2_accVect)
        get_peg_info.setPlanetGM_Py(float(self.planetGM))
        get_peg_info.setPlanetSpinRate_Py(float(self.planetSpinRate))

        return





    def setNumObservations(self,var):
        self.numObs = int(var)
        return

    def setStartLineSlc(self,var):
        self.startLineSlc = int(var)
        return

    def setNumLinesInt(self,var):
        self.numLinesInt = int(var)
        return

    def setNumLinesSlc(self,var):
        self.numLinesSlc = int(var)
        return

    def setNumAzimuthLooksInt(self,var):
        self.numAzimuthLooksInt = int(var)
        return

    def setPrfSlc(self,var):
        self.prfSlc = float(var)
        return

    def setTimeSlc(self,var):
        self.timeSlc = float(var)
        return

    def setTime(self,var):
        self.time = var
        return

    def setPositionVector(self,var):
        self.posVect = var
        return

    def setVelocityVector(self,var):
        self.velVect = var
        return

    def setAccelerationVector(self,var):
        self.accVect = var
        return

    def setPlanetGM(self,var):
        self.planetGM = float(var)
        return

    def setPlanetSpinRate(self,var):
        self.planetSpinRate = float(var)
        return





    def getState(self):
        self.pegLat = get_peg_info.getPegLat_Py()
        self.pegLon = get_peg_info.getPegLon_Py()
        self.pegHgt = get_peg_info.getPegHeight_Py()
        self.pegHead = get_peg_info.getPegHeading_Py()
        self.verticalFit = get_peg_info.getVerticalFit_Py(self.dim1_verticalFit)
        self.horizontalFit = get_peg_info.getHorizontalFit_Py(self.dim1_horizontalFit)
        self.verticalVelocityFit = get_peg_info.getVerticalVelocityFit_Py(self.dim1_verticalVelocityFit)
        self.crossTrackVelocityFit = get_peg_info.getCrossTrackVelocityFit_Py(self.dim1_crossTrackVelocityFit)
        self.alongTrackVelocityFit = get_peg_info.getAlongTrackVelocityFit_Py(self.dim1_alongTrackVelocityFit)
        self.pegRadius = get_peg_info.getPegRadius_Py()
        self.grndSpace = get_peg_info.getGroundSpacing_Py()
        self.transVect = get_peg_info.getTranslationVector_Py(self.dim1_transVect)
        self.transfMat = get_peg_info.getTransformationMatrix_Py(self.dim1_transfMat, self.dim2_transfMat)
        self.intPosition = get_peg_info.getIntPosition_Py(self.dim1_intPosition, self.dim2_intPosition)
        self.intVelocity = get_peg_info.getIntVelocity_Py(self.dim1_intVelocity, self.dim2_intVelocity)
        self.pegVelocity = get_peg_info.getPegVelocity_Py(self.dim1_pegVelocity)
        self.platVel = get_peg_info.getPlatformSCHVelocity_Py(self.dim1_platVel)
        self.platAcc = get_peg_info.getPlatformSCHAcceleration_Py(self.dim1_platAcc)
        self.timeFirstLine = get_peg_info.getTimeFirstScene_Py()

        return




    def getPegLat(self):
        return self.pegLat

    def getPegLon(self):
        return self.pegLon

    def getPegHeight(self):
        return self.pegHgt

    def getPegHeading(self):
        return self.pegHead

    def getVerticalFit(self):
        return self.verticalFit

    def getHorizontalFit(self):
        return self.horizontalFit

    def getVerticalVelocityFit(self):
        return self.verticalVelocityFit

    def getCrossTrackVelocityFit(self):
        return self.crossTrackVelocityFit

    def getAlongTrackVelocityFit(self):
        return self.alongTrackVelocityFit

    def getPegRadius(self):
        return self.pegRadius

    def getGroundSpacing(self):
        return self.grndSpace

    def getTranslationVector(self):
        return self.transVect

    def getTransformationMatrix(self):
        return self.transfMat

    def getIntPosition(self):
        return self.intPosition

    def getIntVelocity(self):
        return self.intVelocity

    def getPegVelocity(self):
        return self.pegVelocity

    def getPlatformSCHVelocity(self):
        return self.platVel

    def getPlatformSCHAcceleration(self):
        return self.platAcc

    def getTimeFirstScene(self):
        return self.timeFirstLine



    def allocateArrays(self):
        if (self.dim1_time == None):
            self.dim1_time = len(self.time)

        if (not self.dim1_time):
            print("Error. Trying to allocate zero size array")

            raise Exception

        get_peg_info.allocate_r_time_Py(self.dim1_time)

        if (self.dim1_posVect == None):
            self.dim1_posVect = len(self.posVect)
            self.dim2_posVect = len(self.posVect[0])

        if (not self.dim1_posVect) or (not self.dim2_posVect):
            print("Error. Trying to allocate zero size array")

            raise Exception

        get_peg_info.allocate_r_xyz1_Py(self.dim1_posVect, self.dim2_posVect)

        if (self.dim1_velVect == None):
            self.dim1_velVect = len(self.velVect)
            self.dim2_velVect = len(self.velVect[0])

        if (not self.dim1_velVect) or (not self.dim2_velVect):
            print("Error. Trying to allocate zero size array")

            raise Exception

        get_peg_info.allocate_r_vxyz1_Py(self.dim1_velVect, self.dim2_velVect)

        #acceleration vector not supported at the moment
        '''
        if (self.dim1_accVect == None):
            self.dim1_accVect = len(self.accVect)
            self.dim2_accVect = len(self.accVect[0])

        if (not self.dim1_accVect) or (not self.dim2_accVect):
            print("Error. Trying to allocate zero size array")

            raise Exception

        get_peg_info.allocate_r_axyz1_Py(self.dim1_accVect, self.dim2_accVect)
        '''
        if (self.dim1_verticalFit == None):
            self.dim1_verticalFit = len(self.verticalFit)

        if (not self.dim1_verticalFit):
            print("Error. Trying to allocate zero size array")

            raise Exception

        get_peg_info.allocate_r_af_Py(self.dim1_verticalFit)

        if (self.dim1_horizontalFit == None):
            self.dim1_horizontalFit = len(self.horizontalFit)

        if (not self.dim1_horizontalFit):
            print("Error. Trying to allocate zero size array")

            raise Exception

        get_peg_info.allocate_r_cf_Py(self.dim1_horizontalFit)

        if (self.dim1_verticalVelocityFit == None):
            self.dim1_verticalVelocityFit = len(self.verticalVelocityFit)

        if (not self.dim1_verticalVelocityFit):
            print("Error. Trying to allocate zero size array")

            raise Exception

        get_peg_info.allocate_r_afdot_Py(self.dim1_verticalVelocityFit)

        if (self.dim1_crossTrackVelocityFit == None):
            self.dim1_crossTrackVelocityFit = len(self.crossTrackVelocityFit)

        if (not self.dim1_crossTrackVelocityFit):
            print("Error. Trying to allocate zero size array")

            raise Exception

        get_peg_info.allocate_r_cfdot_Py(self.dim1_crossTrackVelocityFit)

        if (self.dim1_alongTrackVelocityFit == None):
            self.dim1_alongTrackVelocityFit = len(self.alongTrackVelocityFit)

        if (not self.dim1_alongTrackVelocityFit):
            print("Error. Trying to allocate zero size array")

            raise Exception

        get_peg_info.allocate_r_sfdot_Py(self.dim1_alongTrackVelocityFit)

        if (self.dim1_transVect == None):
            self.dim1_transVect = len(self.transVect)

        if (not self.dim1_transVect):
            print("Error. Trying to allocate zero size array")

            raise Exception

        get_peg_info.allocate_r_transVect_Py(self.dim1_transVect)

        if (self.dim1_transfMat == None):
            self.dim1_transfMat = len(self.transfMat)
            self.dim2_transfMat = len(self.transfMat[0])

        if (not self.dim1_transfMat) or (not self.dim2_transfMat):
            print("Error. Trying to allocate zero size array")

            raise Exception

        get_peg_info.allocate_r_transfMat_Py(self.dim1_transfMat, self.dim2_transfMat)

        if (self.dim1_intPosition == None):
            self.dim1_intPosition = len(self.intPosition)
            self.dim2_intPosition = len(self.intPosition[0])

        if (not self.dim1_intPosition) or (not self.dim2_intPosition):
            print("Error. Trying to allocate zero size array")

            raise Exception

        get_peg_info.allocate_r_intPos_Py(self.dim1_intPosition, self.dim2_intPosition)

        if (self.dim1_intVelocity == None):
            self.dim1_intVelocity = len(self.intVelocity)
            self.dim2_intVelocity = len(self.intVelocity[0])

        if (not self.dim1_intVelocity) or (not self.dim2_intVelocity):
            print("Error. Trying to allocate zero size array")

            raise Exception

        get_peg_info.allocate_r_intVel_Py(self.dim1_intVelocity, self.dim2_intVelocity)


        if (self.dim1_pegVelocity == None):
            self.dim1_pegVelocity = len(self.pegVelocity)

        if (not self.dim1_pegVelocity):
            print("Error. Trying to allocate zero size array")

            raise Exception

        get_peg_info.allocate_r_vxyzpeg_Py(self.dim1_pegVelocity)

        if (self.dim1_platVel == None):
            self.dim1_platVel = len(self.platVel)

        if (not self.dim1_platVel):
            print("Error. Trying to allocate zero size array")

            raise Exception

        get_peg_info.allocate_r_platvel_Py(self.dim1_platVel)

        if (self.dim1_platAcc == None):
            self.dim1_platAcc = len(self.platAcc)

        if (not self.dim1_platAcc):
            print("Error. Trying to allocate zero size array")

            raise Exception

        get_peg_info.allocate_r_platacc_Py(self.dim1_platAcc)


        return





    def deallocateArrays(self):
        get_peg_info.deallocate_r_time_Py()
        get_peg_info.deallocate_r_xyz1_Py()
        get_peg_info.deallocate_r_vxyz1_Py()
        #acceleration vector not supported at the moment
        #get_peg_info.deallocate_r_axyz1_Py()
        get_peg_info.deallocate_r_af_Py()
        get_peg_info.deallocate_r_cf_Py()
        get_peg_info.deallocate_r_afdot_Py()
        get_peg_info.deallocate_r_cfdot_Py()
        get_peg_info.deallocate_r_sfdot_Py()
        get_peg_info.deallocate_r_transVect_Py()
        get_peg_info.deallocate_r_transfMat_Py()
        get_peg_info.deallocate_r_vxyzpeg_Py()
        get_peg_info.deallocate_r_intPos_Py()
        get_peg_info.deallocate_r_intVel_Py()
        get_peg_info.deallocate_r_platvel_Py()
        get_peg_info.deallocate_r_platacc_Py()

        return



    def __init__(self):
        
        Component.__init__(self)
        
        self.startLineSlc = 1
        self.planetGM = CN.EarthGM
        self.planetSpinRate = CN.EarthSpinRate
        
        self.numObs = None
        self.numLinesInt = None
        self.numLinesSlc = None
        self.numAzimuthLooksInt = None
        self.prfSlc = None
        self.timeSlc = None
        self.time = []
        self.dim1_time = None
        self.posVect = []
        self.dim1_posVect = None
        self.dim2_posVect = None
        self.velVect = []
        self.dim1_velVect = None
        self.dim2_velVect = None
        self.accVect = []
        self.dim1_accVect = None
        self.dim2_accVect = None
        self.pegLat = None
        self.pegLon = None
        self.pegHgt = None
        self.pegHead = None
        self.verticalFit = []
        self.dim1_verticalFit = None
        self.horizontalFit = []
        self.dim1_horizontalFit = None
        self.verticalVelocityFit = []
        self.dim1_verticalVelocityFit = None
        self.crossTrackVelocityFit = []
        self.dim1_crossTrackVelocityFit = None
        self.alongTrackVelocityFit = []
        self.dim1_alongTrackVelocityFit = None
        self.pegRadius = None
        self.grndSpace = None
        self.transVect = []
        self.dim1_transVect = None
        self.transfMat = []
        self.dim1_transfMat = None
        self.dim2_transfMat = None
        self.intPosition = []
        self.dim1_intPosition = None
        self.intVelocity = []
        self.dim1_intVelocity = None
        self.pegVelocity = []
        self.dim1_pegVelocity = None
        self.platVel = []
        self.dim1_platVel = None
        self.platAcc = []
        self.dim1_platAcc = None
        self.timeFirstLine = None
        self.dictionaryOfVariables = {'NUM_OBSERVATIONS' : ['self.numObs', 'int','optional'], \
                                      'START_LINE_SLC' : ['self.startLineSlc', 'int','optional'], \
                                      'NUM_LINES_INT' : ['self.numLinesInt', 'int','mandatory'], \
                                      'NUM_LINES_SLC' : ['self.numLinesSlc', 'int','mandatory'], \
                                      'NUM_AZIMUTH_LOOKS_INT' : ['self.numAzimuthLooksInt', 'int','mandatory'], \
                                      'PRF' : ['self.prfSlc', 'float','mandatory'], \
                                      'TIME_SLC' : ['self.timeSlc', 'float','mandatory'], \
                                      'TIME' : ['self.time', 'float','mandatory'], \
                                      'POSITION_VECTOR' : ['self.posVect', 'float','mandatory'], \
                                      'VELOCITY_VECTOR' : ['self.velVect', 'float','mandatory'], \
                                      'ACCELERATION_VECTOR' : ['self.accVect', 'float','optional'], \
                                      'PLANET_GM' : ['self.planetGM', 'float','optional'], \
                                      'PLANET_SPIN_RATE' : ['self.planetSpinRate', 'float','optional']}
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





#end class




if __name__ == "__main__":
    sys.exit(main())
