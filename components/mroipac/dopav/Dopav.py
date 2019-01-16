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




'''
This script creates avg doppler  from the dopplers contained in the raw images. 
'''


from __future__ import print_function
import sys
import os
import math
from iscesys.Compatibility import Compatibility
Compatibility.checkPythonVersion()
from iscesys.Component.Component import Component
from iscesys.Component.InitFromDictionary import InitFromDictionary
class Dopav(Component):
    
    def dopav(self,image1 = None,image2 = None):
        if not (image1 == None):
            trans = {'SQUINT':'SQUINT1','DOPPLER_CENTROID_COEFFICIENTS':'DOPPLER_CENTROID_COEFFICIENTS1','PRF':'PRF1'}
            init = InitFromDictionary(image1,trans)
            self.initComponent(init)
            
        if not (image2 == None):
            trans = {'SQUINT':'SQUINT2','DOPPLER_CENTROID_COEFFICIENTS':'DOPPLER_CENTROID_COEFFICIENTS2','PRF':'PRF2'}
            init = InitFromDictionary(image2,trans)
            self.initComponent(init)
        
        return self.avgDoppler()

    def avgDoppler(self):# average the doppler coefficients from the two raw files
    
        self.checkInitialization()
        prf = (self.prf1 + self.prf2)/2.0 
        if not (len(self.dopplerCentroidCoefficients1) == len(self.dopplerCentroidCoefficients2)):
            print("Error. The two doppler coefficient lists must have the same dimension.")
            raise Exception
        dop = []
        dop1 = self.dopplerCentroidCoefficients1
        dop2 = self.dopplerCentroidCoefficients2
        for i in range(len(self.dopplerCentroidCoefficients1)):
            dop.append((dop1[i]*self.prf1 + dop2[i]*self.prf2)/2.0)
        
        res = self.antennaLength/2.0
        squint = (self.squint1 + self.squint2)/2.0
        self.slAzimuthResolution = res/(1 - (res/self.velocity)*math.fabs(dop1[0]*self.prf1 - dop2[0]*self.prf2))
        dop1 = [0]*len(dop)#otherwise is going to update also self.dopplerCentroidCoefficients1 because of the assignment above. for list the reference is assigned.
        dop2 = [0]*len(dop)
        for i in range(len(dop)):
            
            dop1[i] = dop[i]/self.prf1
            dop2[i] = dop[i]/self.prf2
        dicRet = {}
        dicRet1 = {}
        
        dicRet['DOPPLER_CENTROID_COEFFICIENTS'] = dop1
        dicRet['SQUINT'] = squint
        dicRet['SL_AZIMUT_RESOL'] = self.slAzimuthResolution
        dicRet1['DOPPLER_CENTROID_COEFFICIENTS'] = dop2
        dicRet1['SQUINT'] = squint
        dicRet1['SL_AZIMUT_RESOL'] = self.slAzimuthResolution
        return (dicRet,dicRet1)

    
    
    def setDopplerCentroidCoefficients1(self,var):
        self.dopplerCentroidCoefficients1 = var
        return
    def setDopplerCentroidCoefficients2(self,var):
        self.dopplerCentroidCoefficients2 = var
        return
    def setPRF1(self,var):
        self.PRF1 = float(var)
        return
    def setPRF2(self,var):
        self.PRF2 = float(var)
        return
    def setSquint1(self,var):
        self.squint1 = float(var)
        return
    def setSquint2(self,var):
        self.squint2 = float(var)
        return
        
    def setAntennaLength(self,var):
        self.antennaLength = float(var)
           
    def setVelocity(self,var):
        self.velocity = float(var)
    
    def __init__(self):
        Component.__init__(self)

        # mandatory input variables
        self.dopplerCentroidCoefficients1 = []
        self.dopplerCentroidCoefficients2 = []
        self.prf1 = None
        self.prf2 = None
        self.squint1 = None
        self.squint2 = None
        self.antennaLength = None
        self.velocity = None


        
        self.slAzimuthResolution = None
        
        self.dictionaryOfVariables = {'DOPPLER_CENTROID_COEFFICIENTS1':['self.dopplerCentroidCoefficients1', 'float','mandatory'], \
        'DOPPLER_CENTROID_COEFFICIENTS2':['self.dopplerCentroidCoefficients2', 'float','mandatory'], \
        'PRF1':['self.prf1', 'float','mandatory'], \
        'PRF2':['self.prf2', 'float','mandatory'], \
        'SQUINT1':['self.squint1', 'float','mandatory'], \
        'SQUINT2':['self.squint2', 'float','mandatory'], \
        'ANTENNA_LENGTH':['self.antennaLength', 'float','mandatory'], \
        'VELOCITY':['self.velocity', 'float','mandatory']}
        
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
if __name__ == "__main__":
    sys.exit(main())


