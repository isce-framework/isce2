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
# Author: Walter Szeliga
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




from iscesys.Component.Component import Component
from . import snaphu

ALTITUDE = Component.Parameter(
    'altitude',
    public_name='ALTITUDE',
    default=None,
    type=float,
    mandatory=True,
    intent='input',
    doc='Altitude'
)


AZIMUTH_LOOKS = Component.Parameter(
    'azimuthLooks',
    public_name='AZIMUTH_LOOKS',
    default=1,
    type=int,
    mandatory=True,
    intent='input',
    doc='Number of looks in the azimuth direction'
)


CORR_FILE = Component.Parameter(
    'corrfile',
    public_name='CORR_FILE',
    default=None,
    type=str,
    mandatory=True,
    intent='input',
    doc='Correlation file name'
)


CORR_LOOKS = Component.Parameter(
    'corrLooks',
    public_name='CORR_LOOKS',
    default=None,
    type=int,
    mandatory=True,
    intent='input',
    doc='Correlation looks'
)


COR_FILE_FORMAT = Component.Parameter(
    'corFileFormat',
    public_name='COR_FILE_FORMAT',
    default='ALT_LINE_DATA',
    type=str,
    mandatory=False,
    intent='input',
    doc='Correlation file format'
)


COSTMODE = Component.Parameter(
    'costMode',
    public_name='COSTMODE',
    default='DEFO',
    type=str,
    mandatory=True,
    intent='input',
    doc='Cost function mode. Options are "TOPO","DEFO","SMOOTH".'
)


DEFORMATION_MAX_CYCLES = Component.Parameter(
    'defoMaxCycles',
    public_name='DEFORMATION_MAX_CYCLES',
    default=1.2,
    type=float,
    mandatory=True,
    intent='input',
    doc='Deformation max cycles'
)


DUMP_CONNECTED_COMPONENTS = Component.Parameter(
    'dumpConnectedComponents',
    public_name='DUMP_CONNECTED_COMPONENTS',
    default=True,
    type=bool,
    mandatory=False,
    intent='input',
    doc='Dump the connected component to a file with extension .conncomp'
)


EARTHRADIUS = Component.Parameter(
    'earthRadius',
    public_name='EARTHRADIUS',
    default=0,
    type=float,
    mandatory=True,
    intent='input',
    doc='Earth radius'
)


INIT_METHOD = Component.Parameter(
    'initMethod',
    public_name='INIT_METHOD',
    default='MST',
    type=str,
    mandatory=False,
    intent='input',
    doc='Init method. Options are "MST" or "MCF"'
)


INIT_ONLY = Component.Parameter(
    'initOnly',
    public_name='INIT_ONLY',
    default=False,
    type=bool,
    mandatory=False,
    intent='input',
    doc='Is this is set along with the DUMP_CONNECTED_COMPONENTS flag, then only the' +\
        'connected components are computed and dumped into a file with extension .conncomp'
)


INPUT = Component.Parameter(
    'input',
    public_name='INPUT',
    default=None,
    type=str,
    mandatory=True,
    intent='input',
    doc='Input file name'
)


INT_FILE_FORMAT = Component.Parameter(
    'intFileFormat',
    public_name='INT_FILE_FORMAT',
    default='COMPLEX_DATA',
    type=str,
    mandatory=False,
    intent='input',
    doc='Interferogram file format'
)


MAX_COMPONENTS = Component.Parameter(
    'maxComponents',
    public_name='MAX_COMPONENTS',
    default=32,
    type=int,
    mandatory=False,
    intent='input',
    doc='Max number of components'
)


OUTPUT = Component.Parameter(
    'output',
    public_name='OUTPUT',
    default=None,
    type=str,
    mandatory=True,
    intent='input',
    doc='Output file name'
)


RANGE_LOOKS = Component.Parameter(
    'rangeLooks',
    public_name='RANGE_LOOKS',
    default=1,
    type=int,
    mandatory=True,
    intent='input',
    doc='Number of looks in the range direction'
)


UNW_FILE_FORMAT = Component.Parameter(
    'unwFileFormat',
    public_name='UNW_FILE_FORMAT',
    default='ALT_LINE_DATA',
    type=str,
    mandatory=False,
    intent='input',
    doc='Unwrap file format'
)


WAVELENGTH = Component.Parameter(
    'wavelength',
    public_name='WAVELENGTH',
    default=None,
    type=float,
    mandatory=True,
    intent='input',
    doc='Wave length'
)


WIDTH = Component.Parameter(
    'width',
    public_name='WIDTH',
    default=None,
    type=int,
    mandatory=True,
    intent='input',
    doc='Image width'
)

class Snaphu(Component):

    parameter_list = (
                      ALTITUDE,
                      INPUT,
                      DUMP_CONNECTED_COMPONENTS,
                      WIDTH,
                      EARTHRADIUS,
                      INIT_ONLY,
                      CORR_LOOKS,
                      COR_FILE_FORMAT,
                      CORR_FILE,
                      WAVELENGTH,
                      MAX_COMPONENTS,
                      RANGE_LOOKS,
                      DEFORMATION_MAX_CYCLES,
                      UNW_FILE_FORMAT,
                      OUTPUT,
                      AZIMUTH_LOOKS,
                      INIT_METHOD,
                      COSTMODE,
                      INT_FILE_FORMAT
                     )

    """The Snaphu cost unwrapper"""

    fileFormats = { 'COMPLEX_DATA'  : 1,
                    'FLOAT_DATA'    : 2,
                    'ALT_LINE_DATA' : 3,
                    'ALT_SAMPLE_DATA' : 4}
    
    logging_name = "contrib.Snaphu.Snaphu"


    family = 'snaphu'

    def __init__(self,family='',name=''):
        super(Snaphu, self).__init__(family if family else  self.__class__.family, name=name)
        self.minConnectedComponentFrac = 0.01
        self.connectedComponentCostThreshold = 300
        self.magnitude = None 
        

    def setCorrfile(self, corrfile):
        """Set the correlation filename for unwrapping"""
        self.corrfile = corrfile

    def setDefoMaxCycles(self, ncycles):
        """Set the maximum phase discontinuity expected."""
        self.defoMaxCycles = ncycles

    def setCorrLooks(self, looks):
        """Set the number of looks used for computing correlation"""
        self.corrLooks = looks

    def setInput(self,input):
        """Set the input filename for unwrapping"""
        self.input = input
        
    def setOutput(self,output):
        """Set the output filename for unwrapping"""
        self.output = output
        
    def setWidth(self,width):
        """Set the image width"""
        self.width = width
        
    def setWavelength(self,wavelength):
        """Set the radar wavelength"""
        self.wavelength = wavelength

    def setRangeLooks(self, looks):
        self.rangeLooks = looks

    def setAzimuthLooks(self, looks):
        self.azimuthLooks = looks
   
    def setIntFileFormat(self, instr):
        self.intFileFormat = str(instr)

    def setCorFileFormat(self, instr):
        self.corFileFormat = str(instr)

    def setUnwFileFormat(self, instr):
        self.unwFileFormat = str(instr)

    def setCostMode(self,costMode):
        #moved the selection into prepare otherwise using configurable to
        #init  would not work
        self.costMode = costMode    

    def setInitOnly(self, logic):
        self.initOnly = logic

    def dumpConnectedComponents(self, logic):
        self.dumpConnectedComponents = logic
        
    def setAltitude(self,altitude):
        """Set the satellite altitude"""
        self.altitude = altitude
        
    def setEarthRadius(self,earthRadius):
        """Set the local Earth radius"""
        self.earthRadius = earthRadius

    def setInitMethod(self, method):
        """Set the initialization method."""
        #moved the selection into prepare otherwise using configurable to
        #init  would not work
        self.initMethod = method
       

    def setMaxComponents(self, num):
        """Set the maximum number of connected components."""
        self.maxComponents = num
    
    def prepare(self):
        """Perform some initialization of defaults"""
        
        snaphu.setDefaults_Py()
        snaphu.setInitOnly_Py(int(self.initOnly))
        snaphu.setInput_Py(self.input)
        snaphu.setOutput_Py(self.output)
        if self.magnitude is not None:
            snaphu.setMagnitude_Py(self.magnitude)
        snaphu.setWavelength_Py(self.wavelength)
        
        if not self.costMode in ['TOPO','DEFO','SMOOTH']:
            self.logger.error('Invalid cost mode %s' % (self.costMode))
        #must be one of the 3 above
        snaphu.setCostMode_Py(1 if self.costMode == 'TOPO' else
                             (2 if self.costMode == 'DEFO' else 3))
        snaphu.setAltitude_Py(self.altitude)
        snaphu.setEarthRadius_Py(self.earthRadius)       
        if self.corrfile is not None:
            snaphu.setCorrfile_Py(self.corrfile)

        if self.corrLooks is not None:
            snaphu.setCorrLooks_Py(self.corrLooks)

        if self.defoMaxCycles is not None:
            snaphu.setDefoMaxCycles_Py(self.defoMaxCycles)

        if not self.initMethod in ['MST','MCF']:
            self.logger.error('Invalid init method %s' % (self.initMethod))
        snaphu.setInitMethod_Py(1 if self.initMethod == 'MST' else 2)
                               
        snaphu.setMaxComponents_Py(self.maxComponents)
        snaphu.setRangeLooks_Py(int(self.rangeLooks))
        snaphu.setAzimuthLooks_Py(int(self.azimuthLooks))
        snaphu.setMinConnectedComponentFraction_Py(int(self.minConnectedComponentFrac))
        snaphu.setConnectedComponentThreshold_Py(int(self.connectedComponentCostThreshold))
        snaphu.setIntFileFormat_Py( int(self.fileFormats[self.intFileFormat]))
        snaphu.setCorFileFormat_Py( int(self.fileFormats[self.corFileFormat]))
        snaphu.setUnwFileFormat_Py( int(self.fileFormats[self.unwFileFormat]))
    

    def unwrap(self):
        """Unwrap the interferogram"""       

        ###Connected components can be dumped out in non-initonly mode
        if not self.initOnly and self.dumpConnectedComponents:
            snaphu.setConnectedComponents_Py(self.output+'.conncomp')
#            snaphu.setRegrowComponents_Py(int(True))

        snaphu.snaphu_Py(self.width)
        self._unwrappingCompleted = True

        ##Second pass if initOnly mode was used.
        if self.initOnly and self.dumpConnectedComponents:
            self.growConnectedComponentsOnly()

    def growConnectedComponentsOnly(self,infile=None,outfile=None):
        '''
        Grows the connected components using an unwrapped file.
        '''
        print('Growing connected components on second pass')
        if infile is None:
            inputFile = self.output
        else:
            inputFile = infile

        if outfile is None:
            outputFile = inputFile + '.conncomp'
        else:
            outputFile = outfile

        self.prepare()
        snaphu.setInitOnly_Py(int(False))
        snaphu.setInput_Py(inputFile)
        snaphu.setConnectedComponents_Py(outputFile)
        snaphu.setRegrowComponents_Py(int(True))
        snaphu.setUnwrappedInput_Py(int(True))
        snaphu.snaphu_Py(self.width)
          
