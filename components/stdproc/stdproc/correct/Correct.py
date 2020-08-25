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
from isceobj import Constants as CN
from iscesys.Component.Component import Component,Port
from iscesys.Compatibility import Compatibility
import isceobj.Image as IF #load image factories
from stdproc.stdproc.correct import correct
from isceobj.Util.Polynomial import Polynomial
from isceobj.Util.Poly2D import Poly2D

IS_MOCOMP = Component.Parameter(
    'isMocomp',
    public_name='IS_MOCOMP',
    default=None,
    type=int,
    mandatory=False,
    intent='input',
    doc=''
)


MOCOMP_BASELINE = Component.Parameter(
    'mocompBaseline',
    public_name='MOCOMP_BASELINE',
    default=[],
    type=float,
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


ELLIPSOID_MAJOR_SEMIAXIS = Component.Parameter(
    'ellipsoidMajorSemiAxis',
    public_name='ELLIPSOID_MAJOR_SEMIAXIS',
    default=None,
    type=float,
    mandatory=False,
    intent='input',
    doc=''
)


S1SCH = Component.Parameter(
    's1sch',
    public_name='S1SCH',
    default=[],
    type=float,
    mandatory=True,
    intent='input',
    doc=''
)


RADAR_WAVELENGTH = Component.Parameter(
    'radarWavelength',
    public_name='RADAR_WAVELENGTH',
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


LENGTH = Component.Parameter(
    'length',
    public_name='LENGTH',
    default=None,
    type=int,
    mandatory=True,
    intent='input',
    doc=''
)


RANGE_FIRST_SAMPLE = Component.Parameter(
    'rangeFirstSample',
    public_name='RANGE_FIRST_SAMPLE',
    default=None,
    type=float,
    mandatory=True,
    intent='input',
    doc=''
)


SC = Component.Parameter(
    'sc',
    public_name='SC',
    default=[],
    type=float,
    mandatory=True,
    intent='input',
    doc=''
)


NUMBER_RANGE_LOOKS = Component.Parameter(
    'numberRangeLooks',
    public_name='NUMBER_RANGE_LOOKS',
    default=None,
    type=int,
    mandatory=True,
    intent='input',
    doc=''
)


NUMBER_AZIMUTH_LOOKS = Component.Parameter(
    'numberAzimuthLooks',
    public_name='NUMBER_AZIMUTH_LOOKS',
    default=None,
    type=int,
    mandatory=True,
    intent='input',
    doc=''
)


BODY_FIXED_VELOCITY = Component.Parameter(
    'bodyFixedVelocity',
    public_name='BODY_FIXED_VELOCITY',
    default=None,
    type=float,
    mandatory=True,
    intent='input',
    doc=''
)


SPACECRAFT_HEIGHT = Component.Parameter(
    'spacecraftHeight',
    public_name='SPACECRAFT_HEIGHT',
    default=None,
    type=float,
    mandatory=True,
    intent='input',
    doc=''
)


SLANT_RANGE_PIXEL_SPACING = Component.Parameter(
    'slantRangePixelSpacing',
    public_name='SLANT_RANGE_PIXEL_SPACING',
    default=None,
    type=float,
    mandatory=True,
    intent='input',
    doc=''
)


PRF = Component.Parameter(
    'prf',
    public_name='PRF',
    default=None,
    type=float,
    mandatory=True,
    intent='input',
    doc=''
)


MIDPOINT = Component.Parameter(
    'midpoint',
    public_name='MIDPOINT',
    default=[],
    type=float,
    mandatory=True,
    intent='input',
    doc=''
)


REFERENCE_ORBIT = Component.Parameter(
    'referenceOrbit',
    public_name='REFERENCE_ORBIT',
    default=[],
    type=float,
    mandatory=True,
    intent='input',
    doc=''
)


ELLIPSOID_ECCENTRICITY_SQUARED = Component.Parameter(
    'ellipsoidEccentricitySquared',
    public_name='ELLIPSOID_ECCENTRICITY_SQUARED',
    default=None,
    type=float,
    mandatory=False,
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


WIDTH = Component.Parameter(
    'width',
    public_name='WIDTH',
    default=None,
    type=int,
    mandatory=True,
    intent='input',
    doc=''
)


S2SCH = Component.Parameter(
    's2sch',
    public_name='S2SCH',
    default=[],
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


DOPPLER_CENTROID = Component.Parameter(
    'dopplerCentroidCoeffs',
    public_name='DOPPLER_CENTROID',
    default=0,
    type=float,
    mandatory=True,
    intent='input',
    doc=''
)


class Correct(Component):


    parameter_list = (
                      IS_MOCOMP,
                      MOCOMP_BASELINE,
                      PEG_HEADING,
                      ELLIPSOID_MAJOR_SEMIAXIS,
                      S1SCH,
                      RADAR_WAVELENGTH,
                      PLANET_LOCAL_RADIUS,
                      LENGTH,
                      RANGE_FIRST_SAMPLE,
                      SC,
                      NUMBER_RANGE_LOOKS,
                      NUMBER_AZIMUTH_LOOKS,
                      BODY_FIXED_VELOCITY,
                      SPACECRAFT_HEIGHT,
                      SLANT_RANGE_PIXEL_SPACING,
                      PRF,
                      MIDPOINT,
                      REFERENCE_ORBIT,
                      ELLIPSOID_ECCENTRICITY_SQUARED,
                      PEG_LONGITUDE,
                      WIDTH,
                      S2SCH,
                      PEG_LATITUDE,
                      DOPPLER_CENTROID
                     )


    logging_name = "isce.stdproc.correct"

    family = 'correct'

    def __init__(self,family='',name=''):
        super(Correct, self).__init__(family if family else  self.__class__.family, name=name)
        self.dim1_referenceOrbit = None
        self.dim1_mocompBaseline = None
        self.dim2_mocompBaseline = None
        self.dim1_midpoint = None
        self.dim2_midpoint = None
        self.dim1_s1sch = None
        self.dim2_s1sch = None
        self.dim1_s2sch = None
        self.dim2_s2sch = None
        self.dim1_sc = None
        self.dim2_sc = None
        self.lookSide = None    #Set to right side by default
        self.dopplerCentroidCoeffs = None
        self.polyDoppler = None
        self.dumpRangeFiles = None

        self.heightSchFilename = ''
        self.heightSchCreatedHere = False
        self.heightSchImage = None
        self.heightSchAccessor = None
        self.intFilename = ''
        self.intCreatedHere = False
        self.intImage = None
        self.intAccessor = None
        self.topophaseMphFilename = ''
        self.topophaseMphCreatedHere = False
        self.topophaseMphImage = None
        self.topophaseMphAccessor = None
        self.topophaseFlatFilename = ''
        self.topophaseFlatCreatedHere = False
        self.topophaseFlatImage = None
        self.topophaseFlatAccessor = None
        self.secondaryRangeFilename = ''
        self.secondaryRangeCreatedHere = False
        self.secondaryRangeImage = None
        self.secondaryRangeAccessor = None
        self.referenceRangeFilename = ''
        self.referenceRangeCreatedHere = False
        self.referenceRangeAccessor = None
        self.referenceRangeImage = None
        self.polyDopplerAccessor = None
        
        self.initOptionalAndMandatoryLists()
        return None
    
    def createPorts(self):
        pegPort = Port(name="peg",method=self.addPeg)
        planetPort = Port(name='planet',method=self.addPlanet)        
        framePort = Port(name='frame',method=self.addFrame)
        ifgPort = Port(name='interferogram',method=self.addInterferogram)
        slcPort = Port(name='referenceslc',method=self.addReferenceSlc) #Piyush
        
        self._inputPorts.add(pegPort)
        self._inputPorts.add(planetPort)        
        self._inputPorts.add(framePort)
        self._inputPorts.add(ifgPort)
        self._inputPorts.add(slcPort)  #Piyush
        return None

    # assume that for the images passed no createImage has been called  

    def correct(self, intImage=None,heightSchImage=None,topoMphImage=None,
                topoFlatImage=None):
        for port in self.inputPorts:
            port()
        if not heightSchImage is None:
            self.heightSchImage = heightSchImage
        
        # another way of passing width and length if not using the ports
        if intImage is not None:
            self.intImage = intImage
        
            #if width or length not defined get 'em  from intImage since they 
            #are needed to create the output images
            if self.width is None:
                self.width = self.intImage.getWidth()
            if self.length is None:
                self.length = self.intImage.getLength()
        
        if not topoMphImage is None:
            self.topophaseMphImage = topoMphImage

        if topoFlatImage is not None:
            self.topophaseFlatImage = topoFlatImage


        self.setDefaults() 
        #creates images if not set and call the createImage() (also for the intImage)
        self.createImages()

        self.heightSchAccessor = self.heightSchImage.getImagePointer()
        if self.intImage is not None:
            self.intAccessor = self.intImage.getImagePointer()
        else:
            self.intAccessor = 0

        self.topophaseMphAccessor = self.topophaseMphImage.getImagePointer()

        if self.intImage is not None:
            self.topophaseFlatAccessor = self.topophaseFlatImage.getImagePointer()
        else:
            self.topophaseFlatAccessor = 0

        if self.dumpRangeFiles:
            self.secondaryRangeAccessor = self.secondaryRangeImage.getImagePointer()
            self.referenceRangeAccessor = self.referenceRangeImage.getImagePointer()
        else:
            self.secondaryRangeAccessor = 0
            self.referenceRangeAccessor = 0


        self.polyDopplerAccessor = self.polyDoppler.getPointer()
        self.allocateArrays()
        self.setState()

        correct.correct_Py(self.intAccessor,
                           self.heightSchAccessor,
                           self.topophaseMphAccessor,
                           self.topophaseFlatAccessor,
                           self.referenceRangeAccessor,
                           self.secondaryRangeAccessor)
        self.topophaseMphImage.trueDataType = self.topophaseMphImage.getDataType()
        self.topophaseFlatImage.trueDataType = self.topophaseFlatImage.getDataType()



        self.deallocateArrays()
        #call the finalizeImage() on all the images
        self.destroyImages()
        self.topophaseMphImage.renderHdr()
        self.topophaseFlatImage.renderHdr()
        if self.dumpRangeFiles:
            self.referenceRangeImage.renderHdr()
            self.secondaryRangeImage.renderHdr()

        return


    def setDefaults(self):
        if self.ellipsoidMajorSemiAxis is None:
            self.ellipsoidMajorSemiAxis = CN.EarthMajorSemiAxis

        if self.ellipsoidEccentricitySquared is None:
            self.ellipsoidEccentricitySquared = CN.EarthEccentricitySquared

        if self.lookSide is None:
            self.lookSide = -1

        if self.isMocomp is None:
            self.isMocomp = (8192-2048)/2 
        
        if self.topophaseFlatFilename == '':
            self.topophaseFlatFilename = 'topophase.flat'
            self.logger.warning(
                'The topophase flat file has been given the default name %s' %
                (self.topophaseFlatFilename)
                )

        if self.topophaseMphFilename == '':
            self.topophaseMphFilename = 'topophase.mph'
            self.logger.warning(
            'The topophase mph file has been given the default name %s' %
            (self.topophaseMphFilename)
            )

        if self.dumpRangeFiles is None:
            self.dumpRangeFiles = False

        if self.dumpRangeFiles:
            if self.secondaryRangeFilename == '':
                self.secondaryRangeFilename = 'secondaryrange.rdr'
                self.logger.warning(
                    'Secondary range file has been given the default name %s' %
                    (self.secondaryRangeFilename))
    
            if self.referenceRangeFilename == '':
                self.referenceRangeFilename = 'referencerange.rdr'
                self.logger.warning(
                    'Reference range file has been given the default name %s' %
                    (self.referenceRangeFilename))

        if self.polyDoppler is None:
            polyDop = Poly2D(name=self.name + '_correctPoly')
            polyDop.setNormRange(1.0/(1.0*self.numberRangeLooks))
            polyDop.setNormAzimuth(1.0/(1.0*self.numberAzimuthLooks))
            polyDop.setMeanRange(0.0)
            polyDop.setMeanAzimuth(0.0)
            polyDop.setWidth(self.width)
            polyDop.setLength(self.length)
            polyDop.initPoly(rangeOrder=len(self.dopplerCentroidCoeffs)-1, azimuthOrder=0, coeffs=[self.dopplerCentroidCoeffs])
           
            self.polyDoppler = polyDop

    def destroyImages(self):
        self.intImage.finalizeImage()
        self.heightSchImage.finalizeImage()
        self.topophaseMphImage.finalizeImage()
        self.topophaseFlatImage.finalizeImage()

        if self.dumpRangeFiles:
            self.referenceRangeImage.finalizeImage()
            self.secondaryRangeImage.finalizeImage()

        self.polyDoppler.finalize()

    def createImages(self):
        
        if self.heightSchImage is None and not self.heightSchFilename == '':
            self.heightSchImage = IF.createImage()
            accessMode = 'read'
            dataType = 'FLOAT'
            width = self.width
            self.heightSchImage.initImage(
                self.heightSchFilename, accessMode, width, dataType
            )
        elif self.heightSchImage is None:
            # this should never happen, atleast when using the  
            # correct method. same for other images
            self.logger.error(
            'Must either pass the heightSchImage in the call or set self.heightSchFilename.'
            )
            raise Exception
       
        if self.intImage is not None:
            if (self.topophaseFlatImage is None and
                not self.topophaseFlatFilename == ''
                ):
                self.topophaseFlatImage = IF.createIntImage()
                accessMode = 'write'
                width = self.width
                self.topophaseFlatImage.initImage(self.topophaseFlatFilename,
                                              accessMode,
                                              width)
            elif self.topophaseFlatImage is None:
                self.logger.error(
                    'Must either pass the topophaseFlatImage in the call or set self.topophaseMphFilename.'
                )
        
        if (
            self.topophaseMphImage is None and
            not self.topophaseMphFilename == ''
            ):
            self.topophaseMphImage = IF.createIntImage()
            accessMode = 'write'
            width = self.width
            self.topophaseMphImage.initImage(self.topophaseMphFilename,
                                             accessMode,
                                             width)
        elif self.topophaseMphImage is None:
            self.logger.error(
                'Must either pass the topophaseMphImage in the call or set self.topophaseMphFilename.'
                )

        if self.dumpRangeFiles:
            if (self.secondaryRangeImage is None and not self.secondaryRangeFilename == ''):
                self.secondaryRangeImage = IF.createImage()
                self.secondaryRangeImage.setFilename(self.secondaryRangeFilename)
                self.secondaryRangeImage.setAccessMode('write')
                self.secondaryRangeImage.dataType = 'FLOAT'
                self.secondaryRangeImage.setWidth(self.width)
                self.secondaryRangeImage.bands = 1
                self.secondaryRangeImage.scheme = 'BIL'

            if (self.referenceRangeImage is None and not self.referenceRangeFilename == ''):
                self.referenceRangeImage = IF.createImage()
                self.referenceRangeImage.setFilename(self.referenceRangeFilename)
                self.referenceRangeImage.setAccessMode('write')
                self.referenceRangeImage.dataType = 'FLOAT'
                self.referenceRangeImage.setWidth(self.width)
                self.referenceRangeImage.bands = 1
                self.referenceRangeImage.scheme = 'BIL'
        

        if self.polyDoppler is None:
            self.logger.error('Must pass doppler polynomial in the call to correct')


            
            #one way or another when it gets here the images better be defined
        if self.intImage is not None:
            self.intImage.createImage()#this is passed but call createImage and finalizeImage from here
            self.topophaseFlatImage.createImage()

        self.heightSchImage.createImage()
        self.topophaseMphImage.createImage()

        if self.dumpRangeFiles:
            self.referenceRangeImage.createImage()
            self.secondaryRangeImage.createImage()

        self.polyDoppler.createPoly2D()

    def setState(self):
        correct.setReferenceOrbit_Py(self.referenceOrbit,
                                     self.dim1_referenceOrbit)
        correct.setMocompBaseline_Py(self.mocompBaseline,
                                     self.dim1_mocompBaseline,
                                     self.dim2_mocompBaseline)
        correct.setISMocomp_Py(int(self.isMocomp))
        correct.setEllipsoidMajorSemiAxis_Py(
            float(self.ellipsoidMajorSemiAxis)
            )
        correct.setEllipsoidEccentricitySquared_Py(
            float(self.ellipsoidEccentricitySquared)
            )
        correct.setLength_Py(int(self.length))
        correct.setWidth_Py(int(self.width))
        correct.setRangePixelSpacing_Py(float(self.slantRangePixelSpacing))
        correct.setRangeFirstSample_Py(float(self.rangeFirstSample))
        correct.setSpacecraftHeight_Py(float(self.spacecraftHeight))
        correct.setPlanetLocalRadius_Py(float(self.planetLocalRadius))
        correct.setBodyFixedVelocity_Py(float(self.bodyFixedVelocity))
        correct.setNumberRangeLooks_Py(int(self.numberRangeLooks))
        correct.setNumberAzimuthLooks_Py(int(self.numberAzimuthLooks))
        correct.setPegLatitude_Py(float(self.pegLatitude))
        correct.setPegLongitude_Py(float(self.pegLongitude))
        correct.setPegHeading_Py(float(self.pegHeading))
#        correct.setDopCoeff_Py(self.dopplerCentroidCoeffs)
        correct.setDopCoeff_Py(self.polyDopplerAccessor)
        correct.setPRF_Py(float(self.prf))
        correct.setRadarWavelength_Py(float(self.radarWavelength))
        correct.setMidpoint_Py(self.midpoint,
                               self.dim1_midpoint,
                               self.dim2_midpoint)
        correct.setSch1_Py(self.s1sch, self.dim1_s1sch, self.dim2_s1sch)
        correct.setSch2_Py(self.s2sch, self.dim1_s2sch, self.dim2_s2sch)
        correct.setSc_Py(self.sc, self.dim1_sc, self.dim2_sc)
        correct.setLookSide_Py(int(self.lookSide))

        return None

    def setLookSide(self, var):
        self.lookSide = int(var)
        return

    def setReferenceOrbit(self, var):
        self.referenceOrbit = var
        return

    def setMocompBaseline(self, var):
        self.mocompBaseline = var
        return

    def setISMocomp(self, var):
        self.isMocomp = int(var)
        return

    def setEllipsoidMajorSemiAxis(self, var):
        self.ellipsoidMajorSemiAxis = float(var)
        return

    def setEllipsoidEccentricitySquared(self, var):
        self.ellipsoidEccentricitySquared = float(var)
        return

    def setLength(self, var):
        self.length = int(var)
        return

    def setWidth(self, var):
        self.width = int(var)
        return

    def setRangePixelSpacing(self, var):
        self.slantRangePixelSpacing = float(var)
        return

    def setRangeFirstSample(self, var):
        self.rangeFirstSample = float(var)
        return

    def setSpacecraftHeight(self, var):
        self.spacecraftHeight = float(var)
        return

    def setPlanetLocalRadius(self, var):
        self.planetLocalRadius = float(var)
        return

    def setBodyFixedVelocity(self, var):
        self.bodyFixedVelocity = float(var)
        return

    def setNumberRangeLooks(self, var):
        self.numberRangeLooks = int(var)
        return

    def setNumberAzimuthLooks(self, var):
        self.numberAzimuthLooks = int(var)
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

    def setDopplerCentroidCoeffs(self, var):
        self.dopplerCentroidCoeffs = var
        return

    def setPRF(self, var):
        self.prf = float(var)
        return

    def setRadarWavelength(self, var):
        self.radarWavelength = float(var)
        return

    def setMidpoint(self, var):
        self.midpoint = var
        return

    def setSch1(self, var):
        self.s1sch = var
        return

    def setSch2(self, var):
        self.s2sch = var
        return

    def setSc(self, var):
        self.sc = var
        return

    def setHeightSchFilename(self, var):
        self.heightSchFilename = var
    
    def setInterferogramFilename(self, var):
        self.intFilename = var
    
    def setTopophaseMphFilename(self, var):
        self.topophaseMphFilename = var
    
    def setTopophaseFlatFilename(self, var):
        self.topophaseFlatFilename = var

    def setHeightSchImageImage(self, img):
        self.heightSchImage = img

    def setInterferogramImage(self, img):
        self.intImage = img

    def setTopophaseMphImage(self, img):
        self.topophaseMphImage = img

    def setImageTopophaseFlat(self, img):
        self.topophaseFlatImage = img

    def setPolyDoppler(self, var):
        self.polyDoppler = var
    
    def allocateArrays(self):
        if self.dim1_referenceOrbit is None:
            self.dim1_referenceOrbit = len(self.referenceOrbit)

        if not self.dim1_referenceOrbit:
            print("Error. Trying to allocate zero size array")
            raise Exception

        correct.allocate_s_mocompArray_Py(self.dim1_referenceOrbit)

        if self.dim1_mocompBaseline is None:
            self.dim1_mocompBaseline = len(self.mocompBaseline)
            self.dim2_mocompBaseline = len(self.mocompBaseline[0])

        if (not self.dim1_mocompBaseline) or (not self.dim2_mocompBaseline):
            print("Error. Trying to allocate zero size array")
            raise Exception

        #Recompute length in azimuth to be the minimum of its current value
        #(set from the ifg length in the interferogram port) and the computed
        #maximum value it can have in correct.f to prevent array out of bounds
        #condition in accessing the mocompBaseline.
        self.length = min(self.length,
            int((self.dim1_mocompBaseline - self.isMocomp -
                 self.numberAzimuthLooks/2)/self.numberAzimuthLooks))
        print("Recomputed length = ", self.length)

        correct.allocate_mocbaseArray_Py(self.dim1_mocompBaseline,
                                         self.dim2_mocompBaseline)

        if self.dim1_midpoint is None:
            self.dim1_midpoint = len(self.midpoint)
            self.dim2_midpoint = len(self.midpoint[0])

        if (not self.dim1_midpoint) or (not self.dim2_midpoint):
            print("Error. Trying to allocate zero size array")
            raise Exception

        correct.allocate_midpoint_Py(self.dim1_midpoint, self.dim2_midpoint)

        if self.dim1_s1sch is None:
            self.dim1_s1sch = len(self.s1sch)
            self.dim2_s1sch = len(self.s1sch[0])

        if (not self.dim1_s1sch) or (not self.dim2_s1sch):
            print("Error. Trying to allocate zero size array")
            raise Exception

        correct.allocate_s1sch_Py(self.dim1_s1sch, self.dim2_s1sch)

        if self.dim1_s2sch is None:
            self.dim1_s2sch = len(self.s2sch)
            self.dim2_s2sch = len(self.s2sch[0])

        if (not self.dim1_s2sch) or (not self.dim2_s2sch):
            print("Error. Trying to allocate zero size array")
            raise Exception

        correct.allocate_s2sch_Py(self.dim1_s2sch, self.dim2_s2sch)

        if self.dim1_sc is None:
            self.dim1_sc = len(self.sc)
            self.dim2_sc = len(self.sc[0])

        if (not self.dim1_sc) or (not self.dim2_sc):
            print("Error. Trying to allocate zero size array")
            raise Exception

        correct.allocate_smsch_Py(self.dim1_sc, self.dim2_sc)

        return

    def deallocateArrays(self):
        correct.deallocate_s_mocompArray_Py()
        correct.deallocate_mocbaseArray_Py()
        correct.deallocate_midpoint_Py()
        correct.deallocate_s1sch_Py()
        correct.deallocate_s2sch_Py()
        correct.deallocate_smsch_Py()
        return

    def addPeg(self):
        peg = self._inputPorts.getPort(name='peg').getObject()
        if (peg):            
            try:
                self.planetLocalRadius = peg.getRadiusOfCurvature()
                self.pegLatitude = math.radians(peg.getLatitude())
                self.pegLongitude = math.radians(peg.getLongitude())
                self.pegHeading = math.radians(peg.getHeading())
            except AttributeError as strerr:
                self.logger.error(strerr)
                raise AttributeError
    
    def addPlanet(self):
        planet = self._inputPorts.getPort(name='planet').getObject()
        if (planet):            
            try:
                ellipsoid = planet.get_elp()
                self.ellipsoidMajorSemiAxis = ellipsoid.get_a()
                self.ellipsoidEccentricitySquared = ellipsoid.get_e2()
            except AttributeError as strerr:
                self.logger.error(strerr)
                raise AttributeError
        
    def addFrame(self):
        frame = self._inputPorts.getPort(name='frame').getObject()
        if (frame):            
            try:
                #                self.rangeFirstSample = frame.getStartingRange() - Piyush
                instrument = frame.getInstrument()
                self.slantRangePixelSpacing = instrument.getRangePixelSize()
                self.prf = instrument.getPulseRepetitionFrequency()
                self.radarWavelength = instrument.getRadarWavelength()
            except AttributeError as strerr:
                self.logger.error(strerr)
                raise AttributeError

    #####This part needs to change when formslc is refactored
    #####to use doppler polynomials
    def addReferenceSlc(self): 
        formslc = self._inputPorts.getPort(name='referenceslc').getObject()
        if (formslc):
            try:
                self.rangeFirstSample = formslc.startingRange
            except AttributeError as strerr:
                self.logger.error(strerr)
                raise AttributeError

            self.dopplerCentroidCoeffs = formslc.dopplerCentroidCoefficients

    def addInterferogram(self):
        ifg = self._inputPorts.getPort(name='interferogram').getObject()
        if (ifg):
            try:
                self.intImage = ifg
                self.width = ifg.getWidth()
                self.length = ifg.getLength()
            except AttributeError as strerr:
                self.logger.error(strerr)
                raise AttributeError




    pass
