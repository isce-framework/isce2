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
import logging
from iscesys.Component.Component import Component
from isceobj.Platform.Platform import Platform
from isceobj import Constants
from isceobj.Util.decorators import type_check, force, pickled, logged


PRF = Component.Parameter('PRF',
    public_name='PRF',
    default=None,
    type = float,
    mandatory = True,
    doc = 'Pulse Repetition Frequency')

RANGE_SAMPLING_RATE = Component.Parameter('rangeSamplingRate',
    public_name='RANGE_SAMPLING_RATE',
    default=None,
    type = float,
    mandatory = True,
    doc = 'Range sampling rate')

CHIRP_SLOPE = Component.Parameter('chirpSlope',
    public_name='CHIRP_SLOPE',
    default=None,
    type = float,
    mandatory = True,
    doc = 'Chirp slope of range pulse in Hz / sec')

RADAR_WAVELENGTH = Component.Parameter('radarWavelength',
    public_name='RADAR_WAVELENGTH',
    default=None,
    type = float,
    mandatory = True,
    doc = 'Radar wavelength')

RADAR_FREQUENCY = Component.Parameter('radarFrequency',
    public_name='RADAR_FREQUENCY',
    default=None,
    type = float,
    mandatory = True,
    doc = 'Radar frequency in Hz')

INPHASE_BIAS = Component.Parameter('inPhaseValue',
    public_name='INPHASE_BIAS',
    default=None,
    type = float,
    mandatory = True,
    doc = 'Inphase channel bias')

QUADRATURE_BIAS = Component.Parameter('quadratureValue',
    public_name='QUADRATURE_BIAS',
    default=None,
    type = float,
    mandatory = True,
    doc = 'Quadrature channel bias')

CALTONE_LOCATION = Component.Parameter('caltoneLocation',
    public_name='CALTONE_LOCATION',
    default=None,
    type = float,
    mandatory = True,
    doc = 'Caltone location in Hz')

RANGE_FIRST_SAMPLE = Component.Parameter('rangeFirstSample',
    public_name='RANGE_FIRST_SAMPLE',
    default=None,
    type = float,
    mandatory = True,
    doc = 'Range to the first valid sample')

IQ_FLIP = Component.Parameter('IQFlip',
    public_name='IQ_FLIP',
    default=None,
    type = str,
    mandatory = True,
    doc = 'If the I/Q channels have been flipped')

RANGE_PULSE_DURATION = Component.Parameter('rangePulseDuration',
    public_name='RANGE_PULSE_DURATION',
    default=None,
    type = float,
    mandatory = True,
    doc = 'Range pulse duration')

INCIDENCE_ANGLE = Component.Parameter('incidenceAngle',
    public_name='INCIDENCE_ANGLE',
    default=None,
    type = float,
    mandatory = True,
    doc = 'Incidence angle')

RANGE_PIXEL_SIZE = Component.Parameter('rangePixelSize',
    public_name='RANGE_PIXEL_SIZE',
    default=None,
    type = float,
    mandatory = True,
    doc = 'Range pixel size')

AZIMUTH_PIXEL_SIZE = Component.Parameter('azimuthPixelSize',
    public_name='AZIMUTH_PIXEL_SIZE',
    default=None,
    type = float,
    mandatory = True,
    doc = '')

PULSE_LENGHT = Component.Parameter('pulseLength',
    public_name='PULSE_LENGHT',
    default=None,
    type = float,
    mandatory = True,
    doc = 'Pulse length')


##
# This class allows the creation of a Radar object. The parameters that need
# to be
# set are
#\verbatim
#RANGE_FIRST_SAMPLE': range first sample. Mandatory.
#PRF: pulse repetition frequency. Mandatory.
#CALTONE_LOCATION: caltone location. Optional. Default 0.
#INPHASE_VALUE: in phase value. Mandatory.
#QUADRATURE_VALUE: quadrature value. Mandatory.
#IQ_FLIP: IQ flip flag. Optional. Default 'n'.
#RANGE_SAMPLING_RATE: range sampling rate. Mandatory.

#\endverbatim
#Since the Radar class inherits the Component.Component, the methods of initialization described in the Component package can be used.
#Moreover each parameter can be set with the corresponding accessor method setParameter() (see the class member methods).
@pickled
class Radar(Component):

    family = 'radar'
    logging_name = 'isce.isceobj.radar'
    

    parameter_list = (PRF,
                      RANGE_SAMPLING_RATE,
                      RANGE_FIRST_SAMPLE,
                      CHIRP_SLOPE,
                      RADAR_WAVELENGTH,
                      RADAR_FREQUENCY,
                      IQ_FLIP,
                      INPHASE_BIAS,
                      QUADRATURE_BIAS,
                      CALTONE_LOCATION,
                      RANGE_PULSE_DURATION,
                      INCIDENCE_ANGLE,
                      RANGE_PIXEL_SIZE,
                      AZIMUTH_PIXEL_SIZE,
                      PULSE_LENGHT)


    def _facilities(self):

        self._platform = self.facility(
                '_platform',
                public_name='PLATFORM',
                module='isceobj.Platform.Platform',
                factory='createPlatform',
                args= (),
                mandatory=True,
                doc = "Platform information")

    @logged
    def __init__(self, name=''):
        super(Radar, self).__init__(family=self.__class__.family, name=name)

        return None
    
    def __complex__(self):
        return self.inPhaseValue + (1j) * self.quadratureValue

    @force(float)
    def setRangeFirstSample(self, var):
        self.rangeFirstSample =  var
        pass

    @force(float)
    def setPulseRepetitionFrequency(self, var):
        self.PRF = var
            
    def getPulseRepetitionFrequency(self):
        return self.PRF

    @force(float)
    def setCaltoneLocation(self, var):
        self.caltoneLocation = var
    
    @force(float)
    def setInPhaseValue(self, var):
        self.inPhaseValue = var
        return

    def getInPhaseValue(self):
        return self.inPhaseValue

    @force(float)
    def setQuadratureValue(self, var):
        self.quadratureValue = var
    
    def getQuadratureValue(self):
        return self.quadratureValue

    def setIQFlip(self, var):
        self.IQFlip = str(var)
    
    @force(float)
    def setRangeSamplingRate(self, var):
        self.rangeSamplingRate = var

    def getRangeSamplingRate(self):
        return self.rangeSamplingRate

    @force(float)
    def setChirpSlope(self, var):
        self.chirpSlope = var

    def getChirpSlope(self):
        return self.chirpSlope

    @force(float)
    def setRangePulseDuration(self, var):
        self.rangePulseDuration = var

    def getRangePulseDuration(self):
        return self.rangePulseDuration

    @force(float)
    def setRadarFrequency(self, freq):
        self.radarFrequency = freq
        self.radarWavelength = Constants.lambda2nu(self.radarFrequency)

    def getRadarFrequency(self):
        return self.radarFrequency

    @force(float)
    def setRadarWavelength(self, var):
        self.radarWavelength = var
        self.radarFrequency = Constants.nu2lambda(self.radarWavelength)
        return None

    def getRadarWavelength(self):
        return self.radarWavelength

    @force(float)
    def setIncidenceAngle(self, var):
        self.incidenceAngle = var
        
    def getIncidenceAngle(self):
        return self.incidenceAngle

    @type_check(Platform)
    def setPlatform(self, platform):
        self._platform = platform
        pass
    
    def getPlatform(self):
        return self._platform
    
    @force(float)
    def setRangePixelSize(self, size):
        self.rangePixelSize = size
        
    def getRangePixelSize(self):
        return self.rangePixelSize
    
    @force(float)
    def setAzimuthPixelSize(self, size):
        self.azimuthPixelSize = size
        
    def getAzimuthPixelSize(self):
        return self.azimuthPixelSize
    
    @force(float)
    def setPulseLength(self, rpl):
        self.pulseLength = rpl
    
    def getPulseLength(self):
        return self.pulseLength

    def setBeamNumber(self, num):
        self.beamNumber = num

    def getBeamNumber(self):
        return self.beamNumber


    platform = property(getPlatform  , setPlatform )    


    def __str__(self):
        retstr = "Pulse Repetition Frequency: (%s)\n"
        retlst = (self.PRF,)
        retstr += "Range Sampling Rate: (%s)\n"
        retlst += (self.rangeSamplingRate,)
        retstr += "Radar Wavelength: (%s)\n"
        retlst += (self.radarWavelength,)
        retstr += "Chirp Slope: (%s)\n"
        retlst += (self.chirpSlope,)
        return retstr % retlst



def createRadar():
    return Radar()
