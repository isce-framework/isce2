#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import logging
import logging.config
from iscesys.Component.Component import Component
from iscesys.DateTimeUtil.DateTimeUtil import DateTimeUtil as DTU
from iscesys.Compatibility import Compatibility


REFERENCE_DATE = Component.Parameter('referenceDate',
                                public_name='reference date',
                                default=None,
                                type=str,
                                mandatory=True,
                                doc='reference acquistion date')

SECONDARY_DATE = Component.Parameter('secondaryDate',
                                public_name='secondary date',
                                default=None,
                                type=str,
                                mandatory=True,
                                doc='secondary acquistion date')

MODE_COMBINATION = Component.Parameter('modeCombination',
                                public_name='mode combination',
                                default=None,
                                type=int,
                                mandatory=True,
                                doc='mode combination')

REFERENCE_FRAMES = Component.Parameter('referenceFrames',
                                public_name = 'reference frames',
                                default = None,
                                type=str,
                                container=list,
                                mandatory=False,
                                doc = 'reference frames to process')

SECONDARY_FRAMES = Component.Parameter('secondaryFrames',
                                public_name = 'secondary frames',
                                default = None,
                                type=str,
                                container=list,
                                mandatory=False,
                                doc = 'secondary frames to process')

STARTING_SWATH = Component.Parameter('startingSwath',
                                public_name='starting swath',
                                default=1,
                                type=int,
                                mandatory=False,
                                doc="starting swath to process")

ENDING_SWATH = Component.Parameter('endingSwath',
                                public_name='ending swath',
                                default=5,
                                type=int,
                                mandatory=False,
                                doc="ending swath to process")

BURST_UNSYNCHRONIZED_TIME = Component.Parameter('burstUnsynchronizedTime',
                                public_name = 'burst unsynchronized time',
                                default = None,
                                type = float,
                                mandatory = False,
                                doc = 'burst unsynchronized time in second')

BURST_SYNCHRONIZATION = Component.Parameter('burstSynchronization',
                                public_name = 'burst synchronization',
                                default = None,
                                type = float,
                                mandatory = False,
                                doc = 'average burst synchronization of all swaths and frames in percentage')

RANGE_RESIDUAL_OFFSET_CC = Component.Parameter('rangeResidualOffsetCc',
                                public_name = 'range residual offset estimated by cross correlation',
                                default = None,
                                type = float,
                                mandatory = True,
                                container = list,
                                doc = 'range residual offset estimated by cross correlation')

AZIMUTH_RESIDUAL_OFFSET_CC = Component.Parameter('azimuthResidualOffsetCc',
                                public_name = 'azimuth residual offset estimated by cross correlation',
                                default = None,
                                type = float,
                                mandatory = True,
                                container = list,
                                doc = 'azimuth residual offset estimated by cross correlation')

RANGE_RESIDUAL_OFFSET_SD = Component.Parameter('rangeResidualOffsetSd',
                                public_name = 'range residual offset estimated by spectral diversity',
                                default = None,
                                type = float,
                                mandatory = True,
                                container = list,
                                doc = 'range residual offset estimated by spectral diversity')

AZIMUTH_RESIDUAL_OFFSET_SD = Component.Parameter('azimuthResidualOffsetSd',
                                public_name = 'azimuth residual offset estimated by spectral diversity',
                                default = None,
                                type = float,
                                mandatory = True,
                                container = list,
                                doc = 'azimuth residual offset estimated by spectral diversity')

SWATH_RANGE_OFFSET_GEOMETRICAL_REFERENCE = Component.Parameter('swathRangeOffsetGeometricalReference',
                                public_name = 'swath range offset from geometry reference',
                                default = None,
                                type = float,
                                mandatory = True,
                                container = list,
                                doc = 'swath range offset from geometry reference')

SWATH_AZIMUTH_OFFSET_GEOMETRICAL_REFERENCE = Component.Parameter('swathAzimuthOffsetGeometricalReference',
                                public_name = 'swath azimuth offset from geometry reference',
                                default = None,
                                type = float,
                                mandatory = True,
                                container = list,
                                doc = 'swath azimuth offset from geometry reference')

SWATH_RANGE_OFFSET_MATCHING_REFERENCE = Component.Parameter('swathRangeOffsetMatchingReference',
                                public_name = 'swath range offset from matching reference',
                                default = None,
                                type = float,
                                mandatory = True,
                                container = list,
                                doc = 'swath range offset from matching reference')

SWATH_AZIMUTH_OFFSET_MATCHING_REFERENCE = Component.Parameter('swathAzimuthOffsetMatchingReference',
                                public_name = 'swath azimuth offset from matching reference',
                                default = None,
                                type = float,
                                mandatory = True,
                                container = list,
                                doc = 'swath azimuth offset from matching reference')

SWATH_RANGE_OFFSET_GEOMETRICAL_SECONDARY = Component.Parameter('swathRangeOffsetGeometricalSecondary',
                                public_name = 'swath range offset from geometry secondary',
                                default = None,
                                type = float,
                                mandatory = True,
                                container = list,
                                doc = 'swath range offset from geometry secondary')

SWATH_AZIMUTH_OFFSET_GEOMETRICAL_SECONDARY = Component.Parameter('swathAzimuthOffsetGeometricalSecondary',
                                public_name = 'swath azimuth offset from geometry secondary',
                                default = None,
                                type = float,
                                mandatory = True,
                                container = list,
                                doc = 'swath azimuth offset from geometry secondary')

SWATH_RANGE_OFFSET_MATCHING_SECONDARY = Component.Parameter('swathRangeOffsetMatchingSecondary',
                                public_name = 'swath range offset from matching secondary',
                                default = None,
                                type = float,
                                mandatory = True,
                                container = list,
                                doc = 'swath range offset from matching secondary')

SWATH_AZIMUTH_OFFSET_MATCHING_SECONDARY = Component.Parameter('swathAzimuthOffsetMatchingSecondary',
                                public_name = 'swath azimuth offset from matching secondary',
                                default = None,
                                type = float,
                                mandatory = True,
                                container = list,
                                doc = 'swath azimuth offset from matching secondary')



FRAME_RANGE_OFFSET_GEOMETRICAL_REFERENCE = Component.Parameter('frameRangeOffsetGeometricalReference',
                                public_name = 'frame range offset from geometry reference',
                                default = None,
                                type = float,
                                mandatory = True,
                                container = list,
                                doc = 'frame range offset from geometry reference')

FRAME_AZIMUTH_OFFSET_GEOMETRICAL_REFERENCE = Component.Parameter('frameAzimuthOffsetGeometricalReference',
                                public_name = 'frame azimuth offset from geometry reference',
                                default = None,
                                type = float,
                                mandatory = True,
                                container = list,
                                doc = 'frame azimuth offset from geometry reference')

FRAME_RANGE_OFFSET_MATCHING_REFERENCE = Component.Parameter('frameRangeOffsetMatchingReference',
                                public_name = 'frame range offset from matching reference',
                                default = None,
                                type = float,
                                mandatory = True,
                                container = list,
                                doc = 'frame range offset from matching reference')

FRAME_AZIMUTH_OFFSET_MATCHING_REFERENCE = Component.Parameter('frameAzimuthOffsetMatchingReference',
                                public_name = 'frame azimuth offset from matching reference',
                                default = None,
                                type = float,
                                mandatory = True,
                                container = list,
                                doc = 'frame azimuth offset from matching reference')

FRAME_RANGE_OFFSET_GEOMETRICAL_SECONDARY = Component.Parameter('frameRangeOffsetGeometricalSecondary',
                                public_name = 'frame range offset from geometry secondary',
                                default = None,
                                type = float,
                                mandatory = True,
                                container = list,
                                doc = 'frame range offset from geometry secondary')

FRAME_AZIMUTH_OFFSET_GEOMETRICAL_SECONDARY = Component.Parameter('frameAzimuthOffsetGeometricalSecondary',
                                public_name = 'frame azimuth offset from geometry secondary',
                                default = None,
                                type = float,
                                mandatory = True,
                                container = list,
                                doc = 'frame azimuth offset from geometry secondary')

FRAME_RANGE_OFFSET_MATCHING_SECONDARY = Component.Parameter('frameRangeOffsetMatchingSecondary',
                                public_name = 'frame range offset from matching secondary',
                                default = None,
                                type = float,
                                mandatory = True,
                                container = list,
                                doc = 'frame range offset from matching secondary')

FRAME_AZIMUTH_OFFSET_MATCHING_SECONDARY = Component.Parameter('frameAzimuthOffsetMatchingSecondary',
                                public_name = 'frame azimuth offset from matching secondary',
                                default = None,
                                type = float,
                                mandatory = True,
                                container = list,
                                doc = 'frame azimuth offset from matching secondary')

NUMBER_RANGE_LOOKS1 = Component.Parameter('numberRangeLooks1',
                                public_name='number of range looks 1',
                                default=None,
                                type=int,
                                mandatory=False,
                                doc="number of range looks when forming interferogram")

NUMBER_AZIMUTH_LOOKS1 = Component.Parameter('numberAzimuthLooks1',
                                public_name='number of azimuth looks 1',
                                default=None,
                                type=int,
                                mandatory=False,
                                doc="number of azimuth looks when forming interferogram")

NUMBER_RANGE_LOOKS2 = Component.Parameter('numberRangeLooks2',
                                public_name='number of range looks 2',
                                default=None,
                                type=int,
                                mandatory=False,
                                doc="number of range looks for further multiple looking")

NUMBER_AZIMUTH_LOOKS2 = Component.Parameter('numberAzimuthLooks2',
                                public_name='number of azimuth looks 2',
                                default=None,
                                type=int,
                                mandatory=False,
                                doc="number of azimuth looks for further multiple looking")

NUMBER_RANGE_LOOKS_SIM = Component.Parameter('numberRangeLooksSim',
                                public_name='number of range looks sim',
                                default=None,
                                type=int,
                                mandatory=False,
                                doc="number of range looks when simulating radar image")

NUMBER_AZIMUTH_LOOKS_SIM = Component.Parameter('numberAzimuthLooksSim',
                                public_name='number of azimuth looks sim',
                                default=None,
                                type=int,
                                mandatory=False,
                                doc="number of azimuth looks when simulating radar image")

NUMBER_RANGE_LOOKS_ION = Component.Parameter('numberRangeLooksIon',
                                public_name='number of range looks ion',
                                default=None,
                                type=int,
                                mandatory=False,
                                doc="number of range looks for ionospheric correction")

NUMBER_AZIMUTH_LOOKS_ION = Component.Parameter('numberAzimuthLooksIon',
                                public_name='number of azimuth looks ion',
                                default=None,
                                type=int,
                                mandatory=False,
                                doc="number of azimuth looks for ionospheric correction")

NUMBER_RANGE_LOOKS_SD = Component.Parameter('numberRangeLooksSd',
                                public_name='number of range looks sd',
                                default=None,
                                type=int,
                                mandatory=False,
                                doc="number of range looks for spectral diversity")

NUMBER_AZIMUTH_LOOKS_SD = Component.Parameter('numberAzimuthLooksSd',
                                public_name='number of azimuth looks sd',
                                default=None,
                                type=int,
                                mandatory=False,
                                doc="number of azimuth looks for spectral diversity")

SUBBAND_RADAR_WAVLENGTH = Component.Parameter('subbandRadarWavelength',
                                public_name='lower and upper radar wavelength for ionosphere correction',
                                default=None,
                                type=float,
                                mandatory=False,
                                container = list,
                                doc="lower and upper radar wavelength for ionosphere correction")

RADAR_DEM_AFFINE_TRANSFORM = Component.Parameter('radarDemAffineTransform',
                                public_name = 'radar dem affine transform parameters',
                                default = None,
                                type = float,
                                mandatory = True,
                                container = list,
                                doc = 'radar dem affine transform parameters')

REFERENCE_SLC = Component.Parameter('referenceSlc',
                                public_name='reference slc',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='reference slc file')

SECONDARY_SLC = Component.Parameter('secondarySlc',
                                public_name='secondary slc',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='secondary slc file')

REFERENCE_BURST_PREFIX = Component.Parameter('referenceBurstPrefix',
                                public_name='reference burst prefix',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='reference burst prefix')

SECONDARY_BURST_PREFIX = Component.Parameter('secondaryBurstPrefix',
                                public_name='secondary burst prefix',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='secondary burst prefix')

REFERENCE_MAGNITUDE = Component.Parameter('referenceMagnitude',
                                public_name='reference magnitude',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='reference magnitude file')

SECONDARY_MAGNITUDE = Component.Parameter('secondaryMagnitude',
                                public_name='secondary magnitude',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='secondary magnitude file')

REFERENCE_SWATH_OFFSET = Component.Parameter('referenceSwathOffset',
                                public_name='reference swath offset',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='reference swath offset file')

SECONDARY_SWATH_OFFSET = Component.Parameter('secondarySwathOffset',
                                public_name='secondary swath offset',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='secondary swath offset file')

REFERENCE_FRAME_OFFSET = Component.Parameter('referenceFrameOffset',
                                public_name='reference frame offset',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='reference frame offset file')

SECONDARY_FRAME_OFFSET = Component.Parameter('secondaryFrameOffset',
                                public_name='secondary frame offset',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='secondary frame offset file')

REFERENCE_FRAME_PARAMETER = Component.Parameter('referenceFrameParameter',
                                public_name='reference frame parameter',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='reference frame parameter file')

SECONDARY_FRAME_PARAMETER = Component.Parameter('secondaryFrameParameter',
                                public_name='secondary frame parameter',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='secondary frame parameter file')

REFERENCE_TRACK_PARAMETER = Component.Parameter('referenceTrackParameter',
                                public_name='reference track parameter',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='reference track parameter file')

SECONDARY_TRACK_PARAMETER = Component.Parameter('secondaryTrackParameter',
                                public_name='secondary track parameter',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='secondary track parameter file')

DEM = Component.Parameter('dem',
                                public_name='dem for coregistration',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='dem for coregistration file')

DEM_GEO = Component.Parameter('demGeo',
                                public_name='dem for geocoding',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='dem for geocoding file')

WBD = Component.Parameter('wbd',
                                public_name='water body',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='water body file')

WBD_OUT = Component.Parameter('wbdOut',
                                public_name='output water body',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='output water body file')

INTERFEROGRAM = Component.Parameter('interferogram',
                                public_name='interferogram',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='interferogram file')

AMPLITUDE = Component.Parameter('amplitude',
                                public_name='amplitude',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='amplitude file')

DIFFERENTIAL_INTERFEROGRAM = Component.Parameter('differentialInterferogram',
                                public_name='differential interferogram',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='differential interferogram file')

MULTILOOK_DIFFERENTIAL_INTERFEROGRAM = Component.Parameter('multilookDifferentialInterferogram',
                                public_name='multilook differential interferogram',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='multilook differential interferogram file')

MULTILOOK_DIFFERENTIAL_INTERFEROGRAM_ORIGINAL = Component.Parameter('multilookDifferentialInterferogramOriginal',
                                public_name='original multilook differential interferogram',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='original multilook differential interferogram file')

MULTILOOK_AMPLITUDE = Component.Parameter('multilookAmplitude',
                                public_name='multilook amplitude',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='multilook amplitude file')

MULTILOOK_COHERENCE = Component.Parameter('multilookCoherence',
                                public_name='multilook coherence',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='multilook coherence file')

MULTILOOK_PHSIG = Component.Parameter('multilookPhsig',
                                public_name='multilook phase sigma',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='multilook phase sigma file')

FILTERED_INTERFEROGRAM = Component.Parameter('filteredInterferogram',
                                public_name='filtered interferogram',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='filtered interferogram file')

UNWRAPPED_INTERFEROGRAM = Component.Parameter('unwrappedInterferogram',
                                public_name='unwrapped interferogram',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='unwrapped interferogram file')

UNWRAPPED_MASKED_INTERFEROGRAM = Component.Parameter('unwrappedMaskedInterferogram',
                                public_name='unwrapped masked interferogram',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='unwrapped masked interferogram file')

LATITUDE = Component.Parameter('latitude',
                                public_name='latitude',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='latitude file')

LONGITUDE = Component.Parameter('longitude',
                                public_name='longitude',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='longitude file')

HEIGHT = Component.Parameter('height',
                                public_name='height',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='height file')

LOS = Component.Parameter('los',
                                public_name='los',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='los file')

SIM = Component.Parameter('sim',
                                public_name='sim',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='sim file')

MSK = Component.Parameter('msk',
                                public_name='msk',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='msk file')

RANGE_OFFSET = Component.Parameter('rangeOffset',
                                public_name='range offset',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='range offset file')

AZIMUTH_OFFSET = Component.Parameter('azimuthOffset',
                                public_name='azimuth offset',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='azimuth offset file')


MULTILOOK_LOS = Component.Parameter('multilookLos',
                                public_name='multilook los',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='multilook los file')

MULTILOOK_MSK = Component.Parameter('multilookMsk',
                                public_name='multilook msk',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='multilook msk file')

MULTILOOK_WBD_OUT = Component.Parameter('multilookWbdOut',
                                public_name='multilook wbdOut',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='multilook output water body file')

MULTILOOK_LATITUDE = Component.Parameter('multilookLatitude',
                                public_name='multilook latitude',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='multilook latitude file')

MULTILOOK_LONGITUDE = Component.Parameter('multilookLongitude',
                                public_name='multilook longitude',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='multilook longitude file')

MULTILOOK_HEIGHT = Component.Parameter('multilookHeight',
                                public_name='multilook height',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='multilook height file')

MULTILOOK_ION = Component.Parameter('multilookIon',
                                public_name='multilook ionospheric phase',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='multilook ionospheric phase file')

RECT_RANGE_OFFSET = Component.Parameter('rectRangeOffset',
                                public_name='rectified range offset',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='rectified range offset file')

GEO_INTERFEROGRAM = Component.Parameter('geoInterferogram',
                                public_name='geocoded interferogram',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='geocoded interferogram file')

GEO_MASKED_INTERFEROGRAM = Component.Parameter('geoMaskedInterferogram',
                                public_name='geocoded masked interferogram',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='geocoded masked interferogram file')

GEO_COHERENCE = Component.Parameter('geoCoherence',
                                public_name='geocoded coherence',
                                default=None,
                                type=str,
                                mandatory=False,
                                container = list,
                                doc='geocoded coherence file')

GEO_LOS = Component.Parameter('geoLos',
                                public_name='geocoded los',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='geocoded los file')

GEO_ION = Component.Parameter('geoIon',
                                public_name='geocoded ionospheric phase',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='geocoded ionospheric phase file')
###################################################################

#spectral diversity
INTERFEROGRAM_SD = Component.Parameter('interferogramSd',
                                public_name='spectral diversity interferograms',
                                default=None,
                                type=str,
                                mandatory=False,
                                container = list,
                                doc='spectral diversity interferogram files')

MULTILOOK_INTERFEROGRAM_SD = Component.Parameter('multilookInterferogramSd',
                                public_name='multilook spectral diversity interferograms',
                                default=None,
                                type=str,
                                mandatory=False,
                                container = list,
                                doc='multilook spectral diversity interferogram files')

MULTILOOK_COHERENCE_SD = Component.Parameter('multilookCoherenceSd',
                                public_name='multilook coherence for spectral diversity',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='multilook coherence for spectral diversity file')

FILTERED_INTERFEROGRAM_SD = Component.Parameter('filteredInterferogramSd',
                                public_name='filtered spectral diversity interferograms',
                                default=None,
                                type=str,
                                mandatory=False,
                                container = list,
                                doc='filtered spectral diversity interferogram files')

UNWRAPPED_INTERFEROGRAM_SD = Component.Parameter('unwrappedInterferogramSd',
                                public_name='unwrapped spectral diversity interferograms',
                                default=None,
                                type=str,
                                mandatory=False,
                                container = list,
                                doc='unwrapped spectral diversity interferogram files')

UNWRAPPED_MASKED_INTERFEROGRAM_SD = Component.Parameter('unwrappedMaskedInterferogramSd',
                                public_name='unwrapped masked spectral diversity interferograms',
                                default=None,
                                type=str,
                                mandatory=False,
                                container = list,
                                doc='unwrapped masked spectral diversity interferogram files')

AZIMUTH_DEFORMATION_SD = Component.Parameter('azimuthDeformationSd',
                                public_name='azimuth deformation',
                                default=None,
                                type=str,
                                mandatory=False,
                                container = list,
                                doc='azimuth deformation files')

MASKED_AZIMUTH_DEFORMATION_SD = Component.Parameter('maskedAzimuthDeformationSd',
                                public_name='masked azimuth deformation',
                                default=None,
                                type=str,
                                mandatory=False,
                                container = list,
                                doc='masked azimuth deformation files')

MULTILOOK_WBD_OUT_SD = Component.Parameter('multilookWbdOutSd',
                                public_name='multilook wbdOut for SD',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='multilook output water body for SD file')

MULTILOOK_LATITUDE_SD = Component.Parameter('multilookLatitudeSd',
                                public_name='multilook latitude for SD',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='multilook latitude for SD file')

MULTILOOK_LONGITUDE_SD = Component.Parameter('multilookLongitudeSd',
                                public_name='multilook longitude for SD',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='multilook longitude for SD file')

GEO_COHERENCE_SD = Component.Parameter('geoCoherenceSd',
                                public_name='geocoded coherence for spectral diversity',
                                default=None,
                                type=str,
                                mandatory=False,
                                container = list,
                                doc='geocoded coherence for spectral diversity file')

GEO_AZIMUTH_DEFORMATION_SD = Component.Parameter('geoAzimuthDeformationSd',
                                public_name='geocoded azimuth deformation',
                                default=None,
                                type=str,
                                mandatory=False,
                                container = list,
                                doc='geocoded azimuth deformation files')

GEO_MASKED_AZIMUTH_DEFORMATION_SD = Component.Parameter('geoMaskedAzimuthDeformationSd',
                                public_name='geocoded masked azimuth deformation',
                                default=None,
                                type=str,
                                mandatory=False,
                                container = list,
                                doc='geocoded masked azimuth deformation files')

###################################################################
class Alos2burstProc(Component):
    """
    This class holds the properties, along with methods (setters and getters)
    to modify and return their values.
    """

    parameter_list = (REFERENCE_DATE,
                      SECONDARY_DATE,
                      MODE_COMBINATION,
                      REFERENCE_FRAMES,
                      SECONDARY_FRAMES,
                      STARTING_SWATH,
                      ENDING_SWATH,
                      BURST_UNSYNCHRONIZED_TIME,
                      BURST_SYNCHRONIZATION,
                      RANGE_RESIDUAL_OFFSET_CC,
                      AZIMUTH_RESIDUAL_OFFSET_CC,
                      RANGE_RESIDUAL_OFFSET_SD,
                      AZIMUTH_RESIDUAL_OFFSET_SD,
                      SWATH_RANGE_OFFSET_GEOMETRICAL_REFERENCE,
                      SWATH_AZIMUTH_OFFSET_GEOMETRICAL_REFERENCE,
                      SWATH_RANGE_OFFSET_MATCHING_REFERENCE,
                      SWATH_AZIMUTH_OFFSET_MATCHING_REFERENCE,
                      SWATH_RANGE_OFFSET_GEOMETRICAL_SECONDARY,
                      SWATH_AZIMUTH_OFFSET_GEOMETRICAL_SECONDARY,
                      SWATH_RANGE_OFFSET_MATCHING_SECONDARY,
                      SWATH_AZIMUTH_OFFSET_MATCHING_SECONDARY,
                      FRAME_RANGE_OFFSET_GEOMETRICAL_REFERENCE,
                      FRAME_AZIMUTH_OFFSET_GEOMETRICAL_REFERENCE,
                      FRAME_RANGE_OFFSET_MATCHING_REFERENCE,
                      FRAME_AZIMUTH_OFFSET_MATCHING_REFERENCE,
                      FRAME_RANGE_OFFSET_GEOMETRICAL_SECONDARY,
                      FRAME_AZIMUTH_OFFSET_GEOMETRICAL_SECONDARY,
                      FRAME_RANGE_OFFSET_MATCHING_SECONDARY,
                      FRAME_AZIMUTH_OFFSET_MATCHING_SECONDARY,
                      NUMBER_RANGE_LOOKS1,
                      NUMBER_AZIMUTH_LOOKS1,
                      NUMBER_RANGE_LOOKS2,
                      NUMBER_AZIMUTH_LOOKS2,
                      NUMBER_RANGE_LOOKS_SIM,
                      NUMBER_AZIMUTH_LOOKS_SIM,
                      NUMBER_RANGE_LOOKS_ION,
                      NUMBER_AZIMUTH_LOOKS_ION,
                      NUMBER_RANGE_LOOKS_SD,
                      NUMBER_AZIMUTH_LOOKS_SD,
                      SUBBAND_RADAR_WAVLENGTH,
                      RADAR_DEM_AFFINE_TRANSFORM,
                      REFERENCE_SLC,
                      SECONDARY_SLC,
                      REFERENCE_BURST_PREFIX,
                      SECONDARY_BURST_PREFIX,
                      REFERENCE_MAGNITUDE,
                      SECONDARY_MAGNITUDE,
                      REFERENCE_SWATH_OFFSET,
                      SECONDARY_SWATH_OFFSET,
                      REFERENCE_FRAME_OFFSET,
                      SECONDARY_FRAME_OFFSET,
                      REFERENCE_FRAME_PARAMETER,
                      SECONDARY_FRAME_PARAMETER,
                      REFERENCE_TRACK_PARAMETER,
                      SECONDARY_TRACK_PARAMETER,
                      DEM,
                      DEM_GEO,
                      WBD,
                      WBD_OUT,
                      INTERFEROGRAM,
                      AMPLITUDE,
                      DIFFERENTIAL_INTERFEROGRAM,
                      MULTILOOK_DIFFERENTIAL_INTERFEROGRAM,
                      MULTILOOK_DIFFERENTIAL_INTERFEROGRAM_ORIGINAL,
                      MULTILOOK_AMPLITUDE,
                      MULTILOOK_COHERENCE,
                      MULTILOOK_PHSIG,
                      FILTERED_INTERFEROGRAM,
                      UNWRAPPED_INTERFEROGRAM,
                      UNWRAPPED_MASKED_INTERFEROGRAM,
                      LATITUDE,
                      LONGITUDE,
                      HEIGHT,
                      LOS,
                      SIM,
                      MSK,
                      RANGE_OFFSET,
                      AZIMUTH_OFFSET,
                      MULTILOOK_LOS,
                      MULTILOOK_MSK,
                      MULTILOOK_WBD_OUT,
                      MULTILOOK_LATITUDE,
                      MULTILOOK_LONGITUDE,
                      MULTILOOK_HEIGHT,
                      MULTILOOK_ION,
                      RECT_RANGE_OFFSET,
                      GEO_INTERFEROGRAM,
                      GEO_MASKED_INTERFEROGRAM,
                      GEO_COHERENCE,
                      GEO_LOS,
                      GEO_ION,
                      #spectral diversity
                      INTERFEROGRAM_SD,
                      MULTILOOK_INTERFEROGRAM_SD,
                      MULTILOOK_COHERENCE_SD,
                      FILTERED_INTERFEROGRAM_SD,
                      UNWRAPPED_INTERFEROGRAM_SD,
                      UNWRAPPED_MASKED_INTERFEROGRAM_SD,
                      AZIMUTH_DEFORMATION_SD,
                      MASKED_AZIMUTH_DEFORMATION_SD,
                      MULTILOOK_WBD_OUT_SD,
                      MULTILOOK_LATITUDE_SD,
                      MULTILOOK_LONGITUDE_SD,
                      GEO_COHERENCE_SD,
                      GEO_AZIMUTH_DEFORMATION_SD,
                      GEO_MASKED_AZIMUTH_DEFORMATION_SD)

    facility_list = ()


    family='alos2burstcontext'

    def __init__(self, name='', procDoc=None):
        #self.updatePrivate()

        super().__init__(family=self.__class__.family, name=name)
        self.procDoc = procDoc
        return None

    def setFilename(self, referenceDate, secondaryDate, nrlks1, nalks1, nrlks2, nalks2):

        # if referenceDate == None:
        #     referenceDate = self.referenceDate
        # if secondaryDate == None:
        #     secondaryDate = self.secondaryDate
        # if nrlks1 == None:
        #     nrlks1 = self.numberRangeLooks1
        # if nalks1 == None:
        #     nalks1 = self.numberAzimuthLooks1
        # if nrlks2 == None:
        #     nrlks2 = self.numberRangeLooks2
        # if nalks2 == None:
        #     nalks2 = self.numberAzimuthLooks2

        ms = referenceDate + '-' + secondaryDate
        ml1 = '_{}rlks_{}alks'.format(nrlks1, nalks1)
        ml2 = '_{}rlks_{}alks'.format(nrlks1*nrlks2, nalks1*nalks2)

        self.referenceSlc = referenceDate + '.slc'
        self.secondarySlc = secondaryDate + '.slc'
        self.referenceBurstPrefix = referenceDate
        self.secondaryBurstPrefix = secondaryDate
        self.referenceMagnitude = referenceDate + '.mag'
        self.secondaryMagnitude = secondaryDate + '.mag'
        self.referenceSwathOffset = 'swath_offset_' + referenceDate + '.txt'
        self.secondarySwathOffset = 'swath_offset_' + secondaryDate + '.txt'
        self.referenceFrameOffset = 'frame_offset_' + referenceDate + '.txt'
        self.secondaryFrameOffset = 'frame_offset_' + secondaryDate + '.txt'
        self.referenceFrameParameter = referenceDate + '.frame.xml'
        self.secondaryFrameParameter = secondaryDate + '.frame.xml'
        self.referenceTrackParameter = referenceDate + '.track.xml'
        self.secondaryTrackParameter = secondaryDate + '.track.xml'
        #self.dem = 
        #self.demGeo = 
        #self.wbd = 
        self.interferogram = ms + ml1 + '.int'
        self.amplitude = ms + ml1 + '.amp'
        self.differentialInterferogram = 'diff_' + ms + ml1 + '.int'
        self.multilookDifferentialInterferogram = 'diff_' + ms + ml2 + '.int'
        self.multilookDifferentialInterferogramOriginal = 'diff_' + ms + ml2 + '_ori.int'
        self.multilookAmplitude = ms + ml2 + '.amp'
        self.multilookCoherence = ms + ml2 + '.cor'
        self.multilookPhsig = ms + ml2 + '.phsig'
        self.filteredInterferogram = 'filt_' + ms + ml2 + '.int'
        self.unwrappedInterferogram = 'filt_' + ms + ml2 + '.unw'
        self.unwrappedMaskedInterferogram = 'filt_' + ms + ml2 + '_msk.unw'
        self.latitude = ms + ml1 + '.lat'
        self.longitude = ms + ml1 + '.lon'
        self.height = ms + ml1 + '.hgt'
        self.los = ms + ml1 + '.los'
        self.sim = ms + ml1 + '.sim'
        self.msk = ms + ml1 + '.msk'
        self.wbdOut = ms + ml1 + '.wbd'
        self.rangeOffset = ms + ml1 + '_rg.off'
        self.azimuthOffset = ms + ml1 + '_az.off'
        self.multilookLos = ms + ml2 + '.los'
        self.multilookWbdOut = ms + ml2 + '.wbd'
        self.multilookMsk = ms + ml2 + '.msk'
        self.multilookLatitude = ms + ml2 + '.lat'
        self.multilookLongitude = ms + ml2 + '.lon'
        self.multilookHeight = ms + ml2 + '.hgt'
        self.multilookIon = ms + ml2 + '.ion'
        self.rectRangeOffset = ms + ml1 + '_rg_rect.off'
        self.geoInterferogram = 'filt_' + ms + ml2 + '.unw.geo'
        self.geoMaskedInterferogram = 'filt_' + ms + ml2 + '_msk.unw.geo'
        self.geoCoherence = ms + ml2 + '.cor.geo'
        self.geoLos = ms + ml2 + '.los.geo'
        self.geoIon = ms + ml2 + '.ion.geo'


    def setFilenameSd(self, referenceDate, secondaryDate, nrlks1, nalks1, nrlks_sd, nalks_sd, nsd=3):
        #spectral diversity
        # if referenceDate == None:
        #     referenceDate = self.referenceDate
        # if secondaryDate == None:
        #     secondaryDate = self.secondaryDate
        # if nrlks1 == None:
        #     nrlks1 = self.numberRangeLooks1
        # if nalks1 == None:
        #     nalks1 = self.numberAzimuthLooks1
        # if nrlks_sd == None:
        #     nrlks_sd = self.numberRangeLooksSd
        # if nalks_sd == None:
        #     nalks_sd = self.numberAzimuthLooksSd

        ms = referenceDate + '-' + secondaryDate
        ml1 = '_{}rlks_{}alks'.format(nrlks1, nalks1)
        ml2sd = '_{}rlks_{}alks'.format(nrlks1*nrlks_sd, nalks1*nalks_sd)
        self.interferogramSd = ['sd_{}_'.format(i+1) + ms + ml1 + '.int' for i in range(nsd)]
        self.multilookInterferogramSd = ['sd_{}_'.format(i+1) + ms + ml2sd + '.int' for i in range(nsd)]
        self.multilookCoherenceSd = ['filt_{}_'.format(i+1) + ms + ml2sd + '.cor' for i in range(nsd)]
        self.filteredInterferogramSd = ['filt_{}_'.format(i+1) + ms + ml2sd + '.int' for i in range(nsd)]
        self.unwrappedInterferogramSd = ['filt_{}_'.format(i+1) + ms + ml2sd + '.unw' for i in range(nsd)]
        self.unwrappedMaskedInterferogramSd = ['filt_{}_'.format(i+1) + ms + ml2sd + '_msk.unw' for i in range(nsd)]
        self.azimuthDeformationSd = ['azd_{}_'.format(i+1) + ms + ml2sd + '.unw' for i in range(nsd)]
        self.azimuthDeformationSd.append('azd_' + ms + ml2sd + '.unw')
        self.maskedAzimuthDeformationSd = ['azd_{}_'.format(i+1) + ms + ml2sd + '_msk.unw' for i in range(nsd)]
        self.maskedAzimuthDeformationSd.append('azd_' + ms + ml2sd + '_msk.unw')
        self.multilookWbdOutSd = ms + ml2sd + '.wbd'
        self.multilookLatitudeSd = ms + ml2sd + '.lat'
        self.multilookLongitudeSd = ms + ml2sd + '.lon'
        self.geoCoherenceSd = ['filt_{}_'.format(i+1) + ms + ml2sd + '.cor.geo' for i in range(nsd)]
        self.geoAzimuthDeformationSd = ['azd_{}_'.format(i+1) + ms + ml2sd + '.unw.geo' for i in range(nsd)]
        self.geoAzimuthDeformationSd.append('azd_' + ms + ml2sd + '.unw.geo')
        self.geoMaskedAzimuthDeformationSd = ['azd_{}_'.format(i+1) + ms + ml2sd + '_msk.unw.geo' for i in range(nsd)]
        self.geoMaskedAzimuthDeformationSd.append('azd_' + ms + ml2sd + '_msk.unw.geo')


    def loadProduct(self, xmlname):
        '''
        Load the product using Product Manager.
        '''

        from iscesys.Component.ProductManager import ProductManager as PM

        pm = PM()
        pm.configure()

        obj = pm.loadProduct(xmlname)

        return obj


    def saveProduct(self, obj, xmlname):
        '''
        Save the product to an XML file using Product Manager.
        '''
        
        from iscesys.Component.ProductManager import ProductManager as PM

        pm = PM()
        pm.configure()

        pm.dumpProduct(obj, xmlname)
        
        return None


    def loadTrack(self, reference=True):
        '''
        Load the track using Product Manager.
        '''
        if reference:
            track = self.loadProduct(self.referenceTrackParameter)
        else:
            track = self.loadProduct(self.secondaryTrackParameter)

        track.frames = []
        for i, frameNumber in enumerate(self.referenceFrames):
            os.chdir('f{}_{}'.format(i+1, frameNumber))
            if reference:
                track.frames.append(self.loadProduct(self.referenceFrameParameter))
            else:
                track.frames.append(self.loadProduct(self.secondaryFrameParameter))
            os.chdir('../')

        return track


    def saveTrack(self, track, reference=True):
        '''
        Save the track to XML files using Product Manager.
        '''
        if reference:
            self.saveProduct(track, self.referenceTrackParameter)
        else:
            self.saveProduct(track, self.secondaryTrackParameter)

        for i, frameNumber in enumerate(self.referenceFrames):
            os.chdir('f{}_{}'.format(i+1, frameNumber))
            if reference:
                self.saveProduct(track.frames[i], self.referenceFrameParameter)
            else:
                self.saveProduct(track.frames[i], self.secondaryFrameParameter)
            os.chdir('../')

        return None


    def hasGPU(self):
        '''
        Determine if GPU modules are available.
        '''

        flag = False
        try:
            from zerodop.GPUtopozero.GPUtopozero import PyTopozero
            from zerodop.GPUgeo2rdr.GPUgeo2rdr import PyGeo2rdr
            flag = True
        except:
            pass

        return flag

