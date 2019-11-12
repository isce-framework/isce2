#
# Author: Heresh Fattahi
# Copyright 2017
#


from __future__ import print_function
import os
import logging
import logging.config
from iscesys.Component.Component import Component
from iscesys.DateTimeUtil.DateTimeUtil import DateTimeUtil as DTU
from iscesys.Compatibility import Compatibility
from isceobj.Scene.Frame import FrameMixin

## Master Slave Hash Table
MASTER_SLAVE = {0:'master', 1:'slave', 'master':'master', 'slave':'slave'}


FIRST_SAMPLE_ACROSS = Component.Parameter('firstSampleAcross',
                                public_name='first sample across',
                                default=50,
                                type=int,
                                mandatory=False,
                                doc='')


FIRST_SAMPLE_DOWN = Component.Parameter('first sample down',
                                public_name='firstSampleDown',
                                default=50,
                                type=int,
                                mandatory=False,
                                doc='')


NUMBER_LOCATION_ACROSS = Component.Parameter('numberLocationAcross',
                                public_name='number location across',
                                default=40,
                                type=int,
                                mandatory=False,
                                doc='')


NUMBER_LOCATION_DOWN = Component.Parameter('numberLocationDown',
                                public_name='number location down',
                                default=40,
                                type=int,
                                mandatory=False,
                                doc='')

MASTER_RAW_PRODUCT = Component.Parameter('masterRawProduct',
                                public_name = 'master raw product',
                                default = None,
                                type = str,
                                mandatory = False,
                                doc = 'master raw product xml name')

SLAVE_RAW_PRODUCT = Component.Parameter('slaveRawProduct',
                                public_name = 'slave raw product',
                                default = None,
                                type = str,
                                mandatory = False,
                                doc = 'slave raw product xml name')

MASTER_RAW_CROP_PRODUCT = Component.Parameter('masterRawCropProduct',
                                public_name = 'master raw cropped product',
                                default = None,
                                type = str,
                                mandatory = False,
                                doc = 'master raw cropped product xml name')

SLAVE_RAW_CROP_PRODUCT = Component.Parameter('slaveRawCropProduct',
                                public_name = 'slave raw cropped product',
                                default = None,
                                type = str,
                                mandatory = False,
                                doc = 'slave raw cropped product xml name')
MASTER_SLC_PRODUCT = Component.Parameter('masterSlcProduct',
                                public_name = 'master slc product',
                                default = None,
                                type = str,
                                mandatory = False,
                                doc = 'master slc product xml name')

SLAVE_SLC_PRODUCT = Component.Parameter('slaveSlcProduct',
                                public_name = 'slave slc product',
                                default = None,
                                type = str,
                                mandatory = False,
                                doc = 'slave slc product xml name')

MASTER_SLC_CROP_PRODUCT = Component.Parameter('masterSlcCropProduct',
                                public_name = 'master slc cropped product',
                                default = None,
                                type = str,
                                mandatory = False,
                                doc = 'master slc cropped product xml name')

SLAVE_SLC_CROP_PRODUCT = Component.Parameter('slaveSlcCropProduct',
                                public_name = 'slave slc cropped product',
                                default = None,
                                type = str,
                                mandatory = False,
                                doc = 'slave slc cropped product xml name')


MASTER_GEOMETRY_SYSTEM = Component.Parameter('masterGeometrySystem',
                                public_name = 'master geometry system',
                                default = None,
                                type = str,
                                mandatory = False,
                                doc = 'zero doppler or native doppler')

SLAVE_GEOMETRY_SYSTEM = Component.Parameter('slaveGeometrySystem',
                                public_name = 'slave geometry system',
                                default = None,
                                type = str,
                                mandatory = False,
                                doc = 'zero doppler or native doppler')

GEOMETRY_DIRECTORY = Component.Parameter('geometryDirname',
                                public_name = 'geometry directory name',
                                default = 'geometry',
                                type = str,
                                mandatory = False,
                                doc = 'geometry directory name') 

OFFSETS_DIRECTORY = Component.Parameter('offsetsDirname',
                                public_name = 'offsets directory name',
                                default = 'offsets',
                                type = str,
                                mandatory = False,
                                doc = 'offsets directory name') 

DENSE_OFFSETS_DIRECTORY = Component.Parameter('denseOffsetsDirname',
                                public_name = 'dense offsets directory name',
                                default = 'denseOffsets',
                                type = str,
                                mandatory = False,
                                doc = 'directory name for dense offsets computed from cross correlating two SLC imaged')

COREG_DIRECTORY = Component.Parameter('coregDirname',
                                public_name = 'coreg slc directory name',
                                default = 'coregisteredSlc',
                                type = str,
                                mandatory = False,
                                doc = 'directory that contains coregistered slc') 

COARSE_COREG_FILENAME = Component.Parameter('coarseCoregFilename',
                                public_name = 'coarse coreg slc filename',
                                default='coarse_coreg.slc',
                                type = str,
                                mandatory = False,
                                doc = 'coarse coreg slc name')

REFINED_COREG_FILENAME = Component.Parameter('refinedCoregFilename',
                                public_name = 'refined coreg slc filename',
                                default = 'refined_coreg.slc',
                                type = str,
                                mandatory = False,
                                doc = 'refined coreg slc name')

FINE_COREG_FILENAME = Component.Parameter('fineCoregFilename',
                                public_name='fine coreg slc filename',
                                default='fine_coreg.slc',
                                type = str,
                                mandatory = False,
                                doc = 'fine coreg slc name')

IFG_DIRECTORY = Component.Parameter('ifgDirname',
                                public_name = 'interferogram directory name',
                                default = 'interferogram',
                                type = str,
                                mandatory = False,
                                doc = 'interferogram directory name') 

MISREG_DIRECTORY = Component.Parameter('misregDirname',
                                public_name = 'misregistration directory name',
                                default = 'misreg',
                                type = str,
                                mandatory = False,
                                doc = 'misregistration directory name') 

SPLIT_SPECTRUM_DIRECTORY = Component.Parameter('splitSpectrumDirname',
                                public_name = 'split spectrum directory name',
                                default = 'SplitSpectrum',
                                type=str,
                                mandatory=False,
                                doc = 'split spectrum directory name')

LOWBAND_SLC_DIRECTORY = Component.Parameter('lowBandSlcDirname',
                                public_name = 'low band slc directory name',
                                default = 'lowBand',
                                type = str,
                                mandatory = False,
                                doc = 'directory that contains low-band SLCs after splitting their range spectrum')

IONOSPHERE_DIRECTORY = Component.Parameter('ionosphereDirname',
                                public_name='ionosphere directory',
                                default = 'ionosphere',
                                type=str,
                                mandatory=False,
                                doc = 'directory that contains split spectrum computations')

LOWBAND_RADAR_WAVELENGTH = Component.Parameter('lowBandRadarWavelength',
                                public_name = 'low band radar wavelength',
                                default = None,
                                type = float,
                                mandatory = False,
                                doc = '')


HIGHBAND_SLC_DIRECTORY = Component.Parameter('highBandSlcDirname',
                                public_name = 'high band slc directory name',
                                default = 'highBand',
                                type = str,
                                mandatory = False,                                
                                doc = 'directory that contains high-band SLCs after splitting their range spectrum')

HIGHBAND_RADAR_WAVELENGTH = Component.Parameter('highBandRadarWavelength',
                                public_name = 'high band radar wavelength',
                                default = None,
                                type = float,
                                mandatory = False,
                                doc = '')

COHERENCE_FILENAME = Component.Parameter('coherenceFilename',
                                public_name='coherence name',
                                default='phsig.cor',
                                type=str,
                                mandatory=False,
                                doc='Coherence file name')


CORRELATION_FILENAME = Component.Parameter('correlationFilename',
                                public_name = 'correlation name',
                                default = 'topophase.cor',
                                type = str,
                                mandatory = False,
                                doc = 'Correlation file name')

IFG_FILENAME = Component.Parameter('ifgFilename',
                                public_name='interferogram name',
                                default='topophase.flat',
                                type=str,
                                mandatory=False,
                                doc='Filename of the interferogram')


FILTERED_IFG_FILENAME = Component.Parameter('filtIfgFilename',
                                public_name = 'filtered interferogram name',
                                default = 'filt_topophase.flat',
                                type = str,
                                mandatory = False,
                                doc = 'Filtered interferogram filename')

UNWRAPPED_IFG_FILENAME = Component.Parameter('unwrappedIfgFilename',
                                public_name='unwrapped interferogram name',
                                default='filt_topophase.unw',
                                type=str,
                                mandatory=False,
                                doc='Unwrapped interferogram file name ')


CONNECTED_COMPONENTS_FILENAME = Component.Parameter('connectedComponentsFilename',
                                public_name='connected component filename',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='')


HEIGHT_FILENAME = Component.Parameter('heightFilename',
                                public_name='height file name',
                                default='z.rdr',
                                type=str,
                                mandatory=False,
                                doc='height file name')


GEOCODE_FILENAME = Component.Parameter('geocodeFilename',
                                public_name='geocode file name',
                                default='topophase.geo',
                                type=str,
                                mandatory=False,
                                doc='')


LOS_FILENAME = Component.Parameter('losFilename',
                                public_name='los file name',
                                default='los.rdr',
                                type=str,
                                mandatory=False,
                                doc='')


LAT_FILENAME = Component.Parameter('latFilename',
                                public_name='lat file name',
                                default='lat.rdr',
                                type=str,
                                mandatory=False,
                                doc='')


LON_FILENAME = Component.Parameter('lonFilename',
                                public_name='lon file name',
                                default='lon.rdr',
                                type=str,
                                mandatory=False,
                                doc='')


RANGE_OFFSET_FILENAME = Component.Parameter('rangeOffsetFilename',
                                public_name='range Offset Image Name',
                                default='range.off',
                                type=str,
                                mandatory=False,
                                doc='')

AZIMUTH_OFFSET_FILENAME = Component.Parameter('azimuthOffsetFilename',
                                public_name='azimuth Offset Image Name',
                                default='azimuth.off',
                                type=str,
                                mandatory=False,
                                doc='')


# Modified by V. Brancato 10.07.2019
AZIMUTH_RUBBERSHEET_FILENAME = Component.Parameter('azimuthRubbersheetFilename',
                                public_name='azimuth Rubbersheet Image Name',
                                default = 'azimuth_sheet.off',
                                type=str,
                                mandatory=False,
                                doc='')
				
RANGE_RUBBERSHEET_FILENAME = Component.Parameter('rangeRubbersheetFilename',
                                public_name='range Rubbersheet Image Name',
                                default = 'range_sheet.off',
                                type=str,
                                mandatory=False,
                                doc='')
# End of modification
MISREG_FILENAME = Component.Parameter('misregFilename',
                                public_name='misreg file name',
                                default='misreg',
                                type=str,
                                mandatory=False,
                                doc='misregistration file name')

DENSE_OFFSET_FILENAME = Component.Parameter('denseOffsetFilename',
                                public_name='dense Offset file name',
                                default='denseOffsets',
                                type=str,
                                mandatory=False,
                                doc='file name of dense offsets computed from cross correlating two SLC images')
# Modified by V. Brancato 10.07.2019
FILT_AZIMUTH_OFFSET_FILENAME = Component.Parameter('filtAzimuthOffsetFilename',
                                public_name='filtered azimuth offset filename',
                                default='filtAzimuth.off',
                                type=str,
                                mandatory=False,
                                doc='Filtered azimuth dense offsets')
				
FILT_RANGE_OFFSET_FILENAME = Component.Parameter('filtRangeOffsetFilename',
                                public_name='filtered range offset filename',
                                default='filtRange.off',
                                type=str,
                                mandatory=False,
                                doc='Filtered range dense offsets')
# End of modification
DISPERSIVE_FILENAME = Component.Parameter('dispersiveFilename',
                                public_name = 'dispersive phase filename',
                                default='dispersive.bil',
                                type=str,
                                mandatory=False,
                                doc='Dispersive phase from split spectrum')

NONDISPERSIVE_FILENAME = Component.Parameter('nondispersiveFilename',
                                public_name='nondispersive phase filename',
                                default='nondispersive.bil',
                                type=str,
                                mandatory=False,
                                doc='Non dispersive phase from split spectrum')


OFFSET_TOP = Component.Parameter(
    'offset_top',
    public_name='Top offset location',
    default=None,
    type=int,
    mandatory=False,
    doc='Ampcor-calculated top offset location. Overridden by workflow.'
                                    )

OFFSET_LEFT = Component.Parameter(
    'offset_left',
    public_name='Left offset location',
    default=None,
    type=int,
    mandatory=False,
    doc='Ampcor-calculated left offset location. Overridden by workflow.')

DEM_FILENAME = Component.Parameter('demFilename',
                                public_name='dem image name',
                                default = None,
                                type = str,
                                mandatory = False,
                                doc = 'Name of the dem file')

DEM_CROP_FILENAME = Component.Parameter('demCropFilename',
                                public_name='dem crop filename',
                                default='dem.crop',
                                type=str,
                                mandatory=False,
                                doc='cropped dem file name')


FILTER_STRENGTH = Component.Parameter('filterStrength',
                                public_name='filter Strength',
                                default=0.7,
                                type=float,
                                mandatory=False,
                                doc='')

SECONDARY_RANGE_MIGRATION_FLAG = Component.Parameter('secondaryRangeMigrationFlag',
                                public_name='secondaryRangeMigrationFlag',
                                default=None,
                                type=str,
                                mandatory=False,
                                doc='')


ESTIMATED_BBOX = Component.Parameter('estimatedBbox',
                                public_name='Estimated bounding box',
                                default=None,
                                type = float,
                                container=list,
                                mandatory=False,
                                doc='Bounding box estimated by topo')


class StripmapProc(Component, FrameMixin):
    """
    This class holds the properties, along with methods (setters and getters)
    to modify and return their values.
    """

    parameter_list = (MASTER_RAW_PRODUCT,
                      SLAVE_RAW_PRODUCT,
                      MASTER_RAW_CROP_PRODUCT,
                      SLAVE_RAW_CROP_PRODUCT,
                      MASTER_SLC_PRODUCT,
                      SLAVE_SLC_PRODUCT,
                      MASTER_SLC_CROP_PRODUCT,
                      SLAVE_SLC_CROP_PRODUCT,
                      MASTER_GEOMETRY_SYSTEM,
                      SLAVE_GEOMETRY_SYSTEM,
                      GEOMETRY_DIRECTORY,
                      OFFSETS_DIRECTORY,
                      DENSE_OFFSETS_DIRECTORY,
                      COREG_DIRECTORY,
                      COARSE_COREG_FILENAME,
                      REFINED_COREG_FILENAME,
                      FINE_COREG_FILENAME,
                      IFG_DIRECTORY,
                      MISREG_DIRECTORY,
                      SPLIT_SPECTRUM_DIRECTORY,
                      HIGHBAND_SLC_DIRECTORY,
                      HIGHBAND_RADAR_WAVELENGTH,
                      LOWBAND_SLC_DIRECTORY,
                      IONOSPHERE_DIRECTORY,
                      LOWBAND_RADAR_WAVELENGTH,
                      DEM_FILENAME,
                      DEM_CROP_FILENAME,
                      IFG_FILENAME,
                      FILTERED_IFG_FILENAME,
                      UNWRAPPED_IFG_FILENAME,
                      CONNECTED_COMPONENTS_FILENAME,
                      COHERENCE_FILENAME,
                      CORRELATION_FILENAME,
                      HEIGHT_FILENAME,
                      LAT_FILENAME,
                      LON_FILENAME,
                      LOS_FILENAME,
                      RANGE_OFFSET_FILENAME,
                      AZIMUTH_OFFSET_FILENAME,
                      AZIMUTH_RUBBERSHEET_FILENAME, # Added by V. Brancato 10.07.2019
                      RANGE_RUBBERSHEET_FILENAME,   # Added by V. Brancato 10.07.2019
                      FILT_AZIMUTH_OFFSET_FILENAME, # Added by V. Brancato 10.07.2019
                      FILT_RANGE_OFFSET_FILENAME,   # Added by V. Brancato 10.07.2019
                      DENSE_OFFSET_FILENAME,
                      MISREG_FILENAME,
                      DISPERSIVE_FILENAME,
                      NONDISPERSIVE_FILENAME,
                      OFFSET_TOP,
                      OFFSET_LEFT,
                      FIRST_SAMPLE_ACROSS,
                      FIRST_SAMPLE_DOWN,
                      NUMBER_LOCATION_ACROSS,
                      NUMBER_LOCATION_DOWN,
                      SECONDARY_RANGE_MIGRATION_FLAG,
                      FILTER_STRENGTH,
                      ESTIMATED_BBOX,
                      )

    facility_list = ()

    family='insarcontext'

    def __init__(self, name='', procDoc=None):
        #self.updatePrivate()

        super().__init__(family=self.__class__.family, name=name)
        self.procDoc = procDoc
        return None

    def _init(self):
        """
        Method called after Parameters are configured.
        Determine whether some Parameters still have unresolved
        Parameters as their default values and resolve them.
        """

        #Determine whether the geocode_list still contains Parameters
        #and give those elements the proper value.  This will happen
        #whenever the user doesn't provide as input a geocode_list for
        #this component.
        #for i, x in enumerate(self.geocode_list):
        #    if isinstance(x, Component.Parameter):
        #        y = getattr(self, getattr(x, 'attrname'))
        #        self.geocode_list[i] = y
        return

    def getMasterFrame(self):
        return self._masterFrame

    def getSlaveFrame(self):
        return self._slaveFrame

    def getDemImage(self):
        return self._demImage

    def getNumberPatches(self):
        return self._numberPatches

    def getTopo(self):
        return self._topo

    def setMasterRawImage(self, image):
        self._masterRawImage = image

    def setSlaveRawImage(self, image):
        self._slaveRawImage = image

    def setMasterFrame(self, frame):
        self._masterFrame = frame

    def setSlaveFrame(self, frame):
        self._slaveFrame = frame

    def setMasterSquint(self, squint):
        self._masterSquint = squint

    def setSlaveSquint(self, squint):
        self._slaveSquint = squint

    def setLookSide(self, lookSide):
        self._lookSide = lookSide

    def setDemImage(self, image):
        self._demImage = image

    def setNumberPatches(self, x):
        self._numberPatches = x

    def setTopo(self, topo):
        self._topo = topo

     ## This overides the _FrameMixin.frame
    @property
    def frame(self):
        return self.masterFrame

    # Some line violate PEP008 in order to facilitate using "grep"
    # for development
    masterFrame = property(getMasterFrame, setMasterFrame)
    slaveFrame = property(getSlaveFrame, setSlaveFrame)
    demImage = property(getDemImage, setDemImage) 
    numberPatches = property(getNumberPatches, setNumberPatches)
    topo = property(getTopo, setTopo)

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


    def numberOfLooks(self, frame, posting,  azlooks, rglooks):
        '''
        Compute relevant number of looks.
        '''
        from isceobj.Planet.Planet import Planet
        from isceobj.Constants import SPEED_OF_LIGHT
        import numpy as np

        azFinal = None
        rgFinal = None

        if azlooks is not None:
            azFinal = azlooks

        if rglooks is not None:
            rgFinal = rglooks

        if (azFinal is not None) and (rgFinal is not None):
            return (azFinal, rgFinal)

        if posting is None:
            raise Exception('Input posting is none. Either specify (azlooks, rglooks) or posting in input file')


        elp = Planet(pname='Earth').ellipsoid

        ####First determine azimuth looks
        tmid = frame.sensingMid
        sv = frame.orbit.interpolateOrbit( tmid, method='hermite') #.getPosition()
        llh = elp.xyz_to_llh(sv.getPosition())


        if azFinal is None:
            hdg = frame.orbit.getENUHeading(tmid)
            elp.setSCH(llh[0], llh[1], hdg)
            sch, vsch = elp.xyzdot_to_schdot(sv.getPosition(), sv.getVelocity())
            azFinal = max(int(np.round(posting * frame.PRF / vsch[0])), 1)

        if rgFinal is None:
            pulseLength = frame.instrument.pulseLength
            chirpSlope = frame.instrument.chirpSlope

            #Range Bandwidth
            rBW = np.abs(chirpSlope)*pulseLength

            # Slant Range resolution
            rgres = abs(SPEED_OF_LIGHT / (2.0 * rBW))
            
            r0 = frame.startingRange
            rmax = frame.getFarRange()
            rng =(r0+rmax)/2 

            Re = elp.pegRadCur
            H = sch[2]
            cos_beta_e = (Re**2 + (Re + H)**2 -rng**2)/(2*Re*(Re+H))
            sin_bet_e = np.sqrt(1 - cos_beta_e**2)
            sin_theta_i = sin_bet_e*(Re + H)/rng
            print("incidence angle at the middle of the swath: ", np.arcsin(sin_theta_i)*180.0/np.pi)
            groundRangeRes = rgres/sin_theta_i
            print("Ground range resolution at the middle of the swath: ", groundRangeRes)
            rgFinal = max(int(np.round(posting/groundRangeRes)),1)

        return azFinal, rgFinal


    @property
    def geocode_list(self):

        ###Explicitly build the list of products that need to be geocoded by default
        res = [ os.path.join( self.ifgDirname, self.ifgFilename),  #Unfiltered complex interferogram
                os.path.join( self.ifgDirname, 'filt_' + self.ifgFilename), #Filtered interferogram
                os.path.join( self.ifgDirname, self.coherenceFilename),   #Phase sigma coherence
                os.path.join( self.ifgDirname, self.correlationFilename), #Unfiltered correlation
                os.path.join( self.ifgDirname, swapExtension( 'filt_' + self.ifgFilename, ['.flat', '.int'], '.unw')), #Unwrap
                os.path.join( self.ifgDirname, swapExtension( 'filt_' + self.ifgFilename, ['.flat', '.int'], '.unw'))+'.conncomp', #conncomp
                os.path.join( self.geometryDirname, self.losFilename), #los
              ]

        ###If dispersive components are requested
        res += [ os.path.join( self.ionosphereDirname, self.dispersiveFilename + ".unwCor.filt"),   #Dispersive phase
                 os.path.join( self.ionosphereDirname, self.nondispersiveFilename + ".unwCor.filt"), #Non-dispersive
                 os.path.join( self.ionosphereDirname, 'mask.bil'),  #Mask
               ]
        return res
                
    @property
    def off_geocode_list(self):
        prefix = os.path.join(self.denseOffsetsDirname, self.denseOffsetFilename)

        res = [ prefix + '.bil',
                prefix + '_snr.bil' ]
        return res

###Utility to swap extensions
def swapExtension(infile, inexts, outext):
    found = False

    for ext in inexts:
        if ext in infile:
            outfile = infile.replace(ext, outext)
            found = True
            break

    if not found:
        raise Exception('Did not find extension {0} in file name {1}'.format(str(inexts), infile))

    return outfile

