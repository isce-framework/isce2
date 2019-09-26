#
# Author: Piyush Agram
# Copyright 2016
#

import os
import logging
import logging.config
from iscesys.Component.Component import Component
from iscesys.DateTimeUtil.DateTimeUtil import DateTimeUtil as DTU
from iscesys.Compatibility import Compatibility


OUTPUT_FOLDER = Component.Parameter('outputFolder',
                                public_name='output folder with imagery',
                                default='master',
                                type=str,
                                mandatory=False,
                                doc = 'Directory name of the unpacked GRD product')

GEOMETRY_FOLDER = Component.Parameter('geometryFolder',
                                public_name='folder with geometry products',
                                default='geometry',
                                type=str,
                                mandatory=False,
                                doc='Directory with geometry products')

POLARIZATIONS = Component.Parameter('polarizations',
                                public_name='polarizations',
                                default = [],
                                type = str,
                                container=list,
                                mandatory = False,
                                doc = 'Polarizations in the dataset')


WATER_MASK_FILENAME = Component.Parameter(
        'waterMaskFileName',
        public_name = 'water mask file name',
        default = 'waterMask.msk',
        type=str,
        mandatory=False,
        doc='Filename of the water mask in radar coordinates')

SL_MASK_FILENAME = Component.Parameter(
        'slMaskFileName',
        public_name = 'shadow layover mask file name',
        default = 'slMask.msk',
        type=str,
        mandatory=False,
        doc='Filename of the shadow layover mask in radar coordinates')

LOS_FILENAME = Component.Parameter(
        'losFileName',
        public_name = 'line-of-sight file name',
        default = 'los.rdr',
        type = str,
        mandatory = False,
        doc = 'los file name')

INC_FILENAME = Component.Parameter(
        'incFileName',
        public_name='local incidence angle file name',
        default = 'inc.rdr',
        type = str,
        mandatory = False,
        doc = 'incidence angle file name')

GAMMA0_FILENAME = Component.Parameter(
        'gamma0FileName',
        public_name='Gamma0 backscatter file',
        default = 'gamma0.img',
        type = str,
        mandatory = False,
        doc = 'Unmasked gamma0 backscatter file')

MASKED_GAMMA0_FILENAME = Component.Parameter(
        'maskedGamma0FileName',
        public_name='Masked gamma0 backscatter file',
        default='gamma0_masked.rdr',
        type=str,
        mandatory = False,
        doc = 'Masked gamma0 backscatter file')

BOUNDING_BOX = Component.Parameter(
        'boundingBox',
        public_name='Estimated bounding box',
        default=[],
        type=float,
        container=list,
        doc = 'Estimated bounding box')

GEOCODE_LIST = Component.Parameter('geocode_list',
    public_name='geocode list',
    default=[WATER_MASK_FILENAME,
             SL_MASK_FILENAME,
             LOS_FILENAME,
             INC_FILENAME,
             GAMMA0_FILENAME,
             MASKED_GAMMA0_FILENAME,
             ],
    container=list,
    type=str,
    mandatory=False,
    doc='List of files to geocode'
)


class RtcProc(Component):
    """
    This class holds the properties, along with methods (setters and getters)
    to modify and return their values.
    """

    parameter_list = (OUTPUT_FOLDER,
                      GEOMETRY_FOLDER,
                      POLARIZATIONS,
                      WATER_MASK_FILENAME,
                      SL_MASK_FILENAME,
                      LOS_FILENAME,
                      INC_FILENAME,
                      GAMMA0_FILENAME,
                      MASKED_GAMMA0_FILENAME,
                      BOUNDING_BOX,
                      GEOCODE_LIST)

    facility_list = ()


    family='rtccontext'

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

        outdir  = self.outputFolder
        for i, x in enumerate(self.geocode_list):
            if isinstance(x, Component.Parameter):
                y = getattr(self, getattr(x, 'attrname'))
                self.geocode_list[i] = os.path.join(outdir, y)

        return


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

    def getInputPolarizationList(self, inlist):
        '''
        To be used to get list of swaths that user wants us to process.
        '''
        if len(inlist) == 0:
            return ['HH','HV','VV','VH','RH','RV']
        else:
            return inlist
    
    def getValidPolarizationList(self, inlist):
        '''
        Used to get list of swaths left after applying all filters  - e.g, region of interest.
        '''

        checklist = self.getInputPolarizationList(inlist)

        validlist = [x for x in checklist if x in self.polarizations]

        return validlist


    def getMasterPolarizations(self, masterPol, inlist):
        '''
        Check available list to pick co-pol master if none is provided.
        '''

        validlist = self.getValidPolarizationList(self, inlist)

        if masterPol is None:

            if 'HH' in validlist:
                return 'HH'
            elif 'VV' in validlist:
                return 'VV'
            elif 'RH' in validlist:
                return 'RH'
            else:
                return validlist[0]

        else:
            if masterPol not in validlist:
                raise Exception('Requested master polarization {0} not in available polarizations'.format(masterPol))
            else:
                return masterPol


    def getLooks(self,posting, delaz, delrg, azl, rgl):
        '''
        Return number of looks.
        '''
        import numpy as np
        azlooks = int(np.rint(posting/delaz))
        rglooks = int(np.rint(posting/delrg))
        if azl:
            azlooks = int(azl)

        if rgl:
            rglooks = int(rgl)

        return (azlooks, rglooks)
