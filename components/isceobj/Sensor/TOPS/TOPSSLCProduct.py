#!/usr/bin/env python3

import isce
import datetime
import isceobj
import numpy as np
from iscesys.Component.Component import Component
from iscesys.Traits import datetimeType


####List of parameters
IMAGING_MODE = Component.Parameter('mode',
        public_name = 'imaging mode',
        default = 'TOPS',
        type = str,
        mandatory = False,
        doc = 'Imaging mode')

FOLDER = Component.Parameter('folder',
        public_name = 'folder',
        default = None,
        type = str,
        mandatory = True,
        doc = 'Folder corresponding to single swath of TOPS SLC')

SPACECRAFT_NAME = Component.Parameter('spacecraftName',
    public_name='spacecraft name',
    default=None,
    type = str,
    mandatory = True,
    doc = 'Name of the space craft')

MISSION = Component.Parameter('mission',
        public_name = 'mission',
        default = None,
        type = str,
        mandatory = True,
        doc = 'Mission name')

PROCESSING_FACILITY = Component.Parameter('processingFacility',
    public_name='processing facility',
    default=None,
    type = str,
    mandatory = False,
    doc = 'Processing facility information')

PROCESSING_SYSTEM = Component.Parameter('processingSystem',
    public_name='processing system',
    default=None,
    type = str,
    mandatory = False,
    doc = 'Processing system information')

PROCESSING_SYSTEM_VERSION = Component.Parameter('processingSoftwareVersion',
    public_name='processing software version',
    default=None,
    type = str,
    mandatory = False,
    doc = 'Processing system software version')

ASCENDING_NODE_TIME = Component.Parameter('ascendingNodeTime',
        public_name='ascending node time',
        default=None,
        type=datetimeType,
        mandatory=True,
        doc='Ascending node time corresponding to the acquisition')

SWATH_NUMBERS  = Component.Parameter('swathNumbers',
        public_name = 'swath numbers',
        default = None,
        type = int, 
        mandatory = True,
        container = list,
        doc = 'Swath numbers that are represented by the product')

SWATHS = Component.Facility('swaths',
        public_name='swaths',
        module = 'iscesys.Component',
        factory = 'createTraitSeq',
        args=('swath',),
        mandatory = False,
        doc = 'Trait sequence of swaths products')

class TOPSSLCProduct(Component):
    """A class to represent a burst SLC along a radar track"""
    
    family = 'topsslc'
    logging_name = 'isce.tops.slc'

    facility_list = (SWATHS,)


    parameter_list = (IMAGING_MODE,
                      FOLDER,
                      SPACECRAFT_NAME,
                      MISSION,
                      PROCESSING_FACILITY,
                      PROCESSING_SYSTEM,
                      PROCESSING_SYSTEM_VERSION,
                      ASCENDING_NODE_TIME,
                      SWATH_NUMBERS
                      )


    facility_list = (SWATHS,)


    def __init__(self,name=''):
        super(TOPSSLCProduct, self).__init__(family=self.__class__.family, name=name)
        return None

    @property
    def sensingStart(self):
        return min([x.sensingStart for x in self.swaths])

    @property
    def sensingStop(self):
        return max([x.sensingStop for x in self.swaths])

    @property
    def sensingMid(self):
        return self.sensingStart + 0.5 * (self.sensingStop - self.sensingStart)

    @property
    def startingRange(self):
        return min([x.startingRange for x in self.swaths])

    @property
    def farRange(self):
        return max([x.farRange for x in self.swaths])

    @property
    def midRange(self):
        return 0.5 * (self.startingRange + self.farRange)

    @property
    def orbit(self):
        '''
        For now all bursts have same state vectors.
        This will be the case till we build mechanisms for bursts to share metadata.
        '''
        return self.swaths[0].orbit

    @property
    def numberSwaths(self):
        return len(self.swathNumbers)

    def getBbox(self ,hgtrange=[-500,9000]):
        '''
        Bounding box estimate.
        '''

        ts = [self.sensingStart, self.sensingStop]
        rngs = [self.startingRange, self.farRange]
       
        pos = []
        for ht in hgtrange:
            for tim in ts:
                for rng in rngs:
                    llh = self.orbit.rdr2geo(tim, rng, height=ht)
                    pos.append(llh)

        pos = np.array(pos)

        bbox = [np.min(pos[:,0]), np.max(pos[:,0]), np.min(pos[:,1]), np.max(pos[:,1])]
        return bbox

