#
# Author: Piyush Agram
# Copyright 2016
#

import logging
import isceobj
import mroipac
from mroipac.baseline.Baseline import Baseline
import copy
import os
logger = logging.getLogger('isce.topsinsar.runPreprocessor')

def runPreprocessor(self):
    '''Extract images.
    '''

    virtual = self.useVirtualFiles
    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)


    ###First set maximum number of swaths possible for a sensor.
    self._insar.numberOfSwaths = self.master.maxSwaths 
    swathList = self._insar.getInputSwathList(self.swaths)
    
    catalog.addItem('Input list of swaths to process: ', swathList, 'common')

    self.master.configure()
    self.slave.configure()

    for swath in swathList:
        ###Process master frame-by-frame
        frame = copy.deepcopy(self.master)
        frame.swathNumber = swath
        frame.output = os.path.join(frame.output, 'IW{0}'.format(swath))
        frame.regionOfInterest = self.regionOfInterest

        try:
            master = extract_slc(frame, virtual=virtual)
            success=True
        except Exception as err:
            print('Could not extract swath {0} from {1}'.format(swath, frame.safe))
            print('Generated error: ', err)
            success=False

        if success:
            catalog.addInputsFrom(frame.product, 'master.sensor')
            catalog.addItem('burstWidth_{0}'.format(swath), frame.product.bursts[0].numberOfSamples, 'master')
            catalog.addItem('burstLength_{0}'.format(swath), frame.product.bursts[0].numberOfLines, 'master')
            catalog.addItem('numberOfBursts_{0}'.format(swath), len(frame.product.bursts), 'master')

        
        ###Process slave frame-by-frame
        frame = copy.deepcopy(self.slave)
        frame.swathNumber = swath
        frame.output = os.path.join(frame.output, 'IW{0}'.format(swath))
        frame.regionOfInterest = self.regionOfInterest

        try:
            slave = extract_slc(frame, virtual=virtual)
            success=True
        except Exception as err:
            print('Could not extract swath {0} from {1}'.format(swath, frame.safe))
            print('Generated error: ', err)
            success = False

        if success:
            catalog.addInputsFrom(frame.product, 'slave.sensor')
            catalog.addItem('burstWidth_{0}'.format(swath), frame.product.bursts[0].numberOfSamples, 'slave')
            catalog.addItem('burstLength_{0}'.format(swath), frame.product.bursts[0].numberOfLines, 'slave')
            catalog.addItem('numberOfBursts_{0}'.format(swath), len(frame.product.bursts), 'slave')


    self._insar.masterSlcProduct = self.master.output
    self._insar.slaveSlcProduct = self.slave.output

    catalog.printToLog(logger, "runPreprocessor")
    self._insar.procDoc.addAllFromCatalog(catalog)

def extract_slc(sensor, virtual=False):
#    sensor.configure()
    sensor.extractImage(virtual=virtual)
   
    return sensor.output

