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
    self._insar.numberOfSwaths = self.reference.maxSwaths 
    swathList = self._insar.getInputSwathList(self.swaths)
    
    catalog.addItem('Input list of swaths to process: ', swathList, 'common')

    self.reference.configure()
    self.secondary.configure()

    for swath in swathList:
        ###Process reference frame-by-frame
        frame = copy.deepcopy(self.reference)
        frame.swathNumber = swath
        frame.output = os.path.join(frame.output, 'IW{0}'.format(swath))
        frame.regionOfInterest = self.regionOfInterest

        try:
            reference = extract_slc(frame, virtual=virtual)
            success=True
        except Exception as err:
            print('Could not extract swath {0} from {1}'.format(swath, frame.safe))
            print('Generated error: ', err)
            success=False

        if success:
            catalog.addInputsFrom(frame.product, 'reference.sensor')
            catalog.addItem('burstWidth_{0}'.format(swath), frame.product.bursts[0].numberOfSamples, 'reference')
            catalog.addItem('burstLength_{0}'.format(swath), frame.product.bursts[0].numberOfLines, 'reference')
            catalog.addItem('numberOfBursts_{0}'.format(swath), len(frame.product.bursts), 'reference')

        
        ###Process secondary frame-by-frame
        frame = copy.deepcopy(self.secondary)
        frame.swathNumber = swath
        frame.output = os.path.join(frame.output, 'IW{0}'.format(swath))
        frame.regionOfInterest = self.regionOfInterest

        try:
            secondary = extract_slc(frame, virtual=virtual)
            success=True
        except Exception as err:
            print('Could not extract swath {0} from {1}'.format(swath, frame.safe))
            print('Generated error: ', err)
            success = False

        if success:
            catalog.addInputsFrom(frame.product, 'secondary.sensor')
            catalog.addItem('burstWidth_{0}'.format(swath), frame.product.bursts[0].numberOfSamples, 'secondary')
            catalog.addItem('burstLength_{0}'.format(swath), frame.product.bursts[0].numberOfLines, 'secondary')
            catalog.addItem('numberOfBursts_{0}'.format(swath), len(frame.product.bursts), 'secondary')


    self._insar.referenceSlcProduct = self.reference.output
    self._insar.secondarySlcProduct = self.secondary.output

    catalog.printToLog(logger, "runPreprocessor")
    self._insar.procDoc.addAllFromCatalog(catalog)

def extract_slc(sensor, virtual=False):
#    sensor.configure()
    sensor.extractImage(virtual=virtual)
   
    return sensor.output

