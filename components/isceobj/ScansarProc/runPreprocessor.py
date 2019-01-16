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
logger = logging.getLogger('isce.scansarinsar.runPreprocessor')

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

    ##Processing master first.
    self.master.configure()
    self.master.swaths = swathList
    self.master.virtualFiles = self.useVirtualFiles
    
    ##Assume a single folder is provided for now. 
    ##Leaving room for future extensions.
    self.master.inputDirList = self.master.inputDirList[0]

    swaths = self.master.extractImage()

    self.insar.masterSlcProduct = os.path.dirname(os.path.dirname(swaths[0].image.filename))
    print('Master product: ', self.insar.masterSlcProduct)

    for ind, swath in enumerate(swaths):
        catalog.addInputsFrom(swath, 'master.sensor_{0}'.format(swathList[ind]))
        catalog.addItem('swathWidth_{0}'.format(swathList[ind]), swath.numberOfSamples, 'master')
        catalog.addItem('swathLength_{0}'.format(swathList[ind]), swath.numberOfLines, 'master')
        #catalog.addItem('numberOfBursts_{0}'.format(swathList[ind]), len(swath.burstStartLines), 'master')

        self._insar.saveProduct(swath, os.path.dirname(swath.image.filename) + '.xml')



    ##Processing slave next.
    self.slave.configure()
    self.slave.swaths = swathList
    self.slave.virtualFiles = self.useVirtualFiles
    
    ##Assume a single folder is provided for now. 
    ##Leaving room for future extensions.
    self.slave.inputDirList = self.slave.inputDirList[0]

    swaths = self.slave.extractImage()

    self.insar.slaveSlcProduct = os.path.dirname( os.path.dirname(swaths[0].image.filename))

    print('Slave Product: ', self.insar.slaveSlcProduct)
    for ind, swath in enumerate(swaths):
        catalog.addInputsFrom(swath, 'slave.sensor_{0}'.format(swathList[ind]))
        catalog.addItem('swathWidth_{0}'.format(swathList[ind]), swath.numberOfSamples, 'slave')
        catalog.addItem('swathLength_{0}'.format(swathList[ind]), swath.numberOfLines, 'slave')
        #catalog.addItem('numberOfBursts_{0}'.format(swathList[ind]), len(swath.burstStartLines), 'master')

        self._insar.saveProduct(swath, os.path.dirname(swath.image.filename) + '.xml')




    catalog.printToLog(logger, "runPreprocessor")
    self._insar.procDoc.addAllFromCatalog(catalog)
