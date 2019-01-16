#
# Author: Piyush Agram
# Copyright 2016
#

import logging
import isceobj
import copy
import os
logger = logging.getLogger('isce.grdsar.runPreprocessor')

def runPreprocessor(self):
    '''Extract images.
    '''

    catalog = isceobj.Catalog.createCatalog(self._grd.procDoc.name)

    if len(self.polarizations):
        polListProvided = True
        polList = [x for x in self.polarizations]
    else:
        polListProvided = False
        polList = ['HH', 'HV', 'VV', 'VH', 'RH', 'RV']


    self.master.configure()


    if not os.path.isdir(self.master.output):
        os.makedirs(self.master.output)


    slantRangeExtracted = False
    r0min = 0.
    r0max = 0.

    for pol in polList:
        ###Process master pol-by-pol
        frame = copy.deepcopy(self.master)
        frame.polarization = pol
        frame.output = os.path.join(self.master.output, 'beta_{0}.img'.format(pol))
        frame.slantRangeFile = os.path.join(self.master.output, 'slantrange.img')
        frame.product.startingSlantRange = r0min
        frame.product.endingSlantRange = r0max

        try:
            master = extract_slc(frame, slantRange=(not slantRangeExtracted))
            success=True
            if not slantRangeExtracted:
                r0min = frame.product.startingSlantRange
                r0max = frame.product.endingSlantRange
            slantRangeExtracted = True
        except Exception as err:
            print('Could not extract polarization {0}'.format(pol))
            print('Generated error: ', err)
            success=False
            if polListProvided:
                raise Exception('User requested polarization {0} but not found in input data'.format(pol))



        if success:
            catalog.addInputsFrom(frame.product, 'master.sensor')
            catalog.addItem('numberOfSamples', frame.product.numberOfSamples, 'master')
            catalog.addItem('numberOfLines', frame.product.numberOfLines, 'master')
            catalog.addItem('groundRangePixelSize', frame.product.groundRangePixelSize, 'master')
            self._grd.polarizations.append(pol)

            self._grd.saveProduct( frame.product, os.path.splitext(frame.output)[0] + '.xml')


    self._grd.outputFolder = self.master.output

    catalog.printToLog(logger, "runPreprocessor")
    self._grd.procDoc.addAllFromCatalog(catalog)

def extract_slc(sensor, slantRange=False):
#    sensor.configure()
    sensor.parse()
    sensor.extractImage()
   
    if slantRange:
        sensor.extractSlantRange()

    else:
        img = isceobj.createImage()
        img.load( sensor.slantRangeFile + '.xml')
        img.setAccessMode('READ')
        sensor.product.slantRangeImage = img

    return sensor.output

