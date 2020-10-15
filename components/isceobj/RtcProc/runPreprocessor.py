#
# Author: Piyush Agram
# Copyright 2016
#

import logging
import isceobj
import copy
import os
import inspect
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


    self.reference.configure()

    os.makedirs(self.reference.output, exist_ok=True)


    slantRangeExtracted = False
    r0min = 0.
    r0max = 0.

    for pol in polList:
        ###Process reference pol-by-pol
        frame = copy.deepcopy(self.reference)
        frame.polarization = pol
        frame.output = os.path.join(self.reference.output, 'beta_{0}.img'.format(pol))
        frame.slantRangeFile = os.path.join(self.reference.output, 'slantrange.img')
        frame.product.startingSlantRange = r0min
        frame.product.endingSlantRange = r0max

        try:
            reference = extract_slc(frame, slantRange=(not slantRangeExtracted), removeNoise=self.apply_thermal_noise_correction)
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
            catalog.addInputsFrom(frame.product, 'reference.sensor')
            catalog.addItem('numberOfSamples', frame.product.numberOfSamples, 'reference')
            catalog.addItem('numberOfLines', frame.product.numberOfLines, 'reference')
            catalog.addItem('groundRangePixelSize', frame.product.groundRangePixelSize, 'reference')
            self._grd.polarizations.append(pol)

            self._grd.saveProduct( frame.product, os.path.splitext(frame.output)[0] + '.xml')


    self._grd.outputFolder = self.reference.output

    catalog.printToLog(logger, "runPreprocessor")
    self._grd.procDoc.addAllFromCatalog(catalog)

def extract_slc(sensor, slantRange=False, removeNoise=False):
#    sensor.configure()
    sensor.parse()
    sensor_extractImage_spec = inspect.getfullargspec(sensor.extractImage)
    if "removeNoise" in sensor_extractImage_spec.args or "removeNoise" in sensor_extractImage_spec.kwonlyargs:
        sensor.extractImage(removeNoise=removeNoise)
    else:
        print('Noise removal requested, but sensor does not support noise removal.')
        sensor.extractImage()
   
    if slantRange:
        sensor.extractSlantRange()

    else:
        img = isceobj.createImage()
        img.load( sensor.slantRangeFile + '.xml')
        img.setAccessMode('READ')
        sensor.product.slantRangeImage = img

    return sensor.output

