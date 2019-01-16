#!/usr/bin/env python3

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2012 California Institute of Technology. ALL RIGHTS RESERVED.
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

from isceobj.Location.Offset import OffsetField,Offset
from iscesys.Component.Component import Component,Port
from iscesys.Compatibility import Compatibility
Compatibility.checkPythonVersion()
from isceobj.Histogram import histogram as hist

import logging
logger = logging.getLogger('isce.Util.histogram')


FILE_NAME = Component.Parameter('fileName',
        public_name='FILE_NAME',
        default=None,
        type=str,
        mandatory=False,
        doc = 'Input filename')

NUMBER_BINS = Component.Parameter('numberBins',
        public_name = 'NUMBER_BINS',
        default = 40,
        type=int,
        mandatory=False,
        doc = 'Number of quantile bins')

NULL_VALUE = Component.Parameter('nullValue',
        public_name = 'NULL_VALUE',
        default = 0.0,
        type = float,
        mandatory = False,
        doc = 'Null value in data')

class Histogram(Component):

    family = 'histogram'
    logging_name = 'isce.isceobj.histogram'

    parameter_list = (FILE_NAME,
                      NUMBER_BINS,
                      NULL_VALUE,)


    def histogram(self):
        self.createImage()

        accessor = self.image.getImagePointer()
        if self.image.dataType.upper().startswith('C'):
            raise NotImplementedError('Histograms for complex images have not yet been implemented. Might be more efficient to set up filters for amp / phase for complex images and use those as input')
            self.results = hist.complexHistogram_Py(accessor,self.numberBins, self.nullValue)

        else:
            self.results = hist.realHistogram_Py(accessor, self.numberBins, self.nullValue)

        self.image.finalizeImage()

        return

    def createImage(self):
        '''
        Create an image object to pass to C extension.
        '''
        from isceobj.Util.ImageUtil import ImageLib as IML
        img = IML.loadImage(self.fileName)[0]

        if img.dataType.upper().startswith('C'):
            img.setCaster('read','CDOUBLE')
        else:
            img.setCaster('read', 'DOUBLE')

        self.image = img
        self.image.createImage()

    def getStats(self):
        return self.results

    def __init__(self, name=''):
        super(Histogram,self).__init__(family=self.__class__.family, name=name)
        self.image = None
        self.results = None
        self.dictionaryOfOutputVariables = {}
        self.descriptionOfVariables = {}
        self.mandatoryVariables = []
        self.optionalVariables = []
        self.initOptionalAndMandatoryLists()
        return

#end class


if __name__ == '__main__':
    '''
    Main driver.
    '''
    
    def cmdLineParse():
        '''
        Command line parser.
        '''

        parser = argparse.ArgumentParser(description='Compute histogram')
        parser.add_argument('-i', '--input', dest='infile', type=str, required=True,
            help = 'Input file to analyze')
        parser.add_argument('-b', '--bins', dest='numbins', type=int, default=40,
            help = 'Number of bins')
        parser.add_argument('-n', '--null', dest='nullval', type=float, default=0.0,
            help='Null value for data')
        return parser.parse_args()


    inps = cmdLineParse()
    
    hist = Histogram()
    hist.fileName = inps.infile
    hist.numberBins = inps.numbins
    hist.nullValue = inps.nullval

    hist.histogram()

    ###Show stats for band 1 of the image
    quants, vals = hist.getStats()[0]

    for x,y in zip(quants,vals):
        print('QUANT: {0}   VALUE: {1}'.format(100*x,y))

