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
# Author: Brett George
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



import logging
import argparse
import os

import isce
import isceobj
from isceobj.TopsProc.runBurstIfg import computeCoherence
logger = logging.getLogger('isce.tops.runFilter')

def runFilter(infile, outfile, filterStrength):
    from mroipac.filter.Filter import Filter
    logger.info("Applying power-spectral filter")

    # Initialize the flattened interferogram
    topoflatIntFilename = infile
    intImage = isceobj.createIntImage()
    intImage.load( infile + '.xml')
    intImage.setAccessMode('read')
    intImage.createImage()

    # Create the filtered interferogram
    filtImage = isceobj.createIntImage()
    filtImage.setFilename(outfile)
    filtImage.setWidth(intImage.getWidth())
    filtImage.setAccessMode('write')
    filtImage.createImage()

    objFilter = Filter()
    objFilter.wireInputPort(name='interferogram',object=intImage)
    objFilter.wireOutputPort(name='filtered interferogram',object=filtImage)
    objFilter.goldsteinWerner(alpha=filterStrength)

    intImage.finalizeImage()
    filtImage.finalizeImage()

def runFilter_gaussian(infile, outfile, filterStrength):
    from isceobj import Filter
    
    logger.info("Applying power-spectral filter")

    # Initialize the flattened interferogram
    topoflatIntFilename = infile
    intImage = isceobj.createIntImage()
    intImage.load( infile + '.xml')
    intImage.setAccessMode('read')
    intImage.createImage()

    # Create the filtered interferogram
    filtImage = isceobj.createIntImage()
    filtImage.setFilename(outfile)
    filtImage.setWidth(intImage.getWidth())
    filtImage.setAccessMode('write')
    filtImage.createImage()

    objFilter = Filter()
    objFilter.wireInputPort(name='interferogram',object=intImage)
    objFilter.wireOutputPort(name='filtered interferogram',object=filtImage)
    objFilter.gaussianFilter(filterWidth=10, filterHeight=10, sigma=1)
    
    intImage.finalizeImage()
    filtImage.finalizeImage()
   

def estCoherence(outfile, corfile):
    from mroipac.icu.Icu import Icu

    #Create phase sigma correlation file here
    filtImage = isceobj.createIntImage()
    filtImage.load( outfile + '.xml')
    filtImage.setAccessMode('read')
    filtImage.createImage()

    phsigImage = isceobj.createImage()
    phsigImage.dataType='FLOAT'
    phsigImage.bands = 1
    phsigImage.setWidth(filtImage.getWidth())
    phsigImage.setFilename(corfile)
    phsigImage.setAccessMode('write')
    phsigImage.createImage()

    
    icuObj = Icu(name='sentinel_filter_icu')
    icuObj.configure()
    icuObj.unwrappingFlag = False
    icuObj.useAmplitudeFlag = False
    #icuObj.correlationType = 'NOSLOPE'

    icuObj.icu(intImage = filtImage,  phsigImage=phsigImage)
    phsigImage.renderHdr()

    filtImage.finalizeImage()
    phsigImage.finalizeImage()


def createParser():
    '''
    Create command line parser.
    '''

    parser = argparse.ArgumentParser(description='Filter interferogram and generated coherence layer.')
    parser.add_argument('-i','--input', type=str, required=True, help='Input interferogram',
            dest='infile')
    parser.add_argument('-f','--filt', type=str, default=None, help='Ouput filtered interferogram',
            dest='filtfile')
    parser.add_argument('-c', '--coh', type=str, default='phsig.cor', help='Coherence file',
            dest='cohfile')
    parser.add_argument('-s', '--strength', type=float, default=0.5, help='Filter strength',
            dest='filterstrength')
    parser.add_argument('--slc1', type=str, help="SLC 1", dest='slc1')
    parser.add_argument('--slc2', type=str, help="SLC 2", dest='slc2')
    parser.add_argument('--cc','--complex_coh',type=str, default='fine.cori.full',help='complex coherence file',dest='cpx_cohfile')
    parser.add_argument('-r','--range_looks',type=int, default=9, help= 'range looks', dest='numberRangelooks')
    parser.add_argument('-z','--azimuth_looks',type=int, default=3, help= 'azimuth looks', dest='numberAzlooks')
    return parser

def cmdLineParse(iargs=None):
    parser = createParser()
    return parser.parse_args(args=iargs)


def main(iargs=None):
    inps = cmdLineParse(iargs)

    if inps.filtfile is None:
        inps.filtfile = 'filt_' + inps.infile

    runFilter(inps.infile, inps.filtfile, inps.filterstrength)

    estCoherence(inps.filtfile, inps.cohfile)
    if inps.slc1 and inps.slc2:
        computeCoherence(inps.slc1,inps.slc2,inps.cpx_cohfile)
        from mroipac.looks.Looks import Looks

        print('Multilooking {0} ...'.format(inps.cpx_cohfile))
        
        infile=inps.cpx_cohfile
        inimg = isceobj.createImage()
        inimg.load(infile + '.xml')

        alks=inps.numberAzlooks
        rlks=inps.numberRangelooks
        
        spl = os.path.splitext(inimg.filename)
        #ext = '.{0}alks_{1}rlks'.format(alks, rlks)
        #outname = spl[0] + ext + spl[1]
        outname=spl[0]
        lkObj = Looks()
        lkObj.setDownLooks(alks)
        lkObj.setAcrossLooks(rlks)
        lkObj.setInputImage(inimg)
        lkObj.setOutputFilename(outname)
        lkObj.looks()
        fullfilename=inps.cpx_cohfile
        ret=os.system('rm '+fullfilename)
if __name__ == '__main__':
    
    main()
