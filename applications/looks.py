#!/usr/bin/env python3

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2013 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Piyush Agram
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




import os
import logging
import sys

import isce
import argparse
from isceobj.Image import createImage,createDemImage
from mroipac.looks.Looks import Looks

class customArgparseFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    '''
    For better help message that also shows the defaults.
    '''
    pass

def cmdLineParse():
    '''
    Command Line Parser.
    '''
    parser = argparse.ArgumentParser(description='Take integer number of looks.',
            formatter_class=customArgparseFormatter,
            epilog = '''

Example:

looks.py -i input.file -o output.file -r 4  -a 4

''')
    parser.add_argument('-i','--input', type=str, required=True, help='Input ISCEproduct with a corresponding .xml file.', dest='infile')
    parser.add_argument('-o','--output',type=str, default=None, help='Output ISCE DEproduct with a corresponding .xml file.', dest='outfile')
    parser.add_argument('-r', '--range', type=int, default=1, help='Number of range looks. Default: 1', dest='rglooks')
    parser.add_argument('-a', '--azimuth', type=int, default=1, help='Number of azimuth looks. Default: 1', dest='azlooks')

    values = parser.parse_args()
    if (values.rglooks == 1) and (values.azlooks == 1):
        print('Nothing to do. One look requested in each direction. Exiting ...')
        sys.exit(0)

    return values

def main(inps):
    '''
    The main driver.
    '''

    if inps.infile.endswith('.xml'):
        inFileXml = inps.infile
        inFile = os.path.splitext(inps.infile)[0]
    else:
        inFile = inps.infile
        inFileXml = inps.infile + '.xml'

    if inps.outfile is None:
        spl = os.path.splitext(inFile)
        ext = '.{0}alks_{1}rlks'.format(inps.azlooks, inps.rglooks)
        outFile = spl[0] + ext + spl[1]

    elif inps.outfile.endswith('.xml'):
        outFile = os.path.splitext(inps.outfile)[0]
    else:
        outFile = inps.outfile



    print('Output filename : {0}'.format(outFile))
    #hackish, just to know the image type to instantiate the correct type
    #until we put the info about how to generate the instance in the xml
    from iscesys.Parsers.FileParserFactory import createFileParser
    FP = createFileParser('xml')
    tmpProp, tmpFact, tmpMisc = FP.parse(inFileXml)
    if('image_type' in tmpProp and tmpProp['image_type'] == 'dem'):
        inImage = createDemImage()
    else:
        inImage = createImage()

    inImage.load(inFileXml)
    inImage.filename = inFile

    lkObj = Looks()
    lkObj.setDownLooks(inps.azlooks)
    lkObj.setAcrossLooks(inps.rglooks)
    lkObj.setInputImage(inImage)
    lkObj.setOutputFilename(outFile)
    lkObj.looks()

    return outFile

if __name__ == '__main__':
    '''
    Makes the script executable.
    '''

    inps = cmdLineParse()
    main(inps)
