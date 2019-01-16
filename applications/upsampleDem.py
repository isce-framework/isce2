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
from contrib.demUtils.UpsampleDem import UpsampleDem
from iscesys.Parsers.FileParserFactory import createFileParser
from isceobj.Image import createDemImage

class customArgparseFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    '''
    For better help message that also shows the defaults.
    '''
    pass

def cmdLineParse():
    '''
    Command Line Parser.
    '''
    parser = argparse.ArgumentParser(description='Oversample DEM by integer factor.',
            formatter_class=customArgparseFormatter,
            epilog = '''

Example: 

upsampleDem.py -i input.dem -o output.dem -f 4 4
            
This oversamples the input dem in both lat and lon by a factor of 4.''')
    parser.add_argument('-i','--input', type=str, required=True, help='Input ISCE DEM with a corresponding .xml file.', dest='infile')
    parser.add_argument('-o','--output',type=str, default=None, help='Output ISCE DEM with a corresponding .xml file.', dest='outfile')
    parser.add_argument('-m', '--method', type=str, default='BIQUINTIC', help='Interpolation method out of Akima / Biquintic. Default: biquintic.', dest='method')
    parser.add_argument('-f','--factor',type=int, nargs='+', required=True, help='Oversampling factor in lat and lon (or a single value for both).', dest='factor')

    values = parser.parse_args()
    if len(values.factor) > 2:
        raise Exception('Factor should be a single number or a list of two. Undefined input for -f or --factor : '+str(values.factor))
    elif len(values.factor) == 1:
        values.factor = [values.factor[0], values.factor[0]]

    return values

if __name__ == "__main__":
    inps = cmdLineParse()

    if inps.infile.endswith('.xml'):
        inFileXml = inps.infile
        inFile = os.path.splitext(inps.infile)[0]
    else:
        inFile = inps.infile
        inFileXml = inps.infile + '.xml'

    if inps.outfile.endswith('.xml'):
        outFile = os.path.splitext(inps.outfile)[0]
    else:
        outFile = inps.outfile

    parser = createFileParser('xml')
    prop, fac, misc = parser.parse(inFileXml)


    inImage = createDemImage()
    inImage.init(prop,fac,misc)
    inImage.filename = inFile 
    inImage.createImage()

    upsampObj = UpsampleDem()
    upsampObj.method = inps.method
    upsampObj.setOutputFilename(outFile)
    upsampObj.upsampledem(demImage=inImage, yFactor=inps.factor[0], xFactor=inps.factor[1])
