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





import argparse
import symtable
import math
import numpy as np
import os
import sys

import isce
from isceobj.Util.ImageUtil import ImageLib as IML

#####Global parameters
iMath = {
          'outFile' : None,     ####Output file name
          'outBands' : [],      ####List of out band mmaps
          'outScheme' : 'BSQ',  ####Output scheme
          'equations' : [],     #####List of math equations
          'outType' : 'f',      ####Output datatype
          'width'   : None,     ####Width of images
          'length'  : None,     ####Length of images
          'inBands' : {},       ####Dictionary of input band mmaps
          'inFiles' : {},       ####Dictionary input file mmaps
          'bboxes' :  []        ####Bounding boxes for input mmaps
        }


helpStr = """

ISCE Band image with imageMath.py

Examples:
*********

    1) imageMath.py -e='a*exp(-1.0*J*arg(b))' -o test.int -t cfloat  --a=resampOnlyImage.int --b=topophase.mph
       This uses phase from topophase.mph to correct topophase from the interferograms

    2) imageMath.py -e='a_0;a_1' --a=resampOnlyImage.amp -o test.amp -s BIL
       This converts a BIP image to a BIL image

    3) imageMath.py -e="abs(a);sqrt(b_0**2 + b_1**2)" --a=topophase.flat --b="topophase.mph;3419;float;2;BIP" -o test.mag -s BIL
        This should produce a BIL (RMG) image where both channels are equal. Input the correct width before testing this.

Rules:
******

    0) Input math expressions should be valid python expressions.

    1) A math expression for every band of output image is needed. For a multi-band output image, these expressions are separated by a ;.
       Example: See Example 2  above.

    2) All variable names in the math expressions need to be lower case, single character.  Capital characters and multi-char names are reserved for constants and functions respectively.

    3) The band of multi-band input images are represented by adding  _i  to the variable name, where "i" is the band number. All indices are zero-based (C and python).
       Example : a_0 represents the first band of the image represented by variable "a".

    4) For a single band image, the _0 band notation is optional.
       Example: a_0 and a are equivalent for a single band image.

    5) For every lower case variable in the equations, another input "--varname" is needed. Example shown above where --a and --b are defined.

    6) Variables  can be defined in two ways:
        a) File name (assuming an ISCE .xml file also exists).
           Example --a=resamp.int

        b) Image grammar:  "Filename;width;datatype;bands;scheme"
           Example --a="resamp.int;3200;cfloat;1;BSQ"

                -  Default value for datatype=float
                -  Default value for bands = 1
                -  Default value for scheme = BSQ

        c) In the image grammar: Single character codes for datatypes are case sensitive (Numpy convention) whereas multi-character codes are case-insensitive. Internally, everything is translated to numpy convention by the code before processing.
"""


class NumericStringParser(object):
    '''
    Parse the input expression using Python's inbuilt parser.
    '''
    def __init__(self, num_string):
        '''
        Create a parser object with input string.
        '''
        self.string = num_string
        self._restricted = list(IML.fnDict.keys()) + list(IML.constDict.keys())

    def parse(self):
        '''
        Parse the input expression to get list of identifiers.
        '''

        try:
            symTable = symtable.symtable(self.string, 'string', 'eval')
        except:
            raise IOError('Not a valid python math expression \n' +
                    self.string)

        idents = symTable.get_identifiers()

        known = []
        unknown = []
        for ident in idents:
            if ident not in self._restricted:
                unknown.append(ident)
            else:
                known.append(ident)


        for val in unknown:
            band = val.split('_')[0]
            if len(band)!=1:
                raise IOError('Multi character variables in input expressions represent functions or constants. Unknown function or constant : %s'%(val))

            elif (band.lower() != band):
                raise IOError('Single character upper case letters are used for constant. No available constant named %s'%(val))

        return unknown, known

#######Command line parsing
def detailedHelp():
    '''
    Return the detailed help message.
    '''
    msg = helpStr + '\n\n'+ \
              'Available Functions \n' + \
              '********************\n' + \
              str(IML.fnDict.keys()) + '\n\n' + \
              'Available Constants \n' + \
              '********************\n' + \
              str(IML.constDict.keys()) + '\n\n' + \
              'Available DataTypes -> numpy code mapping  \n' + \
              '*****************************************  \n'+ \
              IML.printNUMPYMap() + '\n'

    return msg

class customArgparseAction(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        '''
        The action to be performed.
        '''
        print(detailedHelp())
        parser.print_help()
        parser.exit()

def firstPassCommandLine():
    '''
    Take a first parse at command line parsing.
    Read only the basic required fields
    '''

    #####Create the generic parser to get equation and output format first
    parser = argparse.ArgumentParser(description='ISCE Band math calculator.',
            formatter_class=IML.customArgparseFormatter)

#    help_parser = subparser.add_
    parser.add_argument('-H','--hh', nargs=0, action=customArgparseAction,
            help='Display detailed help information.')
    parser.add_argument('-e','--eval', type=str, required=True, action='store',
            help='Expression to evaluate.', dest='equation')
    parser.add_argument('-o','--out', type=str, default=None, action='store',
            help='Name of the output file', dest='out')
    parser.add_argument('-s','--scheme',type=str, default='BSQ', action='store',
            help='Output file format.', dest='scheme')
    parser.add_argument('-t','--type', type=str, default='float', action='store',
            help='Output data type.', dest='dtype')
    parser.add_argument('-d','--debug', action='store_true', default=False,
            help='Print debugging statements', dest='debug')
    parser.add_argument('-n','--noxml', action='store_true', default=False,
            help='Do not create an ISCE XML file for the output.', dest='noxml')

    #######Parse equation and output format first
    args, files = parser.parse_known_args()

    #####Check the output scheme for errors
    if args.scheme.upper() not in ['BSQ', 'BIL', 'BIP']:
        raise IOError('Unknown output scheme: %s'%(args.scheme))
    iMath['outScheme'] = args.scheme.upper()

    npType = IML.NUMPY_type(args.dtype)
    iMath['outType'] = npType

    return args, files


def parseInputFile(varname, args):
    '''
    Get the input string corresponding to given variable name.
    '''

    inarg = varname.strip()
    ####Keyname corresponds to specific
    key = '--' + inarg

    if len(varname.strip()) > 1:
        raise IOError('Input variable names should be single characters.\n' +
                'Invalid variable name: %s'%varname)

    if (inarg != inarg.lower()):
        raise IOError('Input variable names should be lower case. \n' +
                'Invalud variable name: %s'%varname)

    #####Create a simple parser
    parser = IML.customArgumentParser(description='Parser for band math.',
            add_help=False)
    parser.add_argument(key, type=str, required=True, action='store',
            help='Input string for a particular variable.', dest='instr')

    try:
        infile, rest = parser.parse_known_args(args)
    except:
        raise SyntaxError('Input file : "%s" not defined on command line'%varname)
    return infile.instr, rest


def createNamespace():
    '''
    Hand utility if you want to use imageMath.py from within other python code.
    '''
    from argparse import Namespace
    g = Namespace()
    g.debug = False
    g.dtype = 'float'
    g.equation = None
    g.hh = None
    g.noxml = False
    g.out = None
    g.scheme = None
    return g

def mergeBbox(inlist):
    '''
    Merge Bboxes of input files.
    '''
    if len(inlist) == 0 :
        return None


    ref = np.array(inlist[0])

    diff = np.zeros((len(inlist), 4))
    for ind in range(1, len(inlist)):
        cand = np.array(inlist[ind])
        diff[ind,: ] = cand - ref

    diff = np.max(np.abs(diff), axis=0)

    if np.any(diff > 1.0e-5):
        print('Bounding boxes dont match. Not adding bbox info.')
        return None
    else:
        return ref

#######The main driver that puts everything together
def main(args, files):
    #######Set up logger appropriately
    logger = IML.createLogger(args.debug, name='imageMath')
    logger.debug('Known: '+ str(args))
    logger.debug('Optional: '+ str(files))


    #######Determine number of input and output bands
    bandList = []
    iMath['equations'] = []
    for ii,expr in enumerate(args.equation.split(';')):

        #####Now parse the equation to get the file names used
        nsp = NumericStringParser(expr.strip())
        logger.debug('Input Expression: %d : %s'%(ii, expr))
        bands, known = nsp.parse()
        logger.debug('Unknown variables: ' + str(bands))
        logger.debug('Known variables: ' + str(known))

        iMath['equations'].append(expr)
        bandList = bandList + bands

    bandList = IML.uniqueList(bandList)

    numOutBands = len(iMath['equations'])
    logger.debug('Number of output bands = %d'%(numOutBands))
    logger.debug('Number of input bands used = %d'%(len(bandList)))
    logger.debug('Input bands used = ' + str(bandList))


    #####Determine unique images from the bandList
    fileList = IML.bandsToFiles(bandList, logger)


    ######Create input memmaps
    for ii,infile in enumerate(fileList):
        if type(files) == list:
            fstr, files = parseInputFile(infile, files)
        else:
            fstr = getattr(files, infile)

        logger.debug('Input string for File %d: %s: %s'%(ii, infile, fstr))

        if len(fstr.split(';')) > 1:
            fmap = IML.mmapFromStr(fstr, logger)
            bbox = None
        else:
            fmap = IML.mmapFromISCE(fstr, logger)
            bbox = IML.getGeoInfo(fstr)


        iMath['inFiles'][infile] = fmap

        if len(fmap.bands) == 1:
            iMath['inBands'][infile] = fmap.bands[0]

        for ii in range(len(fmap.bands)):
            iMath['inBands']['%s_%d'%(infile, ii)] = fmap.bands[ii]

        if bbox is not None:
            iMath['bboxes'].append(bbox)

    if type(files) == list:
        if len(files):
            raise IOError('Unused input variables set:\n'+ ' '.join(files))

    #######Some debugging
    logger.debug('List of available bands: ' + str(iMath['inBands'].keys()))

    ####If used in calculator mode.
    if len(bandList) == 0:
        dataDict=dict(IML.fnDict.items() + IML.constDict.items())
        logger.info('Calculator mode. No output files created')
        for ii, equation in enumerate(iMath['equations']):
            res=eval(expr, dataDict)
            logger.info('Output Band %d : %f '%(ii, res))

        sys.exit(0)
    else:
        if args.out is None:
            raise IOError('Output file has not been defined.')

    #####Check if all bands in bandList have been accounted for
    for band in bandList:
        if band not in iMath['inBands'].keys():
            raise ValueError('Undefined band : %s '%(band))

    ######Check if all the widths match
    widths = [img.width for var,img in iMath['inFiles'].items() ]
    if len(widths) != widths.count(widths[0]):
        logger.debug('Widths of images: ' +
                str([(var, img.name, img.width) for var,img in iMath['inFiles'].items()]))
        raise IOError('Input images are not of same width')

    iMath['width'] = widths[0]
    logger.debug('Output Width =  %d'%(iMath['width']))

    #######Check if all the lengths match
    lengths=[img.length for var,img in iMath['inFiles'].items()]
    if len(lengths) != lengths.count(lengths[0]):
        logger.debug('Lengths of images: ' +
             str([(var, img.name, img.length) for var,img in iMath['inFiles'].items()]))

        raise IOError('Input images are not of the same length')

    iMath['length'] = lengths[0]
    logger.debug('Output Length = %d'%(iMath['length']))

    #####Now create the output file
    outmap = IML.memmap(args.out, mode='write', nchannels=numOutBands,
            nxx=iMath['width'], nyy=iMath['length'], scheme=iMath['outScheme'],
            dataType=iMath['outType'])

    logger.debug('Creating output ISCE mmap with \n' +
            'file = %s \n'%(args.out) +
            'bands = %d \n'%(numOutBands) +
            'width = %d \n'%(iMath['width']) +
            'length = %d \n'%(iMath['length'])+
            'scheme = %s \n'%(iMath['outScheme']) +
            'dtype = %s \n'%(iMath['outType']))

    iMath['outBands'] = outmap.bands

    #####Start evaluating the expressions

    ####Set up the name space to use
    dataDict=dict(IML.fnDict.items() | IML.constDict.items())
    bands = iMath['inBands']
    outBands = iMath['outBands']

    ####Array representing columns
    dataDict['COL'] = np.arange(iMath['width'], dtype=np.float)

    #####Replace ^ by **
    for lineno in range(int(iMath['length'])):

        ####Setting row number
        dataDict['ROW'] = lineno*1.0

        ####Load one line from each of the the bands
        for band in bandList:  #iMath['inBands'].iteritems():
            dataDict[band] = bands[band][lineno,:]

        ####For each output band
        for kk,expr in enumerate(iMath['equations']):
            res = eval(expr, dataDict)
            outBands[kk][lineno,:] = res

    ######Determine common bbox if any
    outputBbox = mergeBbox(iMath['bboxes'])

    ######Render ISCE XML if needed
    if not args.noxml:
        IML.renderISCEXML(args.out, numOutBands,
                iMath['length'], iMath['width'],
                iMath['outType'], iMath['outScheme'],
                bbox = outputBbox,
                descr = ' '.join(sys.argv))


if __name__ == '__main__':
    args, files = firstPassCommandLine()
    print('args: ', args)
    print('files: ', files)
    main(args, files)
