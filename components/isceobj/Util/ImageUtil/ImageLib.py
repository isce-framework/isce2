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
from numpy.lib.stride_tricks import as_strided
import logging
import os
import sys


#######Current list of supported unitary functions - f(x)
fnDict = { 'cos':       np.cos,
           'sin':       np.sin,
           'exp':       np.exp,
           'log':       np.log,
           'log2':      np.log2,
           'log10':     np.log10,
           'tan' :      np.tan,
           'asin':      np.arcsin,
           'acos':      np.arccos,
           'atan':      np.arctan,
           'arg':       np.angle,
           'conj':      np.conj,
           'abs' :      np.abs,
           'round' :    np.round,
           'ceil' :     np.ceil,
           'floor' :    np.floor,
           'real'  :    np.real,
           'imag' :     np.imag,
           'rad':       np.radians,
           'deg':       np.degrees,
           'sqrt':      np.sqrt,
           'mod' :      np.mod
         }

#######Current list of constants
constDict = { "PI"  : np.pi,
              "J"   : np.complex(0.0, 1.0),
              "I"   : np.complex(0.0, 1.0),
              "E"   : np.exp(1.0),
              "NAN" : np.nan,
              "ROW" : None,
              "COL" : None
            }

######To deal with data types
'''
    Translation between user inputs and numpy types.

    Single char codes are case sensitive (Numpy convention).

    Multiple char codes are case insensitive.
'''

####Signed byte
byte_tuple = ('B', 'byte', 'b8', 'b1')

####Unsigned byte
ubyte_tuple = ('B', 'ubyte', 'ub8', 'ub1')

####Short int
short_tuple = ('h', 'i2', 'short', 'int2', 'int16')

####Unsigned short int
ushort_tuple = ('H', 'ui2', 'ushort', 'uint2', 'uint16')

####Integer
int_tuple = ('i', 'i4', 'i32', 'int', 'int32','intc')

####Unsigned int 
uint_tuple = ('I', 'ui4', 'ui32', 'uint', 'uint32', 'uintc')

####Long int
long_tuple = ('l', 'l8', 'l64', 'long', 'long64', 'longc',
            'intpy', 'pyint', 'int64')

####Unsigned long int 
ulong_tuple = ('L', 'ul8', 'ul64', 'ulong', 'ulong64', 'ulongc',
            'uintpy', 'pyuint', 'uint64')

######Float 
float_tuple =('f', 'float', 'single', 'float32', 'real4', 'r4')

######Complex float 
cfloat_tuple = ('F', 'c8','complex','complex64','cfloat')

#####Double
double_tuple = ('d', 'double', 'real8', 'r8', 'float64',
        'floatpy', 'pyfloat')

######Complex Double
cdouble_tuple=('D', 'c16', 'complex128', 'cdouble')

####Mapping to numpy data type
typeDict = {}

for dtuple in (byte_tuple, ubyte_tuple,
              short_tuple, short_tuple,
              int_tuple, uint_tuple,
              long_tuple, ulong_tuple,
              float_tuple, cfloat_tuple,
              double_tuple, cdouble_tuple):

    for dtype in dtuple:
        typeDict[dtype] = dtuple[0]


def NUMPY_type(instr):
    '''
    Translates a given string into a numpy data type string.
    '''

    tstr = instr.strip()

    if len(tstr) == 1:
        key = tstr
    else:
        key = tstr.lower()
   
    try:
        npType = typeDict[key]
    except:
        raise ValueError('Unknown data type provided : %s '%(instr))

    return npType


isceTypeDict = { 
                    "f" : "FLOAT",
                    "F" : "CFLOAT",
                    "d" : "DOUBLE",
                    "h" : "SHORT",
                    "i" : "INT",
                    "l" : "LONG",
                    "B" : "BYTE"
               }


def printNUMPYMap():
    import json
    return json.dumps(typeDict, indent=4, sort_keys=True)

#########Classes and utils to deal with strings ###############
def isNumeric(s):
    '''
    Determine if a string is a number.
    '''
    try:
        i = float(s)
        return True
    except (ValueError, TypeError):
        return False

def uniqueList(seq):
    '''
    Returns a list with unique elements in a list.
    '''
    seen = set()
    seen_add = seen.add
    return [ x for x in seq if x not in seen and not seen_add(x)]

#######Create the logger for the application
def createLogger(debug, name='imageMath'):
    '''
    Creates an appopriate logger.
    '''
#    logging.basicConfig()
    logger = logging.getLogger(name)
    consoleHandler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name) s - %(levelname)s\n%(message)s')
    consoleHandler.setFormatter(formatter)
    if debug:
        logger.setLevel(logging.DEBUG)
        consoleHandler.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        consoleHandler.setLevel(logging.INFO)

    logger.addHandler(consoleHandler)

    return logger

#########Create command line parsers 
class customArgparseFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass

class customArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise Exception(message)

def bandsToFiles(bandList, logger):
    '''
    Take a list of band names and convert it to file names.
    '''
    flist = []
    for band in bandList:
        names = band.split('_')
        if len(names) > 2:
            logger.error('Invalid band name: %s'%band)
        
        if names[0] not in flist:
            flist.append(names[0])

    logger.debug('Number of input files : %d'%len(flist))
    logger.debug('Input files: ' + str(flist))
    return flist


##########Classes and utils for memory maps
class memmap(object):
    '''Create the memap object.'''
    def __init__(self,fname, mode='readonly', nchannels=1, nxx=None, nyy=None, scheme='BSQ', dataType='f'):
        '''Init function.'''

        fsize = np.zeros(1, dtype=dataType).itemsize

        if nxx is None:
            raise ValueError('Undefined file width for : %s'%(fname))

        if mode=='write':
            if nyy is None:
                raise ValueError('Undefined file length for opening file: %s in write mode.'%(fname))
        else:
            try:
                nbytes = os.path.getsize(fname)
            except:
                raise ValueError('Non-existent file : %s'%(fname))

            if nyy is None:
                nyy = nbytes//(fsize*nchannels*nxx)

                if (nxx*nyy*fsize*nchannels) != nbytes:
                    raise ValueError('File size mismatch for %s. Fractional number of lines'(fname))
            elif (nxx*nyy*fsize*nchannels) > nbytes:
                    raise ValueError('File size mismatch for %s. Number of bytes expected: %d'%(nbytes))
             

        self.name = fname
        self.width = nxx
        self.length = nyy

        ####List of memmap objects
        acc = []

        ####Create the memmap for the full file
        nshape = nchannels*nyy*nxx
        omap = np.memmap(fname, dtype=dataType, mode=mode, 
                shape = (nshape,))

        if scheme.upper() == 'BIL':
            nstrides = (nchannels*nxx*fsize, fsize)

            for band in range(nchannels):
                ###Starting offset
                noffset = band*nxx

                ###Temporary view
                tmap = omap[noffset:]

                ####Trick it into creating a 2D array
                fmap = as_strided(tmap, shape=(nyy,nxx), strides=nstrides)

                ###Add to list of objects
                acc.append(fmap)

        elif scheme.upper() == 'BSQ':
            nstrides = (fsize, fsize)

            for band in range(nchannels):
                ###Starting offset
                noffset = band*nxx*nyy

                ###Temporary view
                tmap = omap[noffset:noffset+nxx*nyy]

                ####Reshape into 2D array
                fmap = as_strided(tmap, shape=(nyy,nxx))

                ###Add to lits of objects
                acc.append(fmap)

        elif scheme.upper() == 'BIP':
            nstrides = (nchannels*nxx*fsize,nchannels*fsize)

            for band in range(nchannels):
                ####Starting offset
                noffset = band

                ####Temporary view
                tmap = omap[noffset:]

                ####Trick it into interpreting ot as a 2D array
                fmap = as_strided(tmap, shape=(nyy,nxx), strides=nstrides)

                ####Add to the list of objects
                acc.append(fmap)

        else:
            raise ValueError('Unknown file scheme: %s for file %s'%(scheme,fname))

        ######Assigning list of objects to self.bands
        self.bands = acc

    def flush(self):
        '''
        If mmap opened in write mode, would be useful to have flush functionality on old systems.
        '''

        self.bands[0].base.base.flush()


class memmapGDAL(object):
    '''
    Create a memmap like object from GDAL.
    '''

    from osgeo import gdal

    class BandWrapper:
        '''
        Wrap a GDAL band in a numpy like slicable object.
        '''

        def __init__(self, dataset, band):
            '''
            Init from a GDAL raster band.
            '''

            self.data = dataset.GetRasterBand(band)
            self.width = dataset.RasterXSize
            self.length = data.RasterYSize

        def __getitem__(self, *args):
            
            xmin = max(int(args[0][1].start),0)
            xmax = min(int(args[0][1].stop)+xmin, self.width) - xmin
            ymin = max(int(args[0][0].start),0)
            ymax = min(int(args[0][1].stop)+ymin, self.length) - ymin

            res = self.data.ReadAsArray(xmin, ymin, xmax,ymax)
            return res

        def __del__(self):
            self.data = None


    def __init__(self, fname):
        '''
        Constructor.
        '''
        
        self.name = fname
        self.data = gdal.Open(self.name, gdal.GA_ReadOnly)
        self.width = self.data.RasterXSize
        self.length = self.data.RasterYSize

        self.bands = []
        for ii in range(self.data.RasterCount):
            self.bands.append( BandWrapper(self.data, ii+1))

    def __del__(self):
        self.data = None


def loadImage(fname):
    '''
    Load into appropriate image object.
    '''
    try:
        import iscesys
        import isceobj
        from iscesys.Parsers.FileParserFactory import createFileParser
    except:
        raise ImportError('ISCE has not been installed or is not importable')

    if not fname.endswith('.xml'):
        dataName = fname
        metaName = fname + '.xml'
    else:
        metaName = fname
        dataName = os.path.splitext(fname)[0]

    parser = createFileParser('xml')
    prop,fac,misc = parser.parse(metaName)

    if 'reference' in prop:
        img=isceobj.createDemImage()
        img.init(prop,fac,misc)
    elif 'number_good_bytes' in prop:
        img = isceobj.createRawImage()
        img.init(prop,fac,misc)
    else:
        img = isceobj.createImage()
        img.init(prop,fac,misc)

    img.setAccessMode('READ')
    return img, dataName, metaName


def loadGDALImage(fname):
    '''
    Similar to loadImage but only returns metadata.
    '''

    from osgeo import gdal

    class Dummy(object):
        pass


    ds = gdal.Open(fname, gdal.GA_ReadOnly)
    drv = ds.GetDriver()
    bnd = ds.GetRasterBand(1)

    img = Dummy()
    img.bands = ds.RasterCount 
    img.width = ds.RasterXSize
    img.length = ds.RasterYSize
    img.scheme = drv.GetDescription()
    img.dataType = gdal.GetDataTypeByName(bnd.DataType)

    bnd = None
    drv = None
    ds = None

    return img

def mmapFromISCE(fname, logger=None):
    '''
    Create a file mmap object using information in an ISCE XML.
    '''
    try:
        img, dataName, metaName = loadImage(fname)
        isceFile = True
    except:
        try:
            img = loadGDALImage(fname)
            isceFile=False
            dataName = fname
        except:
            raise Exception('Input file: {0} should either be an ISCE image / GDAL image. Appears to be neither'.format(fname))

    if logger is not None:
        logger.debug('Creating readonly ISCE mmap with \n' +
            'file = %s \n'%(dataName) + 
            'bands = %d \n'%(img.bands) + 
            'width = %d \n'%(img.width) + 
            'length = %d \n'%(img.length)+
            'scheme = %s \n'%(img.scheme) +
            'dtype = %s \n'%(img.dataType))

    if isceFile:
        mObj = memmap(dataName, nchannels=img.bands,
            nxx=img.width, nyy=img.length, scheme=img.scheme,
            dataType=NUMPY_type(img.dataType))
    else:
        mObj = memmapGDAL(dataName)

    return mObj

def getGeoInfo(fname):
    '''
    Get the geobox information for a given image.
    '''
    img = loadImage(fname)[0]
    
    bbox = [img.getFirstLatitude(), img.getFirstLongitude(),
            img.getDeltaLatitude(), img.getDeltaLongitude()]

    if all([x is not None for x in bbox]):
        return bbox
    else:
        return None


def mmapFromStr(fstr, logger):
    '''
    Create a file mmap object using information provided on command line.

    Grammar = 'filename;width;datatype;bands;scheme'
    '''
    def grammarError():
        raise SyntaxError("Undefined image : %s \n" +
                "Grammar='filename;width;datatype;bands;scheme'"%(fstr))

    parms = fstr.split(';')
    logger.debug('Input string: ' + str(parms))
    if len(parms) < 2:
        grammarError()

    try:
        fname = parms[0]
        width = int(parms[1])
        if len(parms)>2:
            datatype = NUMPY_type(parms[2])
        else:
            datatype='f'

        if len(parms)>3:
            bands = int(parms[3])
        else:
            bands = 1

        if len(parms)>4:
            scheme = parms[4].upper()
        else:
            scheme = 'BSQ'

        if scheme not in ['BIL', 'BIP', 'BSQ']:
            raise IOError('Invalid file interleaving scheme: %s'%scheme)
    except:
        grammarError()

    logger.debug('Creating readonly mmap from string with \n' +
            'file = %s \n'%(fname) + 
            'bands = %d \n'%(bands) + 
            'width = %d \n'%(width) + 
            'scheme = %s \n'%(scheme) +
            'dtype = %s \n'%(datatype))


    mObj = memmap(fname, nchannels=bands, nxx=width,
            scheme=scheme, dataType=datatype)

    return mObj

    pass

#######ISCE XML rendering
def renderISCEXML(fname, bands, nyy, nxx, datatype, scheme,
        bbox=None, descr=None):
    '''
    Renders an ISCE XML with the right information.
    '''
    
    try:
        import isce
        import isceobj
    except:
        raise ImportError('ISCE has not been installed or is not importable.')

    
    img = isceobj.createImage()
    img.filename = fname
    img.scheme = scheme
    img.width=nxx
    img.length = nyy
    try:
        img.dataType = isceTypeDict[datatype]
    except:
        try:
            img.dataType = isceTypeDict[NUMPY_type(datatype)]
        except:
            raise Exception('Processing complete but ISCE XML not written as the data type is currently not supported by ISCE Image Api')

    if bbox is not None:
        img.setFirstLatitude(bbox[0])
        img.setFirstLongitude(bbox[1])
        img.setDeltaLatitude(bbox[2])
        img.setDeltaLongitude(bbox[3])

    if descr is not None:
        img.addDescription(descr)

    img.bands = bands
    img.renderVRT()    ###PSA - needed since all reading is now via VRTs
    img.setAccessMode('read')
    img.createImage()
    img.finalizeImage()
    img.renderHdr()
    return


if __name__ == '__main__':
    args, files = firstPassCommandLine()
#    print('args: ', args)
#    print('files: ', files)
    main(args, files)
