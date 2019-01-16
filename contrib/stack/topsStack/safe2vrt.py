#!/usr/bin/env python3

import isce
from osgeo import gdal
import argparse
import numpy as np
import matplotlib.pyplot as plt
from isceobj.Sensor.TOPS.Sentinel1 import Sentinel1
from osgeo import gdal

def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser(description='Stitch 3 swath magnitudes into single image for display.')
    parser.add_argument('-i', '--input', dest='safe', type=str, required=True,
            help='List of SAFE files as input.')
    parser.add_argument('-o', '--output', dest='outvrt', type=str, default='stitched.vrt',
            help='Output VRT file')
    parser.add_argument('-b', '--bbox', dest='bbox', type=float, nargs='*',
            default=None, help='Optional bounding box to use')
    parser.add_argument('-s', '--swaths', dest='swaths', type=int, nargs='*',
            default=[1,2,3], help='Swath numbers to use. Default is to use all.')
    
    inps = parser.parse_args()

    for swt in inps.swaths:
        if swt not in [1,2,3]:
            raise Exception('Swath numbers can only be 1,2 or 3')

    if inps.bbox is not None:
        if len(inps.bbox) != 4:
            raise Exception('Input bbox convention - SNWE. Length of user input {0}'.format(len(inps.bbox)))

        if inps.bbox[1] <= inps.bbox[0]:
            raise Exception('Bbox convention - SNWE. South > North in user input.')

        if inps.bbox[3] <= inps.bbox[2]:
            raise Exception('Bbox convention - SNWE. West > East in user input.')

    inps.safe = inps.safe.strip().split()

    return inps


class Swath(object):
    '''
    Information holder.
    '''

    def __init__(self, reader):
        '''
        Constructor.
        '''

        self.prod = reader.product
        self.tiff = reader.tiff[0]
        self.xsize = None
        self.ysize = None
        self.xoffset = None
        self.yoffset = None

        self.setSizes()

    def setSizes(self):
        '''
        Set xsize and ysize.
        '''

        ds = gdal.Open(self.tiff, gdal.GA_ReadOnly)
        self.xsize = ds.RasterXSize
        self.ysize = ds.RasterYSize 
        ds = None


    def __str__(self):
        '''
        Description.
        '''
        outstr = ''
        outstr  += 'Tiff file: {0}\n'.format(self.tiff)
        outstr  += 'Number of Bursts: {0}\n'.format(self.prod.numberOfBursts)
        outstr  += 'Dimensions: ({0},{1})\n'.format(self.ysize, self.xsize)
        outstr  += 'Burst dims: ({0},{1})\n'.format(self.burstLength, self.burstWidth)
        return outstr

    @property
    def sensingStart(self):
        return self.prod.bursts[0].sensingStart

    @property
    def sensingStop(self):
        return self.prod.bursts[-1].sensingStop

    @property
    def nearRange(self):
        return self.prod.bursts[0].startingRange

    @property
    def dr(self):
        return self.prod.bursts[0].rangePixelSize

    @property
    def dt(self):
        return self.prod.bursts[0].azimuthTimeInterval

    @property
    def burstWidth(self):
        return self.prod.bursts[0].numberOfSamples

    @property
    def burstLength(self):
        return self.prod.bursts[0].numberOfLines

    @property
    def farRange(self):
        return self.nearRange + (self.burstWidth-1)*self.dr


class VRTConstructor(object):
    '''
    Class to construct a large image.
    '''
    def __init__(self, y, x, dtype='CInt16'):
        self.ysize = y
        self.xsize = x
        self.dtype = dtype

        self.tref = None
        self.rref = None
        self.dt = None
        self.dr = None

        ####Counters for tracking
        self.nswaths = 0
        self.nbursts = 0

        ####VRT text handler
        self.vrt = ''

    def setReferenceTime(self, tim):
        self.tref = tim

    def setReferenceRange(self, rng):
        self.rref = rng

    def setTimeSpacing(self, dt):
        self.dt = dt

    def setRangeSpacing(self, dr):
        self.dr = dr

    def initVRT(self):
        '''
        Build the top part of the VRT.
        '''

        head = '''<VRTDataset rasterXSize="{0}" rasterYSize="{1}">
    <VRTRasterBand dataType="{2}" band="1">
        <NoDataValue>0.0</NoDataValue>
'''
        self.vrt += head.format(self.xsize, self.ysize, self.dtype)


    def finishVRT(self):
        '''
        Build the last part of the VRT.
        '''
        tail = '''    </VRTRasterBand>
</VRTDataset>'''

        self.vrt += tail


    def addSwath(self, swath):
        '''
        Add one swath to the VRT.
        '''
        for ind, burst in enumerate(swath.prod.bursts):
            xoff = np.int(np.round( (burst.startingRange - self.rref)/self.dr))
            yoff = np.int(np.round( (burst.sensingStart - self.tref).total_seconds() / self.dt))

            self.addBurst( burst, swath.tiff, yoff, xoff, swath.ysize, swath.xsize)


        self.nswaths += 1



    def addBurst(self, burst, tiff, yoff, xoff, tysize, txsize):
        '''
        Add one burst to the VRT.
        '''

        tyoff = int((burst.burstNumber-1)*burst.numberOfLines + burst.firstValidLine)
        txoff = int(burst.firstValidSample)

        fyoff = int(yoff + burst.firstValidLine)
        fxoff = int(xoff + burst.firstValidSample)

        wysize = int(burst.numValidLines)
        wxsize = int(burst.numValidSamples)

        tmpl = '''        <SimpleSource>
            <SourceFilename relativeToVRT="1">{tiff}</SourceFilename>
            <SourceBand>1</SourceBand>
            <SourceProperties RasterXSize="{txsize}" RasterYSize="{tysize}" DataType="{dtype}" BlockXSize="{txsize}" BlockYSize="1"/>
            <SrcRect xOff="{txoff}" yOff="{tyoff}" xSize="{wxsize}" ySize="{wysize}"/>
            <DstRect xOff="{fxoff}" yOff="{fyoff}" xSize="{wxsize}" ySize="{wysize}"/>
        </SimpleSource>
'''

        self.vrt += tmpl.format( tyoff=tyoff, txoff=txoff,
                                 fyoff=fyoff, fxoff=fxoff,
                                wxsize=wxsize, wysize=wysize,
                                tiff=tiff, dtype=self.dtype,
                                tysize=tysize, txsize=txsize)


        self.nbursts += 1

    def writeVRT(self, outfile):
        '''
        Write VRT to file.
        '''

        with open(outfile, 'w') as fid:
            fid.write(self.vrt)




if __name__ == '__main__':
    '''
    Main driver.
    '''

    #Parse command line inputs
    inps = cmdLineParse()

    ###Number of safe files
    nSafe = len(inps.safe)

    
    ####Parse individual swaths
    swaths = []


    for safe in inps.safe:
        for swathnum in inps.swaths:
            obj = Sentinel1()
            obj.configure()

            obj.safe = [safe]
            obj.swathNumber = swathnum
            obj.output = '{0}-SW{1}'.format(safe,swathnum)

            ###Only parse and no extract
            obj.parse()

            swt = Swath(obj)

            swaths.append(swt)



    ###Identify the 4 corners and dimensions
    topSwath = min(swaths, key = lambda x: x.sensingStart)
    botSwath = max(swaths, key = lambda x: x.sensingStop)
    leftSwath = min(swaths, key = lambda x: x.nearRange)
    rightSwath = max(swaths, key = lambda x: x.farRange)

    totalWidth = int( np.round((rightSwath.farRange - leftSwath.nearRange)/leftSwath.dr + 1))
    totalLength = int(np.round((botSwath.sensingStop - topSwath.sensingStart).total_seconds()/topSwath.dt + 1 ))


    ###Start building the VRT
    builder = VRTConstructor(totalLength, totalWidth)
    builder.setReferenceRange(leftSwath.nearRange)
    builder.setReferenceTime(topSwath.sensingStart)
    builder.setRangeSpacing(topSwath.dr)
    builder.setTimeSpacing(topSwath.dt)


    builder.initVRT()
    for swath in swaths:
        builder.addSwath(swath)

    builder.finishVRT()
    builder.writeVRT('test.vrt')
