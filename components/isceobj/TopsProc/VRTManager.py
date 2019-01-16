#!/usr/bin/env python3
import numpy as np


gdalmap = {'FLOAT': 'Float32',
           'DOUBLE' : 'Float64',
           'CFLOAT' : 'CFloat32',
           'CINT'   : 'CInt16',
           'BYTE'   : 'Byte'}

class Swath(object):
    '''
    Information holder.
    '''

    def __init__(self, product):
        '''
        Constructor.
        '''

        self.prod = product
        self.xsize = None
        self.ysize = None
        self.xoffset = None
        self.yoffset = None

        self.setSizes()

    def setSizes(self):
        '''
        Set xsize and ysize.
        '''

        t0 = self.prod.sensingStart
        dt = self.prod.bursts[0].azimuthTimeInterval
        width = self.prod.bursts[0].numberOfSamples

        tend = self.prod.sensingStop
        nLines = int(np.round((tend-t0).total_seconds() / dt))+1

        self.xsize = width
        self.ysize = nLines


    def __str__(self):
        '''
        Description.
        '''
        outstr = ''
        outstr  += 'Number of Bursts: {0}\n'.format(self.data.numberOfBursts)
        outstr  += 'Dimensions: ({0},{1})\n'.format(self.ysize, self.xsize)
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
    def __init__(self, y, x):
        self.ysize = y
        self.xsize = x
        self.dtype = None

        self.tref = None
        self.rref = None
        self.dt = None
        self.dr = None

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

    def setDataType(self, iscetype):
        self.dtype = gdalmap[iscetype.upper()]

    def initVRT(self):
        '''
        Build the top part of the VRT.
        '''

        head = '''<VRTDataset rasterXSize="{0}" rasterYSize="{1}">'''
        self.vrt += head.format(self.xsize, self.ysize, self.dtype)

    def initBand(self, band=None):

        header='''  <VRTRasterBand dataType="{0}" band="{1}">
        <NoDataValue>0.0</NoDataValue>
'''
        self.vrt += header.format(self.dtype, band)


    def finishBand(self):
        '''
        Build the last part of the VRT.
        '''
        tail = '''    </VRTRasterBand>'''
        self.vrt += tail


    def finishVRT(self):
        tail='''</VRTDataset>'''
        self.vrt += tail


    def addSwath(self, swath, filelist, band = 1, validOnly=True):
        '''
        Add one swath to the VRT.
        '''

        if len(swath.prod.bursts) != len(filelist):
            raise Exception('Number of bursts does not match number of files provided for stitching')


        for ind, burst in enumerate(swath.prod.bursts):
            xoff = np.int(np.round( (burst.startingRange - self.rref)/self.dr))
            yoff = np.int(np.round( (burst.sensingStart - self.tref).total_seconds() / self.dt))

            infile = filelist[ind]
            self.addBurst( burst, infile, yoff, xoff, band=band, validOnly=validOnly)


    def addBurst(self, burst, infile, yoff, xoff, band=1, validOnly=True):
        '''
        Add one burst to the VRT.
        '''

        tysize = burst.numberOfLines
        txsize = burst.numberOfSamples


        if validOnly:
            tyoff = int(burst.firstValidLine)
            txoff = int(burst.firstValidSample)
            wysize = int(burst.numValidLines)
            wxsize = int(burst.numValidSamples)
            fyoff = int(yoff + burst.firstValidLine)
            fxoff = int(xoff + burst.firstValidSample)
        else:
            tyoff = 0
            txoff = 0
            wysize = tysize
            wxsize = txsize
            fyoff = int(yoff)
            fxoff = int(xoff)


        tmpl = '''        <SimpleSource>
            <SourceFilename relativeToVRT="1">{tiff}</SourceFilename>
            <SourceBand>{band}</SourceBand>
            <SourceProperties RasterXSize="{txsize}" RasterYSize="{tysize}" DataType="{dtype}"/>
            <SrcRect xOff="{txoff}" yOff="{tyoff}" xSize="{wxsize}" ySize="{wysize}"/>
            <DstRect xOff="{fxoff}" yOff="{fyoff}" xSize="{wxsize}" ySize="{wysize}"/>
        </SimpleSource>
'''

        self.vrt += tmpl.format( tyoff=tyoff, txoff=txoff,
                                 fyoff=fyoff, fxoff=fxoff,
                                wxsize=wxsize, wysize=wysize,
                                tiff=infile+'.vrt', dtype=self.dtype,
                                tysize=tysize, txsize=txsize,
                                band=band)


    def writeVRT(self, outfile):
        '''
        Write VRT to file.
        '''

        with open(outfile, 'w') as fid:
            fid.write(self.vrt)
