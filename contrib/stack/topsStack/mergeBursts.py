#
# Author: Piyush Agram
# Copyright 2016
#
# Heresh Fattahi, updated for stack processing


import numpy as np 
import os
import isce
import isceobj
import datetime
import logging
import argparse
from isceobj.Util.ImageUtil import ImageLib as IML
from isceobj.Util.decorators import use_api
import s1a_isce_utils as ut
import glob


def createParser():
    '''
    Create command line parser.
    '''

    parser = argparse.ArgumentParser( description='Generate offset field between two Sentinel swaths')
    parser.add_argument('-i', '--inp_master', type=str, dest='master', required=True,
            help='Directory with the master image')

    parser.add_argument('-s', '--stack', type=str, dest='stack', default = None,
                help='Directory with the stack xml files which includes the common valid region of the stack')

    parser.add_argument('-d', '--dirname', type=str, dest='dirname', required=True,
                help='directory with products to merge')

    parser.add_argument('-o', '--outfile', type=str, dest='outfile', required=True,
            help='Output merged file')

    parser.add_argument('-m', '--method', type=str, dest='method', default='avg',
            help = 'Method: top / bot/ avg')

    parser.add_argument('-a', '--aligned', action='store_true', dest='isaligned',
            default=False, help='Use master information instead of coreg for merged grid.')

    parser.add_argument('-l', '--multilook', action='store_true', dest='multilook', default=False,
                    help = 'Multilook the merged products. True or False')

    parser.add_argument('-A', '--azimuth_looks', type=str, dest='numberAzimuthLooks', default=3,
            help = 'azimuth looks')

    parser.add_argument('-R', '--range_looks', type=str, dest='numberRangeLooks', default=9,
            help = 'range looks')

    parser.add_argument('-n', '--name_pattern', type=str, dest='namePattern', default='fine*int',
                help = 'a name pattern of burst products that will be merged. default: fine. it can be lat, lon, los, burst, hgt, shadowMask, incLocal')

    parser.add_argument('-v', '--valid_only', action='store_true', dest='validOnly', default=False,
                help = 'True for SLC, int and coherence. False for geometry files (lat, lon, los, hgt, shadowMask, incLocal).')

    parser.add_argument('-u', '--use_virtual_files', action='store_true', dest='useVirtualFiles', default=False,
                    help = 'writing only a vrt of merged file. Default: True.')

    parser.add_argument('-M', '--multilook_tool', type=str, dest='multilookTool', default='isce',
            help = 'The tool used for multi-looking')

    parser.add_argument('-N', '--no_data_value', type=float, dest='noData', default=None,
            help = 'no data value when gdal is used for multi-looking')

    return parser


def cmdLineParse(iargs = None):
    '''
    Command line parser.
    '''

    parser = createParser()
    inps = parser.parse_args(args=iargs)
    if inps.method not in ['top', 'bot', 'avg']:
        raise Exception('Merge method should be in top / bot / avg')

    return inps

def mergeBurstsVirtual(frame, referenceFrame, fileList, outfile, validOnly=True):
    '''
    Merging using VRTs.
    '''
    
    from VRTManager import Swath, VRTConstructor


    swaths = [Swath(x) for x in frame]
    refSwaths = [Swath(x) for x in referenceFrame]
    ###Identify the 4 corners and dimensions
    #topSwath = min(swaths, key = lambda x: x.sensingStart)
    #botSwath = max(swaths, key = lambda x: x.sensingStop)
    #leftSwath = min(swaths, key = lambda x: x.nearRange)
    #rightSwath = max(swaths, key = lambda x: x.farRange)
    topSwath = min(refSwaths, key = lambda x: x.sensingStart)
    botSwath = max(refSwaths, key = lambda x: x.sensingStop)
    leftSwath = min(refSwaths, key = lambda x: x.nearRange)
    rightSwath = max(refSwaths, key = lambda x: x.farRange)


    totalWidth = int( np.round((rightSwath.farRange - leftSwath.nearRange)/leftSwath.dr + 1))
    totalLength = int(np.round((botSwath.sensingStop - topSwath.sensingStart).total_seconds()/topSwath.dt + 1 ))


    ###Determine number of bands and type
    img  = isceobj.createImage()
    img.load( fileList[0][0] + '.xml')
    bands = img.bands 
    dtype = img.dataType
    img.filename = outfile


    #####Start the builder
    ###Now start building the VRT and then render it
    builder = VRTConstructor(totalLength, totalWidth)
    builder.setReferenceTime( topSwath.sensingStart)
    builder.setReferenceRange( leftSwath.nearRange)
    builder.setTimeSpacing( topSwath.dt )
    builder.setRangeSpacing( leftSwath.dr)
    builder.setDataType( dtype.upper())

    builder.initVRT()


    ####Render XML and default VRT. VRT will be overwritten.
    img.width = totalWidth
    img.length =totalLength
    img.renderHdr()


    for bnd in range(1,bands+1):
        builder.initBand(band = bnd)

        for ind, swath in enumerate(swaths):
            ####Relative path
            relfilelist = [os.path.relpath(x, 
                os.path.dirname(outfile))  for x in fileList[ind]]

            builder.addSwath(swath, relfilelist, band=bnd, validOnly=validOnly)

        builder.finishBand()
    builder.finishVRT()

    with open(outfile + '.vrt', 'w') as fid:
        fid.write(builder.vrt)



def mergeBursts(frame, fileList, outfile,
        method='top'):
    '''
    Merge burst products into single file.
    Simple numpy based stitching
    '''

    ###Check against metadata
    if frame.numberOfBursts != len(fileList):
        print('Warning : Number of burst products does not appear to match number of bursts in metadata')


    t0 = frame.bursts[0].sensingStart
    dt = frame.bursts[0].azimuthTimeInterval
    width = frame.bursts[0].numberOfSamples

    #######
    tstart = frame.bursts[0].sensingStart 
    tend = frame.bursts[-1].sensingStop
    nLines = int( np.round((tend - tstart).total_seconds() / dt)) + 1
    print('Expected total nLines: ', nLines)


    img = isceobj.createImage()
    img.load( fileList[0] + '.xml')
    bands = img.bands
    scheme = img.scheme
    npType = IML.NUMPY_type(img.dataType)

    azMasterOff = []
    for index in range(frame.numberOfBursts):
        burst = frame.bursts[index]
        soff = burst.sensingStart + datetime.timedelta(seconds = (burst.firstValidLine*dt)) 
        start = int(np.round((soff - tstart).total_seconds() / dt))
        end = start + burst.numValidLines

        azMasterOff.append([start,end])

        print('Burst: ', index, [start,end])

        if index == 0:
            linecount = start

    outMap = IML.memmap(outfile, mode='write', nchannels=bands,
            nxx=width, nyy=nLines, scheme=scheme, dataType=npType)

    for index in range(frame.numberOfBursts):
        curBurst = frame.bursts[index]
        curLimit = azMasterOff[index]

        curMap = IML.mmapFromISCE(fileList[index], logging)

        #####If middle burst
        if index > 0:
            topBurst = frame.bursts[index-1]
            topLimit = azMasterOff[index-1]
            topMap = IML.mmapFromISCE(fileList[index-1], logging)

            olap = topLimit[1] - curLimit[0]

            print("olap: ", olap)

            if olap <= 0:
                raise Exception('No Burst Overlap')


            for bb in range(bands):
                topData =  topMap.bands[bb][topBurst.firstValidLine: topBurst.firstValidLine + topBurst.numValidLines,:]

                curData =  curMap.bands[bb][curBurst.firstValidLine: curBurst.firstValidLine + curBurst.numValidLines,:]

                im1 = topData[-olap:,:]
                im2 = curData[:olap,:]

                if method=='avg':
                    data = 0.5*(im1 + im2)
                elif method == 'top':
                    data = im1
                elif method == 'bot':
                    data = im2
                else:
                    raise Exception('Method should be top/bot/avg')

                outMap.bands[bb][linecount:linecount+olap,:] = data

            tlim = olap
        else:
            tlim = 0

        linecount += tlim
            
        if index != (frame.numberOfBursts-1):
            botBurst = frame.bursts[index+1]
            botLimit = azMasterOff[index+1]
            
            olap = curLimit[1] - botLimit[0]

            if olap < 0:
                raise Exception('No Burst Overlap')

            blim = botLimit[0] - curLimit[0]
        else:
            blim = curBurst.numValidLines
       
        lineout = blim - tlim
        
        for bb in range(bands):
            curData =  curMap.bands[bb][curBurst.firstValidLine: curBurst.firstValidLine + curBurst.numValidLines,:]
            outMap.bands[bb][linecount:linecount+lineout,:] = curData[tlim:blim,:] 

        linecount += lineout
        curMap = None
        topMap = None

    IML.renderISCEXML(outfile, bands,
            nLines, width,
            img.dataType, scheme)

    oimg = isceobj.createImage()
    oimg.load(outfile + '.xml')
    oimg.imageType = img.imageType
    oimg.renderHdr()
    try:
        outMap.bands[0].base.base.flush()
    except:
        pass


def multilook(infile, outname=None, alks=5, rlks=15, multilook_tool="isce", no_data=None):
    '''
    Take looks.
    '''

    if multilook_tool=="gdal":

        from osgeo import gdal

        print("multi looking using gdal ...")
        if outname is None:
            spl = os.path.splitext(infile)
            ext = '.{0}alks_{1}rlks'.format(alks, rlks)
            outname = spl[0] + ext + spl[1]
        
        print(infile)
        ds = gdal.Open(infile + ".vrt", gdal.GA_ReadOnly)

        xSize = ds.RasterXSize
        ySize = ds.RasterYSize

        outXSize = xSize/int(rlks)
        outYSize = ySize/int(alks)

        if no_data:
            gdalTranslateOpts = gdal.TranslateOptions(format="ENVI", width=outXSize, height=outYSize, noData=no_data)
        else:
            gdalTranslateOpts = gdal.TranslateOptions(format="ENVI", width=outXSize, height=outYSize)

        gdal.Translate(outname, ds, options=gdalTranslateOpts)       
        ds = None

        
        ds = gdal.Open(outname, gdal.GA_ReadOnly)
        gdal.Translate(outname+".vrt", ds, options=gdal.TranslateOptions(format="VRT"))
        ds = None

    else:
        from mroipac.looks.Looks import Looks

        print('Multilooking {0} ...'.format(infile))

        inimg = isceobj.createImage()
        inimg.load(infile + '.xml')

        if outname is None:
            spl = os.path.splitext(inimg.filename)
            ext = '.{0}alks_{1}rlks'.format(alks, rlks)
            outname = spl[0] + ext + spl[1]

        lkObj = Looks()
        lkObj.setDownLooks(alks)
        lkObj.setAcrossLooks(rlks)
        lkObj.setInputImage(inimg)
        lkObj.setOutputFilename(outname)
        lkObj.looks()

    return outname





#def runMergeBursts(self):
def main(iargs=None):
    '''
    Merge burst products to make it look like stripmap.
    Currently will merge interferogram, lat, lon, z and los.
    '''
    inps=cmdLineParse(iargs)
    virtual = inps.useVirtualFiles
     
    swathList = ut.getSwathList(inps.master)
    referenceFrames = [] 
    frames=[]
    fileList = []
    namePattern = inps.namePattern.split('*')

    for swath in swathList:
        ifg = ut.loadProduct(os.path.join(inps.master , 'IW{0}.xml'.format(swath)))
        if inps.stack:
            stack =  ut.loadProduct(os.path.join(inps.stack , 'IW{0}.xml'.format(swath)))
        if inps.isaligned:
            reference = ifg.reference
        else:
            reference = ifg

        minBurst = ifg.bursts[0].burstNumber
        maxBurst = ifg.bursts[-1].burstNumber


        if minBurst==maxBurst:
            print('Skipping processing of swath {0}'.format(swath))
            continue
        
        if inps.stack:
            minStack = stack.bursts[0].burstNumber
            print('Updating the valid region of each burst to the common valid region of the stack')
        ####Updating the valid region of each burst to the common valid region of the stack
            for ii in range(minBurst, maxBurst + 1):

                ifg.bursts[ii-minBurst].firstValidLine = stack.bursts[ii-minStack].firstValidLine
                ifg.bursts[ii-minBurst].firstValidSample = stack.bursts[ii-minStack].firstValidSample
                ifg.bursts[ii-minBurst].numValidLines = stack.bursts[ii-minStack].numValidLines
                ifg.bursts[ii-minBurst].numValidSamples = stack.bursts[ii-minStack].numValidSamples

        frames.append(ifg)
        referenceFrames.append(reference)
        print('bursts: ', minBurst, maxBurst)
        fileList.append([os.path.join(inps.dirname, 'IW{0}'.format(swath), namePattern[0] + '_%02d.%s'%(x,namePattern[1])) for x in range(minBurst, maxBurst+1)])

    mergedir = os.path.dirname(inps.outfile)
    os.makedirs(mergedir, exist_ok=True)
    
    suffix = '.full'
    if (inps.numberRangeLooks == 1) and (inps.numberAzimuthLooks==1):
        suffix=''


    ####Virtual flag is ignored for multi-swath data
    
        if (not virtual):
            print('User requested for multi-swath stitching.')
            print('Virtual files are the only option for this.')
            print('Proceeding with virtual files.')

    mergeBurstsVirtual(frames, referenceFrames, fileList, inps.outfile+suffix, validOnly=inps.validOnly)

    if (not virtual):
        print('writing merged file to disk ...')
        cmd = 'gdal_translate -of ENVI -co INTERLEAVE=BIL ' + inps.outfile + suffix + '.vrt ' + inps.outfile + suffix 
        os.system(cmd)

    print(inps.multilook)
    if inps.multilook:
        multilook(inps.outfile+suffix, outname = inps.outfile, 
            alks = inps.numberAzimuthLooks, rlks=inps.numberRangeLooks,
            multilook_tool=inps.multilookTool, no_data=inps.noData)

    else:
        print('Skipping multi-looking ....')

if __name__ == '__main__' :
    '''
    Merge products burst-by-burst.
    '''

    main()
