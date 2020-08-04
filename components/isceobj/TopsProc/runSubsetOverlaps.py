#
# Author: Piyush Agram
# Copyright 2016
#


import numpy as np
import os
import isceobj
import copy
import datetime
import logging
from isceobj.Sensor.TOPS import createTOPSSwathSLCProduct
from isceobj.Util.ImageUtil import ImageLib as IML

logger = logging.getLogger('isce.topsinsar.overlaps')


def subset(inname, outname, sliceline, slicepix, 
        virtual=False):
    '''Subset the input image to output image.
    '''

    gdalmap = {'FLOAT': 'Float32',
                'CFLOAT': 'CFloat32',
                'DOUBLE' : 'Float64'}

    inimg = isceobj.createImage()
    inimg.load(inname + '.xml')
    inimg.filename = outname

    inwidth = inimg.width
    inlength = inimg.length
    outwidth = slicepix.stop - slicepix.start
    outlength = sliceline.stop - sliceline.start
    inimg.setWidth(outwidth)
    inimg.setLength(outlength)
    inimg.setAccessMode('READ')
    inimg.renderHdr()

    if not virtual:
        indata = IML.mmapFromISCE(inname, logging).bands[0]
        outdata = indata[sliceline, slicepix]
        outdata.tofile(outname)
        indata = None

    else:

        relpath = os.path.relpath(inname, os.path.dirname(outname))

        rdict = {'outwidth'   : outwidth,
                 'outlength'  : outlength,
                 'inwidth'    : inwidth,
                 'inlength'   : inlength,
                 'xoffset'    : slicepix.start,
                 'yoffset'    : sliceline.start,
                 'dtype'      : gdalmap[inimg.dataType.upper()],
                 'filename'   : relpath + '.vrt'}

       

        tmpl = '''<VRTDataset rasterXSize="{outwidth}" rasterYSize="{outlength}">
    <VRTRasterBand dataType="{dtype}" band="1">
        <NoDataValue>0.0</NoDataValue>
        <SimpleSource>
            <SourceFilename relativeToVRT="1">{filename}</SourceFilename>
            <SourceBand>1</SourceBand>
            <SourceProperties RasterXSize="{inwidth}" RasterYSize="{inlength}" DataType="{dtype}"/>
            <SrcRect xOff="{xoffset}" yOff="{yoffset}" xSize="{outwidth}" ySize="{outlength}"/>
            <DstRect xOff="0" yOff="0" xSize="{outwidth}" ySize="{outlength}"/>
        </SimpleSource>
    </VRTRasterBand>
</VRTDataset>'''

        with open(outname + '.vrt', 'w') as fid:
            fid.write(tmpl.format(**rdict))

    return

def runSubsetOverlaps(self):
    '''
    Create geometry files corresponding to burst overlaps.
    '''
    virtual = self.useVirtualFiles

    if not self.doESD:
        return

    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)

    swathList = self._insar.getValidSwathList(self.swaths)

    for swath in swathList:

        if self._insar.numberOfCommonBursts[swath-1] < 2:
            print('Skipping subset overlap for swath IW{0}'.format(swath))
            continue

        ####Load reference metadata
        mFrame = self._insar.loadProduct( os.path.join(self._insar.referenceSlcProduct, 'IW{0}.xml'.format(swath)))


        ####Output directory for overlap geometry images
        geomdir = os.path.join(self._insar.geometryDirname, 'IW{0}'.format(swath))
        outdir = os.path.join(self._insar.geometryDirname, self._insar.overlapsSubDirname, 'IW{0}'.format(swath))
        subreferencedir = os.path.join(self._insar.referenceSlcProduct, self._insar.overlapsSubDirname, 'IW{0}'.format(swath)) 

        os.makedirs(outdir, exist_ok=True)
        os.makedirs(subreferencedir, exist_ok=True)

        ###Azimuth time interval
        dt = mFrame.bursts[0].azimuthTimeInterval        
        topFrame = createTOPSSwathSLCProduct()
        topFrame.configure()

        bottomFrame = createTOPSSwathSLCProduct()
        bottomFrame.configure()


        numCommon = self._insar.numberOfCommonBursts[swath-1]
        startIndex = self._insar.commonBurstStartReferenceIndex[swath-1]


        ###For each overlap
        for ii in range(numCommon - 1):
            ind = ii + startIndex

            topBurst = mFrame.bursts[ind]
            botBurst = mFrame.bursts[ind+1]

            overlap_start_time = botBurst.sensingStart
            overlap_end_time = topBurst.sensingStop
            catalog.addItem('Overlap {0} start time - IW-{1}'.format(ind,swath), overlap_start_time, 'subset')
            catalog.addItem('Overlap {0} stop time - IW-{1}'.format(ind, swath), overlap_end_time, 'subset')

            nLinesOverlap = int( np.round((overlap_end_time - overlap_start_time).total_seconds() / dt)) + 1
            catalog.addItem('Overlap {0} number of lines - IW-{1}'.format(ind,swath), nLinesOverlap, 'subset')

            length = topBurst.numberOfLines
            width = topBurst.numberOfSamples

            topStart = int ( np.round( (botBurst.sensingStart - topBurst.sensingStart).total_seconds()/dt))+ botBurst.firstValidLine       
            overlapLen = topBurst.firstValidLine + topBurst.numValidLines - topStart
      
            catalog.addItem('Overlap {0} number of valid lines - IW-{1}'.format(ind,swath), overlapLen, 'subset')

            ###Create slice objects for overlaps
            topslicey = slice(topStart, topStart+overlapLen)
            topslicex = slice(0, width)
    

            botslicey = slice(botBurst.firstValidLine, botBurst.firstValidLine + overlapLen)
            botslicex = slice(0, width)

            for prefix in ['lat','lon','hgt']:
                infile = os.path.join(geomdir, prefix + '_%02d.rdr'%(ind+2))
                outfile = os.path.join(outdir, prefix + '_%02d_%02d.rdr'%(ind+1,ind+2))
      
                subset(infile, outfile, botslicey, botslicex, virtual=virtual)

       
            masname1 = topBurst.image.filename
            masname2 = botBurst.image.filename

       
            reference_outname1 = os.path.join(subreferencedir , 'burst_top_%02d_%02d.slc'%(ind+1,ind+2))
            reference_outname2 = os.path.join(subreferencedir , 'burst_bot_%02d_%02d.slc'%(ind+1,ind+2))



            subset(masname1, reference_outname1, topslicey, topslicex,  virtual=virtual)
            subset(masname2, reference_outname2, botslicey, botslicex, virtual=virtual)


            ####TOP frame
#            burst = copy.deepcopy(topBurst)
            burst = topBurst.clone()
            burst.firstValidLine = 0
            burst.numberOfLines = overlapLen
            burst.numValidLines = overlapLen
            burst.sensingStart = topBurst.sensingStart + datetime.timedelta(0,topStart*dt) # topStart*dt
            burst.sensingStop = topBurst.sensingStart + datetime.timedelta(0,(topStart+overlapLen-1)*dt) # (topStart+overlapLen-1)*dt

            ###Replace file name in image
            burst.image.filename = reference_outname1
            burst.image._accessor = None
            burst.image.setLength(overlapLen)
            burst.image.setWidth(width)

            topFrame.bursts.append(burst)

            burst = None


            ####BOTTOM frame
#            burst = copy.deepcopy(botBurst)
            burst = botBurst.clone()
            burst.firstValidLine = 0
            burst.numberOfLines = overlapLen
            burst.numValidLines = overlapLen
            burst.sensingStart = botBurst.sensingStart + datetime.timedelta(seconds=botBurst.firstValidLine*dt)
            burst.sensingStop = botBurst.sensingStart + datetime.timedelta(seconds=(botBurst.firstValidLine+overlapLen-1)*dt)

            ###Replace file name in image
            burst.image.filename = reference_outname2
            burst.image.setLength(overlapLen)
            burst.image.setWidth(width)

            bottomFrame.bursts.append(burst)

            burst = None

            print('Top: ', [x.image.filename for x in topFrame.bursts])
            print('Bottom: ', [x.image.filename for x in bottomFrame.bursts])


        topFrame.numberOfBursts = len(topFrame.bursts)
        bottomFrame.numberOfBursts = len(bottomFrame.bursts)

        self._insar.saveProduct(topFrame, os.path.join(self._insar.referenceSlcOverlapProduct, 'top_IW{0}.xml'.format(swath)))
        self._insar.saveProduct(bottomFrame, os.path.join(self._insar.referenceSlcOverlapProduct, 'bottom_IW{0}.xml'.format(swath)))

    catalog.printToLog(logger, "runSubsetOverlaps")
    self._insar.procDoc.addAllFromCatalog(catalog)



