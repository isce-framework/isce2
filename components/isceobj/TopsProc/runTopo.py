#
# Author: Piyush Agram
# Copyright 2016
#


import numpy as np 
import os
import isceobj
import datetime
import logging

logger = logging.getLogger('isce.topsinsar.topo')

def runTopo(self):

    hasGPU= self.useGPU and self._insar.hasGPU()
    if hasGPU:
        runTopoGPU(self)
    else:
        runTopoCPU(self)



def runTopoCPU(self):
    from zerodop.topozero import createTopozero
    from isceobj.Planet.Planet import Planet

    swathList = self._insar.getValidSwathList(self.swaths)

    ####Catalog for logging
    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)

    ####Load in DEM
    demfilename = self.verifyDEM()
    catalog.addItem('Dem Used', demfilename, 'topo')

    boxes = []
    for swath in swathList:
        #####Load the reference product
        reference = self._insar.loadProduct( os.path.join(self._insar.referenceSlcProduct,  'IW{0}.xml'.format(swath)))


        numCommon  = self._insar.numberOfCommonBursts[swath-1]
        startIndex = self._insar.commonBurstStartReferenceIndex[swath-1]

        if numCommon > 0:
            catalog.addItem('Number of common bursts IW-{0}'.format(swath), self._insar.numberOfCommonBursts[swath-1], 'topo')

            ###Check if geometry directory already exists.
            dirname = os.path.join(self._insar.geometryDirname, 'IW{0}'.format(swath))
            os.makedirs(dirname, exist_ok=True)

            ###For each burst
            for index in range(numCommon):
                ind = index + startIndex
                burst = reference.bursts[ind]

                latname = os.path.join(dirname, 'lat_%02d.rdr'%(ind+1))
                lonname = os.path.join(dirname, 'lon_%02d.rdr'%(ind+1))
                hgtname = os.path.join(dirname, 'hgt_%02d.rdr'%(ind+1))
                losname = os.path.join(dirname, 'los_%02d.rdr'%(ind+1))

                demImage = isceobj.createDemImage()
                demImage.load(demfilename + '.xml')

                #####Run Topo
                planet = Planet(pname='Earth')
                topo = createTopozero()
                topo.slantRangePixelSpacing = burst.rangePixelSize
                topo.prf = 1.0/burst.azimuthTimeInterval
                topo.radarWavelength = burst.radarWavelength
                topo.orbit = burst.orbit
                topo.width = burst.numberOfSamples
                topo.length = burst.numberOfLines
                topo.wireInputPort(name='dem', object=demImage)
                topo.wireInputPort(name='planet', object=planet)
                topo.numberRangeLooks = 1
                topo.numberAzimuthLooks = 1
                topo.lookSide = -1
                topo.sensingStart = burst.sensingStart
                topo.rangeFirstSample = burst.startingRange
                topo.demInterpolationMethod='BIQUINTIC'
                topo.latFilename = latname
                topo.lonFilename = lonname
                topo.heightFilename = hgtname
                topo.losFilename = losname
                topo.topo()

                bbox = [topo.minimumLatitude, topo.maximumLatitude, topo.minimumLongitude, topo.maximumLongitude]
                boxes.append(bbox)

                catalog.addItem('Number of lines for burst {0} - IW-{1}'.format(index,swath), burst.numberOfLines, 'topo')
                catalog.addItem('Number of pixels for bursts {0} - IW-{1}'.format(index,swath), burst.numberOfSamples, 'topo')
                catalog.addItem('Bounding box for burst {0} - IW-{1}'.format(index,swath), bbox, 'topo')

        else:
            print('Skipping Processing for Swath {0}'.format(swath))

        topo = None

    boxes = np.array(boxes)
    bbox = [np.min(boxes[:,0]), np.max(boxes[:,1]), np.min(boxes[:,2]), np.max(boxes[:,3])]
    catalog.addItem('Overall bounding box', bbox, 'topo')


    catalog.printToLog(logger, "runTopo")
    self._insar.procDoc.addAllFromCatalog(catalog)

    return



def runTopoGPU(self):
    '''
    Try with GPU module.
    '''

    from isceobj.Planet.Planet import Planet
    from zerodop.GPUtopozero.GPUtopozero import PyTopozero
    from isceobj import Constants as CN
    from isceobj.Util.Poly2D import Poly2D
    from iscesys import DateTimeUtil as DTU

    swathList = self._insar.getValidSwathList(self.swaths)

    ####Catalog for logging
    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)

    ####Load in DEM
    demfilename = self.verifyDEM()
    catalog.addItem('Dem Used', demfilename, 'topo')

    frames = []
    swaths = []
    swathStarts = []

    for swath in swathList:
        #####Load the reference product
        reference = self._insar.loadProduct( os.path.join(self._insar.referenceSlcProduct,  'IW{0}.xml'.format(swath)))

        numCommon  = self._insar.numberOfCommonBursts[swath-1]
        startIndex = self._insar.commonBurstStartReferenceIndex[swath-1]

        if numCommon > 0:
            catalog.addItem('Number of common bursts IW-{0}'.format(swath), self._insar.numberOfCommonBursts[swath-1], 'topo')


            reference.bursts = reference.bursts[startIndex:startIndex+numCommon]
            reference.numberOfBursts = numCommon

            frames.append(reference)
            swaths.append(swath)
            swathStarts.append(startIndex)

    if len(frames) == 0:
        raise Exception('There is no common region between the two dates to process')

    topSwath = min(frames, key=lambda x: x.sensingStart)
    leftSwath = min(frames, key=lambda x: x.startingRange)
    bottomSwath = max(frames, key=lambda x: x.sensingStop)
    rightSwath = max(frames, key=lambda x: x.farRange)

    r0 = leftSwath.startingRange
    rmax = rightSwath.farRange
    dr = frames[0].bursts[0].rangePixelSize
    t0 = topSwath.sensingStart
    tmax = bottomSwath.sensingStop
    dt = frames[0].bursts[0].azimuthTimeInterval
    wvl = frames[0].bursts[0].radarWavelength
    width = int(np.round((rmax-r0)/dr) + 1)
    lgth = int(np.round((tmax-t0).total_seconds()/dt) + 1)



    polyDoppler = Poly2D(name='topsApp_dopplerPoly')
    polyDoppler.setWidth(width)
    polyDoppler.setLength(lgth)
    polyDoppler.setNormRange(1.0)
    polyDoppler.setNormAzimuth(1.0)
    polyDoppler.setMeanRange(0.0)
    polyDoppler.setMeanAzimuth(0.0)
    polyDoppler.initPoly(rangeOrder=0,azimuthOrder=0, coeffs=[[0.]])
    polyDoppler.createPoly2D()


    slantRangeImage = Poly2D()
    slantRangeImage.setWidth(width)
    slantRangeImage.setLength(lgth)
    slantRangeImage.setNormRange(1.0)
    slantRangeImage.setNormAzimuth(1.0)
    slantRangeImage.setMeanRange(0.)
    slantRangeImage.setMeanAzimuth(0.)
    slantRangeImage.initPoly(rangeOrder=1,azimuthOrder=0,coeffs=[[r0,dr]])
    slantRangeImage.createPoly2D()


    dirname = self._insar.geometryDirname
    os.makedirs(dirname, exist_ok=True)


    latImage = isceobj.createImage()
    latImage.initImage(os.path.join(dirname, 'lat.rdr'), 'write', width, 'DOUBLE')
    latImage.createImage()

    lonImage = isceobj.createImage()
    lonImage.initImage(os.path.join(dirname, 'lon.rdr'), 'write', width, 'DOUBLE')
    lonImage.createImage()

    losImage = isceobj.createImage()
    losImage.initImage(os.path.join(dirname, 'los.rdr'), 'write', width, 'FLOAT', bands=2, scheme='BIL')
    losImage.setCaster('write', 'DOUBLE')
    losImage.createImage()

    heightImage = isceobj.createImage()
    heightImage.initImage(os.path.join(dirname, 'hgt.rdr'),'write',width,'DOUBLE')
    heightImage.createImage()

    demImage = isceobj.createDemImage()
    demImage.load(demfilename + '.xml')
    demImage.setCaster('read', 'FLOAT')
    demImage.createImage()


    orb = self._insar.getMergedOrbit(frames)
    pegHdg = np.radians( orb.getENUHeading(t0))

    elp = Planet(pname='Earth').ellipsoid


    topo = PyTopozero()
    topo.set_firstlat(demImage.getFirstLatitude())
    topo.set_firstlon(demImage.getFirstLongitude())
    topo.set_deltalat(demImage.getDeltaLatitude())
    topo.set_deltalon(demImage.getDeltaLongitude())
    topo.set_major(elp.a)
    topo.set_eccentricitySquared(elp.e2)
    topo.set_rSpace(dr)
    topo.set_r0(r0)
    topo.set_pegHdg(pegHdg)
    topo.set_prf(1.0/dt)
    topo.set_t0(DTU.seconds_since_midnight(t0))
    topo.set_wvl(wvl)
    topo.set_thresh(.05)
    topo.set_demAccessor(demImage.getImagePointer())
    topo.set_dopAccessor(polyDoppler.getPointer())
    topo.set_slrngAccessor(slantRangeImage.getPointer())
    topo.set_latAccessor(latImage.getImagePointer())
    topo.set_lonAccessor(lonImage.getImagePointer())
    topo.set_losAccessor(losImage.getImagePointer())
    topo.set_heightAccessor(heightImage.getImagePointer())
    topo.set_incAccessor(0)
    topo.set_maskAccessor(0)
    topo.set_numIter(25)
    topo.set_idemWidth(demImage.getWidth())
    topo.set_idemLength(demImage.getLength())
    topo.set_ilrl(-1)
    topo.set_extraIter(10)
    topo.set_length(lgth)
    topo.set_width(width)
    topo.set_nRngLooks(1)
    topo.set_nAzLooks(1)
    topo.set_demMethod(5) # BIQUINTIC METHOD
    topo.set_orbitMethod(0) # HERMITE


    # Need to simplify orbit stuff later
    nvecs = len(orb._stateVectors)
    topo.set_orbitNvecs(nvecs)
    topo.set_orbitBasis(1) # Is this ever different?
    topo.createOrbit() # Initializes the empty orbit to the right allocated size
    count = 0
    for sv in orb._stateVectors:
        td = DTU.seconds_since_midnight(sv.getTime())
        pos = sv.getPosition()
        vel = sv.getVelocity()
        topo.set_orbitVector(count,td,pos[0],pos[1],pos[2],vel[0],vel[1],vel[2])
        count += 1

    topo.runTopo()

    latImage.addDescription('Pixel-by-pixel latitude in degrees.')
    latImage.finalizeImage()
    latImage.renderHdr()

    lonImage.addDescription('Pixel-by-pixel longitude in degrees.')
    lonImage.finalizeImage()
    lonImage.renderHdr()

    heightImage.addDescription('Pixel-by-pixel height in meters.')
    heightImage.finalizeImage()
    heightImage.renderHdr()

    descr = '''Two channel Line-Of-Sight geometry image (all angles in degrees). Represents vector drawn from target to platform.
            Channel 1: Incidence angle measured from vertical at target (always +ve).
            Channel 2: Azimuth angle measured from North in Anti-clockwise direction.'''
    losImage.setImageType('bil')
    losImage.addDescription(descr)
    losImage.finalizeImage()
    losImage.renderHdr()

    demImage.finalizeImage()

    if slantRangeImage:
        try:
            slantRangeImage.finalizeImage()
        except:
            pass


    ####Start creating VRTs to point to global topo output
    for swath, frame, istart in zip(swaths, frames, swathStarts):
        outname = os.path.join(dirname, 'IW{0}'.format(swath))

        os.makedirs(outname, exist_ok=True)

        for ind, burst in enumerate(frame.bursts):
            top = int(np.rint((burst.sensingStart - t0).total_seconds()/dt))
            bottom = top + burst.numberOfLines
            left = int(np.rint((burst.startingRange - r0)/dr))
            right = left + burst.numberOfSamples

            
            buildVRT( os.path.join(dirname, 'lat.rdr'),
                      os.path.join(outname, 'lat_%02d.rdr'%(ind+istart+1)),
                      [width, lgth],
                      [top,bottom, left, right],
                      bands=1,
                      dtype='DOUBLE')

            buildVRT( os.path.join(dirname, 'lon.rdr'),
                      os.path.join(outname, 'lon_%02d.rdr'%(ind+istart+1)),
                      [width, lgth],
                      [top,bottom, left, right],
                      bands=1,
                      dtype='DOUBLE')

            buildVRT( os.path.join(dirname, 'hgt.rdr'),
                      os.path.join(outname, 'hgt_%02d.rdr'%(ind+istart+1)),
                      [width, lgth],
                      [top,bottom, left, right],
                      bands=1,
                      dtype='DOUBLE')

            buildVRT( os.path.join(dirname, 'los.rdr'),
                      os.path.join(outname, 'los_%02d.rdr'%(ind+istart+1)),
                      [width, lgth],
                      [top,bottom, left, right],
                      bands=2,
                      dtype='FLOAT')
            
            catalog.addItem('Subset for IW{0}-B{1}'.format(swath, ind+1+istart), 'Lines: {0}-{1} out of {2}, Pixels: {3}-{4} out of {5}'.format(top, bottom, lgth, left, right, width), 'topo')

#            print('IW{0}-B{1}: {2} - {3}/ {4}, {5} - {6} /{7}'.format(swath, ind+1+istart, top, bottom, lgth, left, right, width))

    catalog.printToLog(logger, "runTopo")
    self._insar.procDoc.addAllFromCatalog(catalog)

    return


def buildVRT(srcname, dstname, dims, bbox, bands=1, dtype='FLOAT'):
    '''
    Write a VRT to point to the parent mosaicked file.
    '''

    header='<VRTDataset rasterXSize="{width}" rasterYSize="{lgth}">'
    band = '''    <VRTRasterBand dataType="{dtype}" band="{band}">
        <NoDataValue>0.0</NoDataValue>
        <SimpleSource>
            <SourceFilename relativeToVRT="1">{relpath}</SourceFilename>
            <SourceBand>{band}</SourceBand>
            <SourceProperties RasterXSize="{gwidth}" RasterYSize="{glgth}" DataType="{dtype}"/>
            <SrcRect xOff="{left}" yOff="{top}" xSize="{width}" ySize="{lgth}"/>
            <DstRect xOff="0" yOff="0" xSize="{width}" ySize="{lgth}"/>
        </SimpleSource>
    </VRTRasterBand>
'''
    tail = "</VRTDataset>"


    width = bbox[3] - bbox[2]
    lgth = bbox[1] - bbox[0]

    odtype = dtype
    if dtype.upper() == 'FLOAT':
        dtype = 'Float32'
    elif dtype.upper() == 'DOUBLE':
        dtype = 'Float64'
    elif dtype.upper() == 'BYTE':
        dtype = 'UInt8'
    else:
        raise Exception('Unsupported type {0}'.format(dtype))

    relpath = os.path.relpath(srcname + '.vrt', os.path.dirname(dstname))
    gwidth = dims[0]
    glgth = dims[1]
    left = bbox[2]
    top = bbox[0]

    
    img = isceobj.createImage()
    img.bands = bands
    img.scheme = 'BIL'
    img.setWidth(width)
    img.setLength(lgth)
    img.dataType = odtype
    img.filename = dstname
    img.setAccessMode('READ')
    img.renderHdr()


    with open(dstname + '.vrt', 'w') as fid:
        fid.write( header.format(width=width, lgth=lgth) + '\n')

        for bnd in range(bands):
            fid.write( band.format(width=width, lgth=lgth,
                                   gwidth=gwidth, glgth=glgth,
                                   left=left, top=top,
                                   relpath=relpath, dtype=dtype,
                                   band=bnd+1))

        fid.write(tail + '\n')


