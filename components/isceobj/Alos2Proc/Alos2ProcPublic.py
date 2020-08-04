#!/usr/bin/env python3

#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#


def runCmd(cmd, silent=0):
    import os

    if silent == 0:
        print("{}".format(cmd))
    status = os.system(cmd)
    if status != 0:
        raise Exception('error when running:\n{}\n'.format(cmd))


def find_vrt_keyword(xmlfile, keyword):
    from xml.etree.ElementTree import ElementTree

    value = None
    xmlx = ElementTree(file=open(xmlfile,'r')).getroot()
    #try 10 times
    for i in range(10):
        path=''
        for j in range(i):
            path += '*/'
        value0 = xmlx.find(path+keyword)
        if value0 != None:
            value = value0.text
            break

    return value


def find_vrt_file(xmlfile, keyword, relative_path=True):
    '''
    find file in vrt in another directory
    xmlfile: vrt file
    relative_path: True: return relative (to current directory) path of the file
                   False: return absolute path of the file
    '''
    import os
    #get absolute directory of xmlfile
    xmlfile_dir = os.path.dirname(os.path.abspath(xmlfile))
    #find source file path
    file = find_vrt_keyword(xmlfile, keyword)
    #get absolute path of source file
    file = os.path.abspath(os.path.join(xmlfile_dir, file))
    #get relative path of source file
    if relative_path:
        file = os.path.relpath(file, './')
    return file


def create_xml(fileName, width, length, fileType):
    import isceobj

    if fileType == 'slc':
        image = isceobj.createSlcImage()
    elif fileType == 'int':
        image = isceobj.createIntImage()
    elif fileType == 'amp':
        image = isceobj.createAmpImage()
    elif fileType == 'cor':
        image = isceobj.createOffsetImage()
    elif fileType == 'rmg' or fileType == 'unw':
        image = isceobj.Image.createUnwImage()
    elif fileType == 'byte':
        image = isceobj.createImage()
        image.setDataType('BYTE')
    elif fileType == 'float':
        image = isceobj.createImage()
        image.setDataType('FLOAT')
    elif fileType == 'double':
        image = isceobj.createImage()
        image.setDataType('DOUBLE')

    else:
        raise Exception('format not supported yet!\n')

    image.setFilename(fileName)
    image.extraFilename = fileName + '.vrt'
    image.setWidth(width)
    image.setLength(length)
        
    #image.setAccessMode('read')
    #image.createImage()
    image.renderHdr()
    #image.finalizeImage()


def multilook_v1(data, nalks, nrlks):
    '''
    doing multiple looking
    ATTENSION: original array changed after running this function
    '''

    (length, width)=data.shape
    width2 = int(width/nrlks)
    length2 = int(length/nalks)

    for i in range(1, nalks):
        data[0:length2*nalks:nalks, :] += data[i:length2*nalks:nalks, :]
    for i in range(1, nrlks):
        data[0:length2*nalks:nalks, 0:width2*nrlks:nrlks] += data[0:length2*nalks:nalks, i:width2*nrlks:nrlks]

    return data[0:length2*nalks:nalks, 0:width2*nrlks:nrlks] / nrlks / nalks


def multilook(data, nalks, nrlks):
    '''
    doing multiple looking
    '''
    import numpy as np

    (length, width)=data.shape
    width2 = int(width/nrlks)
    length2 = int(length/nalks)

    data2=np.zeros((length2, width), dtype=data.dtype)
    for i in range(0, nalks):
        data2 += data[i:length2*nalks:nalks, :]
    for i in range(1, nrlks):
        data2[:, 0:width2*nrlks:nrlks] += data2[:, i:width2*nrlks:nrlks]

    return data2[:, 0:width2*nrlks:nrlks] / nrlks / nalks


def cal_coherence_1(inf, win=5):
    '''
    Compute coherence using scipy convolve 2D. Same as "def cal_coherence(inf, win=5):" in funcs.py in insarzd

    #still use standard coherence estimation equation, but with magnitude removed.
    #for example, equation (2) in
    #H. Zebker and K. Chen, Accurate Estimation of Correlation in InSAR Observations, 
    #IEEE GEOSCIENCE AND REMOTE SENSING LETTERS, VOL. 2, NO. 2, APRIL 2005.
    '''
    import numpy as np
    import scipy.signal as ss

    filt = np.ones((win,win))/ (1.0*win*win)
    flag = ss.convolve2d((inf!=0), filt, mode='same')
    angle = inf / (np.absolute(inf)+(inf==0))
    cor = ss.convolve2d(angle, filt, mode='same')
    cor = np.absolute(cor)
    #remove incomplete convolution result
    cor[np.nonzero(flag < 0.999)] = 0.0
    #print(np.max(cor), np.min(cor))
    #cor.astype(np.float32).tofile(f)

    return cor



def computeOffsetFromOrbit(referenceSwath, referenceTrack, secondarySwath, secondaryTrack, referenceSample, referenceLine):
    '''
    compute range and azimuth offsets using orbit. all range/azimuth indexes start with 0
    referenceSample:  reference sample where offset is computed, no need to be integer
    referenceLine:    reference line where offset is computed, no need to be integer
    '''
    import datetime

    pointingDirection = {'right': -1, 'left' :1}

    #compute a pair of range and azimuth offsets using geometry
    #using Piyush's code for computing range and azimuth offsets
    midRange = referenceSwath.startingRange + referenceSwath.rangePixelSize * referenceSample
    midSensingStart = referenceSwath.sensingStart + datetime.timedelta(seconds = referenceLine / referenceSwath.prf)
    llh = referenceTrack.orbit.rdr2geo(midSensingStart, midRange, side=pointingDirection[referenceTrack.pointingDirection])
    slvaz, slvrng = secondaryTrack.orbit.geo2rdr(llh, side=pointingDirection[referenceTrack.pointingDirection])
    ###Translate to offsets
    #at this point, secondary range pixel size and prf should be the same as those of reference
    rgoff = ((slvrng - secondarySwath.startingRange) / referenceSwath.rangePixelSize) - referenceSample
    azoff = ((slvaz - secondarySwath.sensingStart).total_seconds() * referenceSwath.prf) - referenceLine

    return (rgoff, azoff)


def overlapFrequency(centerfreq1, bandwidth1, centerfreq2, bandwidth2):
    startfreq1 = centerfreq1 - bandwidth1 / 2.0
    endingfreq1 = centerfreq1 + bandwidth1 / 2.0

    startfreq2 = centerfreq2 - bandwidth2 / 2.0
    endingfreq2 = centerfreq2 + bandwidth2 / 2.0

    overlapfreq = []
    if startfreq2 <= startfreq1 <= endingfreq2:
        overlapfreq.append(startfreq1)
    if startfreq2 <= endingfreq1 <= endingfreq2:
        overlapfreq.append(endingfreq1)
    
    if startfreq1 < startfreq2 < endingfreq1:
        overlapfreq.append(startfreq2)
    if startfreq1 < endingfreq2 < endingfreq1:
        overlapfreq.append(endingfreq2)

    if len(overlapfreq) != 2:
        #no overlap bandwidth
        return None
    else:
        startfreq = min(overlapfreq)
        endingfreq = max(overlapfreq)
        return [startfreq, endingfreq] 


def readOffset(filename):
    from isceobj.Location.Offset import OffsetField,Offset

    with open(filename, 'r') as f:
        lines = f.readlines()
    #                                          0      1       2       3      4         5             6          7
    #retstr = "%s %s %s %s %s %s %s %s" % (self.x,self.dx,self.y,self.dy,self.snr, self.sigmax, self.sigmay, self.sigmaxy)

    offsets = OffsetField()
    for linex in lines:
        #linexl = re.split('\s+', linex)
        #detect blank lines with only spaces and tabs
        if linex.strip() == '':
            continue

        linexl = linex.split()
        offset = Offset()
        #offset.setCoordinate(int(linexl[0]),int(linexl[2]))
        offset.setCoordinate(float(linexl[0]),float(linexl[2]))
        offset.setOffset(float(linexl[1]),float(linexl[3]))
        offset.setSignalToNoise(float(linexl[4]))
        offset.setCovariance(float(linexl[5]),float(linexl[6]),float(linexl[7]))
        offsets.addOffset(offset)

    return offsets


def writeOffset(offset, fileName):

    offsetsPlain = ''
    for offsetx in offset:
        offsetsPlainx = "{}".format(offsetx)
        offsetsPlainx = offsetsPlainx.split()
        offsetsPlain = offsetsPlain + "{:8d} {:10.3f} {:8d} {:12.3f} {:11.5f} {:11.6f} {:11.6f} {:11.6f}\n".format(
            int(float(offsetsPlainx[0])),
            float(offsetsPlainx[1]),
            int(float(offsetsPlainx[2])),
            float(offsetsPlainx[3]),
            float(offsetsPlainx[4]),
            float(offsetsPlainx[5]),
            float(offsetsPlainx[6]),
            float(offsetsPlainx[7])
            )

    offsetFile = fileName
    with open(offsetFile, 'w') as f:
        f.write(offsetsPlain)


def reformatGeometricalOffset(rangeOffsetFile, azimuthOffsetFile, reformatedOffsetFile, rangeStep=1, azimuthStep=1, maximumNumberOfOffsets=10000):
    '''
    reformat geometrical offset as ampcor output format
    '''
    import numpy as np
    import isceobj

    img = isceobj.createImage()
    img.load(rangeOffsetFile+'.xml')
    width = img.width
    length = img.length

    step = int(np.sqrt(width*length/maximumNumberOfOffsets) + 0.5)
    if step == 0:
        step = 1

    rgoff = np.fromfile(rangeOffsetFile, dtype=np.float32).reshape(length, width)
    azoff = np.fromfile(azimuthOffsetFile, dtype=np.float32).reshape(length, width)

    offsetsPlain = ''
    for i in range(0, length, step):
        for j in range(0, width, step):
            if (rgoff[i][j] == -999999.0) or (azoff[i][j] == -999999.0):
                continue

            offsetsPlain = offsetsPlain + "{:8d} {:10.3f} {:8d} {:12.3f} {:11.5f} {:11.6f} {:11.6f} {:11.6f}\n".format(
                int(j*rangeStep+1),
                float(rgoff[i][j]),
                int(i*azimuthStep+1),
                float(azoff[i][j]),
                float(22.00015),
                float(0.000273),
                float(0.002126),
                float(0.000013)
            )
    with open(reformatedOffsetFile, 'w') as f:
        f.write(offsetsPlain)
            
    return


def cullOffsets(offsets):
    import isceobj
    from iscesys.StdOEL.StdOELPy import create_writer

    distances = (10,5,3,3,3,3,3,3)
    #numCullOffsetsLimits = (100, 75, 50, 50, 50, 50, 50, 50)
    numCullOffsetsLimits = (50, 40, 30, 30, 30, 30, 30, 30)

    refinedOffsets = offsets
    for i, (distance, numCullOffsetsLimit) in enumerate(zip(distances, numCullOffsetsLimits)):

        cullOff = isceobj.createOffoutliers()
        cullOff.wireInputPort(name='offsets', object=refinedOffsets)
        cullOff.setSNRThreshold(2.0)
        cullOff.setDistance(distance)
    
        #set the tag used in the outfile. each message is precided by this tag
        #is the writer is not of "file" type the call has no effect
        stdWriter = create_writer("log", "", True, filename="offoutliers.log")
        stdWriter.setFileTag("offoutliers", "log")
        stdWriter.setFileTag("offoutliers", "err")
        stdWriter.setFileTag("offoutliers", "out")
        cullOff.setStdWriter(stdWriter)

        #run it
        cullOff.offoutliers()

        refinedOffsets = cullOff.getRefinedOffsetField()
        numLeft = len(refinedOffsets._offsets)
        print('Number of offsets left after %2dth culling: %5d'%(i, numLeft))
        if numLeft < numCullOffsetsLimit:
            refinedOffsets = None
    
        stdWriter.finalize()

    return refinedOffsets


def cullOffsetsRoipac(offsets, numThreshold=50):
    '''
    cull offsets using fortran program from ROI_PAC
    numThreshold: minmum number of offsets left
    '''
    import os
    from contrib.alos2proc_f.alos2proc_f import fitoff
    from isceobj.Alos2Proc.Alos2ProcPublic import readOffset
    from isceobj.Alos2Proc.Alos2ProcPublic import writeOffset

    offsetFile = 'offset.off'
    cullOffsetFile = 'cull.off'
    writeOffset(offsets, offsetFile)

    #try different parameters to cull offsets
    breakFlag = 0
    for maxrms in [0.08,  0.16,  0.24]:
        for nsig in [1.5,  1.4,  1.3,  1.2,  1.1,  1.0,  0.9]:
            fitoff(offsetFile, cullOffsetFile, nsig, maxrms, numThreshold)

            #check number of matching points left
            with open(cullOffsetFile, 'r') as ff:
                numCullOffsets = sum(1 for linex in ff)
            if numCullOffsets < numThreshold:
                print('offsets culling with nsig {} maxrms {}:  {} left after culling, too few points'.format(nsig, maxrms, numCullOffsets))
            else:
                print('offsets culling with nsig {} maxrms {}:  {} left after culling, success'.format(nsig, maxrms, numCullOffsets))
                breakFlag = 1
                break
        
        if breakFlag == 1:
            break

    if numCullOffsets < numThreshold:
        refinedOffsets = None
    else:
        refinedOffsets = readOffset(cullOffsetFile)

    os.remove(offsetFile)
    os.remove(cullOffsetFile)

    return refinedOffsets


def meanOffset(offsets):

    rangeOffset = 0.0
    azimuthOffset = 0.0
    i = 0
    for offsetx in offsets:
        i += 1
        rangeOffset += offsetx.dx
        azimuthOffset += offsetx.dy

    rangeOffset /= i
    azimuthOffset /= i

    return (rangeOffset, azimuthOffset)


def fitOffset(inputOffset, order=1, axis='range'):
    '''fit a polynomial to the offset
       order=0 also works, output is mean offset
    '''
    import numpy as np
    index = []
    offset = []
    for a in inputOffset:
        if axis=='range':
            index.append(a.x)
            offset.append(a.dx)
        else:
            index.append(a.y)
            offset.append(a.dy)

    p = np.polyfit(index, offset, order)

    return list(p[::-1])


def topo(swath, track, demFile, latFile, lonFile, hgtFile, losFile=None, incFile=None, mskFile=None, numberRangeLooks=1, numberAzimuthLooks=1, multilookTimeOffset=True):
    import datetime
    import isceobj
    from zerodop.topozero import createTopozero
    from isceobj.Planet.Planet import Planet

    pointingDirection = {'right': -1, 'left' :1}

    demImage = isceobj.createDemImage()
    demImage.load(demFile + '.xml')
    demImage.setAccessMode('read')

    #####Run Topo
    planet = Planet(pname='Earth')
    topo = createTopozero()
    topo.slantRangePixelSpacing = numberRangeLooks * swath.rangePixelSize
    topo.prf = 1.0 / (numberAzimuthLooks * swath.azimuthLineInterval)
    topo.radarWavelength = track.radarWavelength
    topo.orbit = track.orbit
    topo.width = int(swath.numberOfSamples/numberRangeLooks)
    topo.length = int(swath.numberOfLines/numberAzimuthLooks)
    topo.wireInputPort(name='dem', object=demImage)
    topo.wireInputPort(name='planet', object=planet)
    topo.numberRangeLooks = 1 #must be set as 1
    topo.numberAzimuthLooks = 1 #must be set as 1 Cunren
    topo.lookSide = pointingDirection[track.pointingDirection]
    if multilookTimeOffset == True:
        topo.sensingStart = swath.sensingStart + datetime.timedelta(seconds=(numberAzimuthLooks-1.0)/2.0/swath.prf)
        topo.rangeFirstSample = swath.startingRange + (numberRangeLooks-1.0)/2.0 * swath.rangePixelSize
    else:
        topo.sensingStart = swath.sensingStart
        topo.rangeFirstSample = swath.startingRange
    topo.demInterpolationMethod='BIQUINTIC'

    topo.latFilename = latFile
    topo.lonFilename = lonFile
    topo.heightFilename = hgtFile
    if losFile != None:
        topo.losFilename = losFile
    if incFile != None:
        topo.incFilename = incFile
    if mskFile != None:
        topo.maskFilename = mskFile

    topo.topo()

    return list(topo.snwe)


def geo2rdr(swath, track, latFile, lonFile, hgtFile, rangeOffsetFile, azimuthOffsetFile, numberRangeLooks=1, numberAzimuthLooks=1, multilookTimeOffset=True):
    import datetime
    import isceobj
    from zerodop.geo2rdr import createGeo2rdr
    from isceobj.Planet.Planet import Planet

    pointingDirection = {'right': -1, 'left' :1}

    latImage = isceobj.createImage()
    latImage.load(latFile + '.xml')
    latImage.setAccessMode('read')

    lonImage = isceobj.createImage()
    lonImage.load(lonFile + '.xml')
    lonImage.setAccessMode('read')

    hgtImage = isceobj.createDemImage()
    hgtImage.load(hgtFile + '.xml')
    hgtImage.setAccessMode('read')

    planet = Planet(pname='Earth')

    topo = createGeo2rdr()
    topo.configure()
    topo.slantRangePixelSpacing = numberRangeLooks * swath.rangePixelSize
    topo.prf = 1.0 / (numberAzimuthLooks * swath.azimuthLineInterval)
    topo.radarWavelength = track.radarWavelength
    topo.orbit = track.orbit
    topo.width = int(swath.numberOfSamples/numberRangeLooks)
    topo.length = int(swath.numberOfLines/numberAzimuthLooks)
    topo.demLength = hgtImage.length
    topo.demWidth = hgtImage.width
    topo.wireInputPort(name='planet', object=planet)
    topo.numberRangeLooks = 1
    topo.numberAzimuthLooks = 1 #must be set to be 1
    topo.lookSide = pointingDirection[track.pointingDirection]
    if multilookTimeOffset == True:
        topo.sensingStart = swath.sensingStart + datetime.timedelta(seconds=(numberAzimuthLooks-1.0)/2.0*swath.azimuthLineInterval)
        topo.rangeFirstSample = swath.startingRange + (numberRangeLooks-1.0)/2.0*swath.rangePixelSize
    else:
        topo.setSensingStart(swath.sensingStart)
        topo.rangeFirstSample = swath.startingRange
    topo.dopplerCentroidCoeffs = [0.] #we are using zero doppler geometry
    topo.demImage = hgtImage
    topo.latImage = latImage
    topo.lonImage = lonImage
    topo.rangeOffsetImageName = rangeOffsetFile
    topo.azimuthOffsetImageName = azimuthOffsetFile
    topo.geo2rdr()

    return


def waterBodyRadar(latFile, lonFile, wbdFile, wbdOutFile):
    '''
    create water boday in radar coordinates
    '''
    import numpy as np
    import isceobj
    from isceobj.Alos2Proc.Alos2ProcPublic import create_xml

    demImage = isceobj.createDemImage()
    demImage.load(wbdFile + '.xml')
    #demImage.setAccessMode('read')
    wbd=np.memmap(wbdFile, dtype='byte', mode='r', shape=(demImage.length, demImage.width))

    image = isceobj.createImage()
    image.load(latFile+'.xml')
    width = image.width
    length = image.length

    latFp = open(latFile, 'rb')
    lonFp = open(lonFile, 'rb')
    wbdOutFp = open(wbdOutFile, 'wb')
    print("create water body in radar coordinates...")
    for i in range(length):
        if (((i+1)%200) == 0):
            print("processing line %6d of %6d" % (i+1, length), end='\r', flush=True)
        wbdOut = np.zeros(width, dtype='byte')-2
        lat = np.fromfile(latFp, dtype=np.float64, count=width)
        lon = np.fromfile(lonFp, dtype=np.float64, count=width)
        #indexes start with zero
        lineIndex = np.int32((lat - demImage.firstLatitude) / demImage.deltaLatitude + 0.5)
        sampleIndex = np.int32((lon - demImage.firstLongitude) / demImage.deltaLongitude + 0.5)
        inboundIndex = np.logical_and(
            np.logical_and(lineIndex>=0, lineIndex<=demImage.length-1),
            np.logical_and(sampleIndex>=0, sampleIndex<=demImage.width-1)
            )
        #keep SRTM convention. water body. (0) --- land; (-1) --- water; (-2 or other value) --- no data.
        wbdOut = wbd[(lineIndex[inboundIndex], sampleIndex[inboundIndex])]
        wbdOut.astype(np.int8).tofile(wbdOutFp)
    print("processing line %6d of %6d" % (length, length))
    #create_xml(wbdOutFile, width, length, 'byte')

    image = isceobj.createImage()
    image.setDataType('BYTE')
    image.addDescription('water body. (0) --- land; (-1) --- water; (-2) --- no data.')
    image.setFilename(wbdOutFile)
    image.extraFilename = wbdOutFile + '.vrt'
    image.setWidth(width)
    image.setLength(length)
    image.renderHdr()

    del wbd, demImage, image
    latFp.close()
    lonFp.close()
    wbdOutFp.close()


def renameFile(oldname, newname):
    import os
    import isceobj
    img = isceobj.createImage()
    img.load(oldname + '.xml')
    img.setFilename(newname)
    img.extraFilename = newname+'.vrt'
    img.renderHdr()

    os.rename(oldname, newname)
    os.remove(oldname + '.xml')
    os.remove(oldname + '.vrt')


def cal_coherence(inf, win=5, edge=0):
    '''
    compute coherence uisng only interferogram (phase).
    This routine still follows the regular equation for computing coherence,
    but assumes the amplitudes of reference and secondary are one, so that coherence
    can be computed using phase only.

    inf: interferogram
    win: window size
    edge: 0: remove all non-full convolution samples

          1: remove samples computed from less than half convolution
             (win=5 used to illustration below)
             * * *
             * * *
             * * *
             * * *
             * * *

          2: remove samples computed from less than quater convolution
             (win=5 used to illustration below)
             * * *
             * * *
             * * *

          3: remove non-full convolution samples on image edges

          4: keep all samples
    '''
    import numpy as np
    import scipy.signal as ss

    if win % 2 != 1:
        raise Exception('window size must be odd!')
    hwin = np.int(np.around((win - 1) / 2))

    filt = np.ones((win, win))
    amp  = np.absolute(inf)

    cnt = ss.convolve2d((amp!=0), filt, mode='same')
    cor = ss.convolve2d(inf/(amp + (amp==0)), filt, mode='same')
    cor = (amp!=0) * np.absolute(cor) / (cnt + (cnt==0))

    #trim edges
    if edge == 0:
        num = win * win
        cor[np.nonzero(cnt < num)] = 0.0
    elif edge == 1:
        num = win * (hwin+1)
        cor[np.nonzero(cnt < num)] = 0.0
    elif edge == 2:
        num = (hwin+1) * (hwin+1)
        cor[np.nonzero(cnt < num)] = 0.0
    elif edge == 3:
        cor[0:hwin, :] = 0.0
        cor[-hwin:, :] = 0.0
        cor[:, 0:hwin] = 0.0
        cor[:, -hwin:] = 0.0
    else:
        pass

    #print("coherence, max: {} min: {}".format(np.max(cor[np.nonzero(cor!=0)]), np.min(cor[np.nonzero(cor!=0)])))
    return cor


def snaphuUnwrap(track, t, wrapName, corName, unwrapName, nrlks, nalks, costMode = 'DEFO',initMethod = 'MST', defomax = 4.0, initOnly = False):
        #runUnwrap(self,                                           costMode = 'SMOOTH',initMethod = 'MCF', defomax = 2,   initOnly = True)
    '''
    track:       track object
    t:           time for computing earth radius and altitude, normally mid azimuth time
    wrapName:    input interferogram
    corName:     input coherence file
    unwrapName:  output unwrapped interferogram
    nrlks:       number of range looks of the interferogram
    nalks:       number of azimuth looks of the interferogram
    '''
    import datetime
    import numpy as np
    import isceobj
    from contrib.Snaphu.Snaphu import Snaphu
    from isceobj.Planet.Planet import Planet

    corImg = isceobj.createImage()
    corImg.load(corName + '.xml')
    width = corImg.width
    length = corImg.length

    #get altitude
    orbit = track.orbit
    peg = orbit.interpolateOrbit(t, method='hermite')
    refElp = Planet(pname='Earth').ellipsoid
    llh = refElp.xyz_to_llh(peg.getPosition())
    hdg = orbit.getENUHeading(t)
    refElp.setSCH(llh[0], llh[1], hdg)
    earthRadius = refElp.pegRadCur
    altitude   = llh[2]

    rangeLooks = nrlks
    azimuthLooks = nalks
    azfact = 0.8
    rngfact = 0.8
    corrLooks = rangeLooks * azimuthLooks/(azfact*rngfact) 
    maxComponents = 20

    snp = Snaphu()
    snp.setInitOnly(initOnly)
    snp.setInput(wrapName)
    snp.setOutput(unwrapName)
    snp.setWidth(width)
    snp.setCostMode(costMode)
    snp.setEarthRadius(earthRadius)
    snp.setWavelength(track.radarWavelength)
    snp.setAltitude(altitude)
    snp.setCorrfile(corName)
    snp.setInitMethod(initMethod)
    snp.setCorrLooks(corrLooks)
    snp.setMaxComponents(maxComponents)
    snp.setDefoMaxCycles(defomax)
    snp.setRangeLooks(rangeLooks)
    snp.setAzimuthLooks(azimuthLooks)
    if corImg.bands == 1:
        snp.setCorFileFormat('FLOAT_DATA')
    snp.prepare()
    snp.unwrap()

    ######Render XML
    outImage = isceobj.Image.createUnwImage()
    outImage.setFilename(unwrapName)
    outImage.setWidth(width)
    outImage.setAccessMode('read')
    outImage.renderVRT()
    outImage.createImage()
    outImage.finalizeImage()
    outImage.renderHdr()

    #####Check if connected components was created
    if snp.dumpConnectedComponents:
        connImage = isceobj.Image.createImage()
        connImage.setFilename(unwrapName+'.conncomp')
        connImage.setWidth(width)
        connImage.setAccessMode('read')
        connImage.setDataType('BYTE')
        connImage.renderVRT()
        connImage.createImage()
        connImage.finalizeImage()
        connImage.renderHdr()
        del connImage

    del corImg
    del snp
    del outImage

    #remove wired things in no-data area
    amp=np.memmap(unwrapName, dtype='float32', mode='r+', shape=(length*2, width))
    wrap = np.fromfile(wrapName, dtype=np.complex64).reshape(length, width)
    (amp[0:length*2:2, :])[np.nonzero(wrap==0)]=0
    (amp[1:length*2:2, :])[np.nonzero(wrap==0)]=0
    del amp
    del wrap

    return


def snaphuUnwrapOriginal(wrapName, corName, ampName, unwrapName, costMode = 's', initMethod = 'mcf'):
    '''
    unwrap interferogram using original snaphu program
    '''
    import numpy as np
    import isceobj

    corImg = isceobj.createImage()
    corImg.load(corName + '.xml')
    width = corImg.width
    length = corImg.length

    #specify coherence file format in configure file
    snaphuConfFile = 'snaphu.conf'
    if corImg.bands == 1:
        snaphuConf = '''CORRFILEFORMAT        FLOAT_DATA
CONNCOMPFILE        {}
MAXNCOMPS       20'''.format(unwrapName+'.conncomp')

    else:
        snaphuConf = '''CORRFILEFORMAT        FLOAT_DATA
CONNCOMPFILE        {}
MAXNCOMPS       20'''.format(unwrapName+'.conncomp')
    with open(snaphuConfFile, 'w') as f:
        f.write(snaphuConf)
    cmd = 'snaphu {} {} -f {} -{} -o {} -a {} -c {} -v --{}'.format(
        wrapName,
        width,
        snaphuConfFile,
        costMode,
        unwrapName,
        ampName,
        corName,
        initMethod
        )
    runCmd(cmd)
    create_xml(unwrapName, width, length, 'unw')

    connImage = isceobj.Image.createImage()
    connImage.setFilename(unwrapName+'.conncomp')
    connImage.setWidth(width)
    connImage.setAccessMode('read')
    connImage.setDataType('BYTE')
    connImage.renderVRT()
    connImage.createImage()
    connImage.finalizeImage()
    connImage.renderHdr()
    del connImage

    #remove wired things in no-data area
    amp=np.memmap(unwrapName, dtype='float32', mode='r+', shape=(length*2, width))
    wrap = np.fromfile(wrapName, dtype=np.complex64).reshape(length, width)
    (amp[0:length*2:2, :])[np.nonzero(wrap==0)]=0
    (amp[1:length*2:2, :])[np.nonzero(wrap==0)]=0
    del amp
    del wrap

    return


def getBboxGeo(track):
    '''
    get bounding box in geo-coordinate
    '''
    import numpy as np

    pointingDirection = {'right': -1, 'left' :1}

    bboxRdr = getBboxRdr(track)

    rangeMin = bboxRdr[0]
    rangeMax = bboxRdr[1]
    azimuthTimeMin = bboxRdr[2]
    azimuthTimeMax = bboxRdr[3]

    #get bounding box using Piyush's code
    hgtrange=[-500,9000]
    ts = [azimuthTimeMin, azimuthTimeMax]
    rngs = [rangeMin, rangeMax]
    pos = []
    for ht in hgtrange:
        for tim in ts:
            for rng in rngs:
                llh = track.orbit.rdr2geo(tim, rng, height=ht, side=pointingDirection[track.pointingDirection])
                pos.append(llh)
    pos = np.array(pos)
    #               S                 N                 W                 E
    bbox = [np.min(pos[:,0]), np.max(pos[:,0]), np.min(pos[:,1]), np.max(pos[:,1])]
    
    return bbox


def getBboxRdr(track):
    '''
    get bounding box in radar-coordinate
    '''
    import datetime

    numberOfFrames = len(track.frames)
    numberOfSwaths = len(track.frames[0].swaths)

    sensingStartList = []
    sensingEndList = []
    startingRangeList = []
    endingRangeList = []
    for i in range(numberOfFrames):
        for j in range(numberOfSwaths):
            swath = track.frames[i].swaths[j]
            sensingStartList.append(swath.sensingStart)
            sensingEndList.append(swath.sensingStart + datetime.timedelta(seconds=(swath.numberOfLines-1) * swath.azimuthLineInterval))
            startingRangeList.append(swath.startingRange)
            endingRangeList.append(swath.startingRange + (swath.numberOfSamples - 1) * swath.rangePixelSize)
    azimuthTimeMin = min(sensingStartList)
    azimuthTimeMax = max(sensingEndList)
    azimuthTimeMid = azimuthTimeMin+datetime.timedelta(seconds=(azimuthTimeMax-azimuthTimeMin).total_seconds()/2.0)
    rangeMin = min(startingRangeList)
    rangeMax = max(endingRangeList)
    rangeMid = (rangeMin + rangeMax) / 2.0

    bbox = [rangeMin, rangeMax, azimuthTimeMin, azimuthTimeMax]

    return bbox


def filterInterferogram(data, alpha, windowSize, stepSize):
    '''
    a filter wrapper
    '''
    import os
    import numpy as np
    from contrib.alos2filter.alos2filter import psfilt1

    (length, width)=data.shape
    data.astype(np.complex64).tofile('tmp1234.int')
    psfilt1('tmp1234.int', 'filt_tmp1234.int', width, alpha, windowSize, stepSize)
    
    data2 = np.fromfile('filt_tmp1234.int', dtype=np.complex64).reshape(length, width)
    os.remove('tmp1234.int')
    os.remove('filt_tmp1234.int')

    return data2



###################################################################
# these are routines for burst-by-burst ScanSAR interferometry
###################################################################

def mosaicBurstInterferogram(swath, burstPrefix, outputFile, numberOfLooksThreshold=1):
    '''
    take a burst sequence and output mosaicked file
    '''
    import numpy as np

    interferogram = np.zeros((swath.numberOfLines, swath.numberOfSamples), dtype=np.complex64)
    cnt = np.zeros((swath.numberOfLines, swath.numberOfSamples), dtype=np.int8)
    for i in range(swath.numberOfBursts):
        burstFile = burstPrefix + '_%02d.int'%(i+1)
        burstInterferogram = np.fromfile(burstFile, dtype=np.complex64).reshape(swath.burstSlcNumberOfLines, swath.burstSlcNumberOfSamples)
        interferogram[0+swath.burstSlcFirstLineOffsets[i]:swath.burstSlcNumberOfLines+swath.burstSlcFirstLineOffsets[i], :] += burstInterferogram
        cnt[0+swath.burstSlcFirstLineOffsets[i]:swath.burstSlcNumberOfLines+swath.burstSlcFirstLineOffsets[i], :] += (burstInterferogram!=0)

    #trim upper and lower edges with less number of looks
    #############################################################################
    firstLine = 0
    for i in range(swath.numberOfLines):
        if np.sum(cnt[i,:]>=numberOfLooksThreshold) > swath.numberOfSamples/2:
            firstLine = i
            break
    lastLine = swath.numberOfLines - 1
    for i in range(swath.numberOfLines):
        if np.sum(cnt[swath.numberOfLines-1-i,:]>=numberOfLooksThreshold) > swath.numberOfSamples/2:
            lastLine = swath.numberOfLines-1-i
            break
    
    interferogram[:firstLine,:]=0
    interferogram[lastLine+1:,:]=0

    # if numberOfLooksThreshold!= None:
    #     interferogram[np.nonzero(cnt<numberOfLooksThreshold)] = 0
    #############################################################################

    interferogram.astype(np.complex64).tofile(outputFile)
    create_xml(outputFile, swath.numberOfSamples, swath.numberOfLines, 'int')


def mosaicBurstAmplitude(swath, burstPrefix, outputFile, numberOfLooksThreshold=1):
    '''
    take a burst sequence and output the magnitude
    '''
    import numpy as np

    amp = np.zeros((swath.numberOfLines, swath.numberOfSamples), dtype=np.float32)
    cnt = np.zeros((swath.numberOfLines, swath.numberOfSamples), dtype=np.int8)
    for i in range(swath.numberOfBursts):
        burstFile = burstPrefix + '_%02d.slc'%(i+1)
        #azLineOffset = round((swath.burstSlcStartTimes[i] - swath.burstSlcStartTimes[0]).total_seconds() / swath.azimuthLineInterval)
        burstMag = np.absolute(np.fromfile(burstFile, dtype=np.complex64).reshape(swath.burstSlcNumberOfLines, swath.burstSlcNumberOfSamples))
        burstPwr = burstMag * burstMag
        amp[0+swath.burstSlcFirstLineOffsets[i]:swath.burstSlcNumberOfLines+swath.burstSlcFirstLineOffsets[i], :] += burstPwr
        cnt[0+swath.burstSlcFirstLineOffsets[i]:swath.burstSlcNumberOfLines+swath.burstSlcFirstLineOffsets[i], :] += (burstPwr!=0)

    #trim upper and lower edges with less number of looks
    #############################################################################
    firstLine = 0
    for i in range(swath.numberOfLines):
        if np.sum(cnt[i,:]>=numberOfLooksThreshold) > swath.numberOfSamples/2:
            firstLine = i
            break
    lastLine = swath.numberOfLines - 1
    for i in range(swath.numberOfLines):
        if np.sum(cnt[swath.numberOfLines-1-i,:]>=numberOfLooksThreshold) > swath.numberOfSamples/2:
            lastLine = swath.numberOfLines-1-i
            break
    
    amp[:firstLine,:]=0
    amp[lastLine+1:,:]=0

    # if numberOfLooksThreshold!= None:
    #     amp[np.nonzero(cnt<numberOfLooksThreshold)] = 0
    #############################################################################

    np.sqrt(amp).astype(np.float32).tofile(outputFile)
    create_xml(outputFile, swath.numberOfSamples, swath.numberOfLines, 'float')


def resampleBursts(referenceSwath, secondarySwath, 
    referenceBurstDir, secondaryBurstDir, secondaryBurstResampledDir, interferogramDir,
    referenceBurstPrefix, secondaryBurstPrefix, secondaryBurstResampledPrefix, interferogramPrefix, 
    rangeOffset, azimuthOffset, rangeOffsetResidual=0, azimuthOffsetResidual=0):

    import os
    import datetime
    import numpy as np
    import numpy.matlib
    from contrib.alos2proc.alos2proc import resamp

    os.makedirs(secondaryBurstResampledDir, exist_ok=True)
    os.makedirs(interferogramDir, exist_ok=True)

    #get burst file names
    referenceBurstSlc = [referenceBurstPrefix+'_%02d.slc'%(i+1) for i in range(referenceSwath.numberOfBursts)]
    secondaryBurstSlc = [secondaryBurstPrefix+'_%02d.slc'%(i+1) for i in range(secondarySwath.numberOfBursts)]
    secondaryBurstSlcResampled = [secondaryBurstPrefix+'_%02d.slc'%(i+1) for i in range(referenceSwath.numberOfBursts)]
    interferogram = [interferogramPrefix+'_%02d.int'%(i+1) for i in range(referenceSwath.numberOfBursts)]

    length = referenceSwath.burstSlcNumberOfLines
    width = referenceSwath.burstSlcNumberOfSamples
    lengthSecondary = secondarySwath.burstSlcNumberOfLines
    widthSecondary = secondarySwath.burstSlcNumberOfSamples

    #secondary burst slc start times
    secondaryBurstStartTimesSlc = [secondarySwath.firstBurstSlcStartTime + \
                               datetime.timedelta(seconds=secondarySwath.burstSlcFirstLineOffsets[i]*secondarySwath.azimuthLineInterval) \
                               for i in range(secondarySwath.numberOfBursts)]
    #secondary burst raw start times
    secondaryBurstStartTimesRaw = [secondarySwath.firstBurstRawStartTime + \
                               datetime.timedelta(seconds=i*secondarySwath.burstCycleLength/secondarySwath.prf) \
                               for i in range(secondarySwath.numberOfBursts)]


    for i in range(referenceSwath.numberOfBursts):

        ##########################################################################
        # 1. get offsets and corresponding secondary burst
        ##########################################################################
        #range offset
        with open(rangeOffset, 'rb') as f:
            f.seek(referenceSwath.burstSlcFirstLineOffsets[i] * width * np.dtype(np.float32).itemsize, 0)
            rgoffBurst = np.fromfile(f, dtype=np.float32, count=length*width).reshape(length,width)
            if type(rangeOffsetResidual) == np.ndarray:
                residual = rangeOffsetResidual[0+referenceSwath.burstSlcFirstLineOffsets[i]:length+referenceSwath.burstSlcFirstLineOffsets[i],:]
                rgoffBurst[np.nonzero(rgoffBurst!=-999999.0)] += residual[np.nonzero(rgoffBurst!=-999999.0)]
            else:
                rgoffBurst[np.nonzero(rgoffBurst!=-999999.0)] += rangeOffsetResidual
        #azimuth offset
        with open(azimuthOffset, 'rb') as f:
            f.seek(referenceSwath.burstSlcFirstLineOffsets[i] * width * np.dtype(np.float32).itemsize, 0)
            azoffBurst = np.fromfile(f, dtype=np.float32, count=length*width).reshape(length,width)
            if type(azimuthOffsetResidual) == np.ndarray:
                residual = azimuthOffsetResidual[0+referenceSwath.burstSlcFirstLineOffsets[i]:length+referenceSwath.burstSlcFirstLineOffsets[i],:]
                azoffBurst[np.nonzero(azoffBurst!=-999999.0)] += residual[np.nonzero(azoffBurst!=-999999.0)]
            else:
                azoffBurst[np.nonzero(azoffBurst!=-999999.0)] += azimuthOffsetResidual

        #find the corresponding secondary burst
        #get mean offset to use
        #remove BAD_VALUE = -999999.0 as defined in geo2rdr.f90
        #single precision is not accurate enough to compute mean
        azoffBurstMean = np.mean(azoffBurst[np.nonzero(azoffBurst!=-999999.0)], dtype=np.float64)
        iSecondary = -1
        for j in range(secondarySwath.numberOfBursts): 
            if abs(referenceSwath.burstSlcFirstLineOffsets[i] + azoffBurstMean - secondarySwath.burstSlcFirstLineOffsets[j]) < (referenceSwath.burstLength / referenceSwath.prf * 2.0) / referenceSwath.azimuthLineInterval:
                iSecondary = j
                break

        #output zero resampled burst/interferogram if no secondary burst found
        if iSecondary == -1:
            print('\nburst pair, reference: %2d, secondary:  no'%(i+1))
            #output an interferogram with all pixels set to zero
            os.chdir(interferogramDir)
            np.zeros((length, width), dtype=np.complex64).astype(np.complex64).tofile(interferogram[i])
            create_xml(interferogram[i], width, length, 'int')
            os.chdir('../')
            #output a resampled secondary image with all pixels set to zero
            os.chdir(secondaryBurstResampledDir)
            np.zeros((length, width), dtype=np.complex64).astype(np.complex64).tofile(secondaryBurstSlcResampled[i])
            create_xml(secondaryBurstSlcResampled[i], width, length, 'slc')
            os.chdir('../')
            continue
        else:
            print('\nburst pair, reference: %2d, secondary: %3d'%(i+1, iSecondary+1))

        #adjust azimuth offset accordingly, since original azimuth offset assumes reference and secondary start with sensingStart
        azoffBurst -= (secondarySwath.burstSlcFirstLineOffsets[iSecondary]-referenceSwath.burstSlcFirstLineOffsets[i])


        ##########################################################################
        # 2. compute deramp and reramp signals
        ##########################################################################
        cj = np.complex64(1j)
        tbase = (secondaryBurstStartTimesSlc[iSecondary] - (secondaryBurstStartTimesRaw[iSecondary] + \
                 datetime.timedelta(seconds=(secondarySwath.burstLength - 1.0) / 2.0 / secondarySwath.prf))).total_seconds()

        #compute deramp signal
        index1 = np.matlib.repmat(np.arange(widthSecondary), lengthSecondary, 1)
        index2 = np.matlib.repmat(np.arange(lengthSecondary).reshape(lengthSecondary, 1), 1, widthSecondary)
        ka = secondarySwath.azimuthFmrateVsPixel[3] * index1**3 + secondarySwath.azimuthFmrateVsPixel[2] * index1**2 + \
             secondarySwath.azimuthFmrateVsPixel[1] * index1    + secondarySwath.azimuthFmrateVsPixel[0]
        #use the convention that ka > 0
        ka = -ka
        t = tbase + index2*secondarySwath.azimuthLineInterval
        deramp = np.exp(cj * np.pi * (-ka) * t**2)

        #compute reramp signal
        index1 = np.matlib.repmat(np.arange(width), length, 1) + rgoffBurst
        index2 = np.matlib.repmat(np.arange(length).reshape(length, 1), 1, width) + azoffBurst
        ka = secondarySwath.azimuthFmrateVsPixel[3] * index1**3 + secondarySwath.azimuthFmrateVsPixel[2] * index1**2 + \
             secondarySwath.azimuthFmrateVsPixel[1] * index1    + secondarySwath.azimuthFmrateVsPixel[0]
        #use the convention that ka > 0
        ka = -ka
        t = tbase + index2*secondarySwath.azimuthLineInterval
        reramp = np.exp(cj * np.pi * (ka) * t**2)


        ##########################################################################
        # 3. resample secondary burst
        ##########################################################################
        #go to secondary directory to do resampling
        os.chdir(secondaryBurstDir)

        #output offsets
        rgoffBurstFile = "burst_rg.off"
        azoffBurstFile = "burst_az.off"
        rgoffBurst.astype(np.float32).tofile(rgoffBurstFile)
        azoffBurst.astype(np.float32).tofile(azoffBurstFile)

        #deramp secondary burst
        secondaryBurstDerampedFile = "secondary.slc"
        sburst = np.fromfile(secondaryBurstSlc[iSecondary], dtype=np.complex64).reshape(lengthSecondary, widthSecondary)
        (deramp * sburst).astype(np.complex64).tofile(secondaryBurstDerampedFile)
        create_xml(secondaryBurstDerampedFile, widthSecondary, lengthSecondary, 'slc')

        #resampled secondary burst
        secondaryBurstResampFile = 'secondary_resamp.slc'

        #resample secondary burst
        #now doppler has bigger impact now, as it's value is about 35 Hz (azimuth resampling frequency is now only 1/20 * PRF)
        #we don't know if this doppler value is accurate or not, so we set it to zero, which seems to give best resampling result
        #otherwise if it is not accurate and we still use it, it will significantly affect resampling result
        dopplerVsPixel = secondarySwath.dopplerVsPixel
        dopplerVsPixel = [0.0, 0.0, 0.0, 0.0]

        resamp(secondaryBurstDerampedFile, secondaryBurstResampFile, rgoffBurstFile, azoffBurstFile, width, length, 1.0/secondarySwath.azimuthLineInterval, dopplerVsPixel, 
                    rgcoef=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                    azcoef=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                    azpos_off=0.0)

        #read resampled secondary burst and reramp
        sburstResamp = reramp * (np.fromfile(secondaryBurstResampFile, dtype=np.complex64).reshape(length, width))

        #clear up
        os.remove(rgoffBurstFile)
        os.remove(azoffBurstFile)
        os.remove(secondaryBurstDerampedFile)
        os.remove(secondaryBurstDerampedFile+'.vrt')
        os.remove(secondaryBurstDerampedFile+'.xml')
        os.remove(secondaryBurstResampFile)
        os.remove(secondaryBurstResampFile+'.vrt')
        os.remove(secondaryBurstResampFile+'.xml')

        os.chdir('../')


        ##########################################################################
        # 4. dump results
        ##########################################################################
        #dump resampled secondary burst
        os.chdir(secondaryBurstResampledDir)
        sburstResamp.astype(np.complex64).tofile(secondaryBurstSlcResampled[i])
        create_xml(secondaryBurstSlcResampled[i], width, length, 'slc')
        os.chdir('../')

        #dump burst interferogram
        mburst = np.fromfile(os.path.join(referenceBurstDir, referenceBurstSlc[i]), dtype=np.complex64).reshape(length, width)
        os.chdir(interferogramDir)
        (mburst * np.conj(sburstResamp)).astype(np.complex64).tofile(interferogram[i])
        create_xml(interferogram[i], width, length, 'int')
        os.chdir('../')


def create_multi_index(width, rgl):
    import numpy as np
    #create index after multilooking
    #assuming original index start with 0
    #applies to both range and azimuth direction

    widthm = int(width/rgl)

    #create range index: This applies to both odd and even cases, "rgl = 1" case, and "rgl = 2" case
    start_rgindex = (rgl - 1.0) / 2.0
    rgindex0 = start_rgindex + np.arange(widthm) * rgl

    return rgindex0


def create_multi_index2(width2, l1, l2):
    import numpy as np
    #for number of looks of l1 and l2
    #calculate the correponding index number of l2 in the l1 array
    #applies to both range and azimuth direction

    return ((l2 - l1) / 2.0  + np.arange(width2) * l2) / l1










