#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import logging
import numpy as np

import isceobj
from mroipac.ampcor.Ampcor import Ampcor
from isceobj.Alos2Proc.Alos2ProcPublic import runCmd
from contrib.alos2proc.alos2proc import look
from isceobj.Alos2Proc.Alos2ProcPublic import create_xml
from isceobj.Alos2Proc.Alos2ProcPublic import writeOffset
from contrib.alos2proc_f.alos2proc_f import fitoff

logger = logging.getLogger('isce.alos2insar.runRdrDemOffset')

def runRdrDemOffset(self):
    '''estimate between radar image and dem
    '''
    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    self.updateParamemetersFromUser()

    referenceTrack = self._insar.loadTrack(reference=True)
    demFile = os.path.abspath(self._insar.dem)

    insarDir = 'insar'
    os.makedirs(insarDir, exist_ok=True)
    os.chdir(insarDir)

    rdrDemDir = 'rdr_dem_offset'
    os.makedirs(rdrDemDir, exist_ok=True)
    os.chdir(rdrDemDir)

    ##################################################################################################
    #compute dem pixel size
    demImage = isceobj.createDemImage()
    demImage.load(demFile + '.xml')
    #DEM pixel size in meters (appoximate value)
    demDeltaLon = abs(demImage.getDeltaLongitude()) / 0.0002777777777777778 * 30.0
    demDeltaLat = abs(demImage.getDeltaLatitude())  / 0.0002777777777777778 * 30.0

    #number of looks to take in range
    if self._insar.numberRangeLooksSim == None:
        if self._insar.numberRangeLooks1 * referenceTrack.rangePixelSize > demDeltaLon:
            self._insar.numberRangeLooksSim = 1
        else:
            self._insar.numberRangeLooksSim = int(demDeltaLon / (self._insar.numberRangeLooks1 * referenceTrack.rangePixelSize) + 0.5)
    #number of looks to take in azimuth
    if self._insar.numberAzimuthLooksSim == None:
        if self._insar.numberAzimuthLooks1 * referenceTrack.azimuthPixelSize > demDeltaLat:
            self._insar.numberAzimuthLooksSim = 1
        else:
            self._insar.numberAzimuthLooksSim = int(demDeltaLat / (self._insar.numberAzimuthLooks1 * referenceTrack.azimuthPixelSize) + 0.5)

    #simulate a radar image using dem
    simulateRadar(os.path.join('../', self._insar.height), self._insar.sim, scale=3.0, offset=100.0)
    sim = isceobj.createImage()
    sim.load(self._insar.sim+'.xml')

    #take looks
    if (self._insar.numberRangeLooksSim == 1) and (self._insar.numberAzimuthLooksSim == 1):
        simLookFile = self._insar.sim
        ampLookFile = 'amp_{}rlks_{}alks.float'.format(self._insar.numberRangeLooksSim*self._insar.numberRangeLooks1, 
                                                       self._insar.numberAzimuthLooksSim*self._insar.numberAzimuthLooks1)
        cmd = "imageMath.py -e='sqrt(a_0*a_0+a_1*a_1)' --a={} -o {} -t float".format(os.path.join('../', self._insar.amplitude), ampLookFile)
        runCmd(cmd)
    else:
        simLookFile = 'sim_{}rlks_{}alks.float'.format(self._insar.numberRangeLooksSim*self._insar.numberRangeLooks1, 
                                                       self._insar.numberAzimuthLooksSim*self._insar.numberAzimuthLooks1)
        ampLookFile = 'amp_{}rlks_{}alks.float'.format(self._insar.numberRangeLooksSim*self._insar.numberRangeLooks1, 
                                                       self._insar.numberAzimuthLooksSim*self._insar.numberAzimuthLooks1)
        ampTmpFile = 'amp_tmp.float'
        look(self._insar.sim, simLookFile, sim.width, self._insar.numberRangeLooksSim, self._insar.numberAzimuthLooksSim, 2, 0, 1)
        look(os.path.join('../', self._insar.amplitude), ampTmpFile, sim.width, self._insar.numberRangeLooksSim, self._insar.numberAzimuthLooksSim, 4, 1, 1)
 
        width = int(sim.width/self._insar.numberRangeLooksSim)
        length = int(sim.length/self._insar.numberAzimuthLooksSim)
        create_xml(simLookFile, width, length, 'float')
        create_xml(ampTmpFile, width, length, 'amp')

        cmd = "imageMath.py -e='sqrt(a_0*a_0+a_1*a_1)' --a={} -o {} -t float".format(ampTmpFile, ampLookFile)
        runCmd(cmd)
        os.remove(ampTmpFile)
        os.remove(ampTmpFile+'.vrt')
        os.remove(ampTmpFile+'.xml')

    #initial number of offsets to use
    numberOfOffsets = 800
    #compute land ratio to further determine the number of offsets to use
    wbd=np.memmap(os.path.join('../', self._insar.wbdOut), dtype='byte', mode='r', shape=(sim.length, sim.width))
    landRatio = np.sum(wbd[0:sim.length:10, 0:sim.width:10]!=-1) / int(sim.length/10) / int(sim.width/10)
    del wbd
    if (landRatio <= 0.00125):
        print('\n\nWARNING: land area too small for estimating offsets between radar and dem')
        print('do not estimate offsets between radar and dem\n\n')
        self._insar.radarDemAffineTransform = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        catalog.addItem('warning message', 'land area too small for estimating offsets between radar and dem', 'runRdrDemOffset')

        os.chdir('../../')

        catalog.printToLog(logger, "runRdrDemOffset")
        self._insar.procDoc.addAllFromCatalog(catalog)

        return

    #total number of offsets to use
    numberOfOffsets /= landRatio
    #allocate number of offsets in range/azimuth according to image width/length
    width = int(sim.width/self._insar.numberRangeLooksSim)
    length = int(sim.length/self._insar.numberAzimuthLooksSim)
    #number of offsets to use in range/azimuth
    numberOfOffsetsRange = int(np.sqrt(numberOfOffsets * width / length))
    numberOfOffsetsAzimuth = int(length / width * np.sqrt(numberOfOffsets * width / length))

    #this should be better?
    numberOfOffsetsRange = int(np.sqrt(numberOfOffsets))
    numberOfOffsetsAzimuth = int(np.sqrt(numberOfOffsets))


    if numberOfOffsetsRange > int(width/2):
        numberOfOffsetsRange = int(width/2)
    if numberOfOffsetsAzimuth > int(length/2):
        numberOfOffsetsAzimuth = int(length/2)

    if numberOfOffsetsRange < 10:
        numberOfOffsetsRange = 10
    if numberOfOffsetsAzimuth < 10:
        numberOfOffsetsAzimuth = 10

    catalog.addItem('number of range offsets', '{}'.format(numberOfOffsetsRange), 'runRdrDemOffset')
    catalog.addItem('number of azimuth offsets', '{}'.format(numberOfOffsetsAzimuth), 'runRdrDemOffset')

    #matching
    ampcor = Ampcor(name='insarapp_slcs_ampcor')
    ampcor.configure()

    mMag = isceobj.createImage()
    mMag.load(ampLookFile+'.xml')
    mMag.setAccessMode('read')
    mMag.createImage()

    sMag = isceobj.createImage()
    sMag.load(simLookFile+'.xml')
    sMag.setAccessMode('read')
    sMag.createImage()

    ampcor.setImageDataType1('real')
    ampcor.setImageDataType2('real')

    ampcor.setReferenceSlcImage(mMag)
    ampcor.setSecondarySlcImage(sMag)

    #MATCH REGION
    rgoff = 0
    azoff = 0
    #it seems that we cannot use 0, haven't look into the problem
    if rgoff == 0:
        rgoff = 1
    if azoff == 0:
        azoff = 1
    firstSample = 1
    if rgoff < 0:
        firstSample = int(35 - rgoff)
    firstLine = 1
    if azoff < 0:
        firstLine = int(35 - azoff)
    ampcor.setAcrossGrossOffset(rgoff)
    ampcor.setDownGrossOffset(azoff)
    ampcor.setFirstSampleAcross(firstSample)
    ampcor.setLastSampleAcross(width)
    ampcor.setNumberLocationAcross(numberOfOffsetsRange)
    ampcor.setFirstSampleDown(firstLine)
    ampcor.setLastSampleDown(length)
    ampcor.setNumberLocationDown(numberOfOffsetsAzimuth)

    #MATCH PARAMETERS
    ampcor.setWindowSizeWidth(64)
    ampcor.setWindowSizeHeight(64)
    #note this is the half width/length of search area, so number of resulting correlation samples: 8*2+1
    ampcor.setSearchWindowSizeWidth(16)
    ampcor.setSearchWindowSizeHeight(16)

    #REST OF THE STUFF
    ampcor.setAcrossLooks(1)
    ampcor.setDownLooks(1)
    ampcor.setOversamplingFactor(64)
    ampcor.setZoomWindowSize(16)
    #1. The following not set
    #Matching Scale for Sample/Line Directions                       (-)    = 1. 1.
    #should add the following in Ampcor.py?
    #if not set, in this case, Ampcor.py'value is also 1. 1.
    #ampcor.setScaleFactorX(1.)
    #ampcor.setScaleFactorY(1.)

    #MATCH THRESHOLDS AND DEBUG DATA
    #2. The following not set
    #in roi_pac the value is set to 0 1
    #in isce the value is set to 0.001 1000.0
    #SNR and Covariance Thresholds                                   (-)    =  {s1} {s2}
    #should add the following in Ampcor?
    #THIS SHOULD BE THE ONLY THING THAT IS DIFFERENT FROM THAT OF ROI_PAC
    #ampcor.setThresholdSNR(0)
    #ampcor.setThresholdCov(1)
    ampcor.setDebugFlag(False)
    ampcor.setDisplayFlag(False)

    #in summary, only two things not set which are indicated by 'The following not set' above.

    #run ampcor
    ampcor.ampcor()
    offsets = ampcor.getOffsetField()
    ampcorOffsetFile = 'ampcor.off'
    cullOffsetFile = 'cull.off'
    affineTransformFile = 'affine_transform.txt'
    writeOffset(offsets, ampcorOffsetFile)

    #finalize image, and re-create it
    #otherwise the file pointer is still at the end of the image
    mMag.finalizeImage()
    sMag.finalizeImage()

    # #cull offsets
    # import io
    # from contextlib import redirect_stdout
    # f = io.StringIO()
    # with redirect_stdout(f):
    #     fitoff(ampcorOffsetFile, cullOffsetFile, 1.5, .5, 50)
    # s = f.getvalue()
    # #print(s)
    # with open(affineTransformFile, 'w') as f:
    #     f.write(s)

    #cull offsets
    import subprocess
    proc = subprocess.Popen(["python3", "-c", "import isce; from contrib.alos2proc_f.alos2proc_f import fitoff; fitoff('ampcor.off', 'cull.off', 1.5, .5, 50)"], stdout=subprocess.PIPE)
    out = proc.communicate()[0]
    with open(affineTransformFile, 'w') as f:
        f.write(out.decode('utf-8'))

    #check number of offsets left
    with open(cullOffsetFile, 'r') as f:
        numCullOffsets = sum(1 for linex in f)
    if numCullOffsets < 50:
        print('\n\nWARNING: too few points left after culling, {} left'.format(numCullOffsets))
        print('do not estimate offsets between radar and dem\n\n')
        self._insar.radarDemAffineTransform = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        catalog.addItem('warning message', 'too few points left after culling, {} left'.format(numCullOffsets), 'runRdrDemOffset')

        os.chdir('../../')

        catalog.printToLog(logger, "runRdrDemOffset")
        self._insar.procDoc.addAllFromCatalog(catalog)

        return

    #read affine transform parameters
    with open(affineTransformFile) as f:
        lines = f.readlines()
    i = 0
    for linex in lines:
        if 'Affine Matrix ' in linex:
            m11 = float(lines[i + 2].split()[0])
            m12 = float(lines[i + 2].split()[1])
            m21 = float(lines[i + 3].split()[0])
            m22 = float(lines[i + 3].split()[1])
            t1  = float(lines[i + 7].split()[0])
            t2  = float(lines[i + 7].split()[1])
            break
        i += 1    

    self._insar.radarDemAffineTransform = [m11, m12, m21, m22, t1, t2]
    ##################################################################################################

    os.chdir('../../')


    catalog.printToLog(logger, "runRdrDemOffset")
    self._insar.procDoc.addAllFromCatalog(catalog)


def simulateRadar(hgtfile, simfile, scale=3.0, offset=100.0):
    '''
    simulate a radar image by computing gradient of a dem image.
    '''
    import numpy as np
    import isceobj
    from isceobj.Alos2Proc.Alos2ProcPublic import create_xml

    #set chunk length here for efficient processing
    ###############################################
    chunk_length = 1000
    ###############################################

    hgt = isceobj.createImage()
    hgt.load(hgtfile+'.xml')

    chunk_width = hgt.width
    num_chunk = int(hgt.length/chunk_length)
    chunk_length_last = hgt.length - num_chunk * chunk_length

    simData = np.zeros((chunk_length, chunk_width), dtype=np.float32)

    hgtfp = open(hgtfile,'rb')
    simfp = open(simfile,'wb')

    print("simulating a radar image using topography")
    for i in range(num_chunk):
        print("processing chunk %6d of %6d" % (i+1, num_chunk), end='\r', flush=True)
        hgtData = np.fromfile(hgtfp, dtype=np.float64, count=chunk_length*chunk_width).reshape(chunk_length, chunk_width)
        simData[:, 0:chunk_width-1] = scale * np.diff(hgtData, axis=1) + offset
        simData.astype(np.float32).tofile(simfp)

    print("processing chunk %6d of %6d" % (num_chunk, num_chunk))
    if chunk_length_last != 0:
        hgtData = np.fromfile(hgtfp, dtype=np.float64, count=chunk_length_last*chunk_width).reshape(chunk_length_last, chunk_width)
        simData[0:chunk_length_last, 0:chunk_width-1] = scale * np.diff(hgtData, axis=1) + offset
        (simData[0:chunk_length_last, :]).astype(np.float32).tofile(simfp)

    hgtfp.close()
    simfp.close()
    create_xml(simfile, hgt.width, hgt.length, 'float')
