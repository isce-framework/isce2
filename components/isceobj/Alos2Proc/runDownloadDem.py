#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#

import os
import glob
import logging
import numpy as np

import isceobj
from isceobj.Alos2Proc.Alos2ProcPublic import runCmd
from isceobj.Alos2Proc.Alos2ProcPublic import getBboxGeo

logger = logging.getLogger('isce.alos2insar.runDownloadDem')

def runDownloadDem(self):
    '''download DEM and water body
    '''
    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    self.updateParamemetersFromUser()

    referenceTrack = self._insar.loadTrack(reference=True)
    secondaryTrack = self._insar.loadTrack(reference=False)

    bboxGeo = getBboxGeo(referenceTrack)
    bbox = np.array(bboxGeo)
    bboxStr = '{} {} {} {}'.format(np.int(np.floor(bbox[0])), np.int(np.ceil(bbox[1])), np.int(np.floor(bbox[2])), np.int(np.ceil(bbox[3])))


    #get 1 arcsecond dem for coregistration
    if self.dem == None:
        demDir = 'dem_1_arcsec'
        os.makedirs(demDir, exist_ok=True)
        os.chdir(demDir)

        # downloadUrl = 'http://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11'
        # cmd = 'dem.py -a stitch -b {} -k -s 1 -c -f -u {}'.format(
        #        bboxStr,
        #        downloadUrl
        #        )
        # runCmd(cmd)
        # cmd = 'fixImageXml.py -i demLat_*_*_Lon_*_*.dem.wgs84 -f'
        # runCmd(cmd)
        # cmd = 'rm *.hgt* *.log demLat_*_*_Lon_*_*.dem demLat_*_*_Lon_*_*.dem.vrt demLat_*_*_Lon_*_*.dem.xml'
        # runCmd(cmd)

        #replace the above system calls with function calls
        downloadDem(list(bbox), demType='version3', resolution=1, fillingValue=-32768, outputFile=None, userName=None, passWord=None)
        imagePathXml((glob.glob('demLat_*_*_Lon_*_*.dem.wgs84'))[0], fullPath=True)
        filesRemoved = glob.glob('*.hgt*') + glob.glob('*.log') + glob.glob('demLat_*_*_Lon_*_*.dem') + glob.glob('demLat_*_*_Lon_*_*.dem.vrt') + glob.glob('demLat_*_*_Lon_*_*.dem.xml')
        for filex in filesRemoved:
            os.remove(filex)

        os.chdir('../')

        self.dem = glob.glob(os.path.join(demDir, 'demLat_*_*_Lon_*_*.dem.wgs84'))[0]

    #get 3 arcsecond dem for geocoding
    if self.demGeo == None:
        demGeoDir = 'dem_3_arcsec'
        os.makedirs(demGeoDir, exist_ok=True)
        os.chdir(demGeoDir)

        # downloadUrl = 'http://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL3.003/2000.02.11'
        # cmd = 'dem.py -a stitch -b {} -k -s 3 -c -f -u {}'.format(
        #        bboxStr,
        #        downloadUrl
        #        )
        # runCmd(cmd)
        # cmd = 'fixImageXml.py -i demLat_*_*_Lon_*_*.dem.wgs84 -f'
        # runCmd(cmd)
        # cmd = 'rm *.hgt* *.log demLat_*_*_Lon_*_*.dem demLat_*_*_Lon_*_*.dem.vrt demLat_*_*_Lon_*_*.dem.xml'
        # runCmd(cmd)

        #replace the above system calls with function calls
        downloadDem(list(bbox), demType='version3', resolution=3, fillingValue=-32768, outputFile=None, userName=None, passWord=None)
        imagePathXml((glob.glob('demLat_*_*_Lon_*_*.dem.wgs84'))[0], fullPath=True)
        filesRemoved = glob.glob('*.hgt*') + glob.glob('*.log') + glob.glob('demLat_*_*_Lon_*_*.dem') + glob.glob('demLat_*_*_Lon_*_*.dem.vrt') + glob.glob('demLat_*_*_Lon_*_*.dem.xml')
        for filex in filesRemoved:
            os.remove(filex)

        os.chdir('../')

        self.demGeo = glob.glob(os.path.join(demGeoDir, 'demLat_*_*_Lon_*_*.dem.wgs84'))[0]

    #get water body for masking interferogram
    if self.wbd == None:
        wbdDir = 'wbd_1_arcsec'
        os.makedirs(wbdDir, exist_ok=True)
        os.chdir(wbdDir)

        #cmd = 'wbd.py {}'.format(bboxStr)
        #runCmd(cmd)
        download_wbd(np.int(np.floor(bbox[0])), np.int(np.ceil(bbox[1])), np.int(np.floor(bbox[2])), np.int(np.ceil(bbox[3])))
        #cmd = 'fixImageXml.py -i swbdLat_*_*_Lon_*_*.wbd -f'
        #runCmd(cmd)
        #cmd = 'rm *.log'
        #runCmd(cmd)
        
        #replace the above system calls with function calls
        imagePathXml((glob.glob('swbdLat_*_*_Lon_*_*.wbd'))[0], fullPath=True)
        filesRemoved = glob.glob('*.log')
        for filex in filesRemoved:
            os.remove(filex)

        os.chdir('../')

        self.wbd = glob.glob(os.path.join(wbdDir, 'swbdLat_*_*_Lon_*_*.wbd'))[0]

    self._insar.dem = self.dem
    self._insar.demGeo = self.demGeo
    self._insar.wbd = self.wbd


    catalog.printToLog(logger, "runDownloadDem")
    self._insar.procDoc.addAllFromCatalog(catalog)


def downloadDem(bbox, demType='version3', resolution=1, fillingValue=-32768, outputFile=None, userName=None, passWord=None):
    '''
    bbox:        [s, n, w, e]
    demType:     can be 'version3' or 'nasadem'. nasadem is also tested.
    resolution:  1 or 3, NASADEM only available in 1-arc sec resolution
    '''
    import numpy as np
    import isceobj
    from contrib.demUtils import createDemStitcher

    ds = createDemStitcher(demType)
    ds.configure()

    if demType == 'version3':
        if resolution == 1:
            ds._url1 = 'http://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11'
        else:
            ds._url3 = 'http://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL3.003/2000.02.11'
    elif demType == 'nasadem':
        resolution = 1
        #this url is included in the module
        #ds._url1 = 'http://e4ftl01.cr.usgs.gov/MEASURES/NASADEM_HGT.001/2000.02.11'
    else:
        raise Exception('unknown DEM type, currently supported DEM types: version3 and nasadem')

    ds.setUsername(userName)
    ds.setPassword(passWord)

    ds._keepAfterFailed = True
    ds.setCreateXmlMetadata(True)
    ds.setUseLocalDirectory(False)
    ds.setFillingValue(fillingValue)
    ds.setFilling()

    bbox = [np.int(np.floor(bbox[0])), np.int(np.ceil(bbox[1])), np.int(np.floor(bbox[2])), np.int(np.ceil(bbox[3]))]
    if outputFile==None:
        outputFile = ds.defaultName(bbox)

    if not(ds.stitchDems(bbox[0:2],bbox[2:4],resolution,outputFile,'./',keep=True)):
        print('Could not create a stitched DEM. Some tiles are missing')
    else:
        #Apply correction  EGM96 -> WGS84
        demImg = ds.correct()

    #report downloads
    for k,v in list(ds._downloadReport.items()):
        print(k,'=',v)


def download_wbd(s, n, w, e):
    '''
    download water body
    water body. (0) --- land; (-1) --- water; (-2) --- no data.

    set no-value pixel inside of latitude [-56, 60] to -1
    set no-value pixel outside of latitidue [-56, 60] to -2

    look at this figure for SRTM coverage:
    https://www2.jpl.nasa.gov/srtm/images/SRTM_2-24-2016.gif
    '''
    import os
    import numpy as np
    import isceobj
    from iscesys.DataManager import createManager

    latMin = np.floor(s)
    latMax = np.ceil(n)
    lonMin = np.floor(w)
    lonMax = np.ceil(e)

    ############################################################
    #1. download and stitch wbd
    ############################################################
    sw = createManager('wbd')
    sw.configure()

    outputFile = sw.defaultName([latMin,latMax,lonMin,lonMax])
    if os.path.exists(outputFile) and os.path.exists(outputFile+'.xml'):
        print('water body file: {}'.format(outputFile))
        print('exists, do not download and correct')
        return outputFile

    #download and stitch the SWBD tiles
    sw.noFilling = False
    sw._fillingValue = -1
    sw.stitch([latMin,latMax],[lonMin,lonMax])


    ############################################################
    #2. replace 'areas with SRTM but no SWBD' with zeros (land)
    ############################################################
    print('post-process water body file')

    print('get SRTM tiles')
    srtmListFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'srtm_tiles.txt')
    with open(srtmListFile) as f:
        srtmList = f.readlines()
    srtmList = [x[0:7] for x in srtmList]

    #get tiles that have SRTM DEM, but no SWBD, these are mostly tiles that do not have water body
    print('get tiles with SRTM and without SWBD')
    noSwbdListFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'srtm_no_swbd_tiles.txt')
    with open(noSwbdListFile) as f:
        noSwbdList = f.readlines()
    noSwbdList = [x[0:7] for x in noSwbdList]

    print('get SWBD tiles')
    swbdListFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'swbd_tiles.txt')
    with open(swbdListFile) as f:
        swbdList = f.readlines()
    swbdList = [x[0:7] for x in swbdList]


    #read resulting mosaicked water body
    wbdImage = isceobj.createDemImage()
    wbdImage.load(outputFile+'.xml')
    #using memmap instead, which should be faster, since we only have a few pixels to change
    wbd=np.memmap(outputFile, dtype=np.int8, mode='r+', shape=(wbdImage.length, wbdImage.width))

    #replace 'areas with SRTM but no SWBD' with zeros (land)
    names, nlats, nlons = sw.createNameListFromBounds([latMin,latMax],[lonMin,lonMax])
    sign={'S':-1, 'N':1, 'W':-1, 'E':1}
    for tile in names:
        print('checking tile: {}'.format(tile))
        firstLatitude  = sign[tile[0].upper()]*int(tile[1:3])+1
        firstLongitude = sign[tile[3].upper()]*int(tile[4:7])
        lineOffset = np.int32((firstLatitude - wbdImage.firstLatitude) / wbdImage.deltaLatitude + 0.5)
        sampleOffset = np.int32((firstLongitude - wbdImage.firstLongitude) / wbdImage.deltaLongitude + 0.5)

        #first line/sample of mosaicked SWBD is integer lat/lon, but it does not include last integer lat/lon line/sample
        #so here the size is 3600*3600 instead of 3601*3601

        #assuming areas without swbd are water
        if tile[0:7] not in swbdList:
            wbd[0+lineOffset:3600+lineOffset, 0+sampleOffset:3600+sampleOffset] = -1
        #assuming areas with srtm and without swbd are land
        if tile[0:7] in noSwbdList:
            wbd[0+lineOffset:3600+lineOffset, 0+sampleOffset:3600+sampleOffset] = 0


    ############################################################
    #3. set values outside of lat[-56, 60] to -2 (no data)
    ############################################################
    print('check water body file')
    print('set areas outside of lat[-56, 60] to -2 (no data)')
    for i in range(wbdImage.length):
        lat = wbdImage.firstLatitude + wbdImage.deltaLatitude * i
        if lat > 60.0 or lat < -56.0:
            wbd[i, :] = -2
    del wbd, wbdImage


    return outputFile


def imagePathXml(imageFile, fullPath=True):
    import os
    import isceobj
    from isceobj.Util.ImageUtil import ImageLib as IML

    img = IML.loadImage(imageFile)[0]

    dirname  = os.path.dirname(imageFile)
    if fullPath:
        fname = os.path.abspath( os.path.join(dirname, os.path.basename(imageFile)))
    else:
        fname = os.path.basename(imageFile)

    img.filename = fname
    img.setAccessMode('READ')
    img.renderHdr()
