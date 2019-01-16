#!/usr/bin/env python3
# Heresh Fattahi
#

import numpy as np
import argparse
import os
import glob
import isce
import isceobj
import gdal
from gdalconst import GA_ReadOnly
import s1a_isce_utils as ut
from isceobj.Planet.Planet import Planet

GDAL2NUMPY_DATATYPE = {

1 : np.uint8,
2 : np.uint16,
3 : np.int16,
4 : np.uint32,
5 : np.int32,
6 : np.float32,
7 : np.float64,
10: np.complex64,
11: np.complex128,

}

def createParser():
    '''     
    Command line parser.
    '''

    parser = argparse.ArgumentParser( description='filters the densOffset, oversamples it and adds back to the geometry offset')
    parser.add_argument('-i', '--input_directory', dest='input', type=str, default=None,
            help='The directory which contains all pairs (e.g.: ~/hfattahi/process/testSentinel/merged/interferograms). ')
    parser.add_argument('-f', '--file_list', nargs = '+', dest='fileList', type=str, default=None,
                help='A list of files that will be used in pysar e.g.: filt_fine.unw filt_fine.cor')
    parser.add_argument('-o', '--orbit_direction', dest='orbitDirection', type=str, default=None,
                    help='Direction of the orbit: ascending, or descending ')
    parser.add_argument('-x', '--xml_file', dest='xmlFile', type=str, default=None,
                    help='An xml file to extract common metada for the stack: e.g.: master/IW3.xml')
    parser.add_argument('-b', '--baseline_dir', dest='baselineDir', type=str, default=None,
                    help=' directory with baselines ')
    parser.add_argument('-g', '--geometry_dir', dest='geometryDir', type=str, default=None,
                        help=' directory with geometry files ')
    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)

def extractIsceMetadata(xmlFile):
    
    master = ut.loadProduct(xmlFile)
    burst = master.bursts[0]
    burstEnd = master.bursts[-1]
    metadata = {}
    metadata['radarWavelength'] = burst.radarWavelength
    metadata['rangePixelSize'] = burst.rangePixelSize
    metadata['prf'] = burst.prf
    metadata['startUTC'] = burst.burstStartUTC
    metadata['stopUTC'] = burstEnd.burstStopUTC
    metadata['startingRange'] = burst.startingRange
   
    time_seconds = burst.burstStartUTC.hour*3600.0 + burst.burstStartUTC.minute*60.0 + burst.burstStartUTC.second
                    
    metadata['CENTER_LINE_UTC'] = time_seconds
    Vs = np.linalg.norm(burst.orbit.interpolateOrbit(burst.sensingMid, method='hermite').getVelocity())
    metadata['satelliteSpeed'] = Vs
    metadata['azimuthTimeInterval'] = burst.azimuthTimeInterval
    metadata['azimuthPixelSize'] = Vs*burst.azimuthTimeInterval
   
    tstart = burst.sensingStart
    tend   = burstEnd.sensingStop
    tmid = tstart + 0.5*(tend - tstart)

    orbit = burst.orbit
    peg = orbit.interpolateOrbit(tmid, method='hermite')

    refElp = Planet(pname='Earth').ellipsoid
    llh = refElp.xyz_to_llh(peg.getPosition())
    hdg = orbit.getENUHeading(tmid)
    refElp.setSCH(llh[0], llh[1], hdg)

    metadata['earthRadius'] = refElp.pegRadCur

    metadata['altitude'] = llh[2]


    return metadata
def write_rsc(isceFile, dates, metadata, baselineDict):
    rscDict={}

    rscDict['WIDTH'] = metadata['width'] 
    #rscDict['X_FIRST'] =  
    #rscDict['X_STEP'] =
    #rscDict['X_UNIT'] = 

    rscDict['FILE_LENGTH'] = metadata['length'] 
    #rscDict['Y_FIRST'] = 
    #rscDict['Y_STEP'] = 
    #rscDict['Y_UNIT'] = 
    rscDict['WAVELENGTH'] = metadata['radarWavelength'] 
    rscDict['DATE12'] = dates[0][2:] + '-' + dates[1][2:]
    #rscDict['DATE'] = dates[0]

    rscDict['PLATFORM'] = 'Sentinel1'
    rscDict['RANGE_PIXEL_SIZE'] = metadata['rangePixelSize']
    rscDict['AZIMUTH_PIXEL_SIZE'] = metadata['azimuthPixelSize'] 
    rscDict['EARTH_RADIUS'] = metadata['earthRadius']  
    rscDict['CENTER_LINE_UTC'] = metadata['CENTER_LINE_UTC'] 
    rscDict['HEIGHT'] = metadata['altitude'] 
    rscDict['STARTING_RANGE'] = metadata['startingRange']
    rscDict['STARTING_RANGE1'] = metadata['startingRange']
    #rscDict['HEADING'] = 

    #rscDict['LOOK_REF1']=
    #rscDict['LOOK_REF2'] = 
    #rscDict['LAT_REF1'] = 
    #rscDict['LON_REF1'] = 
    #rscDict['LAT_REF2'] =
    #rscDict['LON_REF2'] =
    #rscDict['LAT_REF3'] =
    #rscDict['LON_REF3'] =
    #rscDict['LAT_REF4'] =
    #rscDict['LON_REF4'] = 
    #rscDict['PRF'] = 
    rscDict['ANTENNA_SIDE'] = -1
    #rscDict['HEADING'] = 
    rscDict['ORBIT_DIRECTION'] = metadata['orbitDirection'] 
    rscDict['PROCESSOR'] = 'isce'


    outname = isceFile + '.rsc'
    print('writing ', outname)
    f = open(outname,'w')
    for key in rscDict.keys():
        f.write(key+'    ' + str(rscDict[key])  +'\n')

    f.close()

    outBaselineName = os.path.join(os.path.dirname(isceFile), dates[0][2:] + '_' + dates[1][2:] + '_baseline.rsc')
    f = open(outBaselineName,'w')
    f.write("P_BASELINE_TOP_HDR " + str(baselineDict[dates[1]] - baselineDict[dates[0]]) + '\n')
    f.write("P_BASELINE_BOTTOM_HDR " + str(baselineDict[dates[1]] - baselineDict[dates[0]]) + '\n')
    f.close()

    
    return None

def prepare_stack(inputDir, filePattern, metadata, baselineDict):

    unwDirs = glob.glob(os.path.join(inputDir,'*'))
    isceFile = os.path.join(unwDirs[0], filePattern)
    ds = gdal.Open(isceFile, gdal.GA_ReadOnly)
    length = ds.RasterYSize
    width = ds.RasterXSize
    metadata['length'] = length 
    metadata['width'] = width

    for dirname in unwDirs:
        dates = os.path.basename(dirname).split('_')
        isceFile = os.path.join(dirname,filePattern) #,  metadata)
        print(isceFile)
        write_rsc(isceFile, dates, metadata, baselineDict)
        cmd = "mv " + isceFile + " " + os.path.join(os.path.dirname(isceFile) , "filt_" + dates[0][2:] + '_' + dates[1][2:] + "." + filePattern.split(".")[-1])
        print(cmd)
        os.system(cmd)
        cmd = "mv " + isceFile + ".rsc " + os.path.join(os.path.dirname(isceFile) , "filt_" + dates[0][2:] + '_' + dates[1][2:] + "." + filePattern.split(".")[-1] + ".rsc")
        os.system(cmd)

def read_baseline(baselineFile):
    b=[]
    f = open(baselineFile)
    for line in f:
        l = line.split(":")
        if l[0] == "Bperp (average)":
            b.append(float(l[1]))
    return np.mean(b)

def baselineTimeseries(baselineDir):
    bFiles = glob.glob(os.path.join(baselineDir,'*/*.txt'))
    bFiles = sorted(bFiles)
    bDict={}
    for bFile in bFiles:
        dates = os.path.basename(bFile).split('.txt')[0].split('_')
        bDict[dates[1]] = read_baseline(bFile)

    bDict[dates[0]] = 0
    return bDict

def prepare_geometry(geometryDir):
    demFile = os.path.join(geometryDir, 'hgt.rdr')
    latFile = os.path.join(geometryDir, 'lat.rdr')
    lonFile = os.path.join(geometryDir, 'lon.rdr')
    ds = gdal.Open(demFile, gdal.GA_ReadOnly)
    length = ds.RasterYSize
    width = ds.RasterXSize

    lat = np.memmap(latFile, dtype=np.float64, mode='r', shape=(length,width))
    lon = np.memmap(lonFile, dtype=np.float64, mode='r', shape=(length,width))
    
    print(lat[0,0], lat[0,width-1], lat[length-1,0], lat[length-1,width-1])
    print(lon[0,0], lon[0,width-1], lon[length-1,0], lon[length-1,width-1])
    lat = None
    lon = None
    # This still needs work

def main(iargs=None):

    inps = cmdLineParse(iargs)
    baselineDict = baselineTimeseries(inps.baselineDir)
    metadata = extractIsceMetadata(inps.xmlFile)
    metadata['orbitDirection'] = inps.orbitDirection
    for namePattern in inps.fileList:
        print(namePattern)
        prepare_stack(inps.input, namePattern, metadata, baselineDict)
    
    #prepare_geometry(inps.geometryDir)

if __name__ == '__main__':
    '''
    Main driver.
    '''
    main() 


