#!/usr/bin/env python3

#
# Author: Cunren Liang
# Copyright 2015-present, NASA-JPL/Caltech
#


def loadInsarUserParameters(filename):
    import os
    from isce.applications.alos2App import Alos2InSAR

    #application object cannot recognize extension
    if filename.endswith('.xml'):
        filename = os.path.splitext(filename)[0]

    #here, Alos2InSAR object is only used for reading and storing parameters
    #none of its other attibutes or functions are used.
    insar = Alos2InSAR(name=filename)
    insar.configure()

    return insar


def loadStackUserParameters(filename):
    import os
    from Stack import Stack

    #application object cannot recognize extension
    if filename.endswith('.xml'):
        filename = os.path.splitext(filename)[0]

    stack = Stack(name=filename)
    stack.configure()

    return stack


def loadInsarProcessingParameters(name):
    import os
    import pickle

    from isceobj.Alos2Proc import Alos2Proc

    try:
        toLoad = Alos2Proc()
        toLoad.load(name + '.xml')
        with open(name, 'rb') as f:
            setattr(toLoad, 'procDoc', pickle.load(f))
    except IOError:
        print("Cannot open %s" % (name))
    
    return toLoad


def dumpInsarProcessingParameters(obj, name):
    import os
    import pickle

    ##############################
    #do this to output important paramters to xml (alos2Proc.xml) after each step.
    #self.renderProcDoc()
    ##############################

    os.makedirs(os.path.dirname(name), exist_ok=True)
    try:
        toDump = obj
        toDump.dump(name + '.xml')
        #dump the procDoc separately
        with open(name, 'wb') as f:
            pickle.dump(getattr(toDump, 'procDoc'), f,
                        protocol=pickle.HIGHEST_PROTOCOL)
    except IOError:
        print("Cannot dump %s" % (name))

    return None



def loadProduct(xmlname):
    '''
    Load the product using Product Manager.
    '''

    from iscesys.Component.ProductManager import ProductManager as PM

    pm = PM()
    pm.configure()

    obj = pm.loadProduct(xmlname)

    return obj


def saveProduct(obj, xmlname):
    '''
    Save the product to an XML file using Product Manager.
    '''
    
    from iscesys.Component.ProductManager import ProductManager as PM

    pm = PM()
    pm.configure()

    pm.dumpProduct(obj, xmlname)
    
    return None


def loadTrack(trackDir, date):
    '''
    Load the track using Product Manager.
    trackDir: where *.track.xml is located
    date: YYMMDD
    '''
    import os
    import glob


    frames = sorted(glob.glob(os.path.join(trackDir, 'f*_*/{}.frame.xml'.format(date))))
    track = loadProduct(os.path.join(trackDir, '{}.track.xml'.format(date)))

    track.frames = []
    for x in frames:
        track.frames.append(loadProduct(x))

    return track


def saveTrack(track, date):
    '''
    Save the track to XML files using Product Manager.
    track: track object
    #trackDir: where *.track.xml is located
    date: YYMMDD
    '''
    import os
    import glob

    #dump track object
    #os.chdir(trackDir)
    saveProduct(track, date+'.track.xml')

    for i in range(len(track.frames)):
        #find frame folder
        frameDirs = sorted(glob.glob('f{}_*'.format(i+1)))
        if frameDirs == []:
            frameDir = 'f{}_{}'.format(i+1, track.frames[i].frameNumber)
            print('no existing frame folder found at frame {}, create a frame folder {}'.format(i+1, frameDir))
        else:
            frameDir = frameDirs[0]
        
        #dump frame object
        if track.frames[i].frameNumber != frameDir[-4:]:
            print('frame number in track object {} is different from that in frame folder name: {} at frame {}'.format(
                track.frames[i].frameNumber, frameDir[-4:], i+1))
            print('dumping it to {}'.format(frameDir))

        os.chdir(frameDir)
        saveProduct(track.frames[i], date+'.frame.xml')
        os.chdir('../')


    return None


def datesFromPairs(pairs):
    dates = []
    for x in pairs:
        dateReference = x.split('-')[0]
        dateSecondary = x.split('-')[1]
        if dateReference not in dates:
            dates.append(dateReference)
        if dateSecondary not in dates:
            dates.append(dateSecondary)
    dates = sorted(dates)
    return dates


def stackDateStatistics(idir, dateReference):
    '''
    idir:          input directory where data of each date is located. only folders are recognized
    dateReference: reference date, str type format: 'YYMMDD'
    '''
    import os
    import glob

    #get date folders
    dateDirs = sorted(glob.glob(os.path.join(os.path.abspath(idir), '*')))
    dateDirs = [x for x in dateDirs if os.path.isdir(x)]

    #find index of reference date:
    dates = []
    dateIndexReference = None
    for i in range(len(dateDirs)):
        date = os.path.basename(dateDirs[i])
        dates.append(date)
        if date == dateReference:
            dateIndexReference = i
    if dateIndexReference is None:
        raise Exception('cannot get reference date {} from the data list, pleasae check your input'.format(dateReference))
    else:
        print('reference date index {}'.format(dateIndexReference))

    #use one date to find frames and swaths. any date should work, here we use dateIndexReference
    frames = sorted([x[-4:] for x in glob.glob(os.path.join(dateDirs[dateIndexReference], 'f*_*'))])
    swaths = sorted([int(x[-1]) for x in glob.glob(os.path.join(dateDirs[dateIndexReference], 'f1_*', 's*'))])

    ndate = len(dates)
    nframe = len(frames)
    nswath = len(swaths)

    #print result
    print('\nlist of dates:')
    print(' index      date            frames')
    print('=======================================================')
    for i in range(ndate):
        if dates[i] == dateReference:
            print('  %03d       %s'%(i, dates[i])+'      {}'.format(frames)+'    reference')
        else:
            print('  %03d       %s'%(i, dates[i])+'      {}'.format(frames))
    print('\n')


    #       str list, str list, str list, int list          int
    return (dateDirs,   dates,   frames,   swaths,   dateIndexReference)



def acquisitionModesAlos2():
    '''
    return ALOS-2 acquisition mode
    '''

    spotlightModes = ['SBS']
    stripmapModes = ['UBS', 'UBD', 'HBS', 'HBD', 'HBQ', 'FBS', 'FBD', 'FBQ']
    scansarNominalModes = ['WBS', 'WBD', 'WWS', 'WWD']
    scansarWideModes = ['VBS', 'VBD']
    scansarModes = ['WBS', 'WBD', 'WWS', 'WWD', 'VBS', 'VBD']

    return (spotlightModes, stripmapModes, scansarNominalModes, scansarWideModes, scansarModes)


def hasGPU():
    '''
    Determine if GPU modules are available.
    '''

    flag = False
    try:
        from zerodop.GPUtopozero.GPUtopozero import PyTopozero
        from zerodop.GPUgeo2rdr.GPUgeo2rdr import PyGeo2rdr
        flag = True
    except:
        pass

    return flag


class createObject(object):
    pass


def subbandParameters(track):
    '''
    compute subband parameters
    '''
    #speed of light from: components/isceobj/Planet/AstronomicalHandbook.py
    SPEED_OF_LIGHT = 299792458.0

    #using 1/3, 1/3, 1/3 band split
    radarWavelength = track.radarWavelength
    rangeBandwidth = track.frames[0].swaths[0].rangeBandwidth
    rangeSamplingRate = track.frames[0].swaths[0].rangeSamplingRate
    radarWavelengthLower = SPEED_OF_LIGHT/(SPEED_OF_LIGHT / radarWavelength - rangeBandwidth / 3.0)
    radarWavelengthUpper = SPEED_OF_LIGHT/(SPEED_OF_LIGHT / radarWavelength + rangeBandwidth / 3.0)
    subbandRadarWavelength = [radarWavelengthLower, radarWavelengthUpper]
    subbandBandWidth = [rangeBandwidth / 3.0 / rangeSamplingRate, rangeBandwidth / 3.0 / rangeSamplingRate]
    subbandFrequencyCenter = [-rangeBandwidth / 3.0 / rangeSamplingRate, rangeBandwidth / 3.0 / rangeSamplingRate]

    subbandPrefix = ['lower', 'upper']

    return (subbandRadarWavelength, subbandBandWidth, subbandFrequencyCenter, subbandPrefix)


def formInterferogram(slcReference, slcSecondary, interferogram, amplitude, numberRangeLooks, numberAzimuthLooks):
    import numpy as np
    import isce, isceobj
    from isceobj.Alos2Proc.Alos2ProcPublic import multilook
    from isceobj.Alos2Proc.Alos2ProcPublic import create_xml

    img = isceobj.createImage()
    img.load(slcReference+'.xml')
    width = img.width
    length = img.length

    width2 = int(width / numberRangeLooks)
    length2 = int(length / numberAzimuthLooks)

    fpRef = open(slcReference,'rb')
    fpSec = open(slcSecondary,'rb')
    fpInf = open(interferogram,'wb')
    fpAmp = open(amplitude,'wb')

    for k in range(length2):
        if (((k+1)%200) == 0):
            print("processing line %6d of %6d" % (k+1, length2), end='\r', flush=True)
        ref = np.fromfile(fpRef, dtype=np.complex64, count=numberAzimuthLooks * width).reshape(numberAzimuthLooks, width)
        sec = np.fromfile(fpSec, dtype=np.complex64, count=numberAzimuthLooks * width).reshape(numberAzimuthLooks, width)
        inf = multilook(ref*np.conjugate(sec), numberAzimuthLooks, numberRangeLooks, mean=False)
        amp = np.sqrt(multilook(ref.real*ref.real+ref.imag*ref.imag, numberAzimuthLooks, numberRangeLooks, mean=False)) + 1j * \
              np.sqrt(multilook(sec.real*sec.real+sec.imag*sec.imag, numberAzimuthLooks, numberRangeLooks, mean=False))
        index = np.nonzero( (np.real(amp)==0) + (np.imag(amp)==0) )
        amp[index]=0
        inf.tofile(fpInf)
        amp.tofile(fpAmp)
    print("processing line %6d of %6d" % (length2, length2))
    fpRef.close()
    fpSec.close()
    fpInf.close()
    fpAmp.close()

    create_xml(interferogram, width2, length2, 'int')
    create_xml(amplitude, width2, length2, 'amp')

