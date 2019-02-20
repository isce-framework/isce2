#!/usr/bin/env python3

###Our usual import statements
import numpy as np
import isce
import isceobj
from stdproc.model.enu2los.ENU2LOS import ENU2LOS
import argparse

####Method to load pickle information
####from an insarApp run
def load_pickle(step='topo'):
    '''Loads the pickle from correct as default.'''
    import cPickle
    
    insarObj = cPickle.load(open('PICKLE/{0}'.format(step),'rb'))
    return insarObj


###Create dummy model file if needed
###Use this for simple testing
###Modify values as per your test dataset

def createDummyModel():
    '''Creates a model image.'''
    wid = 401
    lgt = 401
    startLat = 20.0
    deltaLat = -0.025
    startLon = -156.0
    deltaLon = 0.025

    data = np.zeros((lgt,3*wid), dtype=np.float32)
    ###East only
#    data[:,0::3] = 1.0  
    ###North only
#    data[:,1::3] = 1.0
    ###Up only
    data[:,2::3] = 1.0

    data.tofile('model.enu')

    print('Creating model object')
    objModel = isceobj.createDemImage()
    objModel.setFilename('model.enu')
    objModel.setWidth(wid)
    objModel.scheme = 'BIP'
    objModel.setAccessMode('read')
    objModel.imageType='bip'
    objModel.dataType='FLOAT'
    objModel.bands = 3
    dictProp = {'REFERENCE':'WGS84','Coordinate1': \
                   {'size':wid,'startingValue':startLon,'delta':deltaLon}, \
                   'Coordinate2':{'size':lgt,'startingValue':startLat, \
                   'delta':deltaLat},'FILE_NAME':'model.enu'}
    objModel.init(dictProp)
    objModel.renderHdr()





###cmd Line Parser
def cmdLineParser():
    parser = argparse.ArgumentParser(description="Project ENU deformation to LOS in radar coordinates")
    parser.add_argument('-m','--model', dest='model', type=str,
        required=True, 
        help='Input 3 channel FLOAT model file with DEM like info')
    parser.add_argument('-o','--output', dest='output', type=str,
        default='enu2los.rdr', help='Output 1 channel LOS file')
        
    return parser.parse_args()
    
###The main program
if __name__ == '__main__':
    
    ###Parse command line
    inps = cmdLineParser()
    
    ###For testing only
#    createDummyModel()

    ####Load model image
    print('Creating model image')
    modelImg = isceobj.createDemImage()
    modelImg.load(inps.model +'.xml')  ##From cmd line
    
    if (modelImg.bands !=3 ):
        raise Exception('Model input file should be a 3 band image.')
        
    modelImg.setAccessMode('read')
    modelImg.createImage()
    
    
    ####Get geocoded information
    startLon = modelImg.coord1.coordStart
    deltaLon = modelImg.coord1.coordDelta
    startLat = modelImg.coord2.coordStart
    deltaLat = modelImg.coord2.coordDelta
    
    ####Load geometry information from pickle file.
    iObj = load_pickle()
    topo = iObj.getTopo()   #Get info for the dem in radar coords
    
    ####Get the wavelength information.
    ###This is available in multiple locations within insarProc
    #wvl = iObj.getMasterFrame().getInstrument().getRadarWavelength()
    wvl = topo.radarWavelength
    
    
    ####Pixel-by-pixel Latitude image
    print('Creating lat image')
    objLat = isceobj.createImage()
    objLat.load(topo.latFilename+'.xml')
    objLat.setAccessMode('read')
    objLat.createImage()
    
    ####Pixel-by-pixel Longitude image
    print('Creating lon image')
    objLon = isceobj.createImage()
    objLon.load(topo.lonFilename+'.xml')
    objLon.setAccessMode('read')
    objLon.createImage()
    
    #####Pixel-by-pixel LOS information
    print('Creating LOS image')
    objLos = isceobj.createImage()
    objLos.load(topo.losFilename +'.xml')
    objLos.setAccessMode('read')
    objLos.createImage()
    
    ###Check if dimensions are the same
    for img in (objLon, objLos):
        if (img.width != objLat.width) or (img.length != objLat.length):
            raise Exception('Lat, Lon and LOS files are not of the same size.')
            
    
    ####Create an output object
    print ('Creating output image')
    objOut = isceobj.createImage()
    objOut.initImage(inps.output, 'write', objLat.width, type='FLOAT')
    objOut.createImage()
    
    
    print('Actual processing')
    ####The actual processing
    #Stage 1: Construction
    converter = ENU2LOS()
    converter.configure()
    
    #Stage 2: No ports for enu2los
    #Stage 3: Set values 
    converter.setWidth(objLat.width)   ###Radar coords width
    converter.setNumberLines(objLat.length) ###Radar coords length
    converter.setGeoWidth(modelImg.width) ###Geo coords width
    converter.setGeoNumberLines(modelImg.length) ###Geo coords length
    
    ###Set up geo information
    converter.setStartLatitude(startLat)
    converter.setStartLongitude(startLon)
    converter.setDeltaLatitude(deltaLat)
    converter.setDeltaLongitude(deltaLon)
    
    ####Set up output scaling
    converter.setScaleFactor(1.0)   ###Change if ENU not in meters
    converter.setWavelength(4*np.pi)    ###Wavelength for conversion to radians
    
    converter.enu2los(modelImage = modelImg,
                      latImage = objLat,
                      lonImage = objLon,
                      losImage = objLos,
                      outImage = objOut)
                      
    #Step 4: Close the images
    modelImg.finalizeImage()
    objLat.finalizeImage()
    objLon.finalizeImage()
    objLos.finalizeImage()
    objOut.finalizeImage()
    objOut.renderHdr()    ###Create output XML file

