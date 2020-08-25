from coregSwathSLCProduct import coregSwathSLCProduct
import isce
import isceobj
import os
#from isceobj.Sensor.TOPS.coregSwathSLCProduct import coregSwathSLCProduct

class catalog(object):
      def __init__(self):
         pass

      def addItem(self,*args):
          print(' '.join([str(x) for x in args]))
          


def loadProduct(xmlname):
        '''
        Load the product using Product Manager.
        '''

        from iscesys.Component.ProductManager import ProductManager as PM

        pm = PM()
        pm.configure()

        obj = pm.loadProduct(xmlname)

        return obj


def saveProduct( obj, xmlname):
        '''
        Save the product to an XML file using Product Manager.
        '''
       # import shelve
       # import os
       # with shelve.open(os.path.dirname(xmlname) + '/'+ os.path.basename(xmlname)  +'.data') as db:
       #     db['data'] = obj

        from iscesys.Component.ProductManager import ProductManager as PM

        pm = PM()
        pm.configure()

        pm.dumpProduct(obj, xmlname)

        return None

def getRelativeShifts(mFrame, sFrame, minBurst, maxBurst, secondaryBurstStart):
    '''
    Estimate the relative shifts between the start of the bursts.
    '''
    import numpy as np    
    azReferenceOff = {}
    azSecondaryOff = {}
    azRelOff = {}
    tm = mFrame.bursts[minBurst].sensingStart
    dt = mFrame.bursts[minBurst].azimuthTimeInterval
    ts = sFrame.bursts[secondaryBurstStart].sensingStart
    
    for index in range(minBurst, maxBurst):
        burst = mFrame.bursts[index]
        azReferenceOff[index] = int(np.round((burst.sensingStart - tm).total_seconds() / dt))
        
        burst = sFrame.bursts[secondaryBurstStart + index - minBurst]
        azSecondaryOff[secondaryBurstStart + index - minBurst] =  int(np.round((burst.sensingStart - ts).total_seconds() / dt))
        
        azRelOff[secondaryBurstStart + index - minBurst] = azSecondaryOff[secondaryBurstStart + index - minBurst] - azReferenceOff[index]

    
    return azRelOff


def adjustValidSampleLine(reference,  minAz=0, maxAz=0, minRng=0, maxRng=0):
    import numpy as np
    import isce
    import isceobj
    # Valid region in the resampled slc based on offsets
    ####Adjust valid samples and first valid sample here
    print ("Adjust valid samples")
    print('Before: ', reference.firstValidSample, reference.numValidSamples)
    print('Offsets : ', minRng, maxRng)
    if (minRng > 0) and (maxRng > 0):
            reference.numValidSamples -= (int(np.ceil(maxRng)) + 8)
            reference.firstValidSample += 4
    elif (minRng < 0) and  (maxRng < 0):
            reference.firstValidSample -= int(np.floor(minRng) - 4)
            reference.numValidSamples += int(np.floor(minRng) - 8)
    elif (minRng < 0) and (maxRng > 0):
            reference.firstValidSample -= int(np.floor(minRng) - 4)
            reference.numValidSamples += int(np.floor(minRng) - 8) - int(np.ceil(maxRng))

    print('After: ', reference.firstValidSample, reference.numValidSamples)

    ###Adjust valid lines and first valid line here
    print ("Adjust valid lines")
    print('Before: ', reference.firstValidLine, reference.numValidLines)
    print('Offsets : ', minAz, maxAz)
    if (minAz > 0) and (maxAz > 0):
            reference.numValidLines -= (int(np.ceil(maxAz)) + 8)
            reference.firstValidLine += 4
    elif (minAz < 0) and  (maxAz < 0):
            reference.firstValidLine -= int(np.floor(minAz) - 4)
            reference.numValidLines += int(np.floor(minAz) - 8)
    elif (minAz < 0) and (maxAz > 0):
            reference.firstValidLine -= int(np.floor(minAz) - 4)
            reference.numValidLines += int(np.floor(minAz) - 8) - int(np.ceil(maxAz))
    print('After:', reference.firstValidLine, reference.numValidLines)


def adjustValidSampleLine_V2(reference, secondary, minAz=0, maxAz=0, minRng=0, maxRng=0): 
    import numpy as np
    import isce
    import isceobj
    ####Adjust valid samples and first valid sample here
    print ("Adjust valid samples")
    print('Before: ', reference.firstValidSample, reference.numValidSamples)
    print('Offsets : ', minRng, maxRng)

    if (minRng > 0) and (maxRng > 0):
        reference.firstValidSample = secondary.firstValidSample - int(np.floor(maxRng)-4)
        lastValidSample = reference.firstValidSample - 8 + secondary.numValidSamples

        if lastValidSample < reference.numberOfSamples:
            reference.numValidSamples = secondary.numValidSamples - 8
        else:
            reference.numValidSamples = reference.numberOfSamples - reference.firstValidSample

    elif (minRng < 0) and (maxRng < 0):
        reference.firstValidSample = secondary.firstValidSample - int(np.floor(minRng) - 4)
        lastValidSample = reference.firstValidSample + secondary.numValidSamples  - 8
        if lastValidSample < reference.numberOfSamples:
            reference.numValidSamples = secondary.numValidSamples - 8
        else:
            reference.numValidSamples = reference.numberOfSamples - reference.firstValidSample
    elif (minRng < 0) and (maxRng > 0):
        reference.firstValidSample = secondary.firstValidSample - int(np.floor(minRng) - 4)
        lastValidSample = reference.firstValidSample + secondary.numValidSamples + int(np.floor(minRng) - 8) - int(np.ceil(maxRng))
        if lastValidSample < reference.numberOfSamples:
            reference.numValidSamples = secondary.numValidSamples + int(np.floor(minRng) - 8) - int(np.ceil(maxRng))
        else:
            reference.numValidSamples = reference.numberOfSamples - reference.firstValidSample

    reference.firstValidSample = np.max([0, reference.firstValidSample])
 
    print('After: ', reference.firstValidSample, reference.numValidSamples)

    ###Adjust valid lines and first valid line here
    print ("Adjust valid lines")
    print('Before: ', reference.firstValidLine, reference.numValidLines)
    print('Offsets : ', minAz, maxAz)

    if (minAz > 0) and (maxAz > 0):

            reference.firstValidLine = secondary.firstValidLine - int(np.floor(maxAz) - 4)
            lastValidLine = reference.firstValidLine - 8  + secondary.numValidLines
            if lastValidLine < reference.numberOfLines:
               reference.numValidLines = secondary.numValidLines - 8
            else:
               reference.numValidLines = reference.numberOfLines - reference.firstValidLine

    elif (minAz < 0) and  (maxAz < 0):
            reference.firstValidLine = secondary.firstValidLine - int(np.floor(minAz) - 4)
            lastValidLine = reference.firstValidLine + secondary.numValidLines + int(np.floor(minAz) - 8)
            lastValidLine = reference.firstValidLine + secondary.numValidLines  - 8
            if lastValidLine < reference.numberOfLines:
               reference.numValidLines = secondary.numValidLines - 8
            else:
               reference.numValidLines = reference.numberOfLines - reference.firstValidLine

    elif (minAz < 0) and (maxAz > 0):
            reference.firstValidLine = secondary.firstValidLine - int(np.floor(minAz) - 4)
            lastValidLine = reference.firstValidLine + secondary.numValidLines + int(np.floor(minAz) - 8) - int(np.ceil(maxAz))
            if lastValidLine < reference.numberOfLines:
               reference.numValidLines = secondary.numValidLines + int(np.floor(minAz) - 8) - int(np.ceil(maxAz))
            else:
               reference.numValidLines = reference.numberOfLines - reference.firstValidLine

    return reference

def adjustCommonValidRegion(reference,secondary):
    # valid lines between reference and secondary


    reference_lastValidLine = reference.firstValidLine + reference.numValidLines - 1
    reference_lastValidSample = reference.firstValidSample + reference.numValidSamples - 1
    secondary_lastValidLine = secondary.firstValidLine + secondary.numValidLines - 1
    secondary_lastValidSample = secondary.firstValidSample + secondary.numValidSamples - 1

    igram_lastValidLine = min(reference_lastValidLine, secondary_lastValidLine)
    igram_lastValidSample = min(reference_lastValidSample, secondary_lastValidSample)

    reference.firstValidLine = max(reference.firstValidLine, secondary.firstValidLine)
    reference.firstValidSample = max(reference.firstValidSample, secondary.firstValidSample)

    #set to 0 to avoid negative values
    if reference.firstValidLine<0:
        reference.firstValidLine=0
    if reference.firstValidSample<0:
        reference.firstValidSample=0

    reference.numValidLines = igram_lastValidLine - reference.firstValidLine + 1
    reference.numValidSamples = igram_lastValidSample - reference.firstValidSample + 1


def getValidLines(secondary, rdict, inname, misreg_az=0.0, misreg_rng=0.0):
    '''
    Looks at the reference, secondary and azimuth offsets and gets the Interferogram valid lines 
    '''
    import numpy as np
    import isce
    import isceobj

    dimg = isceobj.createSlcImage()
    dimg.load(inname + '.xml')
    shp = (dimg.length, dimg.width)
    az = np.fromfile(rdict['azimuthOff'], dtype=np.float32).reshape(shp)
    az += misreg_az
    aa = np.zeros(az.shape)
    aa[:,:] = az
    aa[aa < -10000.0] = np.nan
    amin = np.nanmin(aa)
    amax = np.nanmax(aa)

    rng = np.fromfile(rdict['rangeOff'], dtype=np.float32).reshape(shp)
    rng += misreg_rng
    rr = np.zeros(rng.shape)
    rr[:,:] = rng
    rr[rr < -10000.0] = np.nan
    rmin = np.nanmin(rr)
    rmax = np.nanmax(rr)

    return amin, amax, rmin, rmax



def asBaseClass(inobj):
    '''
    Return as TOPSSwathSLCProduct.
    '''
    from isceobj.Sensor.TOPS.TOPSSwathSLCProduct import TOPSSwathSLCProduct
    
    
    def topsproduct(cobj):
        obj = TOPSSwathSLCProduct()
        obj.configure()

        for x in obj.parameter_list:
            val = getattr(cobj, x.attrname)
            setattr(obj, x.attrname, val)

        for x in obj.facility_list:
            attrname = x.public_name
            val = getattr(cobj, x.attrname)
            setattr(obj, x.attrname, val)
        
        return obj


    if isinstance(inobj, coregSwathSLCProduct):
        return topsproduct(inobj)

    elif isinstance(inobj, TOPSSwathSLCProduct):
        return inobj
    else:
        raise Exception('Cannot be converted to TOPSSwathSLCProduct')

def getSwathList(indir):

    swathList = []
    for x in [1,2,3]:
       SW = os.path.join(indir,'IW{0}'.format(x))
       if os.path.exists(SW):
          swathList.append(x)

    return swathList



